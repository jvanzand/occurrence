## Perform MCMC simulation for different power law models
import os
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Callable, Tuple

import emcee
import multiprocessing as mp
from scipy.optimize import minimize

import matplotlib.pyplot as plt


@dataclass(frozen=True)
class ModelSpec:
    """Functions and display metadata associated with one model."""

    function: Callable
    ndim: int
    parameter_names: Tuple[str, ...]
    color: str
    initializer: Callable
    prior: Callable

def mcmc(hist_dict,
         model_func_name,
         stack_dim,
         stack_ind=0,
         nwalkers=50,
         nsteps=5000,
         burnin=1000,
         parallel=False,
         save_path="chains.npz",
         random_seed=None,
         ):
    """Fit a registered model to histogram summaries with ``emcee``.

    ``hist_dict`` contains bin centers, occurrence-rate densities, asymmetric
    uncertainties, and the fitted domain. Supplying ``random_seed`` makes
    initialization and sampling reproducible without changing unseeded runs.
    """
    os.makedirs(Path(save_path).parent, exist_ok=True)
    model_spec = get_model_spec(model_func_name)
    rng = np.random.RandomState(random_seed) if random_seed is not None else np.random
    loglik_args = (hist_dict, model_spec.function, model_func_name)

    theta_init = initial_params(model_func_name, hist_dict, random_state=rng)
    pos = theta_init + 1e-3 * rng.randn(nwalkers, model_spec.ndim)
    
    
    # ---- Set up sampler ----
    if parallel:
        with mp.Pool() as pool:
            sampler = emcee.EnsembleSampler(
                nwalkers, model_spec.ndim, loglik_power,
                args=loglik_args, pool=pool
            )

            if random_seed is not None:
                sampler.random_state = rng.get_state()

            # Burn-in
            pos, _, _ = sampler.run_mcmc(pos, burnin, progress=True)
            sampler.reset()

            # Production
            sampler.run_mcmc(pos, nsteps, progress=True)

    else:
        sampler = emcee.EnsembleSampler(nwalkers, model_spec.ndim, loglik_power,
                                        args=loglik_args)

        if random_seed is not None:
            sampler.random_state = rng.get_state()

        # Burn-in
        pos, _, _ = sampler.run_mcmc(pos, burnin, progress=True)
        sampler.reset()

        # Production
        sampler.run_mcmc(pos, nsteps, progress=True)

    # ---- Extract chains ----
    chains = sampler.get_chain()        # shape: (nsteps, nwalkers, ndim)
    log_probs = sampler.get_log_prob()  # same shape

    # Flattened version (often what you want)
    flat_chains = sampler.get_chain(flat=True)
    flat_log_probs = sampler.get_log_prob(flat=True)

    # ---- Save safely (no pickle!) ----
    np.savez_compressed(
        save_path,
        chains=chains,
        log_probs=log_probs,
        flat_chains=flat_chains,
        flat_log_probs=flat_log_probs,
    )

    return sampler
    

def loglik_power(theta, hist_dict, model_func, model_name):
    """
    Log likelihood of a parametric model in
    the M dimension, given a catalog of companion
    posterior draws in the form of a dictionary

    Arguments:
        theta (array of floats): array of model parameters for model_func


        nstars (int): Number of host stars in the sample
        comp_names (list of str): List of companion names
            Must correspond to keys in bin_lam_dict
        bin_lam_dict (dict): Compressed representation of
            posterior samples
        num_cells (int): Number of occurrence cells. Equal to len(lam)
        all_binsizes (array of floats): Logarithmic size of each cell
            in base 10. E.g. if a = 1-10 AU and M = 1-100 M_earth, then
            binsize is 1*2 = 2


        fine_compl_grid (array of floats): Compl value at each M value on
            on a fine grid between min_M and max_M. Compls are from avg. map
    """
    
    AorM_min, AorM_max = hist_dict['lims']
    logprior = log_prior(model_name, theta, AorM_min, AorM_max)
    if np.isinf(logprior):
        return -np.inf 

    model_ORD_vals = model_func(theta, hist_dict['bin_centers'])
    
    # Implement the split gaussian here. If model value is greater, 
    ## use the upper error. If it's lower, use lower err.
    selected_errs = np.where(model_ORD_vals > hist_dict['ORD_vals'],
                             hist_dict['ORD_errs_high'],
                             hist_dict['ORD_errs_low'],
                            )

    loglik = -np.sum((hist_dict['ORD_vals']-model_ORD_vals)**2 / (2*selected_errs**2))

    if np.isnan(loglik):
        print("mcmc_powerlaw.py: Error encountered in loglik", model_ORD_vals, selected_errs)
        import pdb; pdb.set_trace()

    return loglik
    

def FlatLine(theta, AorM):
    """
    Simple constant function for DBIC calc
    """
    
    return theta[0]
    


def PiecewisePower1(theta, AorM):
    """
    Linear model in log-x space:
        y = m * log10(x) + b

    Parameters
    ----------
    theta : tuple
        (m, b) slope and intercept
    x : array-like
        Input values (must be > 0)

    Returns
    -------
    y : array
    """
    slope, y_intercept = theta
    AorM = np.asarray(AorM)

    return slope * np.log10(AorM) + y_intercept



def PiecewisePower2(theta, AorM):
    """
    Broken-line model in log-x space with guaranteed positivity.

    Parameters
    ----------
    theta : (m1, m2, b, log_xt)
        m1 : slope below transition
        m2 : slope above transition
        b  : intercept at log10(x)=0
        log_xt : log10 transition point

    AorM : array-like (>0)

    Returns
    -------
    lam : array (>=0)
    """
    m1, m2, b, log_xt = theta

    logx = np.log10(AorM)

    # hinge function → broken line
    f = b + m1 * logx + (m2 - m1) * np.maximum(0, logx - log_xt)

    return f

def log_gaussian(theta, AorM):
    A, mu, sigma = theta # Sample mu uniformly so that e^mu (ie log-normal median) is sampled logarithmically
    
    return A*np.exp(-(np.log10(AorM)-mu)**2 / (2*sigma**2))

def step(theta, AorM):
    """
    Step model in log-x space: two horizontal lines with one breakpoint.
    
    The model consists of two segments in semi-log space:
    - Horizontal line at C1 for x < breakpoint
    - Horizontal line at C2 for x >= breakpoint
    
    Parameters
    ----------
    theta : (C1, C2, log_bp)
        C1 : occurrence rate at low x values
        C2 : occurrence rate at high x values
        log_bp : ln breakpoint (x value)
    
    AorM : array-like (>0)
        Input values (must be > 0)
    
    Returns
    -------
    lam : array
        Occurrence rate at each input value
    """
    C1, C2, log_bp = theta
    AorM = np.asarray(AorM)
    
    logx = np.log10(AorM)
    
    # Initialize output
    lam = np.zeros_like(logx, dtype=float)
    
    # Region 1: x < bp → y = C1
    mask1 = logx < log_bp
    lam[mask1] = C1
    
    # Region 2: x >= bp → y = C2
    mask2 = logx >= log_bp
    lam[mask2] = C2
    
    return lam


def escarpment(theta, AorM):
    """
    Escarpment model in log-x space: two horizontal lines connected by a sloped line.
    
    The model consists of three segments in semi-log space:
    - Horizontal line at C1 for x < breakpoint1
    - Sloped line from C1 to C2 for breakpoint1 <= x <= breakpoint2
    - Horizontal line at C2 for x > breakpoint2
    
    Parameters
    ----------
    theta : (C1, C2, bp1, bp2)
        C1 : occurrence rate at low x values
        C2 : occurrence rate at high x values
        bp1 : first breakpoint (x value)
        bp2 : second breakpoint (x value)
    
    AorM : array-like (>0)
        Input values (must be > 0)
    
    Returns
    -------
    lam : array
        Occurrence rate at each input value
    """
    C1, C2, log_bp1, log_bp2 = theta
    AorM = np.asarray(AorM)
    
    logx = np.log10(AorM)
    
    # Initialize output
    lam = np.zeros_like(logx, dtype=float)
    
    # Region 1: x < bp1 → y = C1
    mask1 = logx < log_bp1
    lam[mask1] = C1
    
    # Region 2: bp1 <= x <= bp2 → linear interpolation in log space
    mask2 = (logx >= log_bp1) & (logx <= log_bp2)
    if np.any(mask2):
        t = (logx[mask2] - log_bp1) / (log_bp2 - log_bp1)
        lam[mask2] = C1 + (C2 - C1) * t
    
    # Region 3: x > bp2 → y = C2
    mask3 = logx > log_bp2
    lam[mask3] = C2
    
    return lam


def brokenpowerlaw(theta, AorM):
    C, log_a0, beta, gamma = theta
    a0 = 10**log_a0
    return C*AorM**beta * (1 - np.exp(-(AorM/a0)**gamma))



def _init_flat(hist_dict, random_state):
    return np.array([np.mean(hist_dict['ORD_vals'])])


def _init_pp1(hist_dict, random_state):
    min_xval, max_xval = hist_dict['lims']
    ORD_vals = hist_dict['ORD_vals']
    slope = ((ORD_vals[-1] - ORD_vals[0]) /
             (np.log10(max_xval) - np.log10(min_xval)))
    intercept = ORD_vals[-1] - slope*np.log10(max_xval)
    return np.array([slope, intercept])


def _init_pp2(hist_dict, random_state):
    bin_centers = hist_dict['bin_centers']
    min_xval, max_xval = hist_dict['lims']
    ORD_vals = hist_dict['ORD_vals']
    center_val = (min_xval*max_xval)**0.5
    center_bin_ind = np.argmin(abs(bin_centers/center_val - 1))
    center_ORD = ORD_vals[center_bin_ind]
    slope1 = ((center_ORD - ORD_vals[0]) /
              (np.log10(center_val) - np.log10(min_xval)))
    intercept = center_ORD - slope1*np.log10(center_val)
    slope2 = ((ORD_vals[-1] - center_ORD) /
              (np.log10(max_xval) - np.log10(center_val)))
    return np.array([slope1, slope2, intercept, np.log10(center_val)])


def _init_log_gaussian(hist_dict, random_state):
    bin_centers = hist_dict['bin_centers']
    amplitude = np.mean(hist_dict['ORD_vals'])
    mu = np.mean(np.log10(bin_centers))
    sigma = np.log10((bin_centers[-1]/bin_centers[0])**0.25)
    return np.array([amplitude, mu, sigma])


def _init_step(hist_dict, random_state):
    min_xval, max_xval = hist_dict['lims']
    ORD_vals = hist_dict['ORD_vals']
    return np.array([
        np.min(ORD_vals),
        np.max(ORD_vals),
        np.log10((min_xval*max_xval)**0.5),
    ])


def _init_escarpment(hist_dict, random_state):
    bin_centers = hist_dict['bin_centers']
    min_xval, max_xval = hist_dict['lims']
    ORD_vals = hist_dict['ORD_vals']
    log_min, log_max = np.log10(min_xval), np.log10(max_xval)
    log_bp1 = log_min + (log_max - log_min)/3
    log_bp2 = log_min + 2*(log_max - log_min)/3
    middle_mask = (
        (np.log10(bin_centers) >= log_bp1) &
        (np.log10(bin_centers) <= log_bp2)
    )
    slope, intercept = np.polyfit(
        np.log10(bin_centers[middle_mask]), ORD_vals[middle_mask], 1
    )
    C1 = slope*log_bp1 + intercept
    C2 = slope*log_bp2 + intercept
    median_ORD = np.median(ORD_vals)
    C1 = median_ORD if C1 < 0 or C1 > 1 else C1
    C2 = median_ORD if C2 < 0 or C2 > 1 else C2
    return np.array([C1, C2, log_bp1, log_bp2])


def _init_broken_powerlaw(hist_dict, random_state):
    min_xval, max_xval = hist_dict['lims']
    return np.array([
        np.max(hist_dict['ORD_vals']),
        np.log10((min_xval*max_xval)**0.5),
        random_state.uniform(-0.5, 0.5),
        random_state.uniform(-0.5, 0.5),
    ])


def _prior_flat(theta, x_min, x_max):
    return -np.inf if FlatLine(theta, x_min) < 0 else 0.0


def _prior_pp1(theta, x_min, x_max):
    endpoints = [PiecewisePower1(theta, x_min), PiecewisePower1(theta, x_max)]
    return -np.inf if min(endpoints) < 0 or max(endpoints) > 1 else 0.0


def _prior_pp2(theta, x_min, x_max):
    m1, m2, _, log_xt = theta
    values = [
        PiecewisePower2(theta, x_min),
        PiecewisePower2(theta, x_max),
        PiecewisePower2(theta, 10**log_xt),
    ]
    if min(values) < 0 or not np.log10(x_min) <= log_xt <= np.log10(x_max):
        return -np.inf
    if abs(m1) > 100 or abs(m2) > 100:
        return -np.inf
    return -np.log10(abs(m1)) - np.log10(abs(m2))


def _prior_log_gaussian(theta, x_min, x_max):
    amplitude, mu, sigma = theta
    if amplitude < 0 or amplitude > 10*x_max:
        return -np.inf
    if mu < np.log10(x_min) - 15 or mu > np.log10(x_max) + 15:
        return -np.inf
    if sigma < 0 or sigma > np.log10(x_max/x_min):
        return -np.inf
    return 0.0


def _prior_step(theta, x_min, x_max):
    C1, C2, log_bp = theta
    breakpoint = 10**log_bp
    if C1 < 0 or C2 < 0 or C1 > 1 or C2 > 1:
        return -np.inf
    return 0.0 if x_min <= breakpoint <= x_max else -np.inf


def _prior_escarpment(theta, x_min, x_max):
    C1, C2, log_bp1, log_bp2 = theta
    bp1, bp2 = 10**log_bp1, 10**log_bp2
    if C1 < 0 or C2 < 0 or C1 > 1 or C2 > 1:
        return -np.inf
    if bp1 >= bp2:
        return -np.inf
    return 0.0 if x_min <= bp1 <= x_max and x_min <= bp2 <= x_max else -np.inf


def _prior_broken_powerlaw(theta, x_min, x_max):
    C, log_a0, beta, gamma = theta
    if C <= 0 or C > 2:
        return -np.inf
    if abs(log_a0) > 10 or abs(beta) > 4 or abs(gamma) > 4:
        return -np.inf
    return 0.0


MODEL_REGISTRY = {
    'flat': ModelSpec(FlatLine, 1, ('C',), 'black', _init_flat, _prior_flat),
    'pp1': ModelSpec(PiecewisePower1, 2, ('y', 'b'), 'cyan', _init_pp1, _prior_pp1),
    'pp2': ModelSpec(
        PiecewisePower2, 4,
        ('m1', 'm2', 'b1', r'$\log_{10}(x_t)$'), 'goldenrod',
        _init_pp2, _prior_pp2,
    ),
    'logG': ModelSpec(
        log_gaussian, 3, ('A', r'$\mu$', r'$\sigma$'), 'tomato',
        _init_log_gaussian, _prior_log_gaussian,
    ),
    'step': ModelSpec(
        step, 3, ('C1', 'C2', r'$\log_{10}(x_t)$'), 'forestgreen',
        _init_step, _prior_step,
    ),
    'escarpment': ModelSpec(
        escarpment, 4,
        ('C1', 'C2', r'$\log_{10}(x_{t,1})$', r'$\log_{10}(x_{t,2})$'),
        'RoyalBlue', _init_escarpment, _prior_escarpment,
    ),
    'bpl': ModelSpec(
        brokenpowerlaw, 4,
        ('C', r'$\log_{10}(a_0)$', r'$\beta$', r'$\gamma$'), 'deeppink',
        _init_broken_powerlaw, _prior_broken_powerlaw,
    ),
}

# Backward-compatible plotting metadata. New code should use MODEL_REGISTRY.
model_dict = {
    name: [spec.function.__name__, spec.ndim, list(spec.parameter_names), spec.color]
    for name, spec in MODEL_REGISTRY.items()
}


def get_model_spec(model_func_name):
    """Return the registered specification for ``model_func_name``."""
    try:
        return MODEL_REGISTRY[model_func_name]
    except KeyError:
        raise ValueError(f"Unknown model: {model_func_name}")


def initial_params(model_func_name, hist_dict, random_state=None):
    """Generate the legacy histogram-based initial guess for a model."""
    rng = np.random if random_state is None else random_state
    return get_model_spec(model_func_name).initializer(hist_dict, rng)


def log_prior(model_func_name, theta, AorM_min, AorM_max):
    """Evaluate the registered model's legacy parameter prior."""
    return get_model_spec(model_func_name).prior(theta, AorM_min, AorM_max)


def print_power_hard_coded(hist_dict, model_func_name, parameter_sets):
    """Print legacy histogram likelihoods for supplied parameter vectors."""
    print("\n" + "="*60)
    print(f"Hard-coded {model_func_name} Parameter Likelihoods")
    print("="*60)

    model_spec = get_model_spec(model_func_name)
    for i, theta in enumerate(parameter_sets):
        loglik = loglik_power(
            theta, hist_dict, model_spec.function, model_func_name
        )
        param_str = ', '.join(
            f"{name}={val:7.3f}"
            for name, val in zip(model_spec.parameter_names, theta)
        )
        print(f"Set {i+1}: {param_str}  →  lnL = {loglik:10.2f}")

    print("="*60 + "\n")



def calculate_bic(tier123_dir, model_func_name,
                  hist_dict, model_dict, 
                  stack_ind, init_type, 
                  verbose=True):
    """
    Calculate the Bayesian Information Criterion (BIC) for a power law model.
    
    BIC = k * ln(n) - 2 * ln(L_max)
    where:
        k = number of parameters
        n = number of data points (nstars)
        L_max = maximum likelihood
    
    Arguments:
        model_func_name (str): Name of model ('pp1', 'pp2', 'escarpment')
        nstars (int): Number of host stars (treated as number of data points)
        comp_names_inROI (list of str): Companion names in region of interest
        ROIsamples_dict (dict): ROI sample data for companions
        ROIweights_dict (dict): ROI weights for companions
        a_lims (tuple): Semi-major axis limits [a_min, a_max]
        m_lims (tuple): Mass limits [m_min, m_max]
        stack_dim (str): Stack dimension ('a' or 'm')
        interp_fn_avg: Completeness interpolation function
        bin_idx (int or None): Bin index for multi-bin fits. If None, uses old single-bin naming.
        path_to_chains (str): Path to pre-calculated MCMC chains (*.npz file)
    
    Returns:
        bic (float): Bayesian Information Criterion value
        loglik_max (float): Maximum likelihood value
        params_mle (ndarray): Parameters at maximum likelihood
    """
    
    model_spec = get_model_spec(model_func_name)
    model_func = model_spec.function
    ndim = model_spec.ndim

    # Set up arguments for optimization
    optim_args = (
        hist_dict,
        model_func,
        model_func_name,
    )
    
    #import pdb; pdb.set_trace()
    
    if init_type=='chains':
        ## Alternative theta_init: use the max-likelihood params from MCMC
        chain_path = os.path.join(tier123_dir, 'saved_chains/', f'chains_{model_func_name}_bin{stack_ind}.npz')
        chain = np.load(chain_path)['flat_chains']
        theta_init = np.median(chain, axis=0)
    
    elif init_type=='guess':
        theta_init = initial_params(model_func_name, hist_dict)
        
    # Optimize using Powell algorithm (derivative-free)
    # Note: we minimize negative log-likelihood (maximize likelihood)
    result = minimize(
        lambda theta: -loglik_power(theta, *optim_args),
        theta_init,
        method='Powell',
        options={'maxiter': 5000}
    )

    params_mle = result.x
    loglik_max = -result.fun  # Convert back to log-likelihood
    n_data = len(hist_dict['bin_centers']) # Number of histogram bins is the number of "data points"
    # Calculate BIC
    # BIC = k * ln(n) - 2 * ln(L)
    # where L is the likelihood (not log-likelihood), so:
    # BIC = k * ln(n) - 2 * ln(L) = k * ln(n) - 2 * loglik
    bic = ndim * np.log(n_data) - 2 * loglik_max

    if verbose:
        print(f"\n" + "="*70)
        print(f"BIC Calculation for {model_func_name.upper()}")
        print("="*70)
        print(f"Number of parameters (k): {ndim}")
        print(f"Number of data points (n): {n_data}")
        print(f"Maximum log-likelihood: {loglik_max:.4f}")
        print(f"BIC = {ndim} * ln({n_data}) - 2 * {loglik_max:.4f}")
        print(f"BIC = {bic:.4f}")
        print(f"Parameters at MLE: {params_mle}")
        print("="*70 + "\n")
    
    return model_func_name, bic, loglik_max, params_mle



def plot_bics_on_histogram(tier123_dir, model_bic_dict, model_dict,
                           stack_dim, m_unit):
    """
    Plot max-likelihood models (from BIC calculations) overlaid on the ORD histogram.
    
    Arguments:
        tier123_dir (str): Directory path in format 'tier1/tier2/tier3' for loading data
        bic_params_list (list): List of tuples from calculate_bic() outputs, where each tuple is:
                               (model_func_name, bic, loglik_max, params_mle)
        stack_dim (str): Stack dimension ('a' or 'm'). Default: 'm'
        m_unit (str): Mass unit ('earth' or 'jupiter'). Default: 'earth'
    """
   
    from occurrence import plotting_utils as pu
    
    # Load summary dict for histogram
    path_to_summary = os.path.join(tier123_dir, 'saved_dicts/summary_dict.npz')
    summary_dict = dict(np.load(path_to_summary))
    
    # Get the ORD histogram with fig/ax objects
    plot_save_dir = os.path.join(tier123_dir, 'plots/')
    os.makedirs(plot_save_dir, exist_ok=True)
    
    tier1_dir = tier123_dir.split('/')[0]
    fig, ax = pu.plot_occurrence_hist(summary_dict, stack_dim=stack_dim, m_unit=m_unit, mtype=tier1_dir,
                                      rate_type='ORD', title='', return_fig_ax=True,
                                      savepath=None, figsize=(6, 4))
    
    # Handle both single and stacked axes
    xlim_list = []
    if isinstance(ax, np.ndarray):
        axs_list = ax.flatten().tolist()
    elif isinstance(ax, list):
        axs_list = ax
    else:
        axs_list = [ax]
    xlim_list = [ax_i.get_xlim() for ax_i in axs_list]
        
        
    
    
    # Iterate through the models and plot each max-likelihood model
    for idx, key in enumerate(model_bic_dict.keys()):
        #import pdb; pdb.set_trace()
        if 'flat' in key: # Skip flat model, which is used for comparison
            continue
        ax_idx = int(key.split('_')[-1])
        ax_i = axs_list[ax_idx]
        
        _, flat_bic, _, _ = model_bic_dict[f'bic_outputs_flat_{ax_idx}']
        model_func_name, bic, loglik_max, params_mle = model_bic_dict[key]
        delta_bic = flat_bic-bic
            
        model_spec = get_model_spec(model_func_name)
        model_func = model_spec.function
        param_names = model_spec.parameter_names
        color = model_spec.color
        
        ## Plot model on the desired axes
        xlim = xlim_list[ax_idx]
        x_model = np.logspace(np.log10(xlim[0]), np.log10(xlim[1]), 200)
        
        # Calculate model predictions
        y_model = model_func(params_mle, x_model)
        
        # Build label with model name, parameters, BIC, and likelihood
        param_str = ', '.join([f"{name}={val:.2f}" for name, val in zip(param_names, params_mle)])
        #label = f'{model_func_name} ({param_str}): $\Delta$BIC={delta_bic:.1f}, lnL={loglik_max:.1f}'
        label = f'{model_func_name} ({param_str}): $\Delta$BIC={delta_bic:.1f}'
        
        ax_i.plot(
            x_model, y_model,
            color=color,
            linewidth=2.5,
            label=label,
            zorder=90
        )
        
        ax_i.legend(loc='upper right', fontsize=8)
        # Update legend after each model is added
        #for ax_i in axs_list:
        #    ax_i.legend(loc='upper right', fontsize=8)

    # Save the figure
    save_path = os.path.join(plot_save_dir, 'occurrence_ORD_BIC_models.png')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    
    print(f"\nBIC models plotted on histogram:")
    print(f"Saved to: {save_path}\n")
    
    return












