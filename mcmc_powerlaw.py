## Perform MCMC simulation for different power law models
import os
import numpy as np
from pathlib import Path

import emcee
import multiprocessing as mp
from scipy.optimize import minimize

import matplotlib.pyplot as plt

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
    """
    Run MCMC to compute occurrence rates
    Start by extracting and manipulating ingredients 
    to feed to log-likelihood
    
    Arguments:
        nstars (int): Number of host stars in sample
        comp_names_inROI (list of str): List of names of
            companions that fall in the ROI. Must correspond 
            to keys in bin_lam_dict
        cell_dict (dict): Dictionary containing useful info
            related to cell sizes, avg completeness, etc.
        bin_lam_dict (dict): Compressed representation of
            posterior samples, with completeness, priors,
            and lambda indices accounted for. Keys look like
            'planetXX_cellYY_compl_over_prior' and
            'planetXX_cellYY_weight'. Thus, with Np companions
            and Nc cells, there are 2*Np*Nc keys in the dict.
            'compl_over_prior' is an array of floats, and 'weight'
            is a single float.
        
        nwalkers (int): Number of walkers
        nsteps (int): Number of production steps
        burnin (int): Number of burn-in steps
        parallel (bool): Use multiprocessing if True
        save_path (str): Where to save chains
        random_seed (int or None)
            
    """
    os.makedirs(Path(save_path).parent, exist_ok=True)
    # import pdb; pdb.set_trace()
    
    ## Choose ndim and function based on name
    if model_func_name=='pp1':
        model_func = PiecewisePower1
        ndim = 2
    elif model_func_name=='pp2':
        model_func = PiecewisePower2
        ndim = 4
    elif model_func_name=='step':
        model_func = step
        ndim = 3
    elif model_func_name=='escarpment':
        model_func = escarpment
        ndim = 4
        
    
    ############################################################################ 
    ##############################################################################
    #import pdb; pdb.set_trace()
    
    loglik_args = (
        hist_dict,
        model_func,
        model_func_name
    )
    
    
    # ---- Initialize walkers ----
    theta_init = initial_params(model_func_name, hist_dict)
    

    # Small Gaussian ball around initial guess
    pos = theta_init + 1e-3 * np.random.randn(nwalkers, ndim)
    
    
    # ---- Set up sampler ----
    if parallel:
        with mp.Pool() as pool:
            sampler = emcee.EnsembleSampler(
                nwalkers, ndim, loglik_power,
                args=loglik_args, pool=pool
            )

            # Burn-in
            pos, _, _ = sampler.run_mcmc(pos, burnin, progress=True)
            sampler.reset()

            # Production
            sampler.run_mcmc(pos, nsteps, progress=True)

    else:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, loglik_power,
                                        args=loglik_args)

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
    Log likelihood of a 1D piecewise power law in
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
    
    ## Uses AVERAGE of upper/lower uncertainties. Consider using split gaussian in the future
    loglik = -np.sum((hist_dict['ORD_vals']-model_ORD_vals)**2 / (2*hist_dict['ORD_errs']**2))

    if np.isnan(loglik):
        import pdb; pdb.set_trace()

    return loglik
    

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
        log_bp : log10 breakpoint (x value)
    
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


def initial_params(model_func_name, hist_dict):
    """
    Generate initial parameter guesses
    for a model function. Initial guesses
    for each function are empirical, and
    the function is identified by ndim
    """
    
    bin_centers = hist_dict['bin_centers']
    min_xval, max_xval = hist_dict['lims']
    ORD_vals = hist_dict['ORD_vals']
    
    if model_func_name=='pp1':
        
        ## pp1 takes the form lam(x) = slope*log10(x)+b
        ## Initial guess: line connecting the extreme points
        slope = (ORD_vals[-1]-ORD_vals[0])/(np.log10(max_xval)-np.log10(min_xval)) # slope = delta_y/delta_x
        b = ORD_vals[-1]-slope*np.log10(max_xval) # b = y1-slope*x1
        #import pdb; pdb.set_trace()
        
        p0 = [slope, b]

        print(f"Initial params: slope={slope:.3f} and b={b:.3f}")
    
    elif model_func_name=='pp2':
        #import pdb; pdb.set_trace()
        
        ## Idea: divide the histogram down the middle and fit a line to each half
        center_val = (min_xval*max_xval)**0.5
        center_bin_ind = np.argmin(abs(bin_centers/center_val-1)) # Find the nearest bin center in log space
        center_ORD = ORD_vals[center_bin_ind]
        
        slope1 = (center_ORD-ORD_vals[0])/(np.log10(center_val)-np.log10(min_xval)) # slope = delta_y/delta_x
        b1 = center_ORD-slope1*np.log10(center_val) # b = y1-slope*x1
        
        slope2 = (ORD_vals[-1]-center_ORD)/(np.log10(max_xval)-np.log10(center_val)) # slope = delta_y/delta_x
        #b2 = ORD_vals[-1]-slope1*np.log10(max_xval) # b = y1-slope*x1
        log_breakpoint = np.log10(center_val)

        p0 = [slope1, slope2, b1, log_breakpoint]
    
    elif model_func_name=='step':
        
        center_val = (min_xval*max_xval)**0.5
        log_breakpoint = np.log10(center_val)
        C1 = min(ORD_vals)
        C2 = max(ORD_vals)
        
        
        p0 = [C1, C2, log_breakpoint]
    
    elif model_func_name=='escarpment':
        
        C1 = min(ORD_vals)
        C2 = max(ORD_vals)
            
        # Breakpoints in log space
        log_bp1 = np.random.uniform(np.log10(min_xval), np.log10(max_xval))
        log_bp2 = np.random.uniform(log_bp1, np.log10(max_xval))
            
        p0 = [C1, C2, log_bp1, log_bp2]
    
    return p0


def log_prior(model_func_name, theta, AorM_min, AorM_max):
    """
    Calculate the log-prior for model parameters.
    
    Uses uniform priors within allowed bounds (log-prior = 0 if valid, 
    -inf if invalid). Constraints are determined by the same logic as 
    initial_params().
    
    Arguments:
        model_func_name (str): Name of model function ('pp1', 'pp2', 'escarpment')
        theta (array-like): Model parameters
        AorM_list (array-like): Fine grid of a or m values (for determining bounds)
        dlogAorM (float): Log spacing of grid
    
    Returns:
        log_prior (float): Log of prior probability (0 for uniform within bounds, 
                          -inf for parameters outside allowed bounds)
    """
    
    if model_func_name == 'pp1':
        
        # Check if parameters are within bounds
        extreme_val1 = PiecewisePower1(theta, AorM_min)
        extreme_val2 = PiecewisePower1(theta, AorM_max)
        
        if extreme_val1<0 or extreme_val2<0:
            return -np.inf
        if extreme_val1>1 or extreme_val2>1:
            return -np.inf
        
        return 0.0
    
    elif model_func_name == 'pp2':
        m1, m2, b, log_xt = theta
        minL, maxL = np.log10(AorM_min), np.log10(AorM_max)
        
        extreme_val1 = PiecewisePower2(theta, AorM_min)
        extreme_val2 = PiecewisePower2(theta, AorM_max)
        extreme_val3 = PiecewisePower2(theta, 10**log_xt)
        

        if extreme_val1<0 or extreme_val2<0 or extreme_val3<0:
            return -np.inf
        if log_xt<minL or log_xt>maxL:
            return -np.inf
        if abs(m1)>100 or abs(m2)>100:
            return -np.inf
        
        # Log-uniform prior on absolute value of slopes
        logprior=0
        logprior -= np.log(abs(m1))
        logprior -= np.log(abs(m2))
        
        return logprior
    
    elif model_func_name == 'step':
        C1, C2, log_bp = theta
        bp = 10**(log_bp)
        
        if C1<0 or C2<0:
            return -np.inf
        if C1>1 or C2>1:
            return -np.inf
        # Check that bp is within bounds
        if bp<AorM_min or bp>AorM_max:
            return -np.inf
        
        return 0.0
    
    elif model_func_name == 'escarpment':
        C1, C2, log_bp1, log_bp2 = theta
        bp1, bp2 = 10**(log_bp1), 10**(log_bp2)
        
        if C1<0 or C2<0:
            return -np.inf
        if C1>1 or C2>1:
            return -np.inf
        # Check that bp1 < bp2
        if bp1 >= bp2:
            return -np.inf
        if bp1<AorM_min or bp1>AorM_max:
            return -np.inf
        if bp2<AorM_min or bp2>AorM_max:
            return -np.inf
        
        
        return 0.0
    
    else:
        raise ValueError(f"Unknown model: {model_func_name}")


def print_power_hard_coded(nstars, comp_names_inROI, model_func, model_func_name,
                          ROIsamples_dict, ROIweights_dict,
                          dlogAorM, fine_list_AorM, fine_compl_AorM,
                          AorM_min, AorM_max,
                          AorM_ind, parameter_sets):
    """
    Print hard-coded power law parameter sets with their likelihoods.
    
    Arguments:
        nstars (int): Number of host stars
        comp_names_inROI (list of str): Companion names in region of interest
        model_func : The model function (e.g., PiecewisePower1, PiecewisePower2)
        model_func_name (str): Name of model function for label formatting
        ROIsamples_dict (dict): ROI sample data for companions
        ROIweights_dict (dict): ROI weights for companions
        dlogAorM (float): Log spacing of parameter grid
        fine_list_AorM (array): Fine grid of parameter values
        fine_compl_AorM (array): Fine grid of completeness values
        AorM_ind (int): Index indicating dimension (0 for 'a', 1 for 'm')
        parameter_sets (list): List of parameter tuples to evaluate
    """
    
    print("\n" + "="*60)
    print(f"Hard-coded {model_func_name} Parameter Likelihoods")
    print("="*60)
    
    # For label formatting, determine parameter names based on model function
    if model_func_name == 'pp1':
        param_names = ['slope', 'intercept']
    elif model_func_name == 'pp2' or 'Softplus' in model_func_name:
        param_names = ['m1', 'm2', 'b', 'log_xt']
    elif model_func_name == 'step':
        param_names = ['C1', 'C2', 'log_bp']
    elif model_func_name == 'escarpment':
        param_names = ['C1', 'C2', 'bp1', 'bp2']
    else:
        # Generic fallback
        param_names = [f'p{i}' for i in range(len(parameter_sets[0]))]
    
    # Calculate likelihood for each parameter set
    for i, theta in enumerate(parameter_sets):
        loglik = loglik_power(theta, nstars, comp_names_inROI, model_func,
                              model_func_name,
                              ROIsamples_dict, ROIweights_dict,
                              dlogAorM, fine_list_AorM, fine_compl_AorM,
                              AorM_min, AorM_max,
                              AorM_ind)
        
        # Format parameter string
        param_str = ', '.join([f"{name}={val:7.3f}" for name, val in zip(param_names, theta)])
        print(f"Set {i+1}: {param_str}  →  lnL = {loglik:10.2f}")
    
    print("="*60 + "\n")
    
    return



def calculate_bic(tier123_dir, model_func_name,
                  hist_dict, nstars):
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
    
    # Determine model function and parameter count
    if model_func_name == 'pp1':
        model_func = PiecewisePower1
        ndim = 2
    elif model_func_name == 'pp2':
        model_func = PiecewisePower2
        ndim = 4
    elif model_func_name == 'step':
        model_func = step
        ndim = 3
    elif model_func_name == 'escarpment':
        model_func = escarpment
        ndim = 4
    else:
        raise ValueError(f"Cannot calculate BIC for model: {model_func_name}")
        

    # Set up arguments for optimization
    optim_args = (
        hist_dict,
        model_func,
        model_func_name,
    )
    
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

    # Calculate BIC
    # BIC = k * ln(n) - 2 * ln(L)
    # where L is the likelihood (not log-likelihood), so:
    # BIC = k * ln(n) - 2 * ln(L) = k * ln(n) - 2 * loglik
    bic = ndim * np.log(nstars) - 2 * loglik_max

    print(f"\n" + "="*70)
    print(f"BIC Calculation for {model_func_name.upper()}")
    print("="*70)
    print(f"Number of parameters (k): {ndim}")
    print(f"Number of data points (n): {nstars}")
    print(f"Maximum log-likelihood: {loglik_max:.4f}")
    print(f"BIC = {ndim} * ln({nstars}) - 2 * {loglik_max:.4f}")
    print(f"BIC = {bic:.4f}")
    print(f"Parameters at MLE: {params_mle}")
    print("="*70 + "\n")
    
    return model_func_name, bic, loglik_max, params_mle



def plot_bics_on_histogram(tier123_dir, model_bic_dict, stack_dim, m_unit):
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
        
    # Color palette for distinct colors
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    #import pdb; pdb.set_trace()
    # Iterate through the models and plot each max-likelihood model
    
    for idx, key in enumerate(model_bic_dict.keys()):
        ax_idx = int(key.split('_')[-1])
        ax_i = axs_list[ax_idx]
        
        model_func_name, bic, loglik_max, params_mle = model_bic_dict[key]
        
        # Choose model function and parameter names based on model
        if model_func_name == 'pp1':
            model_func = PiecewisePower1
            param_names = ['slope', 'intercept']
            color='RoyalBlue'
        elif model_func_name == 'pp2':
            model_func = PiecewisePower2
            param_names = ['m1', 'm2', 'b', 'log_xt']
            color='tomato'
        elif model_func_name == 'step':
            model_func = step
            param_names = ['C1', 'C2', 'log_bp']
            color='goldenrod'
        elif model_func_name == 'escarpment':
            model_func = escarpment
            param_names = ['C1', 'C2', 'log_bp1', 'log_bp2']
            color='forestgreen'
        else:
            raise ValueError(f"Unknown model: {model_func_name}")
        
        
        
        ## Plot model on the desired axes
        xlim = xlim_list[ax_idx]
        x_model = np.logspace(np.log10(xlim[0]), np.log10(xlim[1]), 200)
        
        # Calculate model predictions
        y_model = model_func(params_mle, x_model)
        
        # Build label with model name, parameters, BIC, and likelihood
        param_str = ', '.join([f"{name}={val:.2f}" for name, val in zip(param_names, params_mle)])
        label = f'{model_func_name} ({param_str}): BIC={bic:.1f}, lnL={loglik_max:.1f}'
        
        ax_i.plot(
            x_model, y_model,
            color=color,
            linewidth=2.5,
            label=label,
            zorder=90
        )
        
        # Update legend after each model is added
        for ax_i in axs_list:
            ax_i.legend(loc='upper right', fontsize=8)

    # Save the figure
    save_path = os.path.join(plot_save_dir, 'occurrence_ORD_BIC_models.png')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    
    print(f"\nBIC models plotted on histogram:")
    print(f"Saved to: {save_path}\n")
    
    return















