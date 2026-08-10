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
    elif model_func_name=='logG':
        model_func = log_gaussian
        ndim = 3
    elif model_func_name=='step':
        model_func = step
        ndim = 3
    elif model_func_name=='escarpment':
        model_func = escarpment
        ndim = 4
    elif model_func_name == 'bpl':
        model_func = brokenpowerlaw
        ndim = 4
    else:
        raise ValueError(f"Unknown model: {model_func_name}")
        
    
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
    a0 = np.exp(log_a0)
    return C*AorM**beta * (1 - np.exp(-(AorM/a0)**gamma))



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
    
    
    if model_func_name=='flat':
        
        C = np.mean(ORD_vals)
        p0 = [C]
        
        
    elif model_func_name=='pp1':
        
        ## pp1 takes the form lam(x) = slope*ln(x)+b
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
    
    
    elif model_func_name=='logG':

        A = np.mean(ORD_vals)
        mu = np.mean(np.log10(bin_centers)) # Take log-mean
        sigma = np.log10((bin_centers[-1]/bin_centers[0])**(1/4)) # Fraction of the total
        
        p0 = [A, mu, sigma]
    
    elif model_func_name=='step':
        
        center_val = (min_xval*max_xval)**0.5
        log_breakpoint = np.log10(center_val)
        C1 = min(ORD_vals)
        C2 = max(ORD_vals)
        
        
        p0 = [C1, C2, log_breakpoint]
    
    elif model_func_name=='escarpment':
    
        ## Idea: divide the histogram into even thirds. Fit a line to the middle third and set the
        ##       height of the first and last at the corresponding levels.
        
        
        # Domain in log-space
        log_min = np.log10(min_xval)
        log_max = np.log10(max_xval)

        # Breakpoints at thirds
        log_bp1 = log_min + (log_max - log_min)/3
        log_bp2 = log_min + 2*(log_max - log_min)/3

        # Select bins in middle third
        middle_mask = (
            (np.log10(bin_centers) >= log_bp1) &
            (np.log10(bin_centers) <= log_bp2)
        )

        middle_x = np.log10(bin_centers[middle_mask])
        middle_y = ORD_vals[middle_mask]

        # Fit line to middle section
        slope_mid, intercept_mid = np.polyfit(middle_x, middle_y, 1)

        # Flat levels determined by evaluating the middle line
        # at the breakpoints
        C1 = slope_mid*log_bp1 + intercept_mid
        C2 = slope_mid*log_bp2 + intercept_mid
        
        if C1<0 or C1>1:
            C1 = np.median(ORD_vals)
        if C2<0 or C2>1:
            C2 = np.median(ORD_vals)

        # Parameter vector
        p0 = [C1, C2, log_bp1, log_bp2]
        
    elif model_func_name=='bpl':

        C = np.max(ORD_vals)
        log_a0 = np.log10((min_xval*max_xval)**0.5) # Middle of the domain
        beta = np.random.uniform(-0.5, 0.5)
        gamma = np.random.uniform(-0.5, 0.5)
        
        p0 = [C, log_a0, beta, gamma]
    
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
    
    if model_func_name == 'flat':
    
        if FlatLine(theta, AorM_min)<0:
            return -np.inf
        return 0.0
    
    elif model_func_name == 'pp1':
        
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
        extreme_val3 = PiecewisePower2(theta, 10**(log_xt))
        

        if extreme_val1<0 or extreme_val2<0 or extreme_val3<0:
            return -np.inf
        if log_xt<minL or log_xt>maxL:
            return -np.inf
        if abs(m1)>100 or abs(m2)>100:
            return -np.inf
        
        # Log-uniform prior on absolute value of slopes
        logprior=0
        logprior -= np.log10(abs(m1))
        logprior -= np.log10(abs(m2))
        
        return logprior
        
    elif model_func_name=='logG':
        
        A, mu, sigma = theta
        
        
        if A<0 or A>10*AorM_max:
            return -np.inf
        if mu<np.log10(AorM_min)-15 or mu>np.log10(AorM_max)+15:
            return -np.inf
        if abs(sigma)>np.log10(AorM_max/AorM_min):
            return -np.inf
        if sigma<0:
            return -np.inf
            
        return 0.0
            
        
    
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
        
    elif model_func_name == 'bpl':
        C, log_a0, beta, gamma = theta
        a0 = 10**(log_a0)
        
        if C<=0:
            return -np.inf
        if C>2:
            return -np.inf
        if abs(log_a0)>10:
            return -np.inf
        if abs(beta)>4:
            return -np.inf
        if abs(gamma)>4:
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
 
     
    try:
        model_info = model_dict[model_func_name]
        param_names = model_info[2]

    except:
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
    
    try:
        model_info = model_dict[model_func_name]
        model_func = eval(model_info[0])
        ndim = model_info[1]
    except:
        raise ValueError(f"Cannot calculate BIC for model: {model_func_name}. Ensure it's in model_dict")

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
            
        try:
            model_info = model_dict[model_func_name]
            model_func = eval(model_info[0])
            param_names = model_info[2]
            color = model_info[3]
        except:
            raise ValueError(f"Unknown model: {model_func_name}")
        
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















