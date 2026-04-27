## Perform MCMC simulation for different power law models
import os
import numpy as np
from pathlib import Path

import emcee
import multiprocessing as mp


## Delete
import matplotlib.pyplot as plt

def mcmc(nstars, comp_names_inROI, model_func_name,
         ROIsamples_dict, ROIweights_dict,
         a_lims, m_lims, stack_dim,
         interp_fn_avg,
             nwalkers=50,
             nsteps=5000,
             burnin=1000,
             parallel=False,
             save_path="chains.npz",
             random_seed=None):
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
    elif model_func_name=='escarpment':
        model_func = escarpment
        ndim = 4
        
    
    ## Need a 2D grid to calculate e^-Lambda integral
    fine_grid_num = 101
    fine_a_spacing = (a_lims[1]/a_lims[0])**(1/(fine_grid_num-1))
    fine_m_spacing = (m_lims[1]/m_lims[0])**(1/(fine_grid_num-1))
    fine_amin, fine_amax = a_lims[0]*(fine_a_spacing**0.5), a_lims[1]/(fine_a_spacing**0.5)
    fine_mmin, fine_mmax = m_lims[0]*(fine_m_spacing**0.5), m_lims[1]/(fine_m_spacing**0.5)
    
    fine_alist = np.logspace(np.log10(fine_amin), np.log10(fine_amax), fine_grid_num-1) # a values, log-spaced
    fine_mlist = np.logspace(np.log10(fine_mmin), np.log10(fine_mmax), fine_grid_num-1) # m values, log-spaced

    A, M = np.meshgrid(fine_alist, fine_mlist, indexing='xy')
    fine_compl_grid = interp_fn_avg((A, M)) # Completeness grid w/ shape (len(fine_mlist), len(fine_alist))
    
    ## Make "a" histograms and stack for different "m" bins
    if stack_dim=='m':
        fine_compl_AorM = np.mean(fine_compl_grid, axis=0) # Average over the "m" dimension
        fine_list_AorM = fine_alist
        AorM_min, AorM_max = a_lims[0], a_lims[-1]
        AorM_ind = 0
        
    ## Make "m" histograms and stack for different "a" bins  
    elif stack_dim=='a':
        fine_compl_AorM = np.mean(fine_compl_grid, axis=1) # Average over the "a" dimension
        fine_list_AorM = fine_mlist
        AorM_min, AorM_max = m_lims[0], m_lims[-1]
        AorM_ind = 1
    
    #total_min = np.min([np.min(ROIsamples_dict[comp_name][1]) for comp_name in ROIsamples_dict.keys()])
    #total_max = np.max([np.max(ROIsamples_dict[comp_name][1]) for comp_name in ROIsamples_dict.keys()])
    #import pdb; pdb.set_trace()

    dlogAorM = np.log10(fine_list_AorM[1]/fine_list_AorM[0]) # log-spacing of a or m grid
    
    ############################################################################
    diagnostic=False
    if diagnostic:
        if model_func_name == 'pp1':
            # PiecewisePower1: (slope, intercept)
            parameter_sets = [
                (-0.13, 0.22),
                (-0.14, 0.21),
                (-0.15, 0.20),
                (-0.16, 0.19),
                (-0.17, 0.18),
            ]
        elif model_func_name == 'pp2' or 'Softplus' in model_func_name:
            # PiecewisePower2: (m1, m2, b, log_xt)
            parameter_sets = [
                (-0.15, 0.0, 0.20, 1.2),
                (-0.1, 0.05, 0.15, 0.7),
                (-0.15, 0.1, 0.12, 0.5),
                (-0.05, 0.15, 0.18, 0.6),
                (-0.2, 0.05, 0.10, 0.4),
            ]
        elif model_func_name == 'escarpment':
            # escarpment: (C1, C2, bp1, bp2)
            parameter_sets = [
                (0.25, 0.05, 3.0, 10.0),
                (0.20, 0.08, 2.5, 8.0),
                (0.30, 0.10, 3.5, 12.0),
                (0.22, 0.06, 3.2, 9.0),
                (0.28, 0.09, 2.8, 11.0),
            ]
        else:
            # Fallback - should not reach here in normal usage
            parameter_sets = []
        #PiecewisePower2([-1, 0.0, 0.15, 1.5], fine_list_AorM)
        print_power_hard_coded(nstars, comp_names_inROI, model_func, model_func_name,
                            ROIsamples_dict, ROIweights_dict, dlogAorM, 
                            fine_list_AorM, fine_compl_AorM, 
                            AorM_min, AorM_max, 
                            AorM_ind, parameter_sets)
    
        tier1_dir='mtrue'
        tier2_dir='allstars'
        tier3_dir='1_10AU'
        plot_hard_coded_on_histogram(tier1_dir, tier2_dir, tier3_dir, 
                                         nstars, comp_names_inROI, model_func, model_func_name,
                                         ROIsamples_dict, ROIweights_dict,
                                         dlogAorM, fine_list_AorM, fine_compl_AorM,
                                         AorM_min, AorM_max, 
                                         AorM_ind, stack_dim, parameter_sets, m_unit='jupiter')
        import pdb; pdb.set_trace()    
    ##############################################################################
    loglik_args = (
        nstars,
        comp_names_inROI,
        model_func,
        model_func_name,
        ROIsamples_dict,
        ROIweights_dict,
        dlogAorM,
        fine_list_AorM,
        fine_compl_AorM,
        AorM_min,
        AorM_max,
        AorM_ind,
    )
    
    
    # ---- Initialize walkers ----
    theta_init = initial_params(model_func_name, fine_list_AorM, dlogAorM)
    

    # Small Gaussian ball around initial guess
    pos = theta_init + 1e-3 * np.random.randn(nwalkers, ndim)

    
    # Clip starting values according to which model_func is being used
    #pos = np.clip(pos, 1e-6, None)
    
    
    # ---- Set up sampler ----
    if parallel:
        with mp.Pool() as pool:
            sampler = emcee.EnsembleSampler(
                nwalkers, ndim, loglik,
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


def loglik_power(theta, nstars, comp_names, model_func,
           model_func_name,
           ROIsamples_dict, ROIweights_dict, 
           dlogAorM, fine_list_AorM, fine_compl_AorM,
           AorM_min, AorM_max, AorM_ind):
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
    
    logprior = log_prior(model_func_name, theta, AorM_min, AorM_max)
    if np.isinf(logprior):
        return -np.inf      
    
    fine_lam_list = model_func(theta, fine_list_AorM)
    rate_map = np.sum(fine_lam_list*dlogAorM) # Occurrence rate is rate density "integrated" over a or m space

    if (rate_map<0) | (rate_map>1): # If occurrence is <0 or >1, reject
        print(f"Params are {theta[0]:.1f}, {theta[1]:.1f}, rate_map is {rate_map:.4f}, dw is {dlogAorM:.3f}")
        return -np.inf

    ## First get the pre-factor e^(-Lambda). We want log-likelihood, so just -Lambda
    ## Lambda is Nstars * integral(lambda*completeness)
    ## Basic Riemann sum of lambda*compl over M space: sum (lambda(M)*compl(M)*dlogM) over the grid
    integral = np.sum(fine_lam_list*fine_compl_AorM*dlogAorM)
    Lambda = nstars*integral

    ## Now the product term
    ## For every companion, do three things:
    ##     First, determine the lambda value at that companion's M value using the escarpment function
    ##     Second, retrieve a list of the completeness/prior values for each of that companion's samples
    ##         (NOTE: these cannot be pre-averaged as in the histogram case because each one will
    ##                be multiplied by a different lambda value)
    ##     Third, retrieve the companion's weight (ie, fraction of samples that fall in the ROI)

    log_prod_term = 0 
    for comp_name in comp_names:
        
        # First, unpack the companion posterior
        comp_sample_array = ROIsamples_dict[comp_name]
        AorM_list = comp_sample_array[AorM_ind]
        cop_list = comp_sample_array[5] # completeness (from single map) over prior
        
        ROIweight = ROIweights_dict[comp_name]
        
        lam_list = model_func(theta, AorM_list)
        if any(lam_list<0):
            neg_ind = np.argmin(lam_list)
            
            print(f"{comp_name}, {theta[0]:.2f}, {theta[1]:.2f}, {lam_list[neg_ind]}, {AorM_list[neg_ind]}")
            return -np.inf
            #log_term=0
        #else:
            ## Calculate this companion's contribution to the likelihood
        mean = np.mean(lam_list*cop_list)
        log_term = ROIweight * np.log(mean+1e-300) # Ensure non-zero for stability
            
        log_prod_term += log_term
        if np.isnan(log_term):
            import pdb; pdb.set_trace()


    loglik = -Lambda + log_prod_term

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
    C1, C2, bp1, bp2 = theta
    AorM = np.asarray(AorM)
    
    logx = np.log10(AorM)
    log_bp1 = np.log10(bp1)
    log_bp2 = np.log10(bp2)
    
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


def initial_params(model_func_name, AorM_list, dlogAorM):
    """
    Generate initial parameter guesses
    for a model function. Initial guesses
    for each function are empirical, and
    the function is identified by ndim
    """
    
    if model_func_name=='pp1':
        
        log_AorM = np.log10(AorM_list)
        minL, maxL = np.min(log_AorM), np.max(log_AorM)
        
        ## pp1 takes the form lam(x) = r*log10(x)+b
        ## From that, we can derive constraints to ensure
        ## lam>0 for all lam, and integral(lam*dlogAorM)<1
        
        # Start with random slopes, then find possible b
        slope = np.random.uniform(-0.5, 0.5)
        
        ## First, all lam should be non-negative
        if slope<=0:
            min_b = -slope*maxL
        else:
            min_b = -slope*minL
        
        ## Second, the integral should be <1
        S1 = np.sum(log_AorM*dlogAorM)
        S2 = dlogAorM*len(AorM_list)
        max_b = (1-slope*S1)/S2
        
        b = np.random.uniform(min_b, max_b)
        
        p0 = [slope, b]
        #p0 = [-0.1, 0.23]
        #import pdb; pdb.set_trace()
        print(f"Initial params: slope={slope:.3f} and b={b:.3f} from {min_b:.2f}-{max_b:.2f} ")
    
    elif model_func_name=='pp2':
        logA = np.log10(AorM_list)
        minL, maxL = np.min(logA), np.max(logA)

        for _ in range(1000):

            # slopes: modest range
            m1 = np.random.uniform(-1.0, 0)
            m2 = np.random.uniform(-1.0, 0)

            # transition in log space
            log_xt = np.random.uniform(minL, maxL)

            # intercept: start near small values
            b = np.random.uniform(0.0, 0.4)

            theta = [m1, m2, b, log_xt]

            lam = PiecewisePower2(theta, AorM_list)

            rate_map = np.sum(lam * dlogAorM)
            
            cond1 = 0 < rate_map < 1
            cond2 = m1<=0
            cond3 = m2<=0
            cond4 = 0 < b < 1
            if cond1 and cond2 and cond3:
                print(f"Initial params: m1={m1:.3f}, m2={m2:.3f}, b={b:.3f}, log_xt={log_xt:.3f},")
                return theta

        raise RuntimeError("Failed to find valid initial params")
    
    elif model_func_name=='escarpment':
        minA, maxA = np.min(AorM_list), np.max(AorM_list)
        
        for _ in range(1000):
            # C1 and C2: occurrence rates (should be positive and their integral < 1)
            C1 = np.random.uniform(0.0, 1.0)
            C2 = np.random.uniform(0.0, C1) # C2<C1
            
            # Breakpoints in linear space
            bp1 = np.random.uniform(minA, maxA)
            bp2 = np.random.uniform(bp1, maxA) # bp2>bp1
            
            # Ensure bp1 < bp2
            #if bp1 >= bp2:
            #    continue
            
            theta = [C1, C2, bp1, bp2]
            lam = escarpment(theta, AorM_list)
            
            rate_map = np.sum(lam * dlogAorM)
            
            # Check conditions
            cond1 = 0 < rate_map < 1
            cond2 = C1 > 0
            cond3 = C2 > 0
            if cond1 and cond2 and cond3:
                print(f"Initial params: C1={C1:.3f}, C2={C2:.3f}, bp1={bp1:.3f}, bp2={bp2:.3f}")
                return theta
        
        raise RuntimeError("Failed to find valid initial params for escarpment")
    
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
        
        return 0.0
    
    elif model_func_name == 'escarpment':
        C1, C2, bp1, bp2 = theta
        
        if C1<0 or C2<0:
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


def plot_hard_coded_on_histogram(tier1_dir, tier2_dir, tier3_dir, 
                                 nstars, comp_names_inROI, model_func, model_func_name,
                                 ROIsamples_dict, ROIweights_dict,
                                 dlogAorM, fine_list_AorM, fine_compl_AorM,
                                 AorM_min, AorM_max,
                                 AorM_ind, stack_dim, parameter_sets, m_unit='earth'):
    """
    Plot hard-coded power law parameter sets overlaid on the ORD histogram.
    
    Arguments:
        tier1_dir, tier2_dir, tier3_dir (str): Directory paths for loading data
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
        stack_dim (str): Stack dimension ('a' or 'm')
        parameter_sets (list): List of parameter tuples to evaluate
        m_unit (str): Mass unit ('earth' or 'jupiter')
    """
    
    from occurrence import plotting_utils as pu
    
    # Color palette for distinct colors
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    
    # Determine parameter names for labels
    if model_func_name == 'pp1':
        param_names = ['slope', 'int']
    elif model_func_name == 'pp2' or 'Softplus' in model_func_name:
        param_names = ['m1', 'm2', 'b', 'log_xt']
    elif model_func_name == 'escarpment':
        param_names = ['C1', 'C2', 'bp1', 'bp2']
    else:
        param_names = [f'p{i}' for i in range(len(parameter_sets[0]))]
    
    # Load summary dict for histogram
    load_save_dir = os.path.join(tier1_dir, tier2_dir, tier3_dir)
    path_to_summary = os.path.join(load_save_dir, 'saved_dicts/summary_dict.npz')
    summary_dict = dict(np.load(path_to_summary))
    
    # Get the ORD histogram with fig/ax objects
    plot_save_dir = os.path.join(load_save_dir, 'plots/')
    os.makedirs(plot_save_dir, exist_ok=True)
    
    fig, ax = pu.plot_occurrence_hist(summary_dict, stack_dim=stack_dim, m_unit=m_unit, mtype=tier1_dir,
                                      rate_type='ORD', title='', return_fig_ax=True,
                                      savepath=None, figsize=(6, 4))
    
    # Handle both single and stacked axes
    if isinstance(ax, np.ndarray):
        axs_list = ax.flatten().tolist()
    elif isinstance(ax, list):
        axs_list = ax
    else:
        axs_list = [ax]
    
    # Calculate likelihoods and prepare labels for each parameter set
    param_info = []
    for theta in parameter_sets:
        loglik = loglik_power(theta, nstars, comp_names_inROI, model_func,
                              model_func_name,
                              ROIsamples_dict, ROIweights_dict,
                              dlogAorM, fine_list_AorM, fine_compl_AorM,
                              AorM_min, AorM_max,
                              AorM_ind)
        param_info.append((theta, loglik))
    
    # Plot each parameter set on the histogram
    for ax_i in axs_list:
        xlim = ax_i.get_xlim()
        x_model = np.logspace(np.log10(xlim[0]), np.log10(xlim[1]), 200)
        
        for (theta, loglik), color in zip(param_info, colors):
            y_model = model_func(theta, x_model)
            
            # Build label with parameter values and likelihood
            param_str = ', '.join([f"{name}={val:.2f}" for name, val in zip(param_names, theta)])
            label = f'{param_str}, lnL={loglik:.1f}'
            
            ax_i.plot(
                x_model, y_model,
                color=color,
                linewidth=2.0,
                label=label,
                zorder=90
            )
        
        ax_i.legend(loc='upper right', fontsize=8)

    
    # Save the figure
    save_path = os.path.join(plot_save_dir, 'occurrence_ORD_hard_coded.png')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    
    print(f"\nHard-coded parameter sets plotted on histogram:")
    print(f"Saved to: {save_path}\n")
    
    return



















