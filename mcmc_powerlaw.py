## Perform MCMC simulation for different power law models
import os
import numpy as np
import pandas as pd
from pathlib import Path

import emcee
import multiprocessing as mp


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
        
    
    ## Need a 2D grid to calculate e^-Lambda integral
    fine_grid_num = 101
    fine_a_spacing = (a_lims[1]/a_lims[0])**(1/(fine_grid_num-1))
    fine_m_spacing = (m_lims[1]/m_lims[0])**(1/(fine_grid_num-1))
    fine_amin, fine_amax = a_lims[0]*(fine_a_spacing**0.5), a_lims[1]/(fine_a_spacing**0.5)
    fine_mmin, fine_mmax = m_lims[0]*(fine_m_spacing**0.5), m_lims[1]/(fine_m_spacing**0.5)
    
    fine_alist = np.logspace(np.log10(fine_amin), np.log10(fine_amax), fine_grid_num-1) # a values, log-spaced
    fine_mlist = np.logspace(np.log10(fine_mmin), np.log10(fine_mmax), fine_grid_num-1) # m values, log-spaced
    #import pdb; pdb.set_trace()
    # fine_alist = np.logspace(np.log10(a_lims[0]), np.log10(a_lims[1]), 100) # a values, log-spaced
    # fine_mlist = np.logspace(np.log10(m_lims[0]), np.log10(m_lims[1]), 100) # m values, log-spaced
    A, M = np.meshgrid(fine_alist, fine_mlist, indexing='xy')
    fine_compl_grid = interp_fn_avg((A, M)) # Completeness grid w/ shape (len(fine_mlist), len(fine_alist))
    
    ## Make "a" histograms and stack for different "m" bins
    if stack_dim=='m':
        fine_compl_AorM = np.mean(fine_compl_grid, axis=0) # Average over the "m" dimension
        fine_list_AorM = fine_alist
        AorM_ind = 0
        
    ## Make "m" histograms and stack for different "a" bins  
    elif stack_dim=='a':
        fine_compl_AorM = np.mean(fine_compl_grid, axis=1) # Average over the "a" dimension
        fine_list_AorM = fine_mlist
        AorM_ind = 1
    
    #import pdb; pdb.set_trace()
    dlogAorM = np.log10(fine_list_AorM[1]/fine_list_AorM[0]) # log-spacing of a or m grid
    # fine_list_AorM = (fine_list_AorM * 10**(dlogAorM/2))[:-1] # Shift values to bin centers; remove the last
    
    

    loglik_args = (
        nstars,
        comp_names_inROI,
        model_func,
        ROIsamples_dict,
        ROIweights_dict,
        dlogAorM,
        fine_list_AorM,
        fine_compl_AorM,
        AorM_ind,
    )
    # import pdb; pdb.set_trace()
    
    
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
           ROIsamples_dict, ROIweights_dict, 
           dlogAorM, fine_list_AorM, fine_compl_AorM,
           AorM_ind):
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
    fine_lam_list = model_func(theta, fine_list_AorM)
    rate_map = np.sum(fine_lam_list*dlogAorM) # Occurrence rate is rate density "integrated" over a or m space

    if (rate_map<0) | (rate_map>1): # If occurrence is <0 or >1, reject
        # print(f"Params are {theta[0]:.1f}, {theta[1]:.1f}, rate_map is {rate_map:.4f}, dw is {dlogAorM:.3f}")
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
        cop_list = comp_sample_array[5] # completeness (from avg. map) over prior
        
        ROIweight = ROIweights_dict[comp_name]
        
        lam_list = model_func(theta, AorM_list)
        if any(lam_list<0):
            return -np.inf
        
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
        print(f"Initial params: slope={slope:.3f} and b={b:.3f} from {min_b:.2f}-{max_b:.2f} ")
    
    return p0

















