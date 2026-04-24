## Perform MCMC simulation specifically for the histogram model
import os
import numpy as np
import pandas as pd
from pathlib import Path

import emcee
import multiprocessing as mp


def mcmc(nstars, comp_names_inROI, cell_dict, bin_lam_dict,
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

    ## Unpack params
    # lam = np.random.uniform(0.001, 0.05, size=cell_dict['num_cells'])
    num_cells = cell_dict['num_cells']
    all_binsizes = cell_dict['all_binsizes']
    avg_cell_compls = cell_dict['avg_compls']
    
    ndim = num_cells
    # ---- Initialize walkers ----
    # Start near small positive values (your prior range)
    lam_init = np.random.uniform(0.001, 0.05, size=ndim)

    # Small Gaussian ball around initial guess
    pos = lam_init + 1e-3 * np.random.randn(nwalkers, ndim)

    # Enforce positivity (important for your likelihood)
    pos = np.clip(pos, 1e-6, None)
    

    loglik_args = (
        nstars,
        comp_names_inROI,
        bin_lam_dict,
        num_cells,
        all_binsizes,
        avg_cell_compls
    )
    # import pdb; pdb.set_trace()
    
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
        sampler = emcee.EnsembleSampler(nwalkers, ndim, loglik_hist,
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


def loglik_hist(lam, nstars, comp_names, bin_lam_dict, num_cells, all_binsizes, avg_cell_compls):
    """
    Log likelihood of a model histogram, given a 
    catalog of companion posterior draws in the
    form of a dictionary
    
    Arguments:
        lam (array of floats): 1D representation of the 2D occurrence
            rate density histogram
        nstars (int): Number of host stars in the sample
        comp_names (list of str): List of companion names
            Must correspond to keys in bin_lam_dict
        bin_lam_dict (dict): Compressed representation of
            posterior samples
        num_cells (int): Number of occurrence cells. Equal to len(lam)
        all_binsizes (array of floats): Logarithmic size of each cell
            in base 10. E.g. if a = 1-10 AU and M = 1-100 M_earth, then
            binsize is 1*2 = 2
        avg_cell_compls (array of floats): Average completeness over each
            cell
    """
    

    rate_map = lam*all_binsizes # Occurrence rate is rate density "integrated" over a/m space
    if ((rate_map<0) | (rate_map>1)).any(): # If occurrence in any cell is <0 or >1, reject
        return -np.inf


    ## First get the pre-factor e^(-Lambda). We want log-likelihood, so just -Lambda
    ## Lambda is Nstars * integral(lambda*completeness)
    ## To integrate lambda*compl over the space, multiply by bin size. Then sum those integrals
    integral = np.sum(all_binsizes*avg_cell_compls*lam)
    Lambda = nstars*integral


    ## Now the product term
    ## For every bin, consider each planet
    ## 	For every planet, retrieve a list of 2 values
    ## 		The first is the averaged (completeness/prior), which can be pre-computed because it
    ##      will be multiplied by a scalar, ie the ORD in that bin
    ##		The second is the weight, which is just the fraction of samples that fall in that bin
	
    log_prod_term = 0 # AKA the sum of all single_bin_sum values
    for bin_ind in range(num_cells):
        lam_single = lam[bin_ind]
	
        single_bin_sum = 0
        for comp_name in comp_names:
             
            # Both compl_over_prior and weight are stored in one entry to save retrieve time
            cop_and_weight = bin_lam_dict[f"{comp_name}_cell{bin_ind}_compl_over_prior_avg_and_weight"]

            if cop_and_weight[1]==0: # Skip if weight=0 (companion does not overlap the bin)
                # print(f'If weight is 0 then copica should be NaN: {comp_over_prior_avg}')
                continue

            mean = lam_single*cop_and_weight[0]
            weighted_mean = mean**cop_and_weight[1] # Raise to power of weight
            single_bin_sum += np.log(weighted_mean+1e-300) # Ensure non-zero for stability

        log_prod_term += single_bin_sum

    loglik = -Lambda + log_prod_term

    if np.isnan(loglik):
        import pdb; pdb.set_trace()
		
    return loglik