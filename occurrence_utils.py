## Utility functions related to calculating completeness
import os
import numpy as np
import pandas as pd
import itertools as itt
import pickle
from pathlib import Path

import emcee
import multiprocessing as mp
from scipy.stats import gaussian_kde

from occurrence import completeness_utils as cu

from line_profiler import profile


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
    lam = np.random.uniform(0.001, 0.05, size=cell_dict['num_cells'])
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
        sampler = emcee.EnsembleSampler(nwalkers, ndim, loglik,
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

def loglik(lam, nstars, comp_names, bin_lam_dict, num_cells, all_binsizes, avg_cell_compls):
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
		
    #import pdb; pdb.set_trace()
    return loglik


def cell_values(a_edges, m_edges, avg_map_fn_path):
    """
    Helper function to produce useful values
    associated with cell edges and sizes
    """
    
    #### Define some useful bin-related values ####
    a_lims_list = [[a_edges[i], a_edges[i+1]] for i in range(len(a_edges)-1)] # [[a0, a1], [a1, a2],...]
    m_lims_list = [[m_edges[i], m_edges[i+1]] for i in range(len(m_edges)-1)]
	
    # First create the pairs in the region of interest (out of order for ease)
    # Then reorder to look like [([a0,a1], [m0,m1]), ([a1,a2], [m0,m1]), ...]
    a_m_lims_pairs_roi_disordered = list(itt.product(m_lims_list, a_lims_list))
    a_m_lims_pairs = [pair[::-1] for pair in a_m_lims_pairs_roi_disordered]
    
    ## Calculate the completeness in each cell. Preserve the order of the lims_pairs
    avg_interp = pickle.load(open(avg_map_fn_path, 'rb'))
    avg_compls = []
    for a_m_lims_pair in a_m_lims_pairs:
        a_lim, m_lim = a_m_lims_pair[0], a_m_lims_pair[1]
        try:
            cell_compl = cu.cell_completeness(a_lim, m_lim, avg_interp)
        except:
            import pdb; pdb.set_trace()
            raise Exception(f"occurrence_utils.cell_values: "\
                                f"Could not compute cell completeness between"\
                                f"a={a_lim} and M={m_lim}. Check for NaNs.")
        avg_compls.append(cell_compl)
    
	
    num_cells = len(a_m_lims_pairs)
    n_abins = len(a_lims_list)
    n_mbins = len(m_lims_list)
	
    a_binsizes = np.log10(a_edges[1:]/a_edges[:-1]) # Logarithmic bin sizes
    m_binsizes = np.log10(m_edges[1:]/m_edges[:-1])

    all_binsizes = np.outer(a_binsizes, m_binsizes).T.flatten()
    
    ## Calculate centers of bins
    # Complicated, but goal is to get an array of [[a_center0, m_center0], [a_center1, m_center0], ...]
    # That order matches the index function lam_ind_assign in likelihood.py. It labels bins in horizontal 
    # rows first, moving up vertically
    a_centers = (a_edges[1:]*a_edges[:-1])**0.5
    m_centers = (m_edges[1:]*m_edges[:-1])**0.5
    bin_center_array = np.array(list(itt.product(m_centers,a_centers)))[:,::-1]
    
    cell_dict = {'num_cells':num_cells,
                 'n_abins':n_abins,
                 'n_mbins':n_mbins, 
                 'a_m_lims_pairs':a_m_lims_pairs,
                 'avg_compls':avg_compls,
                 'all_binsizes':all_binsizes,
                 'bin_centers':bin_center_array}
    
    return cell_dict


def assign_cells(a_list, m_list, a_m_lims_pairs):
    """
    Given lists of a/m values, sort the pairs into
    the cells defined by a_m_lims_pairs. Any pair
    that falls outside the cells is given an ind
    of -1
    """
    a_list = np.asarray(a_list)
    m_list = np.asarray(m_list)
    
    lam_inds = np.full(len(a_list), -1, dtype=int)  # default = -1
    
    for i, ((a_lo, a_hi), (m_lo, m_hi)) in enumerate(a_m_lims_pairs):
        mask = (
            (a_list >= a_lo) & (a_list < a_hi) &
            (m_list >= m_lo) & (m_list < m_hi)
        )
        lam_inds[mask] = i
    
    return lam_inds


def summary_stats(chain_path, cell_dict, bin_lam_dict):
    """
    Load chains and print out the occurrence rate,
    effective number of planets, and average
    completeness in each cell. Store this in a
    dict to be used in other summary outputs
    """
    
    ## First, load chains and extract values
    data = np.load(chain_path)

    # Prefer flat chains if available
    if "flat_chains" in data:
        ORD_samples = data["flat_chains"]
    else:
        chains = data["chains"]  # (nsteps, nwalkers, ndim)
        ORD_samples = chains.reshape(-1, chains.shape[-1])
    
    # Thin samples
    thin=10
    ORD_samples = ORD_samples[::thin]
    OR_samples = ORD_samples*cell_dict['all_binsizes']
    
    summary_dict = summarize_chains(OR_samples, hdi_frac=0.68, grid_size=1000)
    
    #import pdb; pdb.set_trace()
    

    cell_weights = []
    for cell_ind in range(cell_dict['num_cells']):
        all_cop_and_weights = np.array([bin_lam_dict[key] for key in bin_lam_dict.keys() 
                                           if f'cell{cell_ind}_compl_over_prior_avg_and_weight' in key])
        # import pdb; pdb.set_trace()
        all_weights = all_cop_and_weights[:, 1] # Take the weights (ie, frac of each comp in that cell)
        cell_weight = np.sum(all_weights)
        cell_weights.append(cell_weight)
        # print(f"There are {cell_weight:.2f} effective planets in cell{cell_ind}")
    
    summary_dict['cell_weights'] = cell_weights
    summary_dict['cell_compls'] = cell_dict['avg_compls']
    
    ## Transfer these keys to summary_dict for plotting
    summary_dict['a_m_lims_pairs'] = cell_dict['a_m_lims_pairs']
    summary_dict['n_abins'] = cell_dict['n_abins']
    summary_dict['n_mbins'] = cell_dict['n_mbins']
    # import pdb; pdb.set_trace()
    
    modes = summary_dict['mode']
    hdi_low = summary_dict['hdi_low']
    hdi_high = summary_dict['hdi_high']
    cell_weights = summary_dict['cell_weights']
    cell_compls = summary_dict['cell_compls']
    a_m_lims_pairs = summary_dict['a_m_lims_pairs']

    for i in range(len(modes)):
        err_low = modes[i] - hdi_low[i]
        err_high = hdi_high[i] - modes[i]
        err = 0.5 * (err_low + err_high)
        
        a_lims, m_lims = a_m_lims_pairs[i]
        print(
            f"Cell {i}: {a_lims[0]}-{a_lims[1]} AU, {m_lims[0]:.1f}-{m_lims[1]:.1f} Mearth \n"
            f"    OR = {modes[i]:.3f} ± {err:.3f}, \n"
            f"    eff_planets = {cell_weights[i]:.2f}, \n"
            f"    average completeness = {cell_compls[i]:.3f}"
        )
    #import pdb; pdb.set_trace()
    return summary_dict


def summarize_chains(samples, hdi_frac=0.68, grid_size=1000):
    """
    Compute mode and HDI for each parameter in MCMC chains.

    Arguments:
        samples (ndarray): shape (Nsamples, Ndim)
        hdi_frac (float): fraction for HDI (default 0.68)
        grid_size (int): resolution for KDE mode estimation

    Returns:
        summary (list of dict): one per parameter with keys:
            'mode', 'hdi_low', 'hdi_high'
    """

    def compute_hdi(x, frac):
        """Compute highest density interval (HDI)."""
        x_sorted = np.sort(x)
        N = len(x_sorted)
        interval_idx = int(np.floor(frac * N))

        widths = x_sorted[interval_idx:] - x_sorted[:N - interval_idx]
        min_idx = np.argmin(widths)

        return x_sorted[min_idx], x_sorted[min_idx + interval_idx]

    mode_list = []
    hdi_low_list = []
    hdi_high_list = []

    for i in range(samples.shape[1]):
        chain = samples[:, i]

        # --- Mode via KDE ---
        kde = gaussian_kde(chain)
        x_grid = np.linspace(chain.min(), chain.max(), grid_size)
        pdf = kde(x_grid)
        mode = x_grid[np.argmax(pdf)]

        # --- HDI ---
        hdi_low, hdi_high = compute_hdi(chain, hdi_frac)
        
        mode_list.append(mode)
        hdi_low_list.append(hdi_low)
        hdi_high_list.append(hdi_high)
    
    summaries = {'mode':mode_list,
                 'hdi_low':hdi_low_list,
                 'hdi_high':hdi_high_list}

    return summaries
























