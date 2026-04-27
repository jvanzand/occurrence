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


def summary_stats(chain_path, cell_dict, bin_lam_dict, m_unit='earth', verbose=False):
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
    #import pdb; pdb.set_trace()
    ORD_samples = ORD_samples[::thin]
    OR_samples = ORD_samples*cell_dict['all_binsizes']
    
    ORD_summary_dict = summarize_chains(ORD_samples, rate_type='ORD', hdi_frac=0.68, grid_size=1000)
    OR_summary_dict = summarize_chains(OR_samples, rate_type='OR', hdi_frac=0.68, grid_size=1000)
    summary_dict = {**ORD_summary_dict, **OR_summary_dict} # Combine to save both the OR and ORD
    #import pdb; pdb.set_trace()
    

    cell_weights = []
    for cell_ind in range(cell_dict['num_cells']):
        #import pdb; pdb.set_trace()
        all_cop_and_weights = np.array([bin_lam_dict[key] for key in bin_lam_dict.keys() 
                                           if f'cell{cell_ind}_compl_over_prior_avg_and_weight' in key])
        if all_cop_and_weights.size==0: # No companions in cell
            cell_weight=0
        else:
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
    
    if verbose:
        ## Print the occurrence rates, which are easier to interpret than the ORDs
        modes = summary_dict['mode_OR']
        hdi_low = summary_dict['hdi_low_OR']
        hdi_high = summary_dict['hdi_high_OR']
        cell_weights = summary_dict['cell_weights']
        cell_compls = summary_dict['cell_compls']
        a_m_lims_pairs = summary_dict['a_m_lims_pairs']


        m_quant = "M_c" if m_unit in ['earth', 'jupiter'] else "M_c/M_{\star}" if m_unit==None else ''
        m_unit_label = "Mearth" if m_unit=='earth' else "Mjup" if m_unit=='jupiter' else '' 
        for i in range(len(modes)):
            err_low = modes[i] - hdi_low[i]
            err_high = hdi_high[i] - modes[i]
            err = 0.5 * (err_low + err_high)
        
            a_lims, m_lims = a_m_lims_pairs[i]
            print(
                f"Cell {i}: a= {a_lims[0]}-{a_lims[1]} AU, {m_quant}={m_lims[0]:.1f}-{m_lims[1]:.1f} {m_unit_label} \n"
                f"    OR = {modes[i]:.3f} ± {err:.3f}, \n"
                f"    eff_planets = {cell_weights[i]:.2f}, \n"
                f"    average completeness = {cell_compls[i]:.3f}"
            )
    #import pdb; pdb.set_trace()
    return summary_dict


def summarize_chains(samples, rate_type='ORD', hdi_frac=0.68, grid_size=1000):
    """
    Compute mode and HDI for each parameter in MCMC chains.

    Arguments:
        samples (ndarray): shape (Nsamples, Ndim)
        rate_type (str): 'OR' or 'ORD' to label keys
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
    
    summaries = {f'mode_{rate_type}':mode_list,
                 f'hdi_low_{rate_type}':hdi_low_list,
                 f'hdi_high_{rate_type}':hdi_high_list}

    return summaries
























