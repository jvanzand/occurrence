"""Posterior summary utilities for direct occurrence fits."""

import numpy as np
from scipy.stats import gaussian_kde


def summarize_chains(samples, rate_type='ORD', hdi_frac=0.68, grid_size=2000):
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

    mode_list = []
    hdi_low_list = []
    hdi_high_list = []

    for i in range(samples.shape[1]):
        chain_unmasked = samples[:, i]
        nanmask = ~np.isnan(chain_unmasked)
        chain = chain_unmasked[nanmask]
        
        ### Take the middle 98% of the chain to remove chance outliers
        #chain_lim_low, chain_lim_high = np.percentile(chain, [1, 99])
        #clipped_chain = chain[(chain_lim_low<chain) & (chain<chain_lim_high)]
        #chain = clipped_chain

        # --- HDI ---
        hdi_low, hdi_high = compute_hdi(chain, hdi_frac)
        hdi_width = hdi_high-hdi_low # HDI width gives "scale" of chain range

        # --- Mode via KDE ---
        kde = gaussian_kde(chain)
        x_grid = np.linspace(hdi_low-0.5*hdi_width, 
                             hdi_high+0.5*hdi_width, 
                             grid_size) # Only calc. over HDI, not min/max of chain (bc of outliers)
        pdf = kde(x_grid)
        mode = x_grid[np.argmax(pdf)]
        
        ## If the mode and HDI estimation fails using the KDE, just use percentiles
        if hdi_low>mode or mode>hdi_high:
            hdi_low, mode, hdi_high = np.percentile(chain, [0.5*(1-hdi_frac)*100, 50, (hdi_frac+0.5*(1-hdi_frac))*100])
        
        mode_list.append(mode)
        hdi_low_list.append(hdi_low)
        hdi_high_list.append(hdi_high)
    
    summaries = {f'mode_{rate_type}':mode_list,
                 f'hdi_low_{rate_type}':hdi_low_list,
                 f'hdi_high_{rate_type}':hdi_high_list}

    return summaries

def compute_hdi(x, frac):
    """Compute highest density interval (HDI)."""
    x_sorted = np.sort(x)
    N = len(x_sorted)
    interval_idx = int(np.floor(frac * N))

    widths = x_sorted[interval_idx:] - x_sorted[:N - interval_idx] # All widths contain the same # of values
    min_idx = np.argmin(widths) # Smallest width that contains that # of values

    return x_sorted[min_idx], x_sorted[min_idx + interval_idx]


