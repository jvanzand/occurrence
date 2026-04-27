## Utility functions for making plots related to occurrence
import os
import matplotlib.pyplot as plt
import matplotlib.patches as ptch
import numpy as np
import corner
from matplotlib.ticker import FixedLocator, NullLocator, FormatStrFormatter, FuncFormatter
import itertools as itt
from astropy import constants as c

from pathlib import Path


def completeness_plotter(xgrid, ygrid, zgrid, save_path, title, save_plot=True, 
                         a_m_lims_pairs=None, summary_dict=None,
                         ycol='inj_msini', m_unit='earth'):
    """
    Adapted from RVSearch. Plot a completeness map using loaded
    xgrid, ygrid, and zgrid arrays.
                         
    Arguments:
        ycol (str): Determines y-label of completeness plot
                    Can be 'inj_msini', 'inj_mtrue', 'qsini', or 'qtrue'
        m_unit (str): Units listed in y-label of completeness plot
                      Can be 'earth' or 'jupiter'

    Code taken from DG work in:
    ~/my_papers/distant_giants/average_maps/cls_reconstructed/
    """
    
    fig = plt.figure(figsize=(7.5, 5.25))
    plt.subplots_adjust(bottom=0.18, left=0.22, right=0.95)
    
    CS = plt.contourf(xgrid, ygrid, zgrid, 10, cmap=plt.cm.Reds_r, vmax=0.9, alpha=0.7)
    fifty = plt.contour(xgrid, ygrid, zgrid, [0.5])

    
    #import pdb; pdb.set_trace()
    ax = plt.gca()
    ax.set_xscale('log')
    ax.set_yscale('log')

    xticks = plt.xticks()[0]
    xticks = 10**(np.linspace(np.log10(xticks[0]), 
                  np.log10(xticks[-1]), 
                  int(np.log10(xticks[-1]/xticks[0]))+1))
    plt.xticks(xticks, xticks)

    yticks = plt.yticks()[0]
    yticks = 10**(np.linspace(np.log10(yticks[0]), 
                  np.log10(yticks[-1]), 
                  int(np.log10(yticks[-1]/yticks[0]))+1))
    plt.yticks(yticks, yticks)

    plt.xlim(xgrid[0], xgrid[-1])
    plt.ylim(ygrid[0], ygrid[-1])

    if a_m_lims_pairs is not None:
        for i, a_m_lims in enumerate(a_m_lims_pairs):
            alims, mlims = a_m_lims
            a_m_anchor = alims[0], mlims[0]
            a_width, m_width = alims[1]-alims[0], mlims[1]-mlims[0]

            rect = ptch.Rectangle(a_m_anchor, a_width, m_width, linewidth=2.0, edgecolor='k', 
                                                               ls='-', facecolor='None', zorder=101)
            plt.gca().add_patch(rect)
            
    if summary_dict is not None and a_m_lims_pairs is not None:
        a_m_lims_pairs = summary_dict['a_m_lims_pairs']
        print(f"Occurrence-annotated completeness map saved to: {save_path}")
        for i, a_m_lims in enumerate(a_m_lims_pairs): 
            
            alims, mlims = a_m_lims
            a_m_anchor = alims[0], mlims[0]
            a_width, m_width = alims[1]-alims[0], mlims[1]-mlims[0]

            rect = ptch.Rectangle(a_m_anchor, a_width, m_width, linewidth=2.0, edgecolor='k', 
                                                               ls='-', facecolor='None', zorder=101)
            plt.gca().add_patch(rect)
            
            mode = summary_dict['mode_OR'][i]
            hdi_low = summary_dict['hdi_low_OR'][i]
            hdi_high = summary_dict['hdi_high_OR'][i]
            weight = summary_dict['cell_weights'][i]
            compl = summary_dict['cell_compls'][i]

            # Symmetric error
            err = 0.5 * ((mode - hdi_low) + (hdi_high - mode))

            # Annotation string
            text = (
                f"OR = {mode:.3f} ± {err:.3f}\n"
                f"N_eff = {weight:.2f}\n"
                f"C = {compl:.2f}"
            )
            # import pdb; pdb.set_trace()
            a_log_width = alims[1]/alims[0]
            m_log_width = mlims[1]/mlims[0]
            # Place text at center of rectangle
            x_center = alims[0] * a_log_width**0.5
            y_center = mlims[0] * m_log_width**0.5

            plt.text(
                x_center,y_center,
                text,
                ha='center',va='center',
                fontsize=11, zorder=102)

        ## "zoom in" on occurrence region so annotations are clearer
        min_a_cell = a_m_lims_pairs[0][0][0] # First cell, a pair, first element
        min_m_cell = a_m_lims_pairs[0][1][0] # First cell, m pair, first element
        max_a_cell = a_m_lims_pairs[-1][0][-1] # Last cell, a pair, second element
        max_m_cell = a_m_lims_pairs[-1][1][-1] # Last cell, m pair, second element
        plt.xlim([min_a_cell, max_a_cell])
        plt.ylim([min_m_cell, max_m_cell])
        #import pdb; pdb.set_trace()

    
    title_size = 22
    label_size = 18
    tick_size = 16
    cbar = plt.colorbar(mappable=CS, pad=0, label='probability of detection')
    cbar.set_label('probability of detection', size=16)
    cbar.ax.tick_params(labelsize=tick_size)

    xlabel = '$a$ [AU]'

    m_label = '[M$_\oplus$]' if m_unit=='earth' else '[M$_{Jup}$]' if m_unit=='jupiter' else None
    
    if ycol=='inj_msini':
        ylabel = r'M$_p\sin{i}$ '+m_label
    elif ycol=='inj_mtrue':
        ylabel = r'M$_p$ '+m_label
    elif ycol == 'inj_qsini':
        ylabel = r'M$_p\sin{i}$/M$_\star$'
    elif ycol == 'inj_qtrue':
        ylabel = r'M$_p$/M$_\star$'
    #xlabel = '$P$ [days]'
    #ylabel = 'K (m/s)'
    plt.title(title, size=title_size)
    plt.xlabel(xlabel, size=label_size)
    plt.ylabel(ylabel, size=label_size)

    plt.tick_params(axis='both', which='major', labelsize=tick_size)
    #plt.legend()

    plt.grid(True)
    fig.tight_layout(pad=0.1) # Minimize white space around border

    if save_plot:
        image_dpi = 200
        if summary_dict is not None:
            image_dpi=400
        plt.savefig(save_path, dpi=image_dpi)
        plt.close()

    
    return fig

def plot_catalog(tier1_dir, tier2_dir,
                 catalog_path,
                 a_edges=None, m_edges=None, 
                 m_unit='earth', fig_title='Average Completeness',
                 fig_savepath='catalog_and_completeness.png'):
    """
    Plot the planet catalog over the avg. completeness map
    
    Arguments:
        completeness_dir (str): Path to directory holding completeness
                                completeness maps
        catalog_path (str): Path to file holding companion samples
    """
    
    ## Plot the completeness map first
    avg_comp_path = os.path.join(tier1_dir, tier2_dir, 'avg_map/')
    xgrid = np.load(avg_comp_path+"parent_xgrid.npy")
    ygrid = np.load(avg_comp_path+"parent_ygrid.npy")
    zgrid = np.load(avg_comp_path+"parent_zgrid.npy")
    # import pdb; pdb.set_trace()
    
    #### Restructure a_edges and m_edges to make lims_pairs ####
    if a_edges is not None and m_edges is not None:
        a_lims_list = [[a_edges[i], a_edges[i+1]] for i in range(len(a_edges)-1)] # [[a0, a1], [a1, a2],...]
        m_lims_list = [[m_edges[i], m_edges[i+1]] for i in range(len(m_edges)-1)]
	
        # First create the pairs in the region of interest (out of order for ease)
        # Then reorder to look like [([a0,a1], [m0,m1]), ([a1,a2], [m0,m1]), ...]
        a_m_lims_pairs_roi_disordered = list(itt.product(m_lims_list, a_lims_list))
        a_m_lims_pairs = [pair[::-1] for pair in a_m_lims_pairs_roi_disordered]

            
            
    else:
        a_m_lims_pairs = None

    #import pdb; pdb.set_trace()
    ## Normally, you plot completeness using recoveries.csv, which has ycol in it
    ## Here, we are plotting from the x/y/z grids, so we have to provide ycol 'manually'
    ycol = 'inj_'+tier1_dir
    comp_fig = completeness_plotter(xgrid, ygrid, zgrid, 
                            'avg_comp.png', fig_title,
                            save_plot=False,
                            a_m_lims_pairs=a_m_lims_pairs,
                            ycol=ycol,
                            m_unit=m_unit)
    

    sampled_post_prior_compl_dict = dict(np.load(catalog_path))
    comp_names = sampled_post_prior_compl_dict.keys()
    
    ax = plt.gca()
    for cn in comp_names:
        #import pdb; pdb.set_trace()
        a_m_samples = sampled_post_prior_compl_dict[cn]
        
        #post = pd.read_csv(f'planet_posts/{cn}_post.csv')
        # ax.scatter(post.sma_au, post.mass_mearth, s=2)
        ax.scatter(a_m_samples[0], a_m_samples[1], s=2)
        
    parent_dir = Path(fig_savepath).parent.absolute()
    os.makedirs(parent_dir, exist_ok=True)
    comp_fig.savefig(fig_savepath, dpi=300)
    plt.close()
        
    return



def plot_corner_from_file(
    path_to_chains,
    plot_model,
    outpath="corner.png",
    param_names=None,
    thin=10,
    max_samples=50000):
    """
    Load MCMC chains from .npz file and generate a corner plot.
    Handles both histogram and power law models.

    Arguments:
        path_to_chains (str): Path to saved .npz file
        plot_model (str): Model type ('hist' or power law model name like 'pp1')
        plot_dim (str): Dimension being plotted ('a' or 'm'), used for power law models
        outpath (str): Output filename for plot
        param_names (list of str): Labels for parameters (if None, auto-generate based on model)
        thin (int): Thinning factor
        max_samples (int): Max number of samples to plot
    """

    data = np.load(path_to_chains)

    # Prefer flat chains if available
    if "flat_chains" in data:
        samples = data["flat_chains"]
    else:
        chains = data["chains"]  # (nsteps, nwalkers, ndim)
        samples = chains.reshape(-1, chains.shape[-1])

    # Thin samples
    samples = samples[::thin]

    # Subsample if too large
    if samples.shape[0] > max_samples:
        inds = np.random.choice(samples.shape[0], max_samples, replace=False)
        samples = samples[inds]

    ndim = samples.shape[1]

    # Determine parameter names based on model
    if param_names is None:
        if plot_model == 'hist':
            # Histogram model: use ORD labels for each dimension
            param_names = [f"$ORD_{{{i}}}$" for i in range(ndim)]
        elif plot_model == 'pp1':
            # pp1 model: C and dimension reference (a_0 or m_0)
            param_names = ['slope', 'intercept']
        elif plot_model == 'pp2':
            param_names = ['slope1', 'slope2', 'intercept1', 'break_point']
        else:
            # Default fallback
            param_names = [f"$\\theta_{{{i}}}$" for i in range(ndim)]

    # Make corner plot
    fig = corner.corner(
        samples,
        labels=param_names,
        show_titles=True,
        title_fmt=".3f",
        title_kwargs={"fontsize": 10},
    )

    plt.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Corner plot saved to: {outpath}")
    plt.close()
    return




def plot_occurrence_hist(summary_dict, stack_dim, m_unit='earth', mtype='mtrue',
                         rate_type='OR', title='', return_fig_ax=False,
                         savepath='occurrence.png', figsize=(6, 4)):
    """
    Plot occurrence histograms and save to file.

    Parameters
    ----------
    summary_dict : dict
        Your occurrence dictionary.
    stack_dim : str
        'm' → stack over mass (multiple mass histograms vs SMA)
        'a' → stack over SMA (multiple SMA histograms vs mass)
    savepath : str
        Path to save the figure.
    figsize : tuple
        Figure size.
    dpi : int
        Resolution for saved figure.

    Returns
    -------
    fig, ax
    """

    mode = np.array(summary_dict[f'mode_{rate_type}'])
    low = np.array(summary_dict[f'hdi_low_{rate_type}'])
    high = np.array(summary_dict[f'hdi_high_{rate_type}'])
    
    if rate_type=='OR':
        plot_ylabel = 'Occurrence rate\n[Planets per star]'
    elif rate_type=='ORD':
        #plot_ylabel = 'Occurrence rate density\n[Planets/star/$\Delta \log_{10}(a)$/$\Delta \log_{10}(M_c)$]'
        plot_ylabel = 'Occurrence rate density\n[Planets/star/$\Delta \log_{10}(\omega)$]'

    n_a = int(summary_dict['n_abins'])
    n_m = int(summary_dict['n_mbins'])

    pairs = np.array(summary_dict['a_m_lims_pairs'])

    # --- reshape into (mass, sma) ---
    mode = mode.reshape(n_m, n_a)
    low = low.reshape(n_m, n_a)
    high = high.reshape(n_m, n_a)

    # --- extract bin edges ---
    a_edges = np.array([pairs[i][0] for i in range(n_a)])
    a_edges = np.append(a_edges[:, 0], a_edges[-1, 1])

    m_edges = np.array([pairs[i * n_a][1] for i in range(n_m)])
    m_edges = np.append(m_edges[:, 0], m_edges[-1, 1])

    def get_err(m, l, h):
        return np.vstack([m - l, h - m])

    # label/tick sizes for consistent styling
    label_size = 22
    tick_size = 14
    
    
    ## Determine correct mass label
    m_unit_label = r'$M_{\oplus}$' if m_unit=='earth' else r'$M_{Jup}$' if m_unit=='jupiter' else None
    if mtype=='msini':
        mlabel = r'M$_c \sin{i}$ '+f'[{m_unit_label}]'
    elif mtype=='mtrue':
        mlabel = r'M$_c$ '+f'[{m_unit_label}]'
    elif mtype == 'qsini':
        mlabel = r'M$_c \sin{i}$/M$_\star$'
    elif mtype == 'qtrue':
        mlabel = r'M$_c$/M$_\star$'
    
    # Single-axis figure for 1D cases; for multi (stacked) create multiple subplots
    # =========================
    # CASE 1: 1D histogram
    # =========================
    if n_a == 1 or n_m == 1:

        fig, ax = plt.subplots(figsize=figsize)

        if n_a > 1:
            x_edges = a_edges
            y = mode[0]
            err = get_err(mode[0], low[0], high[0])
            xlabel = 'SMA [AU]'
        else:
            x_edges = m_edges
            y = mode[:, 0]
            err = get_err(mode[:, 0], low[:, 0], high[:, 0])
            xlabel = mlabel

        # Use geometric centers for log bins
        centers = np.sqrt(x_edges[:-1] * x_edges[1:])
        widths = np.diff(x_edges)

        # Draw horizontal bar outlines using helper
        x_pairs = np.column_stack((x_edges[:-1], x_edges[1:]))
        all_x, all_y = make_bar_vals(x_pairs, y)
        ax.plot(all_x, all_y, color='k', linewidth=2.5)

        # Error bars at geometric centers (black, thicker, with caps)
        ax.errorbar(centers, y, yerr=err, fmt='none', ecolor='k', elinewidth=2.0, capsize=4)

        # Scatter a black circle at the mode for each bin
        ax.scatter(centers, y, color='k', s=36, zorder=105)

        ax.set_xscale('log')
        # Set ticks at bin edges and format
        tick_label_fmt_fn = int_or_one_decimal if mtype in ['mtrue', 'msini'] \
                       else sci_no_leading_zero if mtype in ['qtrue', 'qsini'] \
                       else None
        ax.xaxis.set_major_locator(FixedLocator(x_edges))
        ax.xaxis.set_major_formatter(FuncFormatter(tick_label_fmt_fn))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.tick_params(axis='both', which='major', labelsize=tick_size)
        
        if mtype in ['qtrue', 'qsini']:
            plt.setp(ax.get_xticklabels(), rotation=-45, ha='left')

        ax.set_xlabel(xlabel)
        ax.set_ylabel(plot_ylabel, fontsize=label_size, labelpad=20)

        # Ensure y-range includes highest error bar
        high_err = err[1]
        top = np.nanmax(y + high_err) if len(y) > 0 else None
        if top is not None and np.isfinite(top):
            ax.set_ylim(0, max(ax.get_ylim()[1], 1.1 * top))
        
        ax.set_title(title)

    # =========================
    # CASE 2: 2D histogram
    # =========================
    else:
        twin_axis_formatter = lambda value: str(int(value)) if value.is_integer() else f"{value:.1f}"
        # For stacked histograms: create individual subplots for each stack element
        if stack_dim == 'm':
            # One subplot per mass bin (rows)
            x_edges = a_edges
            centers = np.sqrt(x_edges[:-1] * x_edges[1:])
            widths = np.diff(x_edges)

            # Make stacked subplots shorter and a bit wider
            stacked_fig_w = figsize[0] * 1.2
            stacked_fig_h = max(0.6 * figsize[1] * n_m, figsize[1])
            fig, ax = plt.subplots(n_m, 1, figsize=(stacked_fig_w, stacked_fig_h), sharex=True)
            if n_m == 1:
                ax = [ax]
            # Reverse axes so the smallest mass interval is plotted at the bottom
            ax = ax[::-1]
            for i in range(n_m):
                ax_i = ax[i]
                y = mode[i]
                err = get_err(mode[i], low[i], high[i])

                # Use helper to draw horizontal-only bar outlines
                x_pairs = np.column_stack((x_edges[:-1], x_edges[1:]))
                all_x, all_y = make_bar_vals(x_pairs, y)
                ax_i.plot(all_x, all_y, color='k', linewidth=2.5)

                # Error bars at geometric centers (black, thicker, with caps)
                centers_local = np.sqrt(x_edges[:-1] * x_edges[1:])
                ax_i.errorbar(centers_local, y, yerr=err, fmt='none', ecolor='k', elinewidth=2.0, capsize=4)

                # Scatter a black circle at the mode for each bin
                ax_i.scatter(centers_local, y, color='k', s=36, zorder=105)

                ax_i.set_xscale('log')
                # Show tick marks at bin edges rather than centers
                ax_i.xaxis.set_major_locator(FixedLocator(x_edges))
                ax_i.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                ax_i.xaxis.set_minor_locator(NullLocator())
                ax_i.tick_params(axis='both', which='major', labelsize=tick_size)

                # Add left-side vertical label showing mass range for this subplot
                m_lo = m_edges[i]
                m_hi = m_edges[i+1]
                lbl = rf"{mlabel}={twin_axis_formatter(m_lo)}-{twin_axis_formatter(m_hi)} [{m_unit_label}]"
                ax2 = ax_i.twinx()
                ax2.set_ylabel(lbl, size=label_size-6, rotation=90)
                ax2.set_yticks([])

                # Ensure y-range includes highest error bar for this subplot
                # per-subplot ylim will be set after plotting to enforce a common range

            # Compute a single y-axis upper limit across all subplots and apply
            global_top = np.nanmax(high)
            if np.isfinite(global_top):
                ytop = 1.1 * global_top
                for ax_i in ax:
                    ax_i.set_ylim(0, max(ax_i.get_ylim()[1], ytop))

            # Set the common y-label for the whole figure
            fig.supylabel(plot_ylabel, fontsize=label_size)
            # Set x-label on bottom-most axis (after reversing it's index 0)
            ax[0].set_xlabel('SMA [AU]', fontsize=label_size)

        elif stack_dim == 'a':
            # One subplot per SMA bin (rows), x-axis is mass
            x_edges = m_edges
            centers = np.sqrt(x_edges[:-1] * x_edges[1:])
            widths = np.diff(x_edges)

            # Make stacked subplots shorter and a bit wider
            stacked_fig_w = figsize[0] * 1.2
            stacked_fig_h = max(0.6 * figsize[1] * n_a, figsize[1])
            fig, ax = plt.subplots(n_a, 1, figsize=(stacked_fig_w, stacked_fig_h), sharex=True)
            if n_a == 1:
                ax = [ax]
            # Reverse axes so the smallest SMA interval is plotted at the bottom
            ax = ax[::-1]
            for i in range(n_a):
                ax_i = ax[i]
                y = mode[:, i]
                err = get_err(mode[:, i], low[:, i], high[:, i])

                # Use helper to draw horizontal-only bar outlines
                x_pairs = np.column_stack((x_edges[:-1], x_edges[1:]))
                all_x, all_y = make_bar_vals(x_pairs, y)
                ax_i.plot(all_x, all_y, color='k', linewidth=2.5)

                centers_local = np.sqrt(x_edges[:-1] * x_edges[1:])
                ax_i.errorbar(centers_local, y, yerr=err, fmt='none', ecolor='k', elinewidth=2.0, capsize=4)
                ax_i.scatter(centers_local, y, color='k', s=36, zorder=105)

                ax_i.set_xscale('log')
                # Show tick marks at bin edges rather than centers
                ax_i.xaxis.set_major_locator(FixedLocator(x_edges))
                ax_i.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                ax_i.xaxis.set_minor_locator(NullLocator())

                # Add left-side vertical label showing SMA range for this subplot
                a_lo = a_edges[i]
                a_hi = a_edges[i+1]
                lbl = rf"SMA={twin_axis_formatter(a_lo)}-{twin_axis_formatter(a_hi)} AU"
                ax2 = ax_i.twinx()
                ax2.set_ylabel(lbl, size=label_size-6, rotation=90)
                ax2.set_yticks([])

                # per-subplot ylim will be set after plotting to enforce a common range

            # Compute a single y-axis upper limit across all subplots and apply
            global_top = np.nanmax(high)
            if np.isfinite(global_top):
                ytop = 1.1 * global_top
                for ax_i in ax:
                    ax_i.set_ylim(0, max(ax_i.get_ylim()[1], ytop))

            # Set the common y-label for the whole figure
            fig.supylabel(plot_ylabel, fontsize=label_size)
            # Set x-label on bottom-most axis (after reversing it's index 0)
            ax[0].set_xlabel(f"{mlabel}", fontsize=label_size)

        else:
            raise ValueError("stack_dim must be 'm' or 'a'")
        
        fig.suptitle(title, fontsize=label_size)

    
    fig.tight_layout(rect=[0,0,1,0.95])
    
    if return_fig_ax:
        return fig, ax
    else:
        fig.savefig(savepath, dpi=300)
        plt.close()
        print(f"Occurrence histogram saved to: {savepath}")
        return



def make_bar_vals(x_pairs, y_vals):
    """
    Helper function to make bar
    plots. 

    Arguments:
        x_pairs (list): List of [x1, x2] pairs representing
            the bin bounds for each x interval
        y50 (list): List of floats corresponding to the 50th
            percentile of the occurrence distribution in the
            bin given by x_pairs
        y16 (list): Same as y50 for the 16th percentile
        y84 (list): Same as y50 for the 84th percentile
        

    Returns:
        all_x_flat, all_y_flat: lists that can be passed
                                directly to plt.plot() to 
                                make a bar plot.
    """

    # Bar graph
    x_pairs = np.asarray(x_pairs)
    all_x_flat = x_pairs.flatten() # [x0, x1, x1, x2]
    all_y_flat = np.array(y_vals).flatten().repeat(2) # Need each y-val twice per bin

    all_x_flat = np.insert(all_x_flat, # Duplicate the first and last elts for plotting
                           [0, len(all_x_flat)], 
                           [all_x_flat[0], all_x_flat[-1]])
    all_y_flat = np.insert(all_y_flat, [0, len(all_y_flat)], [0, 0]) # Dupl. to match a_edges_plot


    return all_x_flat, all_y_flat



def plot_power(fig, ax, model_func_name, save_path, n_draws=50):
    """
    Over-plot the max-likelihood model AND random posterior draws.

    Parameters
    ----------
    n_draws : int
        Number of posterior samples to plot (low-opacity)
    """

    from occurrence import mcmc_powerlaw as mcmc_power

    plot_dir = os.path.dirname(save_path)
    load_dir = os.path.dirname(plot_dir)
    chain_file = os.path.join(load_dir, 'saved_chains', f'chains_{model_func_name}.npz')

    data = np.load(chain_file)
    flat_chains = data['flat_chains']
    flat_log_probs = data['flat_log_probs']

    # --- Max likelihood ---
    ml_idx = np.argmax(flat_log_probs)
    ml_params = flat_chains[ml_idx]
    print("MAX LIKE:", ", ".join(f"{p:.3f}" for p in ml_params))

    # --- Model selection ---
    if model_func_name == 'pp1':
        model_func = mcmc_power.PiecewisePower1
    elif model_func_name == 'pp2':
        model_func = mcmc_power.PiecewisePower2
    elif model_func_name == 'escarpment':
        model_func = mcmc_power.escarpment
    else:
        raise NotImplementedError

    # --- Random posterior draws ---
    rng = np.random.default_rng()
    draw_indices = rng.choice(len(flat_chains), size=n_draws, replace=False)
    posterior_draws = flat_chains[draw_indices]

    # --- Handle axes ---
    if isinstance(ax, np.ndarray):
        axs_list = ax.flatten().tolist()
    elif isinstance(ax, list):
        axs_list = ax
    else:
        axs_list = [ax]

    # --- Plot ---
    for ax_i in axs_list:
        xlim = ax_i.get_xlim()
        x_model = np.logspace(np.log10(xlim[0]), np.log10(xlim[1]), 200)

        # Plot posterior draws (underneath)
        for theta in posterior_draws:
            y_draw = model_func(theta, x_model)
            ax_i.plot(
                x_model, y_draw,
                color='red',
                alpha=0.08,        # low opacity
                linewidth=1.0,
                zorder=10
            )

        # Plot max-likelihood (on top)
        y_model = model_func(ml_params, x_model)
        ax_i.plot(
            x_model, y_model,
            color='red',
            linewidth=2.5,
            label=f'{model_func_name} ML',
            zorder=100
        )

        ax_i.legend(loc='upper right', fontsize=10)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(save_path, dpi=300)
    plt.close(fig)

    print(f"Saved with posterior draws to: {save_path}")




def sci_no_leading_zero(x, pos):
    if x == 0:
        return "0"

    s = f"{x:.1e}"          # e.g. '2.0e-03'
    mant, exp = s.split('e')

    # remove trailing .0 if present
    if mant.endswith('.0'):
        mant = mant[:-2]

    exp = int(exp)          # removes leading zero ? -3

    return f"{mant}e{exp}"

def int_or_one_decimal(x, pos):
    if x == 0:
        return "0"

    # check if effectively an integer (robust to float precision)
    if float(x).is_integer():
        return str(int(x))
    else:
        return f"{x:.1f}"

