## Utility functions for making plots related to occurrence
import os
import numpy as np
import corner

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as ptch
from matplotlib.ticker import FixedLocator, NullLocator, FormatStrFormatter, FuncFormatter
from astropy import constants as c

from pathlib import Path
import itertools as itt

from occurrence import analysis_utils as au
from occurrence import occurrence_utils as ou


def completeness_plotter(xgrid, ygrid, zgrid, save_path, title, save_plot=True, 
                         a_m_lims_pairs=None, summary_dict=None,
                         zoom=False,
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
        #import pdb; pdb.set_trace()
        
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
        zoom=True

    if zoom==True and a_m_lims_pairs is not None:
        ## "zoom in" on occurrence region so annotations are clearer
        min_a_cell = a_m_lims_pairs[0][0][0] # First cell, a pair, first element
        min_m_cell = a_m_lims_pairs[0][1][0] # First cell, m pair, first element
        max_a_cell = a_m_lims_pairs[-1][0][-1] # Last cell, a pair, second element
        max_m_cell = a_m_lims_pairs[-1][1][-1] # Last cell, m pair, second element
        plt.xlim([min_a_cell, max_a_cell])
        plt.ylim([min_m_cell, max_m_cell])
        #import pdb; pdb.set_trace()

    
    title_size = 20
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
    if summary_dict is not None:
        
        ## Collect full interval info.
        mode_single = summary_dict['mode_OR_single'][0]
        hdi_low_single = summary_dict['hdi_low_OR_single'][0]
        hdi_high_single = summary_dict['hdi_high_OR_single'][0]
        weight_single = np.sum(summary_dict['cell_weights'])
        compl_single = summary_dict['cell_compl_single'] # This is originally calculated in ou.cell_values()
        err_single = 0.5 * ((mode_single - hdi_low_single) + (hdi_high_single - mode_single))
        
        #title = title + f"(N$_{{\\rm eff }}$={neff_total:.1f})"
        title = (
                 f"{title} "
                 f"(OR={mode_single:.3f}$\\pm${err_single:.3f}, "
                 f"N$_{{\\rm eff}}$={weight_single:.1f}, "
                 f"$\\bar{{C}}\\approx${compl_single:.2f})"
                )
        
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
                 zoom=False,
                 m_unit='earth',
                 star_df=None, star_param=None,
                 fig_title='Average Completeness',
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
                            zoom=zoom,
                            ycol=ycol,
                            m_unit=m_unit)
    

    sampled_post_prior_compl_dict = dict(np.load(catalog_path))
    comp_names = sampled_post_prior_compl_dict.keys()
    
    ax = plt.gca()
    for cn in comp_names:
        #import pdb; pdb.set_trace()
        a_m_samples = sampled_post_prior_compl_dict[cn]
        
        if star_df is not None:
            #import pdb; pdb.set_trace()
            a_m_samples = np.mean(a_m_samples, axis=1)[:2]
            cps_ident = cn.split('_')[0] # remove '_0', '_1', etc.
            star_row = star_df.query(f"cps_identifier==@cps_ident")[star_param].values
        
        #post = pd.read_csv(f'planet_posts/{cn}_post.csv')
        # ax.scatter(post.sma_au, post.mass_mearth, s=2)
        ax.scatter(a_m_samples[0], a_m_samples[1], s=2)
        
    if star_df is not None:
    
        new_labelsize = 22
        new_ticksize = 20
        
    
        if star_param not in ['Mstar', 'feh', 'age']:
            raise Exception("plotting_utils.py: star_param must be one of Mstar, feh, or age.")
    
        ## Make space on top of plot for colorbar
        comp_fig.subplots_adjust(top=0.82, left=0.14, right=0.98, bottom=0.14)
        
        star_param_dict = {'Mstar':['M$_{\star}$ (M$_{\odot}$)', 'Blues'],
                           'feh':['[Fe/H]', 'seismic'],
                           'age':['Age (Gyr)', 'Greens']}
        star_cmap_label, star_cmap = star_param_dict[star_param]
        
        ## Make colormap
        comp_host_names = [name.split('_')[0] for name in comp_names]
        star_sub_df = star_df[star_df['cps_identifier'].isin(comp_host_names)]
        norm = matplotlib.colors.Normalize(vmin=star_sub_df[star_param].min(),
                                           vmax=star_sub_df[star_param].max())
        #import pdb; pdb.set_trace()
        cmap = plt.get_cmap(star_cmap)
        cmap.set_bad('gray')
        
        for cn in comp_names:
            a_m_samples = sampled_post_prior_compl_dict[cn]
            a_m_samples = np.mean(a_m_samples, axis=1)[:2]
            cps_ident = cn.split('_')[0] # remove '_0', '_1', etc.
            star_param_val = star_sub_df.query(f"cps_identifier==@cps_ident")[star_param].values
            ax.scatter(a_m_samples[0], a_m_samples[1], c=cmap(norm(star_param_val)), edgecolor='k', s=60)

        from mpl_toolkits.axes_grid1 import make_axes_locatable
        import matplotlib as mpl

        #import pdb; pdb.set_trace()
        try:
            cbar_ax = comp_fig.axes[1]
            cbar_ax.yaxis.label.set_size(new_labelsize)
            cbar_ax.tick_params(axis='y', labelsize=new_ticksize)
            
            divider = make_axes_locatable(cbar_ax)
        except:
            divider = make_axes_locatable(ax)
        
        # Get the position of the main axes in figure coordinates
        bbox = ax.get_position()

        # Create a new axes above it
        cax_top = comp_fig.add_axes([
            bbox.x0,          # left
            bbox.y1 + 0.00,  # bottom
            bbox.width,       # same width as the plot
            0.03              # height of the colorbar
        ])

        sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])

        stellar_cbar = comp_fig.colorbar(sm, cax=cax_top, orientation="horizontal")
        stellar_cbar.set_label(star_cmap_label, size=new_labelsize)
        stellar_cbar.ax.xaxis.set_ticks_position("top")
        stellar_cbar.ax.xaxis.set_label_position("top")
        stellar_cbar.ax.tick_params(labelsize=new_ticksize)
        
        ax.set_title('')
        ax.xaxis.label.set_size(new_labelsize)
        ax.yaxis.label.set_size(new_labelsize)
        ax.tick_params(axis='both', labelsize=new_ticksize)
        
        ax.grid(False)
        #import pdb; pdb.set_trace()
        

    
    else:
        for cn in comp_names:
            a_m_samples = sampled_post_prior_compl_dict[cn]
            ax.scatter(a_m_samples[0], a_m_samples[1], s=2)
            
        
    parent_dir = Path(fig_savepath).parent.absolute()
    os.makedirs(parent_dir, exist_ok=True)
    #comp_fig.tight_layout(rect=[0,0,1,0.92])
    
    comp_fig.savefig(fig_savepath, dpi=300)
    plt.close()
        
    return



def plot_corner_from_file(
    path_to_chains,
    plot_model,
    model_dict,
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
        try:
            model_info = model_dict[plot_model]
            param_names = model_info[2]
        except:
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
    plt.close()

    print(f"Corner plot saved to: {outpath}")
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
        mtype_label = 'M$_c \sin{i}$'
        mlabel = mtype_label+' '+f'[{m_unit_label}]'
    elif mtype=='mtrue':
        mtype_label = 'M$_c$'
        mlabel = mtype_label+' '+f'[{m_unit_label}]'
    elif mtype == 'qsini':
        mtype_label = r'M$_c \sin{i}$/M$_\star$'
        mlabel = mtype_label
    elif mtype == 'qtrue':
        mtype_label = r'M$_c$/M$_\star$'
        mlabel = mtype_label

    # Single-axis figure for 1D cases; for multi (stacked) create multiple subplots
    # =========================
    # CASE 1: 1D histogram
    # =========================
    if n_a == 1 or n_m == 1:

        fig, ax = plt.subplots(figsize=figsize)

        if n_a > 1:
            x_edges = a_edges
            y_edges = m_edges
            y = mode[0]
            err = get_err(mode[0], low[0], high[0])
            xparam_label = 'SMA'
            xunit_label = '[AU]'
            xlabel = xparam_label+' '+xunit_label
            
            ## For labeling stack dimension
            yparam_label = mtype_label
            yunit_label = m_unit_label
            
        else:
            x_edges = m_edges
            y_edges = a_edges
            y = mode[:, 0]
            err = get_err(mode[:, 0], low[:, 0], high[:, 0])
            #xparam_label = mtype_label
            #xunit_label = f'[{m_unit_label}]'
            #xlabel = xparam_label+' '+xunit_label
            xlabel = mlabel
            #import pdb; pdb.set_trace()
            
            ## For labeling stack dimension
            yparam_label = 'SMA'
            yunit_label = 'AU'

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
            plt.setp(ax.get_xticklabels(), rotation=90, ha='left')

        ax.set_xlabel(xlabel)
        ax.set_ylabel(plot_ylabel, fontsize=label_size, labelpad=20)

        # Ensure y-range includes highest error bar
        high_err = err[1]
        top = np.nanmax(y + high_err) if len(y) > 0 else None
        if top is not None and np.isfinite(top):
            ax.set_ylim(0, max(ax.get_ylim()[1], 1.1 * top))
            
        # Add right-side vertical label showing SMA or Mass range for this subplot
        #import pdb; pdb.set_trace()
        twin_axis_formatter = lambda value: str(int(value)) if float(value).is_integer() else f"{value:.1f}"
        #import pdb; pdb.set_trace()
        y_lo, y_hi = y_edges
        

        lbl = rf"{yparam_label}={twin_axis_formatter(y_lo)}-{twin_axis_formatter(y_hi)} {yunit_label}"
        ax2 = ax.twinx()
        ax2.set_ylabel(lbl, size=label_size-6, rotation=90)
        ax2.set_yticks([])
        
        ax.set_title(title)

    # =========================
    # CASE 2: 2D histogram
    # =========================
    else:
        twin_axis_formatter = lambda value: str(int(value)) if float(value).is_integer() else f"{value:.1f}"
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

                # Add right-side vertical label showing mass range for this subplot
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
                #ax_i.yaxis.set_minor_locator(NullLocator())
                ax_i.tick_params(axis='both', which='major', labelsize=tick_size)

                # Add right-side vertical label showing SMA range for this subplot
                a_lo = a_edges[i]
                a_hi = a_edges[i+1]
                lbl = rf"SMA={twin_axis_formatter(a_lo)}-{twin_axis_formatter(a_hi)} AU"
                ax2 = ax_i.twinx()
                ax2.set_ylabel(lbl, size=label_size-6, rotation=90)
                ax2.set_yticks([])

                # per-subplot ylim will be set after plotting to enforce a common range
                
            
            
            # Set ticks at bin edges and format
            tick_label_fmt_fn = int_or_one_decimal if mtype in ['mtrue', 'msini'] \
                       else sci_no_leading_zero if mtype in ['qtrue', 'qsini'] \
                       else None
        
        if mtype in ['qtrue', 'qsini']:
            plt.setp(ax[0].get_xticklabels(), rotation=90, ha='left')

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

        ax[-1].set_title(title, fontsize=label_size) # Set title on top figure (suptitle is too high)
        #fig.suptitle(title, fontsize=label_size)

        ## Find and set the widest y lims to apply for all subplots
        min_y_allstack = np.min([ax[i].get_ylim()[0] for i in range(len(ax))])
        max_y_allstack = np.max([ax[i].get_ylim()[1] for i in range(len(ax))])
        for ax_i in ax:
            ax_i.set_ylim(min_y_allstack, max_y_allstack)
    
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



def plot_power(fig, ax, model_func_names, model_dict, save_path, stack_dim='m', n_draws=150):
    """
    Over-plot the max-likelihood model AND random posterior draws.

    Parameters
    ----------
    stack_dim : str
        Dimension along which histograms are stacked ('a' or 'm').
        Used to determine which chain file corresponds to each axis.
    n_draws : int
        Number of posterior samples to plot (low-opacity)
    """

    from occurrence import mcmc_powerlaw as mcmc_power

    plot_dir = os.path.dirname(save_path)
    load_dir = os.path.dirname(plot_dir)
    
    
    # --- Handle axes ---
    if isinstance(ax, np.ndarray):
        axs_list = ax.flatten().tolist()
    elif isinstance(ax, list):
        axs_list = ax
    else:
        axs_list = [ax]
    
    # Determine number of bins from the number of axes
    # When there are multiple stacked histograms, each axis corresponds to one bin
    n_bins = len(axs_list)

    # --- Plot ---
    for ax_idx, ax_i in enumerate(axs_list):
        # Determine which bin this axis corresponds to
        # The axes are reversed in plot_occurrence_hist, so the mapping is:
        # ax_idx 0 -> highest bin index (n_bins - 1)
        # ax_idx 1 -> bin index (n_bins - 2)
        # etc.
        #bin_idx = n_bins - 1 - ax_idx
        bin_idx = ax_idx
        
        if ax_idx==0:
            xlim = ax_i.get_xlim()
            ylim = ax_i.get_ylim()
        #import pdb; pdb.set_trace()
        for model_func_name in model_func_names:
            if model_func_name=='hist':
                continue
        
            # --- Model selection ---
                
            try:
                #import pdb; pdb.set_trace()
                model_info = model_dict[model_func_name]
                model_func = eval('mcmc_power.'+model_info[0])
                param_names = model_info[2]
                plot_clr = model_info[3]
            except:
                raise ValueError(f"Cannot calculate BIC for model: {model_func_name}")
        
            # Load the chain file for this bin
            chain_file = os.path.join(load_dir, 'saved_chains', f'chains_{model_func_name}_bin{bin_idx}.npz')
        
            # Check if file exists; if not, try the old naming scheme (single bin)
            if not os.path.exists(chain_file):
                chain_file = os.path.join(load_dir, 'saved_chains', f'chains_{model_func_name}.npz')
        
            data = np.load(chain_file)
            flat_chains = data['flat_chains']
            flat_log_probs = data['flat_log_probs']

            # --- Max likelihood ---
            ml_idx = np.argmax(flat_log_probs)
            ml_params = flat_chains[ml_idx]
            print(f"BIN {bin_idx} MAX LIKE:", ", ".join(f"{p:.3f}" for p in ml_params))

            # --- Random posterior draws ---
            rng = np.random.default_rng()
            draw_indices = rng.choice(len(flat_chains), size=n_draws, replace=False)
            posterior_draws = flat_chains[draw_indices]

            x_model = np.logspace(np.log10(xlim[0]), np.log10(xlim[1]), 2000)

            # Plot posterior draws (underneath)
            for theta in posterior_draws:
                y_draw = model_func(theta, x_model)
                ax_i.plot(
                    x_model, y_draw,
                    color=plot_clr,
                    alpha=0.08,        # low opacity
                    linewidth=1.0,
                    zorder=10
                )


            #import pdb; pdb.set_trace()
            ## Calculate derived params for each function
            
            full_len = flat_chains.shape[0]
            if full_len>20000: # Don't need huge chains to get summary stats. Take the end.
                model_samples = flat_chains[full_len-10000:,:] # 'burn in' until only 10k samples left
            else:
                model_samples = flat_chains
            
            
            if model_func_name=='logG':
                
                ## Here, we just want the median of the distribution, which is exp(mu). Plot only the central value
                #import pdb; pdb.set_trace()
                log_median_samples = model_samples[:,1]
                #param_dict = ou.summarize_chains(log_median_samples[:,None], rate_type='model')
                
                #param_mode = np.exp(param_dict['mode_model'][0]) # Exponentiate AFTER getting mode to avoid skewing
                #param_str_list = [f'exp($\mu$)={fmt_float_or_sci(param_mode)}']
                
                ## Remove chance outliers

                #median_samples = log_median_samples
                #minv = np.percentile(median_samples, 5, axis=0)
                #maxv = np.percentile(median_samples, 95, axis=0)
                #median_samples_trimmed = median_samples[(median_samples>minv)&(median_samples<maxv)]
                
                median_samples = log_median_samples
                #import pdb; pdb.set_trace()
                
                #median_samples = 10**log_median_samples
                
                param_dict = ou.summarize_chains(median_samples[:,None], rate_type='model', grid_size=2000)
                #param_name_list = ['10$^{\mu}$']
                param_name_list = ['$\mu$']
                #import pdb; pdb.set_trace()
                
                
            if model_func_name=='escarpment':
                #import pdb; pdb.set_trace()
                ## For escarpment, params are [C1, C2, log10BP1, log10BP2]
                ## Here, we want a handful of parameters
                #bp1_samples = 10**model_samples[:,2]
                #bp2_samples = 10**model_samples[:,3]
                log_bp1_samples = model_samples[:,2]
                log_bp2_samples = model_samples[:,3]
                
                ## slope is (y2-y1)/(x2-x1)
                #slope_samples = (model_samples[:,1]-model_samples[:,0])/(model_samples[:,3]-model_samples[:,2])
                #all_samples = np.vstack([bp1_samples, bp2_samples, slope_samples]).T
                slope_samples = (model_samples[:,1]-model_samples[:,0])/(model_samples[:,3]-model_samples[:,2])
                all_samples = np.vstack([log_bp1_samples, log_bp2_samples, slope_samples]).T
                
                #import pdb; pdb.set_trace()
                param_dict = ou.summarize_chains(all_samples, rate_type='model')
                #import pdb; pdb.set_trace()
                
                #param_name_list = ['BP1', 'BP2', 'slope']
                param_name_list = ['$\log_{10}$(BP1)', '$\log_{10}$(BP2)', 'slope']
            
            param_str_list=[]
            for i in range(len(list(param_dict.values())[0])):
        
                param_name = param_name_list[i]
                param_mode = param_dict['mode_model'][i]
                param_err_low = param_mode - param_dict['hdi_low_model'][i]
                param_err_high = param_dict['hdi_high_model'][i] - param_mode
                
                param_val_str = ou.latex_formatter(param_mode, param_err_low, param_err_high)
                
                param_str = f'{param_name}={param_val_str}'
                
                param_str_list.append(param_str)
                

                #import pdb; pdb.set_trace()
                
            legend_label = model_func_name+'\n'+'\n'.join(param_str_list)
            # Plot max-likelihood (on top)
            y_model = model_func(ml_params, x_model)
            ax_i.plot(
                x_model, y_model,
                color=plot_clr,
                linewidth=2.5,
                label=legend_label,
                #label=f'{model_func_name}: '+f', '.join([f"{name}={val:.2f}" for name, val in zip(param_names, ml_params)]),
                #label=f'{model_func_name} ML',
                zorder=100
            )

        ax_i.legend(loc='upper right', fontsize=10)
        ax_i.set_xlim(xlim)
        ax_i.set_ylim(ylim)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(save_path, dpi=300)
    plt.close(fig)

    print(f"Saved with posterior draws to: {save_path}")
    return



def plot_comparison_violins(
    ORD1_array, ORD2_array,
    label1, label2,
    hist_dict,
    diff_exceedance_frac, diff_zscore,
    shape_exceedance_frac, shape_zscore,
    best_fit_ratio,
    stack_dim,
    plot_title,
    save_path,
    m_type,
    m_unit
    ):
    """
    Create split violin plots comparing ORD1 and ORD2 distributions
    in each histogram bin.

    Parameters
    ----------
    ORD1_array : ndarray
        Shape = (Nsamples1, Nbins)

    ORD2_array : ndarray
        Shape = (Nsamples2, Nbins)

    Notes
    -----
    Each bin gets a split violin:
        - left half  = ORD1 samples
        - right half = ORD2 samples
    """

    import numpy as np
    import matplotlib.pyplot as plt
    
    label_size=22

    # -------------------------
    # Validate inputs
    # -------------------------
    if ORD1_array.ndim != 2 or ORD2_array.ndim != 2:
        raise ValueError("ORD arrays must be 2D")

    if ORD1_array.shape[1] != ORD2_array.shape[1]:
        raise ValueError(
            "ORD1_array and ORD2_array must have same number of bins"
        )

    Nbins = ORD1_array.shape[1]

    # -------------------------
    # Histogram geometry
    # -------------------------
    if stack_dim == 'a':
        stack_ind = 1
        n_nonstack_bins = hist_dict['n_abins']
        xlabel = 'Mass'

    elif stack_dim == 'm':
        stack_ind = 0
        n_nonstack_bins = hist_dict['n_mbins']
        xlabel = 'SMA [AU]'

    hist_lims_pairs = (
        hist_dict['a_m_lims_pairs'][:, stack_ind][::n_nonstack_bins]
    )

    bin_centers = (
        hist_lims_pairs[:, 0] * hist_lims_pairs[:, 1]
    ) ** 0.5

    # -------------------------
    # Labels
    # -------------------------
    ylabel = 'Occurrence rate density\n[Planets/star/$\Delta \log_{10}(\omega)$]'

    # -------------------------
    # Figure
    # -------------------------
    fig, ax = plt.subplots(figsize=(7, 5))

    # -------------------------
    # Violin widths
    # -------------------------
    bin_widths = (
        hist_lims_pairs[:, 1] - hist_lims_pairs[:, 0]
    )

    violin_widths = 0.35 * bin_widths

    # -------------------------
    # Build split violins
    # -------------------------
    all_vals = []

    for i in range(Nbins):
    
        violin_width = violin_widths[i]

        left_data = ORD1_array[:, i]
        right_data = ORD2_array[:, i]

        all_vals.extend(left_data)
        all_vals.extend(right_data)

        pos = bin_centers[i]

        # -------------------------
        # Left violin (ORD1)
        # -------------------------
        vp_left = ax.violinplot(
            left_data,
            positions=[pos],
            widths=violin_width,
            showmeans=False,
            showmedians=False,
            showextrema=False,
        )

        for body in vp_left['bodies']:

            verts = body.get_paths()[0].vertices

            # Keep only LEFT half
            verts[:, 0] = np.minimum(verts[:, 0], pos)

            body.set_facecolor('green')
            body.set_edgecolor('black')
            body.set_alpha(0.7)

        # -------------------------
        # Right violin (ORD2)
        # -------------------------
        vp_right = ax.violinplot(
            right_data,
            positions=[pos],
            widths=violin_width,
            showmeans=False,
            showmedians=False,
            showextrema=False,
        )

        for body in vp_right['bodies']:

            verts = body.get_paths()[0].vertices

            # Keep only RIGHT half
            verts[:, 0] = np.maximum(verts[:, 0], pos)

            body.set_facecolor('magenta')
            body.set_edgecolor('black')
            body.set_alpha(0.7)
            
        # -------------------------
        # Another Right violin (ORD2 * c)
        # -------------------------
        vp_right = ax.violinplot(
            right_data*best_fit_ratio,
            positions=[pos],
            widths=violin_width,
            showmeans=False,
            showmedians=False,
            showextrema=False,
        )

        for body in vp_right['bodies']:

            verts = body.get_paths()[0].vertices

            # Keep only RIGHT half
            verts[:, 0] = np.maximum(verts[:, 0], pos)

            body.set_facecolor('magenta')
            body.set_edgecolor('black')
            body.set_alpha(0.2)
        #import pdb; pdb.set_trace()

        # -------------------------
        # Annotate P(ORD1 > ORD2)
        # -------------------------
        y_text = max(
            np.nanmax(left_data),
            np.nanmax(right_data)
        )

        y_range = np.nanmax(all_vals) - np.nanmin(all_vals)

        #ax.text(
        #    pos,
        #    y_text + 0.03 * y_range,
        #    f"{frac_greater_array[i]:.3f}",
        #    ha='center',
        #    va='bottom',
        #    fontsize=16,
        #    color='black',
        #)

    # -------------------------
    # Axes formatting
    # -------------------------
    ax.set_xscale('log')

    bin_edges = np.unique(hist_lims_pairs.flatten())

    ax.set_xticks(bin_edges)
    
    
    # Set ticks at bin edges and format
    
    tick_label_fmt_fn = int_or_one_decimal if m_type == 'm' \
                       else sci_no_leading_zero if m_type == 'q' \
                       else None

    
    ax.xaxis.set_major_locator(FixedLocator(bin_edges))
    ax.xaxis.set_major_formatter(FuncFormatter(tick_label_fmt_fn))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.tick_params(axis='both', which='major', labelsize=14)
    

    if m_type == 'q':

        plt.setp(ax.get_xticklabels(), rotation=90, ha='left')

        ax.set_xlabel(r'M$_c$/M$_{\star}$', fontsize=label_size)

    elif m_type == 'm':

        x_unit = (
            '[$M_{\oplus}$]'
            if m_unit == 'earth'
            else '[$M_{Jup}$]'
        )

        ax.set_xlabel(r'M$_c$ ' + x_unit, fontsize=label_size)

    ax.xaxis.set_minor_locator(plt.NullLocator())

    # -------------------------
    # Y limits
    # -------------------------
    all_vals = np.array(all_vals)

    ymin = np.nanmin(all_vals)
    ymax = np.nanmax(all_vals)

    yrange = ymax - ymin

    ax.set_ylim(
        ymin - 0.05 * yrange,
        ymax + 0.15 * yrange,
    )

    # -------------------------
    # Labels
    # -------------------------
    ax.set_ylabel(ylabel, fontsize=label_size)

    ax.set_title(plot_title, fontsize=12)

    # Manual legend
    import matplotlib
    from matplotlib.patches import Patch

    legend_handles = [
        Patch(facecolor='green', edgecolor='black', label=f'{label1}', alpha=0.7),
        Patch(facecolor='magenta', edgecolor='black', label=f'{label2}', alpha=0.7),
        Patch(facecolor='magenta', edgecolor='black', label=f'c$\\times${label2}', alpha=0.2),
    ]

    legend = ax.legend(handles=legend_handles, fontsize=16)
    
    ## Annotate with exceedance fractions
    #import pdb; pdb.set_trace()
    
    annot_str = (
        f"Pval for null hypothesis 'A=B': "
            f"{fmt_float_or_sci(diff_exceedance_frac)} ({diff_zscore:.2f}$\\sigma$)\n"
        f"Pval for null hypothesis 'A=cB': "
            f"{fmt_float_or_sci(shape_exceedance_frac)} ({shape_zscore:.2f}$\\sigma$)\n"
            f"                          c={best_fit_ratio:.2f}"
            )
            
    offset = matplotlib.text.OffsetFrom(legend, (1.0, 0.0))
    ax.annotate(annot_str, xy=(0,0),size=14,
                xycoords='figure fraction', xytext=(0,-20), textcoords=offset, 
                horizontalalignment='right', verticalalignment='top')

    fig.tight_layout()

    fig.savefig(save_path, dpi=300)
    plt.close()

    return



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
        
def fmt_float_or_sci(x):
    return f"{x:.1e}" if abs(x) < 0.005 else f"{x:.2f}" 
        
def multiple_hist_dist(tier1_list, tier2_types, tier3_list,
                       stack_dim, m_unit):
    """
    Iterate over arguments to run au.hist_dist
    multiple times
    """
    
    for t1_val in tier1_list:
        for t2_type in tier2_types:
            for t3_val in tier3_list:
                 tier123_dir1 = os.path.join(t1_val, 'high'+t2_type, t3_val)
                 tier123_dir2 = os.path.join(t1_val, 'low'+t2_type, t3_val)
                 
                 label_t2 = '$M_{star}$' if t2_type=='Mstar' \
                      else '[Fe/H]' if t2_type=='FeH' \
                      else '$lR^{\prime}_{HK}$' if t2_type=='Act' \
                      else None
                 label1 = 'High '+label_t2
                 label2 = 'Low '+label_t2
                 
                 au.hist_dist(tier123_dir1=tier123_dir1,
                              tier123_dir2=tier123_dir2,
                              label1=label1,
                              label2=label2,
                              stack_dim=stack_dim,
                              m_unit=m_unit,
                              make_plot=True)
                 #import pdb; pdb.set_trace()
                     
    return









        
        
        
        
        
        
        
  

