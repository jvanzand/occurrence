## Utility functions for making plots related to occurrence
import matplotlib.pyplot as plt
import matplotlib.patches as ptch
import numpy as np
import corner
from matplotlib.ticker import FixedLocator, NullLocator, FormatStrFormatter


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
            
    if summary_dict is not None:
        a_m_lims_pairs = summary_dict['a_m_lims_pairs']
        for i, a_m_lims in enumerate(a_m_lims_pairs): 
            
            alims, mlims = a_m_lims
            a_m_anchor = alims[0], mlims[0]
            a_width, m_width = alims[1]-alims[0], mlims[1]-mlims[0]

            rect = ptch.Rectangle(a_m_anchor, a_width, m_width, linewidth=2.0, edgecolor='k', 
                                                               ls='-', facecolor='None', zorder=101)
            plt.gca().add_patch(rect)
            
            mode = summary_dict['mode'][i]
            hdi_low = summary_dict['hdi_low'][i]
            hdi_high = summary_dict['hdi_high'][i]
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
                fontsize=3, zorder=102)

    
    title_size = 22
    label_size = 18
    tick_size = 16
    cbar = plt.colorbar(mappable=CS, pad=0, label='probability of detection')
    cbar.set_label('probability of detection', size=16)
    cbar.ax.tick_params(labelsize=tick_size)

    xlabel = '$a$ [AU]'

    m_label = '[M$_\oplus$]' if m_unit=='earth' else '[M$_{Jup}$]' if m_unit=='jupiter' else None
    
    if ycol=='inj_msini':
        ylabel = r'M$\sin{i_p}$ '+m_label
    elif ycol=='inj_mtrue':
        ylabel = r'M$_p$ '+m_label
    elif ycol == 'inj_qsini':
        ylabel = r'M$\sin{i_p}$/M$_\star$'
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


def plot_corner_from_file(
    filepath,
    cell_dict,
    outpath="corner.png",
    param_names=None,
    thin=10,
    max_samples=50000):
    """
    Load MCMC chains from .npz file and generate a corner plot.

    Arguments:
        filepath (str): Path to saved .npz file
        cell_dict (dict): Dictionary containing bin sizes
        outpath (str): Output filename for plot
        param_names (list of str): Labels for parameters
        thin (int): Thinning factor
        max_samples (int): Max number of samples to plot
    """

    data = np.load(filepath)

    # Prefer flat chains if available
    if "flat_chains" in data:
        ORD_samples = data["flat_chains"]
    else:
        chains = data["chains"]  # (nsteps, nwalkers, ndim)
        ORD_samples = chains.reshape(-1, chains.shape[-1])

    # Thin samples
    ORD_samples = ORD_samples[::thin]
    OR_samples = ORD_samples*cell_dict['all_binsizes']

    # Subsample if too large
    if OR_samples.shape[0] > max_samples:
        inds = np.random.choice(OR_samples.shape[0], max_samples, replace=False)
        OR_samples = OR_samples[inds]

    ndim = OR_samples.shape[1]

    # Default parameter names
    if param_names is None:
        param_names = [f"$\\lambda_{{{i}}}$" for i in range(ndim)]

    # Make corner plot
    fig = corner.corner(
        OR_samples,
        labels=param_names,
        show_titles=True,
        title_fmt=".3f",
        title_kwargs={"fontsize": 10},
    )

    plt.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Corner plot saved to: {outpath}")
    # import pdb; pdb.set_trace()
    return



def plot_occurrence_hist(summary_dict, stack_dim='m', savepath='occurrence.png',
                         figsize=(6, 4), dpi=300):
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

    mode = np.array(summary_dict['mode'])
    low = np.array(summary_dict['hdi_low'])
    high = np.array(summary_dict['hdi_high'])

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
    label_size = 24
    tick_size = 18
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
            xlabel = 'SMA'
        else:
            x_edges = m_edges
            y = mode[:, 0]
            err = get_err(mode[:, 0], low[:, 0], high[:, 0])
            xlabel = 'Mass'

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
        ax.xaxis.set_major_locator(FixedLocator(x_edges))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax.xaxis.set_minor_locator(NullLocator())

        unit = 'AU' if xlabel == 'SMA' else r'$M_{\oplus}$'
        ax.set_xlabel(f"{xlabel} [{unit}]")
        ax.set_ylabel('Occurrence rate\n[Planets per star]', fontsize=label_size, labelpad=20)

        # Ensure y-range includes highest error bar
        high_err = err[1]
        top = np.nanmax(y + high_err) if len(y) > 0 else None
        if top is not None and np.isfinite(top):
            ax.set_ylim(0, max(ax.get_ylim()[1], 1.1 * top))

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
            fig, axs = plt.subplots(n_m, 1, figsize=(stacked_fig_w, stacked_fig_h), sharex=True)
            if n_m == 1:
                axs = [axs]
            # Reverse axes so the smallest mass interval is plotted at the bottom
            axs = axs[::-1]
            for i in range(n_m):
                ax_i = axs[i]
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

                # Add left-side vertical label showing mass range for this subplot
                m_lo = m_edges[i]
                m_hi = m_edges[i+1]
                lbl = rf"M={twin_axis_formatter(m_lo)}-{twin_axis_formatter(m_hi)} $M_{{\oplus}}$"
                ax2 = ax_i.twinx()
                ax2.set_ylabel(lbl, size=label_size-6, rotation=90)
                ax2.set_yticks([])

                # Ensure y-range includes highest error bar for this subplot
                # per-subplot ylim will be set after plotting to enforce a common range

            # Compute a single y-axis upper limit across all subplots and apply
            global_top = np.nanmax(high)
            if np.isfinite(global_top):
                ytop = 1.1 * global_top
                for ax_i in axs:
                    ax_i.set_ylim(0, max(ax_i.get_ylim()[1], ytop))

            # Set the common y-label for the whole figure
            fig.supylabel('Occurrence rate\n[Planets per star]', fontsize=label_size)
            # Set x-label on bottom-most axis (after reversing it's index 0)
            axs[0].set_xlabel('SMA [AU]', fontsize=label_size)

        elif stack_dim == 'a':
            # One subplot per SMA bin (rows), x-axis is mass
            x_edges = m_edges
            centers = np.sqrt(x_edges[:-1] * x_edges[1:])
            widths = np.diff(x_edges)

            # Make stacked subplots shorter and a bit wider
            stacked_fig_w = figsize[0] * 1.2
            stacked_fig_h = max(0.6 * figsize[1] * n_a, figsize[1])
            fig, axs = plt.subplots(n_a, 1, figsize=(stacked_fig_w, stacked_fig_h), sharex=True)
            if n_a == 1:
                axs = [axs]
            # Reverse axes so the smallest SMA interval is plotted at the bottom
            axs = axs[::-1]
            for i in range(n_a):
                ax_i = axs[i]
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
                for ax_i in axs:
                    ax_i.set_ylim(0, max(ax_i.get_ylim()[1], ytop))

            # Set the common y-label for the whole figure
            fig.supylabel('Occurrence rate\n[Planets per star]', fontsize=label_size)
            # Set x-label on bottom-most axis (after reversing it's index 0)
            axs[0].set_xlabel("Mass [$M_{\oplus}$]", fontsize=label_size)

        else:
            raise ValueError("stack_dim must be 'm' or 'a'")

    fig.tight_layout()
    fig.savefig(savepath, dpi=dpi)
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





