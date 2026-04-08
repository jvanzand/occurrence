## Utility functions for making plots related to occurrence
import matplotlib.pyplot as plt
import matplotlib.patches as ptch
import numpy as np
import corner


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


