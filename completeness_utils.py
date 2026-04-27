## Utility functions related to calculating and plotting completeness
import os
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.interpolate as sci
import pickle
from astropy import constants as c
import matplotlib.pyplot as plt

import occurrence.plotting_utils as plot_utils
import occurrence.rvsearch_borrowed as rvsb



def build_interpolators(maps_dir, star_subset):
    """
    Build and save interpolation functions for each system.
    """
    for subdir in os.listdir(maps_dir):
        if subdir not in star_subset:
            continue

        subdir_path = os.path.join(maps_dir, subdir)
        if not os.path.isdir(subdir_path):
            continue

        try:
            xgrid = np.load(os.path.join(subdir_path, "xgrid.npy"))
            ygrid = np.load(os.path.join(subdir_path, "ygrid.npy"))
            zgrid = np.load(os.path.join(subdir_path, "zgrid.npy"))
        except FileNotFoundError:
            continue

        # Build interpolator
        z_T = zgrid.T
        interpolator = sci.RegularGridInterpolator(
            (xgrid, ygrid), z_T,
            bounds_error=False, fill_value=0
        )

        # Save interpolator
        with open(os.path.join(subdir_path, "interp_fn.pkl"), "wb") as f:
            pickle.dump(interpolator, f)
            
    return


def average_map(maps_dir, avg_map_dir, star_subset, ycol='inj_msini', m_unit='earth'):
    """
    Load interpolators, compute average completeness map, and save outputs.
    """
    os.makedirs(avg_map_dir, exist_ok=True)

    xmins, xmaxs, ymins, ymaxs = [], [], [], []

    # --- Determine global bounds ---
    for subdir in os.listdir(maps_dir):
        if subdir not in star_subset:
            continue

        subdir_path = os.path.join(maps_dir, subdir)
        if not os.path.isdir(subdir_path):
            continue

        try:
            xgrid = np.load(os.path.join(subdir_path, "xgrid.npy"))
            ygrid = np.load(os.path.join(subdir_path, "ygrid.npy"))
        except FileNotFoundError:
            continue

        xmins.append(np.min(xgrid))
        xmaxs.append(np.max(xgrid))
        ymins.append(np.min(ygrid))
        ymaxs.append(np.max(ygrid))

    fac = 1.0001
    xmin = np.log10(fac * np.min(xmins))
    xmax = np.log10((1/fac) * np.max(xmaxs))
    ymin = np.log10(fac * np.min(ymins))
    ymax = np.log10((1/fac) * np.max(ymaxs))

    grid_num = 50
    parent_xgrid = np.logspace(xmin, xmax, grid_num)
    parent_ygrid = np.logspace(ymin, ymax, grid_num)

    parent_zgrid = np.zeros((grid_num, grid_num))
    tracker = np.zeros((grid_num, grid_num))

    # --- Loop over systems ---
    for subdir in os.listdir(maps_dir):
        if subdir not in star_subset:
            continue

        subdir_path = os.path.join(maps_dir, subdir)
        interp_path = os.path.join(subdir_path, "interp_fn.pkl")

        if not os.path.exists(interp_path):
            continue

        with open(interp_path, "rb") as f:
            interpolator = pickle.load(f)

        xgrid = np.load(os.path.join(subdir_path, "xgrid.npy"))
        ygrid = np.load(os.path.join(subdir_path, "ygrid.npy"))
        #import pdb; pdb.set_trace()
        xinds = np.where((xgrid.min() < parent_xgrid) & (xgrid.max() > parent_xgrid))[0]
        yinds = np.where((ygrid.min() < parent_ygrid) & (ygrid.max() > parent_ygrid))[0]

        X, Y = np.meshgrid(parent_xgrid[xinds], parent_ygrid[yinds], indexing='xy')
        points = np.column_stack([X.ravel(), Y.ravel()])
        vals = interpolator(points).reshape(X.shape)
        
        tracker[np.ix_(yinds, xinds)] += ~np.isnan(vals) # Add +1 to grid for each non-nan
        vals = np.nan_to_num(vals, nan=0.0) # Convert NaN vals to 0 so they don't affect the sum
        parent_zgrid[np.ix_(yinds, xinds)] += vals # Add vals to parent_zgrid
        

    ## Make map contribution plot
    fig, ax = plt.subplots()

    im = ax.imshow(tracker, origin="lower")
    fig.colorbar(im, ax=ax)
    ax.set_title("Number of maps contributing\nto each cell")

    fig.savefig(os.path.join(avg_map_dir, "map_contribution.png"), dpi=300)
    plt.close(fig)
    #############################


    tracker[tracker < 1e-6] = np.nan
    parent_zgrid /= tracker

    # --- Save outputs ---
    np.save(f'{avg_map_dir}/parent_xgrid', parent_xgrid)
    np.save(f'{avg_map_dir}/parent_ygrid', parent_ygrid)
    np.save(f'{avg_map_dir}/parent_zgrid', parent_zgrid)

    # Save average interpolator
    avg_interp = sci.RegularGridInterpolator(
        (parent_xgrid, parent_ygrid), parent_zgrid.T,
        bounds_error=False, fill_value=0
    )

    with open(os.path.join(avg_map_dir, "interp_fn.pkl"), "wb") as f:
        pickle.dump(avg_interp, f)

    # Plot
    plot_save_path = os.path.join(avg_map_dir, 'average_completeness.png')
    plot_utils.completeness_plotter(
        parent_xgrid, parent_ygrid, parent_zgrid,
        plot_save_path, 'Average Completeness',
        save_plot=True, ycol=ycol, m_unit=m_unit
    )

    return



def single_map_maker(system_name, recoveries_path, save_dir, mstar, 
                     ycol='inj_msini', m_unit='earth', 
                     fill_nans=True, trends_count=False):
    """
    Calculate completeness map from a recoveries.csv file
    and save the map, plus x, y, and z grids used to plot.

    Arguments:
        recoveries_path (str): Path to recoveries file to generate
                                completeness map
        save_dir (str): Path to save completeness products
        mstar (float): Mass of host star (M_sun)

    """

    os.makedirs(save_dir, exist_ok=True)


    xcol = 'inj_au'
    xlabel = '$a$ [AU]'
    
    m_label = '[M$_\oplus$]' if m_unit=='earth' else '[M$_{Jup}$]' if m_unit=='jupiter' else None
    
    if ycol=='inj_msini':
        ylabel = r'M$_p \sin{i}$ '+m_label
    elif ycol=='inj_mtrue':
        ylabel = r'M$_p$ '+m_label
    elif ycol == 'inj_qsini':
        ylabel = r'M$_p \sin{i}$/M$_\star$'
    elif ycol == 'inj_qtrue':
        ylabel = r'M$_p$/M$_\star$'

    #xcol = 'inj_period'
    #xlabel = '$P$ [days]'
    #ycol = 'inj_k'
    #ylabel = 'K (m/s)'
    #print("Plotting {} vs. {}".format(ycol, xcol))

    comp = rvsb.Completeness.from_csv(recoveries_path, xcol=xcol,
                                      ycol=ycol, mstar=mstar,
                                      y_unit=m_unit)

    #############################################


    cplt = rvsb.CompletenessPlots(comp, fill_nans=fill_nans, trends_count=trends_count)
    cplt.save_comp_grids(save_dir)
    

    
    save_single_plots=True
    if save_single_plots:
        fig = cplt.completeness_plot(title=system_name,
                                     xlabel=xlabel,
                                     ylabel=ylabel)

        saveto = os.path.join(save_dir, f'{system_name}_recoveries.png')

        fig.savefig(saveto, dpi=200)
        print("Recovery plot saved to {}".format(
               os.path.abspath(saveto)))
        plt.close(fig)
    else:
        print(f"Saved grids for {system_name}")

    return

def _process_single_star(args):
    """
    Helper function for parallel map creation.
    Wraps single_map_maker() above
    """
    row, path_to_recoveries, maps_save_path, maps_ycol, m_unit = args

    starname = row.star_name
    mstar = row.Mstar
    

    single_recoveries_path = os.path.join(
        path_to_recoveries, f'{starname}_recoveries.csv'
    )
    single_star_save_dir = os.path.join(maps_save_path, starname)

    single_map_maker(
        starname,
        single_recoveries_path,
        single_star_save_dir,
        mstar,
        ycol=maps_ycol,
        m_unit=m_unit,
        fill_nans=True
    )
    
    return


def recoveries_combiner(recoveries_path, sys_names, save_dir):
    """
	Combine recoveries dfs from a set of individual 
    recoveries files
    
    Arguments:
        recoveries_path (str): Path to directory containing individual recoveries files
        sys_names (list): List of systems to be included in combined recoveries file
    
    """
    combined_recs = pd.DataFrame({})
    for sys_name in sys_names:
        recs_orig = pd.read_csv(recoveries_path+f'{sys_name}_recoveries.csv')
		
        ## Append to master csv file
        combined_recs = combined_recs.append(recs_orig)
			
    print("Master recs is long", len(combined_recs))

    # Leave all masses in original units (default: M_earth)
    os.makedirs(save_dir, exist_ok=True)
    combined_recs.to_csv(os.path.join(save_dir, 'combined_recs.csv'), index=False)
    
    return


def recs_msini_converter(recoveries_path, save_file):
    """
    Convert a recoveries.csv file from Msini to Mtrue by
    marginalizing over inclination

    Arguments:
        recoveries_path (str): Path to a single recoveries.csv file
        save_file (str): Path to save new recoveries.csv file

    """
    save_dir = Path(save_file).parent # Ensure save file location exists
    os.makedirs(save_dir, exist_ok=True)


    ## Get original recoveries
    recs_orig = pd.read_csv(recoveries_path)


    ndraws = 10 # Num of realizations to marginalize over inclination
    msini = recs_orig['inj_msini']
    sini_vals = np.sin(np.arccos(np.random.uniform(size=ndraws))) # Cos-uniform dist
    m_true = (msini.to_numpy()[:,None] / sini_vals[None,:]).ravel() # Multiple M calcs per Msini

    # Create duplicate entries for other orb. elements to match new length of m_true
    au = recs_orig['inj_au'].repeat(ndraws)
    ecc = recs_orig['inj_e'].repeat(ndraws)
    recovered = recs_orig['recovered'].repeat(ndraws)

    # import pdb; pdb.set_trace()
    labels = ['inj_mtrue', 'inj_au', 'inj_e', 'recovered'] # Leave msini label for compatibility
    vals = [m_true, au, ecc, recovered]
    mtrue_recs = pd.DataFrame({label:val for label, val in zip(labels,vals)})

    mtrue_recs.to_csv(save_file, index=False)

    return

def recs_mass_ratio_converter(recoveries_path, save_file, mstar, m_unit='earth'):
    """
    Convert a recoveries.csv file from Msini or Mtrue to
    mass ratio (M/Mstar) by dividing every injection by
    the stellar mass.
    
    Arguments:
        recoveries_path (str): Path to a single recoveries.csv file
        save_file (str): Path to save new recoveries.csv file
        mstar (float): Stellar mass (solar masses)
        m_unit (str): Units of mass parameter. Either 'earth' or 'jupiter'
    
    """

    save_dir = Path(save_file).parent # Ensure save file location exists
    os.makedirs(save_dir, exist_ok=True)

    ## Get recoveries file with masses
    recs_orig = pd.read_csv(recoveries_path)
    
    if 'inj_mtrue' in recs_orig.columns:
        mass_col = 'inj_mtrue'
        qcol = 'inj_qtrue'
    elif 'inj_msini' in recs_orig.columns:
        mass_col = 'inj_msini'
        qcol = 'inj_qsini'
    else:
        raise Exception("recoveries.csv must contain either 'inj_msini' or 'inj_mtrue' column")
        
    if m_unit=='earth':
        mstar = mstar*c.M_sun.cgs.value/c.M_earth.cgs.value
    elif m_unit=='jupiter':
        mstar = mstar*c.M_sun.cgs.value/c.M_jup.cgs.value
    else:
        raise Exception('completeness_utils.recs_mass_ratio_converter: Specify a mass unit for recoveries.csv')
    
    
    recs_orig[qcol] = recs_orig[mass_col]/mstar
    recs_orig.to_csv(save_file, index=False)
    
    return


def cell_completeness(xlims, ylims, interp_fn):
    """
    Calculate the average completeness in a rectangular cell of parameter space.

    Arguments:
        xlims (tuple of floats): Min and max separations defining cell 
				(units are whatever the grid is in; for
				this project, AU)
        ylims (tuple of floats): Min and max M values defining cell
				(units are whatever the grid is in; for
				this project, MJup)
        interp_fn (fn): Function that accepts (a,m) vals and outputs completeness

    Returns:
        avg (float): Average completeness in grid cell
    """

    sub_xgrid = np.linspace(xlims[0], xlims[1], 100)
    sub_ygrid = np.linspace(ylims[0], ylims[1], 100)
    

    counter = 0
    total = 0
    for xval in sub_xgrid:
        for yval in sub_ygrid:
            counter += 1
            interp_val = interp_fn((xval, yval))
            if np.isnan(interp_val):
                raise Exception("NaN encountered in completeness_utils.cell_completeness")
                # import pdb; pdb.set_trace()
                total += nan_fill # Catch any Nans, although they should have been filled above
            else:
                total += interp_val


    if counter==0:
        avg = 0.001
        print("All NANs encountered in bin with xlim={} AU and ylim={} M_E".format(xlims, ylims))

    else:
        avg = total/counter

    return avg


def fill_completeness_nans(A, direction='high-mass'):
    """
    Function to fill NaN values in completeness maps.
    There is no exact way to do this; the real answer
    is that completeness should be well-measured in
    whatever region you want to measure occurrence in.
    However, in some cases it's obvious what the
    completeness is, even if it's not measured.
    
    This function takes a map (zgrid) and finds any
    place where a NaN is directly above (ie, at higher
    mass than) a cell with completeness=1, and fills
    all NaN
    """
    A = A.copy()

    nan_mask = np.isnan(A)


    ## Remember, the arrays fed into this fn are flipped in the y-direction
    ## That means that mass increases downward, so filling in completeness
    ## for high-mass cells means looking for compl=1 ABOVE (at lower mass than)
    ## a given NaN value.
    if direction=='high-mass':
        # Compare with value above
        above = np.roll(A, shift=1, axis=0) # All rows shifted down, to see what's above
        above[0, :] = np.nan # Set top row to NaN so it can't affect rows below

        start_mask = nan_mask & (above >= 0.90) # "start" cells. Nan values with 1 above them

        # Cumulative sum, so walk down a column and see if there are any "start" cells
        # that are NaN and have 1 above them. Once we hit one, the cumsum will always 
        # be >0
        activation = np.cumsum(start_mask, axis=0) > 0

        # Fill in any NaN value that is below a "start" cell
        fill_mask = activation & nan_mask

        A[fill_mask] = 1
    
    elif direction=='low-mass':
        # Compare with value below
        below = np.roll(A, shift=-1, axis=0) # Shift all rows up this time, to see what's below
        below[-1, :] = np.nan # Set bottom row to NaN

        start_mask = nan_mask & (below < 0.001) # "start" cells have 0 below them (ie, at higher mass)

        # To use the cumsum idea in the reverse direction, we have to reverse the mask
        # and fix it later
        start_up_reverse = start_mask[::-1]
        activation_up_rev = np.cumsum(start_up_reverse, axis=0) > 0
        activation = activation_up_rev[::-1]

        fill_mask = activation & nan_mask

        A[fill_mask] = 0

    return A























