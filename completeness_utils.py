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


def map_interpolator(path, save_folder, star_subset, ycol='inj_msini', m_unit='earth'):
    """
    Function to average all computed completeness maps.
    - Also saves the interpolation function for each 
      individual map in that map's sub-directory.
    For xgrid and ygrid, the units match the input units.
    
    Arguments:
        path (str): Path to a parent dir containing subdirs for each system.
                    The subdirs should include files named "XXXX_xgrid.npy",
                    "XXXX_ygrid.npy", and "XXXX_zgrid.npy", where "XXXX"
                    is the target name.
        save_folder (str): Folder to save output arrays
        star_subset (list): List of stars to be included in average map
    
        ycol (str): Determines y-label of completeness plot
                    Can be 'inj_msini', 'inj_mtrue', 'qsini', or 'qtrue'
        m_unit (str): Units listed in y-label of completeness plot
                      Can be 'earth' or 'jupiter'

    Returns:
        parent_xgrid (1D array): x-values of completeness array
        parent_ygrid (1D array): y-values of completeness array
        parent_zgrid (2D array): Completeness at each (x, y) pair
    """
    
    os.makedirs(save_folder, exist_ok=True)
    # To average maps, we must line them up so we are averaging at the same (x,y) values.
    
    # First, find the global max/min y-values of all grids to define size of "parent" grid.
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []

    ni, nf = 0, None # Choose a subset of systems, if desired. 0, None to get all systems.
    #import pdb; pdb.set_trace()

    for subdir in os.listdir(path)[ni:nf]:
        if subdir not in star_subset:
            continue
        
        subdir_path = os.path.join(path, subdir)
        if not os.path.isdir(subdir_path):
            continue
            
        for file in os.listdir(subdir_path):
            file_path = os.path.join(subdir_path, file)

            if file_path.endswith("xgrid.npy"):
                xlist = np.load(file_path)
                xmins.append(min(xlist))
                xmaxs.append(max(xlist))
            if file_path.endswith("ygrid.npy"):
                ylist = np.load(file_path)
                ymins.append(min(ylist))
                ymaxs.append(max(ylist))

    # Find absolute bounds and go a little beyond them for buffer. Downloaded completeness grids are logarithmic, so make sure we provide logarithmic bounds as well.
    fac = 1.0001
    
    xmin = np.log10(fac*np.min(xmins))
    xmax = np.log10((1/fac)*np.max(xmaxs))
    ymin = np.log10(fac*np.min(ymins))
    ymax = np.log10((1/fac)*np.max(ymaxs))
    

    print("X lims", 10**xmin, 10**xmax)
    print("Y lims", 10**ymin, 10**ymax)
    grid_num = 50
    parent_xgrid = np.logspace(xmin, xmax, grid_num)
    parent_ygrid = np.logspace(ymin, ymax, grid_num)
    
    # Make parent grid of zeros to add interpolated values to
    parent_zgrid = np.zeros((grid_num, grid_num))
    parent_zgrid_tracker = np.zeros((grid_num, grid_num)) # Second grid to track where I've added values
    
    for subdir in os.listdir(path)[ni:nf]:
        if subdir not in star_subset:
            continue
        subdir_path = os.path.join(path, subdir)
        if os.path.isdir(subdir_path) and len(os.listdir(subdir_path))>0:
            system_name = subdir_path.split("/")[-1]
            
            target_xgrid = np.load(subdir_path+"/xgrid.npy")
            target_ygrid = np.load(subdir_path+"/ygrid.npy")
            target_zgrid = np.load(subdir_path+"/zgrid.npy")
        
    
        # Interpolation function based on the target's original map
        # Note: transpose zgrid to match correct (x,y) pairs to z values
        z_T = target_zgrid.T
        interpolator = sci.RegularGridInterpolator((target_xgrid, target_ygrid), z_T, 
                                                    bounds_error=False, fill_value=0)

        # Save the interpolation function for easy retrieval later
        with open(subdir_path+"/interp_fn.pkl", 'wb') as file_handle:
            pickle.dump(interpolator, file_handle)

        # Find the parent x-values that fall within the target grid. Same for y. These are the new approximation of the target x/y grids.
        target_xinds = np.where((min(target_xgrid)<parent_xgrid) 
                              & (max(target_xgrid)>parent_xgrid))[0]
        new_target_xgrid = parent_xgrid[target_xinds]

        target_yinds = np.where((min(target_ygrid)<parent_ygrid) 
                              & (max(target_ygrid)>parent_ygrid))[0]
        new_target_ygrid = parent_ygrid[target_yinds]

        print(subdir_path)
        print("X", min(target_xinds), max(target_xinds))
        print("Y", min(target_yinds), max(target_yinds))


        # Now loop through the new target grid, interpolate the value, and place it at the proper location in the parent grid.
        # i and j begin at 0. They let us iterate through the new_target_x/y_grids. xind and yind track where we are within parent_grid, so we can place the new interpolated values at the right place.
        for i, xind in enumerate(target_xinds):
            for j, yind in enumerate(target_yinds):
    
                target_grid_xval = new_target_xgrid[i] # Desired x-coord to interpolate
                target_grid_yval = new_target_ygrid[j] # Desired y-coord to interpolate
                interp_val = interpolator((target_grid_xval, target_grid_yval))
           
                
                if np.isnan(interp_val): # If nan value, don't increment the tracker
                    pass
                # Make parent_zgrid with [yind, x_ind] so y is the vertical axis and x the horizontal
                else:
                    parent_zgrid[yind, xind] += interp_val
                    parent_zgrid_tracker[yind, xind] += 1 # Add 1 to the tracker grid

    zero_inds = np.where(parent_zgrid_tracker<1e-6)
    print("ZERO INDS", np.shape(zero_inds))
    
    parent_zgrid_tracker[zero_inds] = np.nan # Set to NaN to avoid div by 0 error
    
    plt.imshow(parent_zgrid_tracker, origin="lower")
    plt.colorbar()
    plt.title("Number of maps used to calculate \n average in each cell")
    plt.savefig("{}/map_contribution.png".format(save_folder), dpi=300)
    plt.close()

    parent_zgrid = parent_zgrid / parent_zgrid_tracker # Average each point's completeness by dividing by number of maps that contributed only to that point

    ## Save completeness map
    np.save(f'{save_folder}/parent_xgrid', parent_xgrid)
    np.save(f'{save_folder}/parent_ygrid', parent_ygrid)
    np.save(f'{save_folder}/parent_zgrid', parent_zgrid)
    
    # Save average grid too
    z_T_avg = parent_zgrid.T
    avg_interpolator = sci.RegularGridInterpolator((parent_xgrid, parent_ygrid), z_T_avg,
                                                    bounds_error=False, fill_value=0)

    # Save the interpolation function for easy retrieval later
    with open(save_folder+"/interp_fn.pkl", 'wb') as file_handle:
        pickle.dump(avg_interpolator, file_handle)
    
    # import pdb; pdb.set_trace()
    plot_save_path = os.path.join(save_folder, 'average_completeness.png')
    plot_utils.completeness_plotter(parent_xgrid, parent_ygrid, parent_zgrid, 
                                    plot_save_path, 'Average Completeness', save_plot=True,
                                    ycol=ycol, m_unit=m_unit)
    

    return parent_xgrid, parent_ygrid, parent_zgrid



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
    fig = cplt.completeness_plot(title=system_name,
                                 xlabel=xlabel,
                                 ylabel=ylabel)

    #saveto = os.path.join(save_dir,'{}_recoveries.png'.format(system_name))
    saveto = os.path.join(save_dir, f'{system_name}_recoveries.png')

    fig.savefig(saveto, dpi=200)
    print("Recovery plot saved to {}".format(
           os.path.abspath(saveto)))

    return

def _process_single_star(args):
    """
    Helper function for parallel map creation.
    Wraps single_map_maker() above
    """
    row, path_to_recoveries, maps_save_path, maps_ycol, m_unit = args

    starname = row.star_name
    mstar = row.mstar

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























