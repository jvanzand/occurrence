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


def average_map(maps_dir, avg_map_dir, star_subset, ycol='inj_msini', m_unit='earth',
                xcol='inj_au'):
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
        save_plot=True, ycol=ycol, m_unit=m_unit, xcol=xcol
    )

    return



# Injection column for each x coordinate a Tier 1 can use: semi-major axis
# (AU) or orbital period (days, as injected)
X_COLUMNS = {"a": "inj_au", "p": "inj_period"}


def single_map_maker(system_name, recoveries_path, save_dir, mstar, 
                     ycol='inj_msini', m_unit='earth', 
                     fill_nans=True, trends_count=False,
                     save_plot=False, xcol='inj_au'):
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


    xlabel = plot_utils.X_LABELS[xcol]
    
    m_label = '[M$_\oplus$]' if m_unit=='earth' else '[M$_{Jup}$]' if m_unit=='jupiter' else None
    
    if ycol=='inj_msini':
        ylabel = r'M$_p \sin{i}$ '+m_label
    elif ycol=='inj_mtrue':
        ylabel = r'M$_p$ '+m_label
    elif ycol == 'inj_qsini':
        ylabel = r'M$_p \sin{i}$/M$_\star$'
    elif ycol == 'inj_qtrue':
        ylabel = r'M$_p$/M$_\star$'


    comp = rvsb.Completeness.from_csv(recoveries_path, xcol=xcol,
                                      ycol=ycol, mstar=mstar,
                                      y_unit=m_unit)

    #############################################


    cplt = rvsb.CompletenessPlots(comp, fill_nans=fill_nans, trends_count=trends_count)
    cplt.save_comp_grids(save_dir)
    

    
    if save_plot:
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
    (row, path_to_recoveries, maps_save_path, maps_ycol, m_unit,
     avg_map_only, save_single_plots, maps_xcol) = args

    starname = row.star_name
    mstar = row.Mstar
    
    if avg_map_only:
        starname='average'

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
        fill_nans=True,
        save_plot=save_single_plots,
        xcol=maps_xcol,
    )
    
    return


def recs_msini_converter(recoveries_path, save_file, keep_period=False):
    """
    Convert a recoveries.csv file from Msini to Mtrue by
    marginalizing over inclination

    Arguments:
        recoveries_path (str): Path to a single recoveries.csv file
        save_file (str): Path to save new recoveries.csv file
        keep_period (bool): Also keep inj_period, for period-based Tier 1s

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

    labels = ['inj_mtrue', 'inj_au', 'inj_e', 'recovered'] # Leave msini label for compatibility
    vals = [m_true, au, ecc, recovered]
    # Period-based Tier 1s also need the injected period; inclination does
    # not change it. Other runs omit it so their files stay as before.
    if keep_period:
        labels.append('inj_period')
        vals.append(recs_orig['inj_period'].repeat(ndraws))
    mtrue_recs = pd.DataFrame({label:val for label, val in zip(labels,vals)})

    mtrue_recs.to_csv(save_file, index=False)

    return

def recs_mass_ratio_converter(recoveries_path, save_file, mstar):
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
        
    ## Assuming recoveries file is in earth masses, calculate mstar in m_earth
    mstar = mstar*c.M_sun.cgs.value/c.M_earth.cgs.value
    
    
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
    import warnings

    sub_xgrid = np.logspace(np.log10(xlims[0]), np.log10(xlims[1]), 100)
    sub_ygrid = np.logspace(np.log10(ylims[0]), np.log10(ylims[1]), 100)
    

    counter = 0
    total = 0
    for xval in sub_xgrid:
        for yval in sub_ygrid:
            counter += 1
            interp_val = interp_fn((xval, yval))
            if np.isnan(interp_val):
                warnings.warn("NaN encountered in completeness_utils.cell_completeness")
                total += 1 # Assume NaNs are encountered in high-mass regions where q~1
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























# ---------------------------------------------------------------------------
# Eccentricity-sliced completeness
#
# Completeness depends on eccentricity as well as on a and m. Each star has
# only ~1000 injections (x10 inclination draws), drawn with e ~ Beta(0.867,
# 3.03), too few for a 3D moving average. Instead, injections are split into
# eccentricity slices, and each slice gets an ordinary (a, m) map from the same
# moving-average estimator and NaN-fill rules as the original maps, on a grid
# shared by all slices. Completeness is then piecewise constant in e within a
# slice. With a single slice spanning [0, 1] the result is exactly the
# original map.
# ---------------------------------------------------------------------------

# Narrower slices where injections are dense (low e), wider where sparse
ECC_SLICE_EDGES = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0)
SLICED_MAPS_SUFFIX = "_eslices"
SLICED_MAP_FILE = "sliced_map.npz"
SLICED_INTERP_FILE = "sliced_interp.pkl"


def eccentricity_slice_index(e, e_edges):
    """Return the slice holding each eccentricity (e = 1 joins the last)."""
    e = np.asarray(e, dtype=float)
    index = np.searchsorted(np.asarray(e_edges), e, side="right") - 1
    return np.clip(index, 0, len(e_edges) - 2)


class SlicedCompleteness:
    """Completeness piecewise constant in e, interpolated in (a, m).

    ``zgrids[s]`` is slice ``s``'s map on (``xgrid`` = a, ``ygrid`` = m), with
    rows along m as in the original maps. Calling the object with
    ``(a, m, e)`` returns completeness, like the original 2D interpolators
    (0 outside the grid, NaN where a slice has no estimate).
    """

    def __init__(self, xgrid, ygrid, zgrids, e_edges):
        self.xgrid = np.asarray(xgrid, dtype=float)
        self.ygrid = np.asarray(ygrid, dtype=float)
        self.zgrids = np.asarray(zgrids, dtype=float)
        self.e_edges = np.asarray(e_edges, dtype=float)
        if self.zgrids.shape != (len(self.e_edges) - 1, len(self.ygrid),
                                 len(self.xgrid)):
            raise ValueError("zgrids must have shape (slices, ny, nx)")
        self._interpolators = [
            sci.RegularGridInterpolator(
                (self.xgrid, self.ygrid), zgrid.T,
                bounds_error=False, fill_value=0,
            )
            for zgrid in self.zgrids
        ]

    @property
    def n_slices(self):
        return len(self.e_edges) - 1

    def slice_interpolator(self, index):
        """The 2D (a, m) interpolator of one slice."""
        return self._interpolators[index]

    def __call__(self, points):
        a, m, e = (np.asarray(values, dtype=float) for values in points)
        a, m, e = np.broadcast_arrays(a, m, e)
        result = np.empty(a.shape, dtype=float)
        index = eccentricity_slice_index(e, self.e_edges)
        for slice_index in np.unique(index):
            selected = index == slice_index
            result[selected] = self._interpolators[slice_index](
                (a[selected], m[selected])
            )
        return result

    def save(self, directory, extra=None):
        """Write the grids (npz) and the pickled interpolator."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.savez(directory / SLICED_MAP_FILE, xgrid=self.xgrid,
                 ygrid=self.ygrid, zgrids=self.zgrids, e_edges=self.e_edges,
                 **(extra or {}))
        with open(directory / SLICED_INTERP_FILE, "wb") as stream:
            pickle.dump(self, stream)


def load_sliced(directory):
    """Load a :class:`SlicedCompleteness` saved by :meth:`save`."""
    with open(Path(directory) / SLICED_INTERP_FILE, "rb") as stream:
        return pickle.load(stream)


def moving_average_grid(xinj, yinj, good, xlim, ylim, resolution=30,
                        xlogwin=0.5, ylogwin=0.5, fill_nans=True):
    """Vectorized copy of ``rvsb.Completeness.completeness_grid``.

    Same estimator and same result (checked by tests): on a log grid, the
    completeness at each point is the recovered fraction of injections in a
    box ``xlogwin`` x ``ylogwin`` dex wide, NaN where the box holds five or
    fewer injections or the point lies outside the injections' mass range at
    that separation; NaNs are then filled (1 above full completeness, 0 below
    zero completeness). Counting with sorted arrays instead of a Python loop
    over pandas masks makes it about 50 times faster.
    """
    xinj = np.asarray(xinj, dtype=float)
    yinj = np.asarray(yinj, dtype=float)
    good = np.asarray(good, dtype=bool)
    xgrid = np.logspace(np.log10(xlim[0]), np.log10(xlim[1]), resolution)
    ygrid = np.logspace(np.log10(ylim[0]), np.log10(ylim[1]), resolution)
    ylow = 10**(np.log10(ygrid) - ylogwin/2)
    yhigh = 10**(np.log10(ygrid) + ylogwin/2)
    z = np.full((len(ygrid), len(xgrid)), np.nan)
    for i, x in enumerate(xgrid):
        xlow = 10**(np.log10(x) - xlogwin/2)
        xhigh = 10**(np.log10(x) + xlogwin/2)
        in_x = (xinj <= xhigh) & (xinj >= xlow)
        ybox = np.sort(yinj[in_x])
        if ybox.size == 0:
            continue
        ygood = np.sort(yinj[in_x & good])
        # Counts with both box edges included, as in the original
        boxall = (np.searchsorted(ybox, yhigh, side="right") -
                  np.searchsorted(ybox, ylow, side="left"))
        boxgood = (np.searchsorted(ygood, yhigh, side="right") -
                   np.searchsorted(ygood, ylow, side="left"))
        inside = (ygrid <= ybox[-1]) & (ygrid >= ybox[0]) & (boxall > 5)
        z[inside, i] = boxgood[inside]/boxall[inside]
    if fill_nans:
        z = fill_completeness_nans(z, direction="high-mass")
        z = fill_completeness_nans(z, direction="low-mass")
    return xgrid, ygrid, z


def sliced_map_maker(recoveries_path, save_dir, ycol="inj_mtrue",
                     m_unit="earth", e_edges=ECC_SLICE_EDGES):
    """Make one star's eccentricity-sliced completeness map.

    Every slice uses the same (a, m) grid, set by the full injection set's
    range as in the original maps, and the same estimator: a moving average
    over 0.5-dex windows needing more than five injections, with NaNs above
    full completeness filled with 1 and below zero completeness with 0.
    Returns the number of injections (after inclination draws) per slice.
    """
    recoveries = pd.read_csv(recoveries_path)
    if "inj_e" not in recoveries.columns:
        raise KeyError(f"{recoveries_path} has no inj_e column")
    # Converts the mass column to the requested unit, as the original maps do
    full = rvsb.Completeness(recoveries, xcol="inj_au", ycol=ycol,
                             y_unit=m_unit)
    converted = full.recoveries
    xlim = (converted["inj_au"].min(), converted["inj_au"].max())
    ylim = (converted[ycol].min(), converted[ycol].max())

    index = eccentricity_slice_index(converted["inj_e"], e_edges)
    zgrids, counts = [], []
    for slice_index in range(len(e_edges) - 1):
        subset = converted[index == slice_index].reset_index(drop=True)
        counts.append(len(subset))
        if len(subset) == 0:
            zgrids.append(None)
            continue
        xgrid, ygrid, zgrid = moving_average_grid(
            subset["inj_au"], subset[ycol], subset["recovered"], xlim, ylim
        )
        zgrids.append(zgrid)
    template = next(z for z in zgrids if z is not None)
    zgrids = [np.full_like(template, np.nan) if z is None else z
              for z in zgrids]
    sliced = SlicedCompleteness(xgrid, ygrid, zgrids, e_edges)
    sliced.save(save_dir, extra={"counts": np.asarray(counts)})
    return np.asarray(counts)


def _process_single_star_sliced(args):
    """Helper for parallel sliced-map creation."""
    (starname, path_to_recoveries, maps_save_path, ycol, m_unit,
     e_edges) = args
    return starname, sliced_map_maker(
        os.path.join(path_to_recoveries, f"{starname}_recoveries.csv"),
        os.path.join(maps_save_path, starname), ycol=ycol, m_unit=m_unit,
        e_edges=e_edges,
    )


def sliced_average_map(maps_dir, avg_map_dir, star_subset,
                       e_edges=ECC_SLICE_EDGES):
    """Average the stars' sliced maps, slice by slice.

    Each slice is averaged exactly as :func:`average_map` averages the
    original maps: on a 50x50 log grid spanning the stars' grids, counting
    only stars with a non-NaN value at each point. Points where no star has
    an estimate in a slice (sparse, high-e slices) take the value of the
    nearest lower-e slice; their number is returned per slice so the fill
    can be reported.
    """
    avg_map_dir = Path(avg_map_dir)
    avg_map_dir.mkdir(parents=True, exist_ok=True)
    star_maps = []
    for star in star_subset:
        path = Path(maps_dir) / star
        if (path / SLICED_INTERP_FILE).exists():
            star_maps.append(load_sliced(path))
    if not star_maps:
        raise FileNotFoundError(f"no sliced maps for the stars in {maps_dir}")
    for sliced in star_maps:
        if not np.array_equal(sliced.e_edges, e_edges):
            raise ValueError("stars' sliced maps use different e slices")

    fac = 1.0001
    xmin = np.log10(fac*min(s.xgrid.min() for s in star_maps))
    xmax = np.log10(max(s.xgrid.max() for s in star_maps)/fac)
    ymin = np.log10(fac*min(s.ygrid.min() for s in star_maps))
    ymax = np.log10(max(s.ygrid.max() for s in star_maps)/fac)
    grid_num = 50
    parent_xgrid = np.logspace(xmin, xmax, grid_num)
    parent_ygrid = np.logspace(ymin, ymax, grid_num)

    n_slices = len(e_edges) - 1
    zgrids = np.zeros((n_slices, grid_num, grid_num))
    trackers = np.zeros((n_slices, grid_num, grid_num))
    for sliced in star_maps:
        xinds = np.where((sliced.xgrid.min() < parent_xgrid) &
                         (sliced.xgrid.max() > parent_xgrid))[0]
        yinds = np.where((sliced.ygrid.min() < parent_ygrid) &
                         (sliced.ygrid.max() > parent_ygrid))[0]
        X, Y = np.meshgrid(parent_xgrid[xinds], parent_ygrid[yinds],
                           indexing="xy")
        for slice_index in range(n_slices):
            vals = sliced.slice_interpolator(slice_index)(
                np.column_stack([X.ravel(), Y.ravel()])
            ).reshape(X.shape)
            trackers[slice_index][np.ix_(yinds, xinds)] += ~np.isnan(vals)
            zgrids[slice_index][np.ix_(yinds, xinds)] += np.nan_to_num(vals)

    with np.errstate(invalid="ignore", divide="ignore"):
        zgrids = np.where(trackers > 0, zgrids/trackers, np.nan)
    # Cells where no star covers the whole grid stay NaN in every slice
    filled_from_lower = np.zeros(n_slices, dtype=int)
    for slice_index in range(1, n_slices):
        missing = np.isnan(zgrids[slice_index]) & ~np.isnan(
            zgrids[slice_index - 1]
        )
        filled_from_lower[slice_index] = int(missing.sum())
        zgrids[slice_index][missing] = zgrids[slice_index - 1][missing]

    average = SlicedCompleteness(parent_xgrid, parent_ygrid, zgrids, e_edges)
    average.save(avg_map_dir, extra={
        "stars_per_point": trackers,
        "filled_from_lower_slice": filled_from_lower,
        "n_stars": np.asarray(len(star_maps)),
    })
    return average, filled_from_lower


# Grid sizes for engine maps: points along a fitted e axis, and integration
# cells across a marginalized a or m range
ENGINE_E_POINTS = 201
MARGINAL_LOG_CELLS = 120


def _dimension_nodes(name, dimension, sliced):
    """Evaluation points (and weights, if marginalized) for one dimension.

    A fitted dimension is evaluated on the sliced map's own grid (``a`` and
    ``m``) or on a fine grid (``e``), so the engine interpolates over the
    whole map. A marginalized dimension is integrated over its bounds:
    weights are the population distribution's exact probability in each cell
    (cells are the e slices themselves for ``e``, since completeness is
    constant within a slice, and fine log cells for ``a`` and ``m``), and
    completeness is evaluated at each cell's center.
    """
    if dimension.role == "fit":
        if name == "a":
            return sliced.xgrid, None
        if name == "m":
            return sliced.ygrid, None
        return np.linspace(0.0, 1.0, ENGINE_E_POINTS), None

    low, high = dimension.bounds
    distribution = dimension.distribution
    if name == "e":
        edges = np.unique(np.clip(sliced.e_edges, low, high))
        centers = 0.5*(edges[1:] + edges[:-1])
    else:
        edges = np.logspace(np.log10(low), np.log10(high),
                            MARGINAL_LOG_CELLS + 1)
        centers = np.sqrt(edges[1:]*edges[:-1])
    weights = distribution.cell_probabilities(edges)
    return centers, weights


def engine_completeness(spec, sliced):
    """Completeness on the engine's (x, y) slots, marginalized per ``spec``.

    Evaluates the sliced map over every combination of the three dimensions'
    nodes, then integrates out each marginalized dimension with its
    population weights (the independence assumption: the marginalized
    dimensions follow their distributions regardless of the fitted ones).
    When the y slot holds a marginalized dimension, the result does not
    depend on y and is repeated at its two bounds. Returns a 2D interpolator
    in engine coordinates (eccentricity as u = 10**e), with 0 outside the
    grid like the original average maps.

    Raises if a marginalized dimension's range meets a point with no
    completeness estimate (NaN): averaging over the points that do have one
    would silently bias the result (see ECCENTRICITY_PLAN.md, step 3).
    """
    nodes, weights = {}, {}
    for name in ("a", "m", "e"):
        nodes[name], weights[name] = _dimension_nodes(
            name, spec.dimensions[name], sliced
        )
    A, M, E = np.meshgrid(nodes["a"], nodes["m"], nodes["e"], indexing="ij")
    values = sliced((A, M, E))

    marginalized_axes = [axis for axis, name in enumerate(("a", "m", "e"))
                         if weights[name] is not None]
    if marginalized_axes:
        # A NaN in any cell of a marginalized range poisons that output point
        bad = np.isnan(values).any(axis=tuple(marginalized_axes))
        kept_axes = [axis for axis in range(3) if axis not in marginalized_axes]
        fitted_names = [("a", "m", "e")[axis] for axis in kept_axes]
        bad_within = _within_fit_bounds(spec, fitted_names, nodes, bad)
        if np.any(bad_within):
            raise ValueError(
                f"completeness is undefined at {int(bad_within.sum())} of "
                f"{bad_within.size} points of the fitted range ("
                f"{spec.describe()}), so the marginalized dimensions cannot "
                "be averaged there. Narrow the marginalized bounds or widen "
                "the maps' coverage."
            )
    if marginalized_axes:
        # Product of the marginalized dimensions' cell probabilities, then a
        # single sum over all of their axes
        total_weight = np.ones(values.shape)
        for axis in marginalized_axes:
            shape = [1, 1, 1]
            shape[axis] = -1
            total_weight = total_weight*weights[("a", "m", "e")[axis]].reshape(
                shape
            )
        values = np.sum(values*total_weight, axis=tuple(marginalized_axes))

    fitted = [name for name in ("a", "m", "e") if weights[name] is None]
    if spec.y_is_marginalized:
        (x_name,) = fitted
        x_points = spec.to_engine(x_name, nodes[x_name])
        y_points = spec.to_engine(spec.y, spec.dimensions[spec.y].bounds)
        z = np.repeat(values[:, None], 2, axis=1)
    else:
        x_name, y_name = spec.x, spec.y
        x_points = spec.to_engine(x_name, nodes[x_name])
        y_points = spec.to_engine(y_name, nodes[y_name])
        z = values if fitted.index(x_name) == 0 else values.T
    return sci.RegularGridInterpolator(
        (x_points, y_points), z, bounds_error=False, fill_value=0
    )


def _within_fit_bounds(spec, fitted_names, nodes, flags):
    """Keep only flags at nodes inside the fitted dimensions' bin edges."""
    mask = np.ones(flags.shape, dtype=bool)
    for axis, name in enumerate(fitted_names):
        low, high = spec.dimensions[name].bounds
        inside = (nodes[name] >= low) & (nodes[name] <= high)
        shape = [1]*flags.ndim
        shape[axis] = -1
        mask &= inside.reshape(shape)
    return flags & mask
