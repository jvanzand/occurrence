## RVSearch functionality that I need for completeness work
## I don't want the occurrence code to depend on RVSearch or radvel, so I'm bringing this code here
import os
import numpy as np
import pandas as pd
import pylab as pl
from scipy.interpolate import RegularGridInterpolator
from astropy import constants as c
Mj2Me = (c.M_jup/c.M_earth).value

import occurrence.radvel_borrowed as rvb
from occurrence import completeness_utils as cu

class Completeness(object):
    """Calculate completeness surface from a suite of injections

    Args:
        recoveries (DataFrame): DataFrame with injection/recovery tests from Injections.save
    """

    def __init__(self, recoveries, xcol='inj_au', ycol='inj_msini',
                 mstar=None, searches=None, y_unit='earth'):
        """Object to handle a suite of injection/recovery tests

        Args:
            recoveries (DataFrame): DataFrame of injection/recovery tests from Injections class
            mstar (float): (optional) stellar mass to use in conversion from p, k to au, msini
            xcol (string): (optional) column name for independent variable. Completeness grids and
                interpolator will work in these axes
            ycol (string): (optional) column name for dependent variable. Completeness grids and
                interpolator will work in these axes

        """
        self.recoveries = recoveries
        self.searches = searches


        #########################################################################################
        # Do the below only when inj_period is in the columns
        # Why? inj_period is only missing when inj_mtrue has already been added, and at that point, msini is not needed so skip the block below
        if 'inj_period' in self.recoveries.columns and 'inj_msini' not in self.recoveries.columns:
            if mstar is not None:

                self.mstar = np.zeros_like(self.recoveries['inj_period']) + mstar

                self.recoveries['inj_msini'] = rvb.Msini(self.recoveries['inj_k'],
                                                                  self.recoveries['inj_period'],
                                                                  self.mstar, self.recoveries['inj_e'],
                                                                  Msini_units='earth')
                self.recoveries['rec_msini'] = rvb.Msini(self.recoveries['rec_k'],
                                                                  self.recoveries['rec_period'],
                                                                  self.mstar, self.recoveries['rec_e'],
                                                                  Msini_units='earth')

                self.recoveries['inj_au'] = rvb.semi_major_axis(self.recoveries['inj_period'], mstar)
                self.recoveries['rec_au'] = rvb.semi_major_axis(self.recoveries['rec_period'], mstar)

        self.xcol = xcol
        self.ycol = ycol
        
        ## Convert recoveries file to correct y unit. Assume initial unit is m_earth
        if y_unit=='jupiter':
            self.recoveries[ycol] = self.recoveries[ycol]/Mj2Me # Convert Me to Mj

        self.grid = None
        self.interpolator = None
        
        #self.trends_count = trends_count

    @classmethod
    def from_csv(cls, recovery_file, *args, **kwargs):
        """Read recoveries and create Completeness object"""
        recoveries = pd.read_csv(recovery_file)
        
        return cls(recoveries, *args, **kwargs)

    def completeness_grid(self, xlim, ylim, resolution=30, xlogwin=0.5, ylogwin=0.5, 
                          fill_nans=True, trends_count=False):
        """Calculate completeness on a fine grid

        Compute a 2D moving average in loglog space

        Args:
            xlim (tuple): min and max x limits
            ylim (tuple): min and max y limits
            resolution (int): (optional) grid is sampled at this resolution
            xlogwin (float): (optional) x width of moving average
            ylogwin (float): (optional) y width of moving average

        """
        xgrid = np.logspace(np.log10(xlim[0]),
                            np.log10(xlim[1]),
                            resolution)
        ygrid = np.logspace(np.log10(ylim[0]),
                            np.log10(ylim[1]),
                            resolution)

        xinj = self.recoveries[self.xcol]
        yinj = self.recoveries[self.ycol]

        # If trends_count, then ONLY trends count. Treat resolved as non-detections.
        if trends_count:
            #good = self.recoveries[['recovered', 'trend_pref']].any(axis=1) # Is either one True?
            good = self.recoveries['trend_pref']
        else:
            good = self.recoveries['recovered']


        z = np.zeros((len(ygrid), len(xgrid)))
        last = 0
        for i,x in enumerate(xgrid):
            for j,y in enumerate(ygrid):
                xlow = 10**(np.log10(x) - xlogwin/2)
                xhigh = 10**(np.log10(x) + xlogwin/2)
                ylow = 10**(np.log10(y) - ylogwin/2)
                yhigh = 10**(np.log10(y) + ylogwin/2)

                xbox = yinj[np.where((xinj <= xhigh) & (xinj >= xlow))[0]]
                if len(xbox) == 0 or y > max(xbox) or y < min(xbox):
                    z[j, i] = np.nan
                    continue

                boxall = np.where((xinj <= xhigh) & (xinj >= xlow) &
                                  (yinj <= yhigh) & (yinj >= ylow))[0]
                boxgood = np.where((xinj[good] <= xhigh) &
                                   (xinj[good] >= xlow) & (yinj[good] <= yhigh) &
                                   (yinj[good] >= ylow))[0]
                # print(x, y, xlow, xhigh, ylow, yhigh, len(boxgood), len(boxall))
                #print("{:.2f}, {:.2f}, {:.2f}, {:.2f}".format(xlow, xhigh, ylow, yhigh))
                if len(boxall) > 5:
                    z[j, i] = float(len(boxgood))/len(boxall)
                    last = float(len(boxgood))/len(boxall)
                else:
                    # import pdb; pdb.set_trace()
                    z[j, i] = np.nan

        self.grid = (xgrid, ygrid, z)
        
        ## Fill in completeness=1 for a/m values directly above (at higher mass than)
        ## areas where completeness=1 already. Similarly, fill 0 for low masses.
        if fill_nans:
            z_fill_high = cu.fill_completeness_nans(z, direction='high-mass')
            z_fill_both = cu.fill_completeness_nans(z_fill_high, direction='low-mass')
            z = z_fill_both
            
            # import matplotlib.pyplot as plt
            # import pdb; pdb.set_trace()

        return (xgrid, ygrid, z)

    def interpolate(self, x, y, refresh=False):
        """Interpolate completeness surface

        Interpolate completeness surface at x, y. X, y should be in the same
        units as self.xcol and self.ycol

        Args:
            x (array): x points to interpolate to
            y (array): y points to interpolate to
            refresh (bool): (optional) refresh the interpolator?

        Returns:
            array : completeness value at x and y

        """
        if self.interpolator is None or refresh:
            assert self.grid is not None, "Must run Completeness.completeness_grid before interpolating."
            gi = cartesian_product(self.grid[0], self.grid[1])
            zi = self.grid[2].T
            self.interpolator = RegularGridInterpolator((self.grid[0], self.grid[1]), zi)

        return self.interpolator((x, y))


def cartesian_product(*arrays):
    """
        Generate a cartesian product of input arrays.

    Args:
        arrays (arrays): 1-D arrays to form the cartesian product of.

    Returns:
        array: cartesian product of input arrays
    """

    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[..., i] = a

    return arr.reshape(-1, la)




class CompletenessPlots(object):
    """Class to plot results of injection/recovery tests

    Args:
        completeness (inject.Completeness): completeness object
        planets (numpy array): masses and semi-major axes for planets
        searches (list): list of rvsearch.Search objects. If present, overplot planets detected in
            all Search objects.

    """
    def __init__(self, completeness, searches=None, fill_nans=True, trends_count=False):
        self.comp = completeness
        self.trends_count = trends_count
        if isinstance(searches, list):
            self.searches = searches
        else:
            self.searches = [searches]
        

        self.xlim = (min(completeness.recoveries[completeness.xcol]),
                     max(completeness.recoveries[completeness.xcol]))

        self.ylim = (min(completeness.recoveries[completeness.ycol]),
                     max(completeness.recoveries[completeness.ycol]))

        self.xgrid, self.ygrid, self.comp_array = completeness.completeness_grid(self.xlim, self.ylim,
                                                                                 fill_nans=fill_nans,
                                                                                 trends_count=trends_count)
        

    def save_comp_grids(self, save_dir):
        """
        Save completeness grids for later loading, plotting,
        and interpolation
        """
        
        ## FOR NOW: ignore all trend sensitivity. Will need to fix path details when I have trend maps
        # if self.trends_count:
        #     np.save(os.path.join(save_dir, 'xgrid_trend'), self.xgrid)
        #     np.save(os.path.join(save_dir, 'ygrid_trend'), self.ygrid)
        #     np.save(os.path.join(save_dir, 'comp_array_trend'), self.comp_array)
        #
        # else:
        np.save(os.path.join(save_dir, 'xgrid'), self.xgrid)
        np.save(os.path.join(save_dir, 'ygrid'), self.ygrid)
        np.save(os.path.join(save_dir, 'zgrid'), self.comp_array)
        

    def completeness_plot(self, title='', xlabel='', ylabel='',
                          colorbar=True, hide_points=False):
        """Plot completeness contours

        Args:
            title (string): (optional) plot title
            xlabel (string): (optional) x-axis label
            ylabel (string): (optional) y-axis label
            colorbar (bool): (optional) plot colorbar
            hide_points (bool): (optional) if true hide individual injection/recovery points
            trends_count (bool): (optional) If true, injections recovered only as trends also count
            y_unit (str): (optional) Either 'earth' or 'jupiter' to determine y units of cplt plot
        """
        
        # If trends_count, then ONLY trends count. Treat resolved as non-detections.
        if self.trends_count==True:
            good_cond = 'recovered == False & trend_pref == True'
            bad_cond = '(recovered == True) or (recovered == False & trend_pref == False)'
            #trend_only_cond = 'recovered == False & trend_pref == True'
            good_color = "g."

            # if save_grids:
            #     ## Save completeness map
            #     np.save('xgrid_trend', self.xgrid)
            #     np.save('ygrid_trend', self.ygrid)
            #     np.save('comp_array_trend', self.comp_array)
            
        else:
            good_cond = 'recovered == True'
            bad_cond = 'recovered == False'
            #trend_only_cond = 'recovered == False & recovered == True' # Empty
            good_color = "b."


            # if save_grids:
            #     ## Save completeness map
            #     np.save('xgrid_resolved', self.xgrid)
            #     np.save('ygrid_resolved', self.ygrid)
            #     np.save('comp_array_resolved', self.comp_array)
            
        good = self.comp.recoveries.query(good_cond)
        bad = self.comp.recoveries.query(bad_cond)
        #trend_only = self.comp.recoveries.query(trend_only_cond)

        fig = pl.figure(figsize=(7.5, 5.25))
        pl.subplots_adjust(bottom=0.18, left=0.22, right=0.95)


        CS = pl.contourf(self.xgrid, self.ygrid, self.comp_array, 10, cmap=pl.cm.Reds_r, vmin=0, vmax=0.9)

        # Plot 50th percentile.
        fifty = pl.contour(self.xgrid, self.ygrid, self.comp_array, [0.5])
        if not hide_points:
            pl.plot(good[self.comp.xcol], good[self.comp.ycol], good_color, alpha=0.3, label='recovered')
            pl.plot(bad[self.comp.xcol], bad[self.comp.ycol], 'r.', alpha=0.3, label='missed')
            #pl.plot(trend_only[self.comp.xcol], trend_only[self.comp.ycol], 'g.', alpha=0.3, label='trend')
        ax = pl.gca()
        ax.set_xscale('log')
        ax.set_yscale('log')
        #import pdb; pdb.set_trace()


        ### IRRELEVANT for general occurrence work. Not interested in single system detections.
        # ## Code for plotting real detected planets as black dots
        # if self.comp.xcol == 'inj_au' and self.comp.ycol == 'inj_msini' and self.searches[0] is not None:
        #     for search in self.searches:
        #         post = search.post
        #         synthparams = post.params.basis.to_synth(post.params)
        #         for i in range(search.num_planets):
        #             p = i + 1
        #             if search.mcmc:
        #                 msini = post.medparams['mpsini{:d}'.format(p)]
        #                 msini_err1 = -post.uparams['mpsini{:d}_err1'.format(p)]
        #                 msini_err2 = post.uparams['mpsini{:d}_err2'.format(p)]
        #                 a = post.medparams['a{:d}'.format(p)]
        #                 a_err1 = -post.uparams['a{:d}_err1'.format(p)]
        #                 a_err2 = post.uparams['a{:d}_err2'.format(p)]
        #
        #                 a_err = np.array([[a_err1, a_err2]]).transpose()
        #                 msini_err = np.array([[msini_err1, msini_err2]]).transpose()
        #
        #                 ax.errorbar([a], [msini], xerr=a_err, yerr=msini_err, fmt='ko', ms=10)
        #             else:
        #                 per = synthparams['per{:d}'.format(p)].value
        #                 k = synthparams['k{:d}'.format(p)].value
        #                 e = synthparams['e{:d}'.format(p)].value
        #                 msini = radvel.utils.Msini(k, per, search.mstar, e)
        #                 a = radvel.utils.semi_major_axis(per, search.mstar)
        #
        #                 ax.plot(a, msini, 'ko', ms=10)

        # else:
        #     warnings.warn('Overplotting detections not implemented for the current axis selection: x={}, y={}'.format(self.comp.xcol, self.comp.ycol))

        labelsize=18
        ticksize=14

        
        xticks = pl.xticks()[0]
        xticks = 10**(np.linspace(np.log10(xticks[0]), 
                      np.log10(xticks[-1]), 
                      int(np.log10(xticks[-1]/xticks[0]))+1))
        pl.xticks(xticks, xticks)

        yticks = pl.yticks()[0]
        yticks = 10**(np.linspace(np.log10(yticks[0]), 
                      np.log10(yticks[-1]), 
                      int(np.log10(yticks[-1]/yticks[0]))+1))
        pl.yticks(yticks, yticks)
        

        pl.xlim(self.xlim[0], self.xlim[1])
        pl.ylim(self.ylim[0], self.ylim[1])

        fontsize = 18
        pl.title(title, size=labelsize)
        pl.xlabel(xlabel, size=labelsize)
        pl.ylabel(ylabel, size=labelsize)
        
        pl.tick_params(axis='both', which='major', labelsize=ticksize)

        pl.grid(True)


        if colorbar:
            cb = pl.colorbar(mappable=CS, pad=0)
            cb.ax.set_ylabel('probability of detection', fontsize=labelsize)
            cb.ax.tick_params(labelsize=ticksize)

        fig = pl.gcf()

        return fig





























