## Main module for running an end-to-end occurrence calculation
## Optionally create all preliminary products (average completeness maps,
## interpolation functions, catalog samples, prior weights), or skip if
## already done. Then do an occurrence fit.

import os
import glob
import numpy as np
import pandas as pd
import pickle
import multiprocessing as mp

from occurrence import completeness_utils as cu
from occurrence import sampling_utils as su
from occurrence import plotting_utils as pu
from occurrence import fit_utils as dfu
from occurrence import mcmc

from occurrence.completeness_utils import _process_single_star

def prep_recoveries_files(tier1_dir,
                          star_df,
                          master_rec_dir,
                          recoveries_mtype,
                          m_dir_to_make_q=None,
                          tier1_kind=None,
                          keep_period=False,
                          ):
    """
    Prepare recoveries.csv files for occurrence
    calculations.
    Starting from a file with an inj_msini column,
    optionally compute inj_mtrue, inj_qsini, and/or
    inj_qtrue.
    
    Arguments:
        convert_recs_msini_mtrue (bool): Whether to convert 
                          recoveries files from true mass to msini
        convert_recs_m_q (bool): Whether to convert 
                          recoveries files from mass to mass ratio (initial
                          mass may be msini or mtrue)
        msini_rec_dir_to_make_mtrue (str): Directory where msini recoveries.csv
                          files are stored
        m_dir_to_make_q (str): Directory to find mass recoveries files, which
                          can be used to calculate mass ratio q
        keep_period (bool): Keep inj_period in the converted files, for
                          Tier 1s whose x coordinate is orbital period
    """

    
    tier1_label = tier1_kind or os.path.basename(os.path.normpath(tier1_dir))
    if recoveries_mtype=='msini':

        if 'true' in tier1_label: # If tier1 is mtrue or qtrue, convert msini to mtrue
            for starname in star_df.star_name:
                recoveries_file = os.path.join(master_rec_dir, starname+'_recoveries.csv')
                mtrue_recoveries_save_file = os.path.join(tier1_dir, 'mtrue_recoveries/', starname+'_recoveries.csv')
            
                cu.recs_msini_converter(recoveries_file, mtrue_recoveries_save_file,
                                        keep_period=keep_period)
            
        else:
            os.makedirs(
                os.path.join(tier1_dir, 'msini_recoveries'), exist_ok=True
            )
            keep_cols = ['inj_msini', 'inj_au', 'inj_e', 'recovered']
            if keep_period:
                keep_cols.append('inj_period')
            for starname in star_df.star_name:
                recoveries_file = os.path.join(master_rec_dir, starname+'_recoveries.csv')
                msini_recoveries_save_file = os.path.join(tier1_dir, 'msini_recoveries/', starname+'_recoveries.csv')
            
                rec_file = pd.read_csv(recoveries_file)[keep_cols]
                rec_file.to_csv(msini_recoveries_save_file, index=False)
    
        if 'q' in tier1_label: # If tier1 is qsini or qtrue, convert m to q
            for i in range(len(star_df)):
                row = star_df.iloc[i]
                starname = row.star_name
                mstar = row.Mstar
            
                dirname = tier1_label+'_recoveries'
            
                q_recoveries_save_file = os.path.join(tier1_dir, dirname, starname+'_recoveries.csv')
                recoveries_file = os.path.join(m_dir_to_make_q, starname+'_recoveries.csv')
            
                cu.recs_mass_ratio_converter(recoveries_file, q_recoveries_save_file, mstar)
            
    elif recoveries_mtype=='mtrue':
        os.makedirs(
            os.path.join(tier1_dir, 'mtrue_recoveries'), exist_ok=True
        )
        if 'sini' in tier1_label:
            raise Exception("main.prep_recoveries_files: Cannot calculate Msini completeness from Mtrue recoveries files")

        for starname in star_df.star_name:
            recoveries_file = os.path.join(master_rec_dir, starname+'_recoveries.csv')
            mtrue_recoveries_save_file = os.path.join(tier1_dir, 'mtrue_recoveries/', starname+'_recoveries.csv')
            
            rec_file = pd.read_csv(recoveries_file)
            rec_file.to_csv(mtrue_recoveries_save_file, index=False)
    
        if 'q' in tier1_label: # If tier1 is qtrue, convert m to q
            for i in range(len(star_df)):
                row = star_df.iloc[i]
                starname = row.star_name
                mstar = row.Mstar
            
                dirname = tier1_label+'_recoveries'
            
                q_recoveries_save_file = os.path.join(tier1_dir, dirname, starname+'_recoveries.csv')
                recoveries_file = os.path.join(m_dir_to_make_q, starname+'_recoveries.csv')
            
                cu.recs_mass_ratio_converter(recoveries_file, q_recoveries_save_file, mstar)
            
            
    return

def prep_maps(tier1_dir,
              star_df,
              path_to_recoveries,
              m_unit='earth',
              avg_map_only=False,
              save_single_plots=False,
              tier1_kind=None,
              xcol='inj_au'):
    """
    Prepare both single-system and average completeness 
    maps and calculate corresponding interpolation
    functions.
                  
    Arguments:
        make_maps (bool): Whether to convert the specified
              recoveries files into x/y/z grids and plot completeness
        make_interps (bool): Whether to calculate the interpolation
              functions associated with the specified maps


        path_to_recoveries (str): Path to diretory containing
              recoveries.csv files to make maps with


        maps_ycol (str): Which value to use in completeness map creation.
              Can be 'inj_msini', 'inj_mtrue', 'qsini', or 'qtrue'
        m_unit (str): Units of mass column, either 'earth' or 'jupiter'
        star_df (dataframe): Pandas df with columns 'star_name', 'mstar',
              and 'comp_list'. comp_list is a list of the names of all 
              companions orbiting that star. mstar is in solar masses
                   
    """

    tier1_label = tier1_kind or os.path.basename(os.path.normpath(tier1_dir))
    maps_save_label = 'saved_maps_'+tier1_label
    maps_save_path = os.path.join(tier1_dir, maps_save_label)
    maps_ycol = f"inj_{tier1_label}"
    
    if mp.cpu_count()>100:
        ncores=30
    else:
        ncores = int(mp.cpu_count()/2)

    args_list = [
        (row, path_to_recoveries, maps_save_path, maps_ycol, m_unit,
         avg_map_only, save_single_plots, xcol)
        for _, row in star_df.iterrows()
        ]

    with mp.Pool(ncores) as pool:
        pool.map(_process_single_star, args_list)
    
    
    ## Make and save individual interp functions for sensitivity maps
    cu.build_interpolators(maps_save_path, star_df.star_name.to_list())
    
    return


def make_average_map(tier1_dir, tier2_dir,
                     star_df,
                     ycol,
                     m_unit='earth',
                     tier1_kind=None,
                     xcol='inj_au'):
    """
    Calculate average map from a subset of
    pre-computed completeness maps
    """
    
    tier1_label = tier1_kind or os.path.basename(os.path.normpath(tier1_dir))
    path_to_maps = os.path.join(tier1_dir, f"saved_maps_{tier1_label}")
    avg_map_dir = os.path.join(tier1_dir, tier2_dir, 'avg_map/')
    
    cu.average_map(path_to_maps, avg_map_dir, star_df.star_name.to_list(), ycol=ycol, m_unit=m_unit,
                   xcol=xcol)
    
    return
    
def periods_from_samples(post_sample_dict, star_df, m_unit):
    """Replace each companion's semi-major axis samples (AU) with periods.

    Kepler's third law with the host mass from ``star_df`` and each draw's
    companion mass. Only the drawn samples are converted (500 per companion
    in prep_post_draws), not the full posterior chains. ``m_unit`` None
    means masses in M_sun, as the samplers return for mass-ratio Tier 1s.
    """
    converted = {}
    for comp_name, samples in post_sample_dict.items():
        mstar = _host_mass(star_df, comp_name)
        converted[comp_name] = np.array([
            su.orbital_period(samples[0], samples[1], mstar,
                              m_unit or 'solar'),
            samples[1],
        ])
    return converted


def _host_mass(star_df, comp_name):
    """Return the stellar mass (M_sun) of the star hosting ``comp_name``."""
    rows = star_df[star_df['comp_list'].apply(lambda names: comp_name in names)]
    if len(rows) != 1:
        raise ValueError(
            f"{comp_name} must belong to exactly one star in star_df; "
            f"found {len(rows)}"
        )
    return float(rows.Mstar.iloc[0])


def prep_post_draws(tier1_dir, tier2_dir,
                    star_df, comp_post_dir,
                    sampling_func,
                    saved_maps_dir=None, m_unit='earth',
                    fig_title='Catalog Posteriors',
                    avg_map_only=False,
                    fill_single_nan_with_average=True,
                    tier1_kind=None,
                    x_param='a'):

    """
    Sample from companion posteriors according to user-specified
    approach               
                         
    Arguments:
        sample_posts (bool): Whether to sample from companion
            posteriors. This also involves computing the prior
            probability and the system-specific completeness
            for each sample.
    
        star_df (dataframe): Pandas df with columns 'star_name', 'mstar',
              and 'comp_list'. comp_list is a list of the names of all 
              companions orbiting that star. mstar is in solar masses
        comp_post_dir (str): Path to companion posteriors. Naming
            convention depends on custom post sampling function
        sampling_func (function): Function from sampling_utils.py
                             that samples posteriors
        x_param (str): 'a' keeps the semi-major axis samples (AU); 'p'
                             replaces them with orbital periods (days) from
                             Kepler's third law, using the host mass in
                             star_df and each draw's companion mass. The
                             log-uniform interim prior in a is also
                             log-uniform in P (constant Jacobian, 3/2), so
                             the prior values are unchanged.
    """
    
    #### CUSTOMIZE your own sampler to match the posterior format ####
    ## The output of custom sampler should be a dict whose keys are companion names
    ## and whose values are 2xN arrays, where the first/second sub-array is SMA/mass samples
    post_sample_dict = sampling_func(comp_post_dir, star_df, num_samples=500, m_unit=m_unit) # First sample posteriors
    ## If using mass ratio, convert masses to q
    tier1_label = tier1_kind or os.path.basename(os.path.normpath(tier1_dir))
    if saved_maps_dir is None:
        saved_maps_dir = os.path.join(
            tier1_dir, f"saved_maps_{tier1_label}"
        )
    if x_param == 'p':
        # Before any conversion of mass to mass ratio
        post_sample_dict = periods_from_samples(post_sample_dict, star_df,
                                                m_unit)
    elif x_param != 'a':
        raise ValueError("x_param must be 'a' or 'p'")
    if 'q' in tier1_label:

        for comp_name in post_sample_dict.keys():
            row = star_df[star_df['comp_list'].apply(lambda name_list: comp_name in name_list)]
            mstar = row.Mstar.values
                
            old_samples = post_sample_dict[comp_name]
            new_samples = np.array([old_samples[0], old_samples[1]/mstar]) # Same 'a' samples; convert m-->q
            post_sample_dict[comp_name] = new_samples
                
    post_prior_sample_dict = su.interim_prior(post_sample_dict, prior_type='loguniform') # Then calculate prior at each draw. Each value is a 3xN array of SMA samples, mass samples, and prior values
    # Now add on completeness values
    sampled_post_with_compls = su.include_post_completeness(post_prior_sample_dict,
                                                            star_df,
                                                            tier1_dir, tier2_dir,
                                                            saved_maps_dir=saved_maps_dir,
                                                            avg_map_only=avg_map_only,
                                                            fill_single_nan_with_average=fill_single_nan_with_average)
    ## Saves dict with companion names as key names
    ## Each value is a 7xN array, with prior densities defined relative to
    ## dlog10(a) dlog10(m), of:
    ## [a_list, m_list, avg_compls, single_star_compls,
    ##  compl_over_prior_avg, compl_over_prior_single, interim_prior]
    ## Probably the only compl array I'll use is compl_over_prior_single. compl_over_prior_avg is to test whether using avg completeness changes the answer. The two completeness arrays are for testing/sanity checks.
    saved_dict_dir = os.path.join(tier1_dir, tier2_dir, 'sampled_post_prior_compl.npz')
    np.savez(saved_dict_dir, **sampled_post_with_compls)
        
        
    return
    
    
def prep_fit_materials(
        tier1_dir,
        tier2_dir,
        tier3_dir,
        x_bounds,
        y_bounds,
        star_df,
        completeness_type='single',
        integration_resolution=(100, 100),
        use_average_completeness=True,
        interim_prior_fn=None,
        tier1_kind=None,
        spec=None):
    """Prepare unbinned samples and survey exposure for a direct fit.

    ``spec`` is the run's ``DimensionSpec``; for a non-legacy spec the
    catalog and average map are its own Tier 2 products (checked against
    its record), the bounds are in engine coordinates, and the record is
    saved in ``fit_data.npz``.

    This Stage 2 entry point supports the current ``(a, mass)`` catalog order
    and one rectangular x-y region. It writes ``fit_data.npz`` but
    does not evaluate a likelihood or run MCMC. The average-map exposure is
    the default because some individual maps may contain NaNs in the ROI;
    individual-map summation remains available with strict validation.
    """
    explicit = spec is not None and not spec.legacy
    if explicit:
        from occurrence import dimensions as dm
        paths = dimension_tier2_paths(tier1_dir, tier2_dir, spec)
        catalog_path = paths["catalog"]
        with np.load(catalog_path) as data:
            saved = dm.spec_from_json(str(data[dfu.DIMENSIONS_KEY]))
        dm.check_matching(spec, saved, catalog_path, include_fit_edges=False)
        if not use_average_completeness:
            raise NotImplementedError(
                "runs with explicit dimension roles use the sample-average "
                "completeness for the expected number of detections"
            )
    else:
        catalog_path = os.path.join(
            tier1_dir, tier2_dir, 'sampled_post_prior_compl.npz'
        )
    companions = dfu.load_catalog(
        catalog_path=catalog_path,
        x_bounds=x_bounds,
        y_bounds=y_bounds,
        completeness_type=completeness_type,
        interim_prior_fn=interim_prior_fn,
    )

    if use_average_completeness:
        average_path = (
            os.path.join(paths["engine_average"], 'interp_fn.pkl')
            if explicit else
            os.path.join(tier1_dir, tier2_dir, 'avg_map', 'interp_fn.pkl')
        )
        with open(average_path, 'rb') as stream:
            average_completeness = pickle.load(stream)
        exposure = dfu.build_exposure_grid(
            x_bounds=x_bounds,
            y_bounds=y_bounds,
            resolution=integration_resolution,
            average_completeness=average_completeness,
            nstars=len(star_df),
        )
    else:
        tier1_label = tier1_kind or os.path.basename(os.path.normpath(tier1_dir))
        maps_dir = os.path.join(tier1_dir, f'saved_maps_{tier1_label}')
        interpolators = []
        for star_name in star_df.star_name:
            interp_path = os.path.join(maps_dir, star_name, 'interp_fn.pkl')
            with open(interp_path, 'rb') as stream:
                interpolators.append(pickle.load(stream))
        exposure = dfu.build_exposure_grid(
            x_bounds=x_bounds,
            y_bounds=y_bounds,
            resolution=integration_resolution,
            completeness_interpolators=interpolators,
        )

    save_path = os.path.join(
        tier1_dir, tier2_dir, tier3_dir, 'saved_dicts', 'fit_data.npz'
    )
    dfu.save_fit_data(save_path, companions, exposure,
                      dimensions_json=spec.to_json() if explicit else None)
    return save_path


def plot_piecewise(
        tier1_dir,
        tier2_dir,
        tier3_dir,
        nstars,
        stack_dim,
        m_unit='earth',
        mtype=None,
        title='Piecewise-constant fit',
        plot_occurrence=True,
        plot_density=True,
        plot_corner=True,
        plot_catalog_roi=False,
        plot_roi_occurrence=False,
        plot_uncorrected_occurrence_mle=False,
        occurrence_legend_loc="upper right",
        xcol='inj_au'):
    """Load and plot a saved direct piecewise-constant fit."""
    base_dir = os.path.join(tier1_dir, tier2_dir, tier3_dir)
    if mtype is None:
        mtype = os.path.basename(os.path.normpath(tier1_dir))
    return mcmc.plot_piecewise_results(
        fit_path=os.path.join(
            base_dir, 'saved_dicts', 'fit_data.npz'
        ),
        chain_path=os.path.join(
            base_dir, 'saved_chains', 'chains_piecewise.npz'
        ),
        output_dir=os.path.join(base_dir, 'plots'),
        nstars=nstars,
        stack_dim=stack_dim,
        m_unit=m_unit,
        mtype=mtype,
        title=title,
        plot_occurrence=plot_occurrence,
        plot_density=plot_density,
        plot_corner=plot_corner,
        plot_catalog_roi=plot_catalog_roi,
        plot_roi_occurrence=plot_roi_occurrence,
        plot_uncorrected_occurrence_mle=plot_uncorrected_occurrence_mle,
        occurrence_legend_loc=occurrence_legend_loc,
        tier1_dir=tier1_dir,
        tier2_dir=tier2_dir,
        xcol=xcol,
    )


def plot_smooth(
        tier1_dir,
        tier2_dir,
        tier3_dir,
        model_name,
        stack_dim,
        a_edges,
        m_edges,
        m_unit='earth',
        title='Smooth-model fit',
        plot_occurrence=True,
        plot_cumulative=False,
        plot_density=True,
        plot_corner=True,
        model_plot_style='credible',
        n_posterior_draws=100,
        plot_random_seed=None,
        plot_uncorrected_occurrence_mle=False,
        nstars=None,
        occurrence_legend_loc="upper right",
        mtype=None):
    """Load, plot, and save one smooth direct model.

    ``mtype`` is the Tier 1 mass type ('mtrue', 'msini', 'qtrue', 'qsini');
    by default it is the Tier 1 folder name.
    """
    import matplotlib.pyplot as plt

    base_dir = os.path.join(tier1_dir, tier2_dir, tier3_dir)
    # Saved material bounds identify only the outer range. Chain discovery is
    # intentionally filename-based so plotting also works after a fit-only run.
    chain_dir = os.path.join(base_dir, 'saved_chains')
    chain_paths = glob.glob(
        os.path.join(chain_dir, f'chains_{model_name}_bin*.npz')
    )
    chain_paths.sort(
        key=lambda path: int(os.path.splitext(path)[0].rsplit('bin', 1)[1])
    )
    if not chain_paths:
        raise FileNotFoundError(
            f"no saved {model_name} chains in {chain_dir!r}"
        )
    figures = {}
    if plot_density:
        figures['density'] = plt.subplots(figsize=(6, 4))
    if plot_occurrence:
        figures['occurrence'] = plt.subplots(figsize=(6, 4))
    if plot_cumulative:
        figures['cumulative'] = plt.subplots(figsize=(6, 4))
    model_edges = m_edges if stack_dim == 'a' else a_edges
    paths = mcmc.add_smooth_model_to_figures(
        chain_paths=chain_paths,
        model_name=model_name,
        figures=figures,
        output_dir=os.path.join(base_dir, 'plots'),
        stack_dim=stack_dim,
        m_unit=m_unit,
        title=title,
        plot_occurrence=plot_occurrence,
        plot_cumulative=plot_cumulative,
        plot_density=plot_density,
        plot_corner=plot_corner,
        model_plot_style=model_plot_style,
        n_posterior_draws=n_posterior_draws,
        plot_random_seed=plot_random_seed,
        model_edges=model_edges,
    )
    if plot_uncorrected_occurrence_mle:
        mcmc.add_uncorrected_mle_to_figures(
            figures, os.path.join(base_dir, 'saved_dicts', 'fit_data.npz'),
            a_edges, m_edges, nstars, stack_dim,
        )
    paths.update(mcmc.save_model_figures(
        figures=figures, output_dir=os.path.join(base_dir, 'plots'),
        stack_dim=stack_dim, model_edges=model_edges, title=title,
        m_unit=m_unit,
        mtype=mtype or os.path.basename(os.path.normpath(tier1_dir)),
        occurrence_legend_loc=occurrence_legend_loc,
    ))
    return paths


def plot_models(
        tier1_dir,
        tier2_dir,
        tier3_dir,
        nstars,
        stack_dim,
        a_edges,
        m_edges,
        plot_models,
        m_unit='earth',
        title='Occurrence fit',
        plot_occurrence=True,
        plot_cumulative=False,
        plot_density=True,
        plot_corner=True,
        plot_catalog_roi=False,
        plot_roi_occurrence=False,
        model_plot_style='credible',
        n_posterior_draws=100,
        plot_random_seed=None,
        plot_uncorrected_occurrence_mle=False,
        occurrence_legend_loc="upper right",
        mtype=None,
        xcol='inj_au'):
    """Plot selected direct models together while retaining separate corners.

    ``mtype`` is the Tier 1 mass type (by default the Tier 1 folder name) and
    ``xcol`` its injection x column ('inj_au' or 'inj_period').
    """
    selected = list(plot_models)
    supported = {'piecewise'} | mcmc.SMOOTH_MODELS
    unknown = set(selected) - supported
    if unknown:
        raise ValueError(f"unsupported plot models: {sorted(unknown)}")
    if selected == ['piecewise']:
        return plot_piecewise(
            tier1_dir=tier1_dir, tier2_dir=tier2_dir, tier3_dir=tier3_dir,
            nstars=nstars, stack_dim=stack_dim,
            m_unit=m_unit, title=title, plot_occurrence=plot_occurrence,
            plot_density=plot_density, plot_corner=plot_corner,
            plot_catalog_roi=plot_catalog_roi,
            plot_roi_occurrence=plot_roi_occurrence,
            plot_uncorrected_occurrence_mle=plot_uncorrected_occurrence_mle,
            occurrence_legend_loc=occurrence_legend_loc,
            mtype=mtype,
            xcol=xcol,
        )
    if len(selected) == 1 and selected[0] != 'piecewise':
        return plot_smooth(
            tier1_dir=tier1_dir, tier2_dir=tier2_dir, tier3_dir=tier3_dir,
            model_name=selected[0],
            stack_dim=stack_dim, a_edges=a_edges, m_edges=m_edges,
            m_unit=m_unit,
            title=title, plot_occurrence=plot_occurrence,
            plot_cumulative=plot_cumulative,
            plot_density=plot_density, plot_corner=plot_corner,
            model_plot_style=model_plot_style,
            n_posterior_draws=n_posterior_draws,
            plot_random_seed=plot_random_seed,
            plot_uncorrected_occurrence_mle=plot_uncorrected_occurrence_mle,
            nstars=nstars,
            occurrence_legend_loc=occurrence_legend_loc,
            mtype=mtype,
        )
    base_dir = os.path.join(tier1_dir, tier2_dir, tier3_dir)
    base_figures = {}
    model_edges = m_edges if stack_dim == 'a' else a_edges
    mtype = mtype or os.path.basename(os.path.normpath(tier1_dir))
    paths = {}
    if 'piecewise' in selected:
        fit_path = os.path.join(
            base_dir, 'saved_dicts', 'fit_data.npz'
        )
        piecewise_chain = os.path.join(
            base_dir, 'saved_chains', 'chains_piecewise.npz'
        )
        summary_path = os.path.join(
            base_dir, 'saved_dicts', 'summary_dict_piecewise.npz'
        )
        summary = mcmc.summarize_piecewise_file(
            fit_path, piecewise_chain, nstars, save_path=summary_path
        )
        if plot_density:
            base_figures['density'] = pu.plot_occurrence_hist(
                summary, stack_dim=stack_dim, m_unit=m_unit, mtype=mtype,
                rate_type='ORD', title=title, return_fig_ax=True,
            )
        if plot_occurrence:
            base_figures['occurrence'] = pu.plot_occurrence_hist(
                summary, stack_dim=stack_dim, m_unit=m_unit, mtype=mtype,
                rate_type='OR', title=title, return_fig_ax=True,
            )
        if plot_cumulative:
            base_figures['cumulative'] = mcmc.piecewise_cumulative_figure(
                piecewise_chain, stack_dim=stack_dim, title=title, m_unit=m_unit,
            )
        paths['piecewise'] = mcmc.plot_piecewise_results(
            fit_path, piecewise_chain, os.path.join(base_dir, 'plots'),
            nstars, stack_dim, m_unit=m_unit, mtype=mtype, title=title,
            plot_occurrence=False, plot_density=False, plot_corner=plot_corner,
            plot_catalog_roi=plot_catalog_roi,
            plot_roi_occurrence=plot_roi_occurrence,
            tier1_dir=tier1_dir, tier2_dir=tier2_dir,
            summary=summary, xcol=xcol,
        )
    else:
        import matplotlib.pyplot as plt
        if plot_density:
            base_figures['density'] = plt.subplots(figsize=(6, 4))
        if plot_occurrence:
            base_figures['occurrence'] = plt.subplots(figsize=(6, 4))
        if plot_cumulative:
            base_figures['cumulative'] = plt.subplots(figsize=(6, 4))
    for model_name in [name for name in selected if name != 'piecewise']:
        chain_dir = os.path.join(base_dir, 'saved_chains')
        chain_paths = glob.glob(os.path.join(
            chain_dir, f'chains_{model_name}_bin*.npz'
        ))
        chain_paths.sort(
            key=lambda path: int(os.path.splitext(path)[0].rsplit('bin', 1)[1])
        )
        if not chain_paths:
            raise FileNotFoundError(
                f"no saved {model_name} chains in {chain_dir!r}"
            )
        paths[model_name] = mcmc.add_smooth_model_to_figures(
            chain_paths=chain_paths, model_name=model_name,
            figures=base_figures, output_dir=os.path.join(base_dir, 'plots'),
            stack_dim=stack_dim, m_unit=m_unit, title=title,
            plot_occurrence=False, plot_cumulative=plot_cumulative,
            plot_density=plot_density, plot_corner=plot_corner,
            model_plot_style=model_plot_style,
            n_posterior_draws=n_posterior_draws,
            plot_random_seed=plot_random_seed,
            model_edges=model_edges,
        )
    if plot_uncorrected_occurrence_mle:
        mcmc.add_uncorrected_mle_to_figures(
            base_figures,
            os.path.join(base_dir, 'saved_dicts', 'fit_data.npz'),
            a_edges, m_edges, nstars, stack_dim,
        )
    paths['combined'] = mcmc.save_model_figures(
        figures=base_figures, output_dir=os.path.join(base_dir, 'plots'),
        stack_dim=stack_dim, model_edges=model_edges, title=title,
        m_unit=m_unit, mtype=mtype,
        occurrence_legend_loc=occurrence_legend_loc,
    )
    return paths


# ---------------------------------------------------------------------------
# Runs with explicit dimension roles (see ``dimensions``)
#
# These steps replace ``prep_maps``/``make_average_map``/``prep_post_draws``
# for runs whose ``dimensions`` setting fits or marginalizes eccentricity.
# Legacy runs never call them, so their products are unchanged.
#
# Independence assumption: marginalized dimensions are averaged over fixed
# population distributions, assuming they are independent of the fitted
# dimensions (see ``marginal_distributions``).
# ---------------------------------------------------------------------------

def sliced_maps_dir(tier1_dir, tier1_label):
    """Folder of the stars' eccentricity-sliced maps."""
    return os.path.join(
        tier1_dir, f"saved_maps_{tier1_label}{cu.SLICED_MAPS_SUFFIX}"
    )


def prep_sliced_maps(tier1_dir, star_df, path_to_recoveries, tier1_label,
                     m_unit='jupiter', e_edges=cu.ECC_SLICE_EDGES):
    """Make every star's eccentricity-sliced completeness map (Tier 1).

    Uses the inclination-marginalized recoveries already written for the
    original maps. Returns {star: injections per slice}.
    """
    save_path = sliced_maps_dir(tier1_dir, tier1_label)
    os.makedirs(save_path, exist_ok=True)
    ycol = f"inj_{tier1_label}"
    args_list = [
        (star, path_to_recoveries, save_path, ycol, m_unit, tuple(e_edges))
        for star in star_df.star_name
    ]
    ncores = 30 if mp.cpu_count() > 100 else max(1, int(mp.cpu_count()/2))
    with mp.Pool(ncores) as pool:
        counts = dict(pool.map(cu._process_single_star_sliced, args_list))
    np.savez(os.path.join(save_path, "slice_counts.npz"),
             stars=np.asarray(list(counts)),
             counts=np.asarray(list(counts.values())),
             e_edges=np.asarray(e_edges))
    return counts


def dimension_tier2_paths(tier1_dir, tier2_dir, spec):
    """Paths of the Tier 2 products for one dimension spec."""
    base = os.path.join(tier1_dir, tier2_dir)
    return {
        "sliced_average": os.path.join(base, "avg_map" + cu.SLICED_MAPS_SUFFIX),
        "engine_average": os.path.join(base, f"avg_map_{spec.token}"),
        "catalog": os.path.join(
            base, f"sampled_post_prior_compl_{spec.token}.npz"
        ),
    }


def make_dimension_average_maps(tier1_dir, tier2_dir, spec, star_df,
                                tier1_label):
    """Average the sliced maps over a sample and marginalize them per spec.

    Writes the sliced average (shared by every spec) and the engine map for
    this spec, with its dimension record. Returns the engine interpolator.
    """
    paths = dimension_tier2_paths(tier1_dir, tier2_dir, spec)
    maps_dir = sliced_maps_dir(tier1_dir, tier1_label)
    if os.path.exists(os.path.join(paths["sliced_average"],
                                   cu.SLICED_INTERP_FILE)):
        sliced_average = cu.load_sliced(paths["sliced_average"])
    else:
        sliced_average, filled = cu.sliced_average_map(
            maps_dir, paths["sliced_average"], star_df.star_name.to_list()
        )
        if filled.any():
            print(f"{tier2_dir}: filled {int(filled.sum())} sliced-average "
                  "points with no estimate from the next lower e slice "
                  f"(per slice: {filled.tolist()})")
    engine_map = cu.engine_completeness(spec, sliced_average)
    os.makedirs(paths["engine_average"], exist_ok=True)
    with open(os.path.join(paths["engine_average"], "interp_fn.pkl"),
              "wb") as stream:
        pickle.dump(engine_map, stream)
    with open(os.path.join(paths["engine_average"], "dimensions.json"),
              "w") as stream:
        stream.write(spec.to_json())
    return engine_map


def plot_dimension_average_maps(tier1_dir, tier2_dir, spec, star_names,
                                tier1_label, overwrite=False):
    """Save images of a spec's Tier 2 average maps, as the original maps do.

    Writes ``sliced_average_maps.png`` (the sample-average map in each e
    slice, titled with the median injections per star in the slice) next to
    the sliced average, and ``average_completeness.png`` (the completeness
    the fit uses, in its fitted dimensions) next to the engine map. Existing
    images are kept unless ``overwrite``. Returns the paths written.
    """
    paths = dimension_tier2_paths(tier1_dir, tier2_dir, spec)
    sliced_image = os.path.join(paths["sliced_average"],
                                "sliced_average_maps.png")
    engine_image = os.path.join(paths["engine_average"],
                                "average_completeness.png")
    written = []
    sliced = cu.load_sliced(paths["sliced_average"])
    if overwrite or not os.path.exists(sliced_image):
        maps_dir = sliced_maps_dir(tier1_dir, tier1_label)
        star_counts = []
        for star in star_names:
            path = os.path.join(maps_dir, star, cu.SLICED_MAP_FILE)
            if os.path.exists(path):
                with np.load(path) as data:
                    star_counts.append(data["counts"])
        counts = np.median(star_counts, axis=0) if star_counts else None
        mass = "M_c" if "true" in tier1_label else r"M_c \sin i"
        repeated = (", counting each orbit's 10 inclination copies"
                    if tier1_label.endswith("true") else "")
        pu.plot_sliced_average_maps(
            sliced, sliced_image,
            title=(f"{tier2_dir}: average completeness of {len(star_names)} "
                   f"stars in each e slice\nInjections: median per star in "
                   f"the slice{repeated}"),
            y_label=rf"${mass}$ [$M_{{\rm Jup}}$]",
            counts=counts,
        )
        written.append(sliced_image)
    if overwrite or not os.path.exists(engine_image):
        with open(os.path.join(paths["engine_average"], "interp_fn.pkl"),
                  "rb") as stream:
            engine_map = pickle.load(stream)
        units = {"a": " AU", "m": r" $M_{\rm Jup}$", "e": ""}
        averaged = ", ".join(
            f"{name} = {low:g}-{high:g}{units[name]} "
            f"({spec.dimensions[name].distribution.form})"
            for name in spec.marginalized
            for low, high in [spec.dimensions[name].bounds]
        )
        pu.plot_engine_completeness(
            spec, engine_map, engine_image,
            title=(f"{tier2_dir}: completeness the fit uses\n"
                   f"averaged over {averaged}"),
            e_edges=sliced.e_edges,
        )
        written.append(engine_image)
    return written


def prep_dimension_post_draws(tier1_dir, tier2_dir, spec, star_df,
                              comp_post_dir, sampling_func, tier1_label,
                              m_unit='jupiter', num_samples=500,
                              fill_single_nan_with_average=True):
    """Sample companion posteriors in a, m, and e and weight them per spec.

    Draws ``num_samples`` (a, m, e) samples per companion with
    ``sampling_func(..., include_ecc=True)``, weights them with
    :func:`weigh_dimension_samples`, and saves the catalog. Returns its path.
    """
    samples = sampling_func(comp_post_dir, star_df, num_samples=num_samples,
                            m_unit=m_unit, include_ecc=True)
    paths = dimension_tier2_paths(tier1_dir, tier2_dir, spec)
    output, fallback_total = weigh_dimension_samples(
        spec, samples, star_df, sliced_maps_dir(tier1_dir, tier1_label),
        cu.load_sliced(paths["sliced_average"]),
        fill_single_nan_with_average=fill_single_nan_with_average,
    )
    if fallback_total:
        print(f"{tier2_dir}: {fallback_total} posterior draws used the sample-"
              "average completeness where the host's sliced map is undefined "
              "or zero")
    np.savez(paths["catalog"], **output)
    return paths["catalog"]


def weigh_dimension_samples(spec, samples, star_df, host_maps_dir, average,
                            fill_single_nan_with_average=True):
    """Turn (a, m, e) posterior samples into a catalog for one spec.

    ``samples`` maps companion names to ``[a, m, e]`` arrays (M_Jup, AU);
    ``host_maps_dir`` holds the stars' sliced maps and ``average`` is the
    sample's sliced average. For each draw, records:
      * the engine coordinates of the x and y slots (e as 10**e);
      * completeness at the draw's (a, m, e) from the host's own sliced map
        and from the sample average. Host values that are undefined (NaN)
        or exactly zero fall back to the average. NaN fallback is the
        original code's rule; zeros are added because slicing by e leaves
        each moving-average box only a few injections, so "0 of N
        recovered" is common where the true completeness is small but
        positive, and a detected companion cannot have zero completeness;
      * the marginal factor: the product, over marginalized dimensions, of
        the population density at the draw divided by the posterior's
        interim prior there. Both are densities in the dimension's own
        coordinate (log10 for a and m, linear for e), where the orvara
        interim priors are flat, so the factor is the density itself up to a
        constant. It is zero outside a marginalized range. This assumes the
        marginalized dimensions are independent of the fitted ones.
    The main array keeps the original seven-row layout (x, y, average and
    host completeness, their ratios to the prior, and the prior), so the
    fit code reads it unchanged. Returns (catalog dict, number of draws that
    fell back to the average completeness, whether undefined or zero).
    """
    output = {dfu.DIMENSIONS_KEY: np.asarray(spec.to_json())}
    fallback_total = 0
    for star_name, comp_list in zip(star_df.star_name, star_df.comp_list):
        present = [name for name in comp_list if name in samples]
        if not present:
            continue
        host = cu.load_sliced(os.path.join(host_maps_dir, star_name))
        for comp_name in present:
            a, m, e = (np.asarray(values, dtype=float)
                       for values in samples[comp_name])
            average_completeness = average((a, m, e))
            host_completeness = host((a, m, e))
            if fill_single_nan_with_average:
                undefined = ~np.isfinite(host_completeness) | (
                    host_completeness <= 0
                )
                fallback = undefined & np.isfinite(average_completeness)
                host_completeness[fallback] = average_completeness[fallback]
                fallback_total += int(fallback.sum())
            physical = {"a": a, "m": m, "e": e}
            factor = np.ones_like(a)
            for name in spec.marginalized:
                distribution = spec.dimensions[name].distribution
                with np.errstate(invalid="ignore"):
                    factor = factor*distribution.density(physical[name])
            factor[~np.isfinite(factor)] = 0.0
            prior = np.ones_like(a)
            x = spec.to_engine(spec.x, physical[spec.x])
            y = spec.to_engine(spec.y, physical[spec.y])
            output[comp_name] = np.vstack([
                x, y, average_completeness, host_completeness,
                average_completeness/prior, host_completeness/prior, prior,
            ])
            separator = dfu.AUXILIARY_SEPARATOR
            output[comp_name + separator + "marginal_factor"] = factor
            for name, values in physical.items():
                output[comp_name + separator + name] = values
    return output, fallback_total
