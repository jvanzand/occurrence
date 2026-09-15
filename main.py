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
from occurrence import direct_fit_utils as dfu
from occurrence import mcmc_direct

from occurrence.completeness_utils import _process_single_star

def prep_recoveries_files(tier1_dir,
                          star_df,
                          master_rec_dir,
                          recoveries_mtype,
                          m_dir_to_make_q=None
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
    """

    
    if recoveries_mtype=='msini':

        if 'true' in tier1_dir: # If tier1_dir is mtrue or qtrue, then convert msini to mtrue
            for starname in star_df.star_name:
                recoveries_file = os.path.join(master_rec_dir, starname+'_recoveries.csv')
                mtrue_recoveries_save_file = os.path.join(tier1_dir, 'mtrue_recoveries/', starname+'_recoveries.csv')
            
                cu.recs_msini_converter(recoveries_file, mtrue_recoveries_save_file)
            
        else:
            os.makedirs(
                os.path.join(tier1_dir, 'msini_recoveries'), exist_ok=True
            )
            keep_cols = ['inj_msini', 'inj_au', 'inj_e', 'recovered']
            for starname in star_df.star_name:
                recoveries_file = os.path.join(master_rec_dir, starname+'_recoveries.csv')
                msini_recoveries_save_file = os.path.join(tier1_dir, 'msini_recoveries/', starname+'_recoveries.csv')
            
                rec_file = pd.read_csv(recoveries_file)[keep_cols]
                rec_file.to_csv(msini_recoveries_save_file, index=False)
    
        if 'q' in tier1_dir: # If tier1_dir is qsini or qtrue, then convert m to q
            for i in range(len(star_df)):
                row = star_df.iloc[i]
                starname = row.star_name
                mstar = row.Mstar
            
                dirname = tier1_dir+'_recoveries'
            
                q_recoveries_save_file = os.path.join(tier1_dir, dirname, starname+'_recoveries.csv')
                recoveries_file = os.path.join(m_dir_to_make_q, starname+'_recoveries.csv')
            
                cu.recs_mass_ratio_converter(recoveries_file, q_recoveries_save_file, mstar)
            
    elif recoveries_mtype=='mtrue':
        os.makedirs(
            os.path.join(tier1_dir, 'mtrue_recoveries'), exist_ok=True
        )
        if 'sini' in tier1_dir:
            raise Exception("main.prep_recoveries_files: Cannot calculate Msini completeness from Mtrue recoveries files")

        for starname in star_df.star_name:
            recoveries_file = os.path.join(master_rec_dir, starname+'_recoveries.csv')
            mtrue_recoveries_save_file = os.path.join(tier1_dir, 'mtrue_recoveries/', starname+'_recoveries.csv')
            
            rec_file = pd.read_csv(recoveries_file)
            rec_file.to_csv(mtrue_recoveries_save_file, index=False)
    
        if 'q' in tier1_dir: # If tier1_dir is qtrue, then convert m to q
            for i in range(len(star_df)):
                row = star_df.iloc[i]
                starname = row.star_name
                mstar = row.Mstar
            
                dirname = tier1_dir+'_recoveries'
            
                q_recoveries_save_file = os.path.join(tier1_dir, dirname, starname+'_recoveries.csv')
                recoveries_file = os.path.join(m_dir_to_make_q, starname+'_recoveries.csv')
            
                cu.recs_mass_ratio_converter(recoveries_file, q_recoveries_save_file, mstar)
            
            
    return

def prep_maps(tier1_dir,
              star_df,
              path_to_recoveries,
              m_unit='earth',
              avg_map_only=False,
              save_single_plots=False):
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

    maps_save_label = 'saved_maps_'+tier1_dir
    maps_save_path = os.path.join(tier1_dir, maps_save_label)
    maps_ycol = f"inj_{tier1_dir}"
    
    if mp.cpu_count()>100:
        ncores=30
    else:
        ncores = int(mp.cpu_count()/2)

    args_list = [
        (row, path_to_recoveries, maps_save_path, maps_ycol, m_unit,
         avg_map_only, save_single_plots)
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
                     m_unit='earth'):
    """
    Calculate average map from a subset of
    pre-computed completeness maps
    """
    
    path_to_maps = os.path.join(tier1_dir, f"saved_maps_{tier1_dir}")
    avg_map_dir = os.path.join(tier1_dir, tier2_dir, 'avg_map/')
    
    cu.average_map(path_to_maps, avg_map_dir, star_df.star_name.to_list(), ycol=ycol, m_unit=m_unit)
    
    return
    
def prep_post_draws(tier1_dir, tier2_dir,
                    star_df, comp_post_dir,
                    sampling_func,
                    saved_maps_dir=None, m_unit='earth',
                    fig_title='Catalog Posteriors',
                    avg_map_only=False,
                    fill_single_nan_with_average=True):

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
    """
    
    #if avg_map_only:
    #    Mstar = star_df.Mstar.mean()
    #    cols = star_df.columns
    #    avg_df = pd.DataFrame([['average', Mstar, [], 0]], columns=cols)
    #    star_df_compl = avg_df
    #else:
    #    star_df_compl=star_df
        
    #### CUSTOMIZE your own sampler to match the posterior format ####
    ## The output of custom sampler should be a dict whose keys are companion names
    ## and whose values are 2xN arrays, where the first/second sub-array is SMA/mass samples
    post_sample_dict = sampling_func(comp_post_dir, star_df, num_samples=500, m_unit=m_unit) # First sample posteriors
    #from copy import deepcopy; pp_test = deepcopy(post_sample_dict)
    ## If using mass ratio, convert masses to q
    #if "qtrue" in saved_maps_dir or "qsini" in saved_maps_dir:
    if 'q' in tier1_dir:

        for comp_name in post_sample_dict.keys():
            row = star_df[star_df['comp_list'].apply(lambda name_list: comp_name in name_list)]
            mstar = row.Mstar.values
                
            old_samples = post_sample_dict[comp_name]
            new_samples = np.array([old_samples[0], old_samples[1]/mstar]) # Same 'a' samples; convert m-->q
            post_sample_dict[comp_name] = new_samples
                
    post_prior_sample_dict = su.interim_prior(post_sample_dict, prior_type='loguniform') # Then calculate prior at each draw. Each value is a 3xN array of SMA samples, mass samples, and prior values
    #qq_test = deepcopy(post_prior_sample_dict)
    # Now add on completeness values
    sampled_post_with_compls = su.include_post_completeness(post_prior_sample_dict,
                                                            star_df,
                                                            tier1_dir, tier2_dir,
                                                            avg_map_only=avg_map_only,
                                                            fill_single_nan_with_average=fill_single_nan_with_average)
    #rr_test = deepcopy(sampled_post_with_compls)                                                  
    ## Saves dict with companion names as key names
    ## Each value is a 7xN array, with prior densities defined relative to
    ## dlog10(a) dlog10(m), of:
    ## [a_list, m_list, avg_compls, single_star_compls,
    ##  compl_over_prior_avg, compl_over_prior_single, interim_prior]
    ## Probably the only compl array I'll use is compl_over_prior_single. compl_over_prior_avg is to test whether using avg completeness changes the answer. The two completeness arrays are for testing/sanity checks.
    saved_dict_dir = os.path.join(tier1_dir, tier2_dir, 'sampled_post_prior_compl.npz')
    np.savez(saved_dict_dir, **sampled_post_with_compls)
        
        
    return
    
    
def prep_direct_fit_materials(
        tier1_dir,
        tier2_dir,
        tier3_dir,
        x_bounds,
        y_bounds,
        star_df,
        completeness_type='single',
        integration_resolution=(100, 100),
        use_average_completeness=True,
        interim_prior_fn=None):
    """Prepare unbinned samples and survey exposure for a direct fit.

    This Stage 2 entry point supports the current ``(a, mass)`` catalog order
    and one rectangular x-y region. It writes ``direct_fit_data.npz`` but
    does not evaluate a likelihood or run MCMC. The average-map exposure is
    the default because some individual maps may contain NaNs in the ROI;
    individual-map summation remains available with strict validation.
    """
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
        average_path = os.path.join(
            tier1_dir, tier2_dir, 'avg_map', 'interp_fn.pkl'
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
        tier1_label = os.path.basename(os.path.normpath(tier1_dir))
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
        tier1_dir, tier2_dir, tier3_dir, 'saved_dicts', 'direct_fit_data.npz'
    )
    dfu.save_direct_fit_data(save_path, companions, exposure)
    return save_path


def plot_direct_piecewise(
        tier1_dir,
        tier2_dir,
        tier3_dir,
        nstars,
        stack_dim,
        m_unit='earth',
        mtype=None,
        title='Direct piecewise-constant fit',
        plot_occurrence=True,
        plot_density=True,
        plot_corner=True,
        plot_catalog_roi=False,
        plot_roi_occurrence=False):
    """Load and plot a saved direct piecewise-constant fit."""
    base_dir = os.path.join(tier1_dir, tier2_dir, tier3_dir)
    if mtype is None:
        mtype = os.path.basename(os.path.normpath(tier1_dir))
    return mcmc_direct.plot_piecewise_results(
        direct_fit_path=os.path.join(
            base_dir, 'saved_dicts', 'direct_fit_data.npz'
        ),
        chain_path=os.path.join(
            base_dir, 'saved_chains', 'chains_direct_piecewise.npz'
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
        tier1_dir=tier1_dir,
        tier2_dir=tier2_dir,
    )


def plot_direct_smooth(
        tier1_dir,
        tier2_dir,
        tier3_dir,
        model_name,
        stack_dim,
        a_edges,
        m_edges,
        m_unit='earth',
        title='Direct smooth-model fit',
        plot_occurrence=True,
        plot_cumulative=False,
        plot_density=True,
        plot_corner=True,
        model_plot_style='credible',
        n_posterior_draws=100,
        plot_random_seed=None):
    """Load, plot, and save one smooth direct model."""
    import matplotlib.pyplot as plt

    base_dir = os.path.join(tier1_dir, tier2_dir, tier3_dir)
    # Saved material bounds identify only the outer range. Chain discovery is
    # intentionally filename-based so plotting also works after a fit-only run.
    chain_dir = os.path.join(base_dir, 'saved_chains')
    chain_paths = glob.glob(
        os.path.join(chain_dir, f'chains_direct_{model_name}_bin*.npz')
    )
    chain_paths.sort(
        key=lambda path: int(os.path.splitext(path)[0].rsplit('bin', 1)[1])
    )
    if not chain_paths:
        raise FileNotFoundError(
            f"no saved direct {model_name} chains in {chain_dir!r}"
        )
    figures = {}
    if plot_density:
        figures['density'] = plt.subplots(figsize=(6, 4))
    if plot_occurrence:
        figures['occurrence'] = plt.subplots(figsize=(6, 4))
    if plot_cumulative:
        figures['cumulative'] = plt.subplots(figsize=(6, 4))
    model_edges = m_edges if stack_dim == 'a' else a_edges
    paths = mcmc_direct.add_smooth_model_to_figures(
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
    paths.update(mcmc_direct.save_direct_model_figures(
        figures=figures, output_dir=os.path.join(base_dir, 'plots'),
        stack_dim=stack_dim, model_edges=model_edges, title=title,
        m_unit=m_unit,
    ))
    return paths


def plot_direct_models(
        tier1_dir,
        tier2_dir,
        tier3_dir,
        nstars,
        stack_dim,
        a_edges,
        m_edges,
        plot_models,
        m_unit='earth',
        title='Direct occurrence fit',
        plot_occurrence=True,
        plot_cumulative=False,
        plot_density=True,
        plot_corner=True,
        plot_catalog_roi=False,
        plot_roi_occurrence=False,
        model_plot_style='credible',
        n_posterior_draws=100,
        plot_random_seed=None):
    """Plot selected direct models together while retaining separate corners."""
    selected = list(plot_models)
    supported = {'piecewise', 'logG', 'escarpment', 'sigmoid', 'bpl'}
    unknown = set(selected) - supported
    if unknown:
        raise ValueError(f"unsupported direct plot models: {sorted(unknown)}")
    if selected == ['piecewise']:
        return plot_direct_piecewise(
            tier1_dir=tier1_dir, tier2_dir=tier2_dir, tier3_dir=tier3_dir,
            nstars=nstars, stack_dim=stack_dim,
            m_unit=m_unit, title=title, plot_occurrence=plot_occurrence,
            plot_density=plot_density, plot_corner=plot_corner,
            plot_catalog_roi=plot_catalog_roi,
            plot_roi_occurrence=plot_roi_occurrence,
        )
    if len(selected) == 1 and selected[0] != 'piecewise':
        return plot_direct_smooth(
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
        )
    base_dir = os.path.join(tier1_dir, tier2_dir, tier3_dir)
    base_figures = {}
    model_edges = m_edges if stack_dim == 'a' else a_edges
    paths = {}
    if 'piecewise' in selected:
        direct_fit_path = os.path.join(
            base_dir, 'saved_dicts', 'direct_fit_data.npz'
        )
        piecewise_chain = os.path.join(
            base_dir, 'saved_chains', 'chains_direct_piecewise.npz'
        )
        summary_path = os.path.join(
            base_dir, 'saved_dicts', 'summary_dict_direct_piecewise.npz'
        )
        summary = mcmc_direct.summarize_piecewise_file(
            direct_fit_path, piecewise_chain, nstars, save_path=summary_path
        )
        mtype = os.path.basename(os.path.normpath(tier1_dir))
        if plot_density:
            base_figures['density'] = pu.plot_occurrence_hist(
                summary, stack_dim=stack_dim, m_unit=m_unit, mtype=mtype,
                rate_type='ORD', title=title, return_fig_ax=True,
            )
        if plot_cumulative:
            base_figures['cumulative'] = mcmc_direct.piecewise_cumulative_figure(
                piecewise_chain, stack_dim=stack_dim, title=title, m_unit=m_unit,
            )
        paths['piecewise'] = mcmc_direct.plot_piecewise_results(
            direct_fit_path, piecewise_chain, os.path.join(base_dir, 'plots'),
            nstars, stack_dim, m_unit=m_unit, mtype=mtype, title=title,
            plot_occurrence=False, plot_density=False, plot_corner=plot_corner,
            plot_catalog_roi=plot_catalog_roi,
            plot_roi_occurrence=plot_roi_occurrence,
            tier1_dir=tier1_dir, tier2_dir=tier2_dir,
            summary=summary,
        )
    else:
        import matplotlib.pyplot as plt
        if plot_density:
            base_figures['density'] = plt.subplots(figsize=(6, 4))
        if plot_cumulative:
            base_figures['cumulative'] = plt.subplots(figsize=(6, 4))
    for model_name in [name for name in selected if name != 'piecewise']:
        chain_dir = os.path.join(base_dir, 'saved_chains')
        chain_paths = glob.glob(os.path.join(
            chain_dir, f'chains_direct_{model_name}_bin*.npz'
        ))
        chain_paths.sort(
            key=lambda path: int(os.path.splitext(path)[0].rsplit('bin', 1)[1])
        )
        if not chain_paths:
            raise FileNotFoundError(
                f"no saved direct {model_name} chains in {chain_dir!r}"
            )
        paths[model_name] = mcmc_direct.add_smooth_model_to_figures(
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
    paths['combined'] = mcmc_direct.save_direct_model_figures(
        figures=base_figures, output_dir=os.path.join(base_dir, 'plots'),
        stack_dim=stack_dim, model_edges=model_edges, title=title,
        m_unit=m_unit,
    )
    return paths

