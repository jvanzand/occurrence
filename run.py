## Module to run the mass ratio occurrence calculation
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from types import SimpleNamespace
from concurrent.futures import ProcessPoolExecutor, as_completed

from occurrence import plotting_utils as pu
from occurrence import main
from occurrence import mcmc_direct

# Use plot template
plt.style.use(os.path.join(os.path.dirname(__file__), 'matplotlibrc'))
  

    
def make_tier1(tier1_config, star_df_full, recoveries_dir, 
               recoveries_mtype='mtrue', 
               avg_map_only=False,
               save_single_plots=False):
    """
    Makes only the tier1 dict
    """
    t1 = SimpleNamespace(**tier1_config)
    
    y_param = f"{t1.m_or_q}{t1.true_or_sini}"
    m_dir_to_make_q = os.path.join(t1.t1_dir, f"m{t1.true_or_sini}_recoveries")
    path_to_recoveries = os.path.join(t1.t1_dir, f"{y_param}_recoveries")
    
    ## Optionally use only one recoveries file to represent completeness for the whole sample
    if avg_map_only:
        Mstar = star_df_full.Mstar.mean()
        cols = star_df_full.columns
        avg_df = pd.DataFrame([['average', Mstar, [], 0]], columns=cols)
        star_df_full = avg_df
    
    t1_exists = os.path.exists(t1.t1_dir)
    if not t1_exists:
        main.prep_recoveries_files(tier1_dir=t1.t1_dir,
                                   star_df=star_df_full,
                                   master_rec_dir=recoveries_dir,
                                   recoveries_mtype=recoveries_mtype,
                                   m_dir_to_make_q=m_dir_to_make_q,
                                   )
        main.prep_maps(tier1_dir=t1.t1_dir,
                       star_df=star_df_full,
                       path_to_recoveries=path_to_recoveries,
                       m_unit=t1.mass_unit,
                       avg_map_only=avg_map_only,
                       save_single_plots=save_single_plots)
    return


def make_tier2(tier2_config, star_df_full, comp_post_dir, sampling_func, avg_map_only=False):
    """
    Makes only the tier2 dict
    """
    
    t2 = SimpleNamespace(**tier2_config)
                        
                        
    if t2.star_df_query is not None:
        star_df = star_df_full.query(t2.star_df_query) 
    else:
        star_df = star_df_full
    nstars = len(star_df)
    
    # Make star_df_map either with all recoveries files or just with avg
    if avg_map_only:
        Mstar = star_df_full.Mstar.mean()
        cols = star_df_full.columns
        avg_df = pd.DataFrame([['average', Mstar, [], 0]], columns=cols)
        star_df_compl = avg_df
    else:
        star_df_compl = star_df
    
    #if t2.t1_true_or_sini=='true':
    #    comp_post_dir='/data/user/judahvz/planet_bd/orvara/judah/burned_chains_am_only/'
    #elif t2.t1_true_or_sini=='sini':
    #    comp_post_dir='/data/user/judahvz/planet_bd/orvara/judah/burned_chains_amsini_only/'

    y_param = f"{t2.t1_m_or_q}{t2.t1_true_or_sini}"
    saved_maps_dir = os.path.join(t2.t1_dir, f"saved_maps_{y_param}")
    ycol = f"inj_{y_param}"
    
    t2_exists = _tier2_artifacts_exist(t2.t1_dir, t2.t2_dir)
    print("Checking if exists: ", t2.t1_dir, t2.t2_dir, t2_exists)
    if not t2_exists:
        main.make_average_map(tier1_dir=t2.t1_dir,
                              tier2_dir=t2.t2_dir,
                              star_df=star_df_compl,
                              ycol=ycol,
                              m_unit=t2.t1_mass_unit)

        main.prep_post_draws(tier1_dir=t2.t1_dir,
                             tier2_dir=t2.t2_dir,
                             star_df=star_df, comp_post_dir=comp_post_dir,
                             sampling_func=sampling_func,
                             saved_maps_dir=saved_maps_dir, m_unit=t2.t1_mass_unit,
                             avg_map_only=avg_map_only)
                             
        ## Plot companion catalog
        catalog_path = os.path.join(t2.t1_dir, t2.t2_dir, 'sampled_post_prior_compl.npz')
        plot_save_path = os.path.join(t2.t1_dir, t2.t2_dir, 'catalog_and_completeness.png')          
        pu.plot_catalog(tier1_dir=t2.t1_dir,
                        tier2_dir=t2.t2_dir,
                        catalog_path=catalog_path,
                        m_unit=t2.t1_mass_unit, fig_title=f'Average Completeness for {nstars} Stars',
                        fig_savepath=plot_save_path)
        ##########################
    
    return


def _tier2_artifacts_exist(tier1_dir, tier2_dir):
    """Return whether the Tier 2 products required by direct fitting exist."""
    tier2_path = os.path.join(tier1_dir, tier2_dir)
    required_paths = (
        os.path.join(tier2_path, "sampled_post_prior_compl.npz"),
        os.path.join(tier2_path, "avg_map", "interp_fn.pkl"),
    )
    return all(os.path.isfile(path) for path in required_paths)


def run_direct_multiple(
        tier1_list,
        tier2_list,
        tier3_list,
        a_edges,
        m_edges,
        star_df,
        tier2_df_cuts_dict,
        recoveries_dir=None,
        recoveries_mtype="mtrue",
        comp_post_dir=None,
        sampling_func=None,
        run_models_list=("piecewise",),
        plot_models_list=("piecewise",),
        stack_dim="a",
        m_unit="earth",
        run_fits=True,
        make_plots=True,
        plot_occurrence=True,
        plot_cumulative=False,
        plot_density=True,
        plot_corner=True,
        plot_catalog_roi=False,
        plot_roi_occurrence=False,
        completeness_type="single",
        integration_resolution=(100, 100),
        use_average_completeness=True,
        nwalkers=50,
        nsteps=5000,
        burnin=1000,
        parallel_fits=False,
        parallel_mcmc=False,
        random_seed=None,
        logg_amplitude_bounds=(1e-6, 10.0),
        logg_sigma_bounds=None,
        escarpment_amplitude_bounds=(1e-6, 10.0),
        sigmoid_amplitude_bounds=(1e-6, 10.0),
        sigmoid_width_bounds=None,
        bpl_amplitude_bounds=(1e-6, 10.0),
        bpl_slope_bounds=(-4.0, 4.0),
        max_integrated_occurrence=1.0,
        model_plot_style="credible",
        n_posterior_draws=100,
        plot_random_seed=None,
        prepare_missing=True,
        avg_map_only=False,
        plot_tier1_maps=False):
    """Run and plot direct occurrence fits for multiple tier combinations.

    Registered models are ``piecewise``, ``logG``, ``escarpment``, ``sigmoid``,
    and ``bpl``.
    Missing Tier 1 products
    are generated from ``recoveries_dir`` when ``prepare_missing`` is true.
    Set ``run_fits=False`` to regenerate plots from saved chains. Fit-level and
    MCMC-level parallelism are controlled separately.
    """
    supported_models = {"piecewise", "logG", "escarpment", "sigmoid", "bpl"}
    run_models = list(run_models_list)
    plot_models = list(plot_models_list)
    unknown = (set(run_models) | set(plot_models)) - supported_models
    if unknown:
        raise ValueError(
            f"unsupported direct models: {sorted(unknown)}; "
            f"available models: {sorted(supported_models)}"
        )
    if stack_dim not in {"a", "m"}:
        raise ValueError("stack_dim must be 'a' or 'm'")

    tier1_configurations = []
    for tier1_dir in tier1_list:
        mass_unit = m_unit if tier1_dir in {"mtrue", "msini"} else None
        true_or_sini = "true" if "true" in tier1_dir else "sini"
        m_or_q = "m" if "m" in tier1_dir else "q" if "q" in tier1_dir else None
        if m_or_q is None:
            raise ValueError(
                f"cannot determine mass or mass-ratio coordinate from {tier1_dir!r}"
            )
        tier1_configurations.append({
            "mass_unit": mass_unit,
            "m_or_q": m_or_q,
            "true_or_sini": true_or_sini,
            "t1_dir": tier1_dir,
        })

    missing_tier1 = [
        configuration for configuration in tier1_configurations
        if not os.path.isdir(configuration["t1_dir"])
    ]
    if missing_tier1 and not prepare_missing:
        raise FileNotFoundError(
            "required Tier 1 directories are missing: " +
            str([item["t1_dir"] for item in missing_tier1])
        )
    if missing_tier1:
        if recoveries_dir is None:
            raise ValueError(
                "recoveries_dir is required to create missing Tier 1 products"
            )
        preparation_arguments = (
            star_df, recoveries_dir, recoveries_mtype, avg_map_only,
            plot_tier1_maps,
        )
        if parallel_fits and len(missing_tier1) > 1:
            _run_futures_in_parallel(
                make_tier1, missing_tier1, preparation_arguments, "direct Tier 1"
            )
        else:
            for configuration in missing_tier1:
                make_tier1(configuration, *preparation_arguments)

    configurations = []
    tier2_configurations = []
    for tier1_dir in tier1_list:
        mass_unit = m_unit if tier1_dir in {"mtrue", "msini"} else None
        true_or_sini = "true" if "true" in tier1_dir else "sini"
        m_or_q = "m" if "m" in tier1_dir else "q" if "q" in tier1_dir else None
        for tier2_dir in tier2_list:
            try:
                cut_config, title = tier2_df_cuts_dict[tier2_dir]
            except KeyError:
                raise ValueError(f"no Tier 2 configuration for {tier2_dir!r}")
            query = cut_config.get("star_df_query")
            selected_stars = star_df.query(query).copy() if query else star_df.copy()
            tier2_configurations.append({
                "t1_dir": tier1_dir,
                "t2_dir": tier2_dir,
                "star_df_query": query,
                "t1_mass_unit": mass_unit,
                "t1_true_or_sini": true_or_sini,
                "t1_m_or_q": m_or_q,
            })
            for tier3_dir in tier3_list:
                configurations.append({
                    "tier1_dir": tier1_dir,
                    "tier2_dir": tier2_dir,
                    "tier3_dir": tier3_dir,
                    "a_edges": np.asarray(a_edges, dtype=float),
                    "m_edges": np.asarray(m_edges, dtype=float),
                    "star_df": selected_stars,
                    "title": title,
                    "run_models": run_models,
                    "plot_models": plot_models,
                    "stack_dim": stack_dim,
                    "m_unit": m_unit,
                    "run_fits": run_fits,
                    "make_plots": make_plots,
                    "plot_occurrence": plot_occurrence,
                    "plot_cumulative": plot_cumulative,
                    "plot_density": plot_density,
                    "plot_corner": plot_corner,
                    "plot_catalog_roi": plot_catalog_roi,
                    "plot_roi_occurrence": plot_roi_occurrence,
                    "completeness_type": completeness_type,
                    "integration_resolution": integration_resolution,
                    "use_average_completeness": use_average_completeness,
                    "nwalkers": nwalkers,
                    "nsteps": nsteps,
                    "burnin": burnin,
                    "parallel_mcmc": parallel_mcmc,
                    "random_seed": random_seed,
                    "logg_amplitude_bounds": logg_amplitude_bounds,
                    "logg_sigma_bounds": logg_sigma_bounds,
                    "escarpment_amplitude_bounds": escarpment_amplitude_bounds,
                    "sigmoid_amplitude_bounds": sigmoid_amplitude_bounds,
                    "sigmoid_width_bounds": sigmoid_width_bounds,
                    "bpl_amplitude_bounds": bpl_amplitude_bounds,
                    "bpl_slope_bounds": bpl_slope_bounds,
                    "max_integrated_occurrence": max_integrated_occurrence,
                    "model_plot_style": model_plot_style,
                    "n_posterior_draws": n_posterior_draws,
                    "plot_random_seed": plot_random_seed,
                })

    missing_tier2 = [
        configuration for configuration in tier2_configurations
        if not _tier2_artifacts_exist(
            configuration["t1_dir"], configuration["t2_dir"]
        )
    ]
    if missing_tier2 and not prepare_missing:
        missing_paths = [
            os.path.join(item["t1_dir"], item["t2_dir"])
            for item in missing_tier2
        ]
        raise FileNotFoundError(
            f"required Tier 2 products are missing: {missing_paths}"
        )
    if missing_tier2:
        if comp_post_dir is None or sampling_func is None:
            raise ValueError(
                "comp_post_dir and sampling_func are required to create missing Tier 2 products"
            )
        preparation_arguments = (
            star_df, comp_post_dir, sampling_func, avg_map_only
        )
        if parallel_fits and len(missing_tier2) > 1:
            _run_futures_in_parallel(
                make_tier2, missing_tier2, preparation_arguments, "direct Tier 2"
            )
        else:
            for configuration in missing_tier2:
                make_tier2(configuration, *preparation_arguments)

    if parallel_fits and len(configurations) > 1:
        with ProcessPoolExecutor() as executor:
            futures = {
                executor.submit(_run_direct_configuration, configuration): index
                for index, configuration in enumerate(configurations)
            }
            results = [None]*len(configurations)
            for future in as_completed(futures):
                index = futures[future]
                results[index] = future.result()
                configuration = configurations[index]
                print(
                    "Finished direct fit "
                    f"{configuration['tier1_dir']}/"
                    f"{configuration['tier2_dir']}/"
                    f"{configuration['tier3_dir']}"
                )
    else:
        results = [
            _run_direct_configuration(configuration)
            for configuration in configurations
        ]
    return results


def _run_futures_in_parallel(function, configurations, shared_args, label):
    """Run a prerequisite phase concurrently and propagate worker failures."""
    with ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(function, configuration, *shared_args): configuration
            for configuration in configurations
        }
        for future in as_completed(futures):
            configuration = futures[future]
            future.result()
            print(
                f"Finished {label} "
                f"{configuration['t1_dir']}/{configuration['t2_dir']}"
            )


def _run_direct_configuration(configuration):
    """Execute one configuration produced by :func:`run_direct_multiple`."""
    result = {
        "tier1": configuration["tier1_dir"],
        "tier2": configuration["tier2_dir"],
        "tier3": configuration["tier3_dir"],
        "nstars": len(configuration["star_df"]),
        "chains": {},
        "plots": {},
    }
    if configuration["run_fits"]:
        material_path = main.prep_direct_fit_materials(
            tier1_dir=configuration["tier1_dir"],
            tier2_dir=configuration["tier2_dir"],
            tier3_dir=configuration["tier3_dir"],
            x_bounds=(configuration["a_edges"][0], configuration["a_edges"][-1]),
            stack_bounds=(configuration["m_edges"][0], configuration["m_edges"][-1]),
            star_df=configuration["star_df"],
            completeness_type=configuration["completeness_type"],
            integration_resolution=configuration["integration_resolution"],
            use_average_completeness=configuration["use_average_completeness"],
        )
        result["materials"] = material_path
        for model_name in configuration["run_models"]:
            if model_name == "piecewise":
                chain_path = os.path.join(
                    configuration["tier1_dir"], configuration["tier2_dir"],
                    configuration["tier3_dir"], "saved_chains",
                    "chains_direct_piecewise.npz",
                )
                mcmc_direct.fit_piecewise_file(
                    direct_fit_path=material_path,
                    x_edges=configuration["a_edges"],
                    stack_edges=configuration["m_edges"],
                    nwalkers=configuration["nwalkers"],
                    nsteps=configuration["nsteps"],
                    burnin=configuration["burnin"],
                    parallel=configuration["parallel_mcmc"],
                    save_path=chain_path,
                    random_seed=configuration["random_seed"],
                )
                result["chains"][model_name] = chain_path
            elif model_name in {"logG", "escarpment", "sigmoid", "bpl"}:
                amplitude_key = {
                    "logG": "logg_amplitude_bounds",
                    "escarpment": "escarpment_amplitude_bounds",
                    "sigmoid": "sigmoid_amplitude_bounds",
                    "bpl": "bpl_amplitude_bounds",
                }[model_name]
                width_bounds = (
                    configuration["sigmoid_width_bounds"]
                    if model_name == "sigmoid"
                    else configuration["logg_sigma_bounds"]
                )
                _, chain_paths = mcmc_direct.fit_smooth_file(
                    direct_fit_path=material_path,
                    a_edges=configuration["a_edges"],
                    m_edges=configuration["m_edges"],
                    stack_dim=configuration["stack_dim"],
                    model_name=model_name,
                    save_dir=os.path.join(
                        configuration["tier1_dir"], configuration["tier2_dir"],
                        configuration["tier3_dir"], "saved_chains",
                    ),
                    nwalkers=configuration["nwalkers"],
                    nsteps=configuration["nsteps"],
                    burnin=configuration["burnin"],
                    parallel=configuration["parallel_mcmc"],
                    random_seed=configuration["random_seed"],
                    amplitude_bounds=configuration[amplitude_key],
                    slope_bounds=configuration["bpl_slope_bounds"],
                    width_bounds=width_bounds,
                    max_integrated_occurrence=configuration[
                        "max_integrated_occurrence"
                    ],
                )
                result["chains"][model_name] = chain_paths

    if configuration["make_plots"] and configuration["plot_models"]:
        result["plots"] = main.plot_direct_models(
            tier1_dir=configuration["tier1_dir"],
            tier2_dir=configuration["tier2_dir"],
            tier3_dir=configuration["tier3_dir"],
            nstars=len(configuration["star_df"]),
            stack_dim=configuration["stack_dim"],
            a_edges=configuration["a_edges"],
            m_edges=configuration["m_edges"],
            plot_models=configuration["plot_models"],
            m_unit=configuration["m_unit"],
            title=(
                f"{len(configuration['star_df'])} Stars "
                f"({configuration['title']})"
            ),
            plot_occurrence=configuration["plot_occurrence"],
            plot_cumulative=configuration["plot_cumulative"],
            plot_density=configuration["plot_density"],
            plot_corner=configuration["plot_corner"],
            plot_catalog_roi=configuration["plot_catalog_roi"],
            plot_roi_occurrence=configuration["plot_roi_occurrence"],
            model_plot_style=configuration["model_plot_style"],
            n_posterior_draws=configuration["n_posterior_draws"],
            plot_random_seed=configuration["plot_random_seed"],
        )
    return result
