## Module to run the mass ratio occurrence calculation
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from types import SimpleNamespace
from concurrent.futures import ProcessPoolExecutor, as_completed
import traceback
from copy import deepcopy


github_path = '/data/user/judahvz/GitHub/'
occurrence_path = os.path.join(github_path, 'occurrence/')
sys.path.insert(0, github_path)
from occurrence import analysis_utils as au
from occurrence import occurrence_utils as ou
from occurrence import completeness_utils as cu
from occurrence import plotting_utils as pu
from occurrence import main

from astropy import constants as c
Mj2Me = (c.M_jup/c.M_earth).value
Mj2Ms = (c.M_jup/c.M_sun).value

# Use plot template
plt.style.use(os.path.join(occurrence_path, 'matplotlibrc'))
  

    
def make_tier1(tier1_config, star_df_full, recoveries_dir, recoveries_m_unit):
    """
    Makes only the tier1 dict
    """
    #import pdb; pdb.set_trace()
    t1 = SimpleNamespace(**tier1_config)
    
    y_param = f"{t1.m_or_q}{t1.true_or_sini}"
    m_dir_to_make_q = os.path.join(t1.t1_dir, f"m{t1.true_or_sini}_recoveries")
    path_to_recoveries = os.path.join(t1.t1_dir, f"{y_param}_recoveries")
    
    t1_exists = os.path.exists(t1.t1_dir)
    if not t1_exists:
        main.prep_recoveries_files(tier1_dir=t1.t1_dir,
                                   star_df=star_df_full,
                                   msini_rec_dir_to_make_mtrue=recoveries_dir,
                                   m_dir_to_make_q=m_dir_to_make_q,
                                   recoveries_m_unit=recoveries_m_unit)
        
        main.prep_maps(tier1_dir=t1.t1_dir,
                       star_df=star_df_full,
                       path_to_recoveries=path_to_recoveries,
                       m_unit=t1.mass_unit)
    return


def make_tier2(tier2_config, star_df_full):
    """
    Makes only the tier2 dict
    """
    
    t2 = SimpleNamespace(**tier2_config)
                        
                        
    if t2.star_df_query is not None:
        star_df = star_df_full.query(t2.star_df_query) 
    else:
        star_df = star_df_full
    nstars = len(star_df)
    
    if t2.t1_true_or_sini=='true':
        comp_post_dir='/data/user/judahvz/planet_bd/orvara/judah/burned_chains_am_only/'
    elif t2.t1_true_or_sini=='sini':
        comp_post_dir='/data/user/judahvz/planet_bd/orvara/judah/burned_chains_amsini_only/'
        #raise Exception('sini not yet fixed. Resolve problem in burned_chains_amsini_only/msini_chain_maker. Jul2 notes.')
    y_param = f"{t2.t1_m_or_q}{t2.t1_true_or_sini}"
    saved_maps_dir = os.path.join(t2.t1_dir, f"saved_maps_{y_param}")
    ycol = f"inj_{y_param}"
    
    t2_exists = os.path.exists(os.path.join(t2.t1_dir, t2.t2_dir))
    print("Checking if exists: ", t2.t1_dir, t2.t2_dir, t2_exists)
    if not t2_exists:
        main.make_average_map(tier1_dir=t2.t1_dir,
                              tier2_dir=t2.t2_dir,
                              star_df=star_df,
                              ycol=ycol,
                              m_unit=t2.t1_mass_unit)

        main.prep_post_draws(tier1_dir=t2.t1_dir,
                             tier2_dir=t2.t2_dir,
                             star_df=star_df, comp_post_dir=comp_post_dir,
                             saved_maps_dir=saved_maps_dir, m_unit=t2.t1_mass_unit)
                             
        #print("DIRS??", t2.t1_dir, t2.t2_dir)
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


def make_tier3(tier3_config, star_df_full):
    """
    Makes only the tier3 dict
    """
    
    t3 = SimpleNamespace(**tier3_config)
    
    if t3.t2_star_df_query is not None:
        star_df = star_df_full.query(t3.t2_star_df_query) 
    else:
        star_df = star_df_full
    nstars = len(star_df)
    
    
    #import pdb; pdb.set_trace()
    ## Prepare materials that are specific not only to the stars and companions,
    ## but also to the region of parameter space under consideration
    t3_exists = os.path.exists(os.path.join(t3.t1_dir, t3.t2_dir, t3.t3_dir))
    if not t3_exists:

        main.prep_occurrence_materials(tier1_dir=t3.t1_dir,
                                       tier2_dir=t3.t2_dir, 
                                       tier3_dir=t3.t3_dir,
                                       a_edges=t3.a_edges, 
                                       m_or_q_edges=t3.m_or_q_edges,
                                       stack_dim=t3.stack_dim,
                                       star_df=star_df,
                                       compl_type='single',
                                       m_unit=t3.t1_mass_unit,)

        ## Plot catalog with occurrence cells
        catalog_path = os.path.join(t3.t1_dir, t3.t2_dir, t3.t3_dir, 
                                    'saved_dicts/sampled_post_prior_compl_lam_inROI.npz')       
        plot_save_path = os.path.join(t3.t1_dir, t3.t2_dir, t3.t3_dir, 'plots/catalog_inROI_and_completeness.png')          
        pu.plot_catalog(tier1_dir=t3.t1_dir,
                        tier2_dir=t3.t2_dir,
                        catalog_path=catalog_path,
                        a_edges=t3.a_edges,
                        m_edges=t3.m_or_q_edges,
                        m_unit=t3.t1_mass_unit, fig_title='',
                        fig_savepath=plot_save_path)
        #####################################
    #import pdb; pdb.set_trace()
    print("Run mcmc?", t3.run_mcmc, t3.t1_dir, t3.t2_dir, t3.t3_dir)
    if t3.run_mcmc:
        print("Going to run MCMC", t3.t1_dir, t3.t2_dir, t3.t3_dir)
        nstars = len(star_df)
        
        
        main.run_mcmc(tier1_dir=t3.t1_dir,
                      tier2_dir=t3.t2_dir, 
                      tier3_dir=t3.t3_dir,
                      run_models=t3.run_models,
                      a_edges=t3.a_edges,
                      m_edges=t3.m_or_q_edges,
                      stack_dim=t3.stack_dim,
                      nstars=nstars, parallel=False,
                      nwalkers=50, nsteps=6000, burnin=2000)
        
        
        #if t3.plot.summary_stats:
        main.summary_stats(tier1_dir=t3.t1_dir,
                           tier2_dir=t3.t2_dir,
                           tier3_dir=t3.t3_dir,
                           nstars=nstars,
                           verbose=False)
        
        main.make_results_plots(tier1_dir=t3.t1_dir,
                                tier2_dir=t3.t2_dir, 
                                tier3_dir=t3.t3_dir, 
                                nstars=nstars,
                                run_models=t3.run_models,
                                plot_models=t3.plot_models,
                                plot_hist_corner=True,
                                plot_model_corners=True,
                                stack_dim=t3.stack_dim, m_unit=t3.t1_mass_unit,
                                hist_title=f"{nstars} Stars ({t3.occurrence_hist_title})")
    
    if t3.run_bic_compare:

        main.bic_compare(tier1_dir=t3.t1_dir,
                         tier2_dir=t3.t2_dir, 
                         tier3_dir=t3.t3_dir,
                         run_models=t3.run_models,
                         stack_dim=t3.stack_dim,
                         m_unit=t3.t1_mass_unit)
    
    if t3.plot_only: # Assuming calculations have been done, simply regenerate plots
    
        #if t3.plot.plot_catalogs:
        ## Plot companion catalog
        catalog_path = os.path.join(t3.t1_dir, t3.t2_dir, 'sampled_post_prior_compl.npz')
        plot_save_path = os.path.join(t3.t1_dir, t3.t2_dir, 'catalog_and_completeness.png')          
        pu.plot_catalog(tier1_dir=t3.t1_dir,
                        tier2_dir=t3.t2_dir,
                        catalog_path=catalog_path,
                        m_unit=t3.t1_mass_unit, fig_title=f'Average Completeness for {nstars} Stars',
                        fig_savepath=plot_save_path)
                        
                        
        ## Plot catalog with occurrence cells
        catalog_path = os.path.join(t3.t1_dir, t3.t2_dir, t3.t3_dir, 
                                    'saved_dicts/sampled_post_prior_compl_lam_inROI.npz')       
        plot_save_path = os.path.join(t3.t1_dir, t3.t2_dir, t3.t3_dir, 'plots/catalog_inROI_and_completeness.png')          
        pu.plot_catalog(tier1_dir=t3.t1_dir,
                        tier2_dir=t3.t2_dir,
                        catalog_path=catalog_path,
                        a_edges=t3.a_edges,
                        m_edges=t3.m_or_q_edges,
                        m_unit=t3.t1_mass_unit, fig_title='',
                        fig_savepath=plot_save_path)
        
        #if t3.plot.summary_stats:
        main.summary_stats(tier1_dir=t3.t1_dir,
                           tier2_dir=t3.t2_dir, 
                           tier3_dir=t3.t3_dir,
                           nstars=nstars,
                           verbose=False)
        #import pdb; pdb.set_trace()
        main.make_results_plots(tier1_dir=t3.t1_dir,
                                tier2_dir=t3.t2_dir, 
                                tier3_dir=t3.t3_dir, 
                                nstars=nstars,
                                run_models=t3.run_models,
                                plot_models=t3.plot_models,
                                plot_hist_corner=True,
                                plot_model_corners=True,
                                stack_dim=t3.stack_dim, m_unit=t3.t1_mass_unit,
                                hist_title=f"{nstars} Stars ({t3.occurrence_hist_title})")
       

    return
    


def run_multiple(tier1_list, tier2_list, tier3_list,
                 a_edges, m_edges,
                 recoveries_dir, recoveries_m_unit,
                 star_df,
                 tier2_df_cuts_dict,
                 run_mcmc, run_bic_compare,
                 run_models_list,
                 plot_models_list,
                 stack_dim,
                 m_unit,
                 plot_only,
                 do_single_cells):
    """
    Create/organize the parameters to
    run multiple experiments
    """
    
    def get_params(t1, t2, t3):
        """Organize subdicts into one dict"""
        ## Add a 'dir' key to each dict
        t1_config = tier1_configs[t1]
        t1_config['dir'] = t1
    
        t2_config = tier2_configs[(t1, t2)]
        t2_config['dir'] = t2
    
        t3_config = tier3_configs[(t1, t2, t3)]
        t3_config['dir'] = t3
    
        plot_dict = {'plot_catalogs':True,
                     'summary_stats':True,
                     'plot_hist_corner':True,
                     'plot_model_corners':True}
    
        return {'tier1':t1_config, 
                'tier2':t2_config, 
                'tier3':t3_config,
                'plot':plot_dict}

    
    tier1_configs={}
    tier2_configs={}
    tier3_configs={}
    
    for t1_val in tier1_list:
    
        mass_unit = m_unit if t1_val in ['mtrue', 'msini'] else None
        true_or_sini = 'true' if 'true' in t1_val else 'sini'
        if 'm' in t1_val:
            m_or_q = 'm'
        elif 'q' in t1_val:
            m_or_q = 'q'
            m_edges = np.array(m_edges)*Mj2Ms
        
        t1_subdict = {'mass_unit':mass_unit,
                      'm_or_q':m_or_q,
                      'true_or_sini':true_or_sini,
                      't1_dir':t1_val}
        
        tier1_configs[t1_val] = deepcopy(t1_subdict)
    
        for t2_val in tier2_list:
            t2_subdict, hist_title = deepcopy(tier2_df_cuts_dict)[t2_val]
            t2_subdict['t1_dir'] = t1_val
            t2_subdict['t1_mass_unit'] = mass_unit
            t2_subdict['t1_m_or_q'] = m_or_q
            t2_subdict['t1_true_or_sini'] = true_or_sini
            t2_subdict['t2_dir'] = t2_val

            tier2_configs[(t1_val, t2_val)] = deepcopy(t2_subdict)
            
            for t3_val in tier3_list:
            
                t3_subdict = {'run_mcmc':run_mcmc,
                              'run_models':run_models_list,
                              'run_bic_compare':run_bic_compare,
                              'a_edges' :np.array(a_edges),
                              'm_or_q_edges' :np.array(m_edges),
                              'occurrence_hist_title' :hist_title,
                              'plot_models':plot_models_list,
                              'stack_dim':stack_dim,
                              'plot_only':plot_only,
                              't1_mass_unit':mass_unit,
                              't2_star_df_query':t2_subdict['star_df_query'],
                              't1_dir':t1_val,
                              't2_dir':t2_val,
                              't3_dir':t3_val}
                #import pdb; pdb.set_trace()
                tier3_configs[(t1_val, t2_val, t3_val)] = deepcopy(t3_subdict)
                
                if do_single_cells:
                    t3_subdict_single = {'run_mcmc':run_mcmc,
                                         'run_models':['hist'],
                                         'run_bic_compare':False,
                                         'a_edges' :np.array([a_edges[0], a_edges[-1]]),
                                         'm_or_q_edges' :np.array([m_edges[0], m_edges[-1]]),
                                         'occurrence_hist_title' :hist_title,
                                         'plot_models':[],
                                         'stack_dim':stack_dim,
                                         'plot_only':plot_only,
                                         't1_mass_unit':mass_unit,
                                         't2_star_df_query':t2_subdict['star_df_query'],
                                         't1_dir':t1_val,
                                         't2_dir':t2_val,
                                         't3_dir':t3_val+'_single_cell'}
                    tier3_configs[(t1_val, t2_val, t3_val+'_single_cell')] = t3_subdict_single
          
  
    ## Make star_df_full to create directories ##
    ##########################################
                       
    # DF w/ at least 3 cols --- star_name, Mstar, and comp_list --- plus any more needed for cuts (Mstar, FeH, etc.)
    #import pdb; pdb.set_trace()
    #star_df_full = make_star_df()
    star_df_full = star_df

    
    ## Now create all tier1 directories in parallel, bc they don't depend on each other
    if len(tier1_configs) > 1:
        run_parallel(make_tier1, [tier1_configs, star_df_full, recoveries_dir, recoveries_m_unit], "tier1")
    else:
        print("RUNNING single tier1")
        make_tier1(next(iter(tier1_configs.values())), star_df_full, recoveries_dir, recoveries_m_unit)
    
   

    ## Now create all tier2 directories in parallel
    #import pdb; pdb.set_trace()
    if len(tier2_configs) > 1:
        run_parallel(make_tier2, [tier2_configs, star_df_full], "tier2")
    else:
        print("RUNNING single tier2")
        make_tier2(next(iter(tier2_configs.values())), star_df_full)


    ## Now create all tier3 directories in parallel
    if len(tier3_configs) > 1:
        run_parallel(make_tier3, [tier3_configs, star_df_full], "tier3")
    else:
        print("RUNNING single tier3")
        make_tier3(next(iter(tier3_configs.values())), star_df_full)         
                
    
    return





def run_parallel_(func, configs, star_df, label):
    """
    Run func(config, star_df) for each config in parallel.

    Raises the first exception encountered after printing the
    associated configuration and traceback.
    """

    print(f"RUNNING parallel {label}")

    with ProcessPoolExecutor() as executor:

        futures = {
            executor.submit(func, cfg, star_df): name
            for name, cfg in configs.items()
        }

        for future in as_completed(futures):
            name = futures[future]
            try:
                future.result()
                print(f"Finished {name}")
            except Exception:
                print(f"\n run.py: ERROR while processing '{name}'")
                print("Configuration:")
                print(configs[name])
                traceback.print_exc()
                raise
    return        


def run_parallel(func, arg_list, label):
    """
    Run func(*args) for each item in arg_list in parallel.

    The first argument in each args tuple is assumed to be a
    configuration object containing the job name.

    Raises the first exception encountered after printing the
    associated configuration and traceback.
    """

    print(f"RUNNING parallel {label}")

    with ProcessPoolExecutor() as executor:

        futures = {
            executor.submit(func, *args): args
            for args in arg_list
        }

        for future in as_completed(futures):
            args = futures[future]
            config = args[0]
            name = config.items()[0]

            try:
                future.result()
                print(f"Finished {name}")

            except Exception:
                print(f"\n run.py: ERROR while processing '{name}'")
                print("Configuration:")
                print(config)
                traceback.print_exc()
                raise

    return








