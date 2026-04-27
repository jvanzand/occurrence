## Main module for running an end-to-end occurrence calculation
## Optionally create all preliminary products (average completeness maps,
## interpolation functions, catalog samples, prior weights), or skip if
## already done. Then do an occurrence fit.

import os
import numpy as np
import pandas as pd
import pickle
from astropy import constants as c
import multiprocessing as mp

from occurrence import completeness_utils as cu
from occurrence import sampling_utils as su
from occurrence import occurrence_utils as ou
from occurrence import plotting_utils as pu
from occurrence import mcmc_histogram as mcmc_hist
from occurrence import mcmc_powerlaw as mcmc_power

from occurrence.completeness_utils import _process_single_star

def prep_recoveries_files(tier1_dir,
                          star_df,
                          #convert_recs_msini_mtrue=False,
                          #convert_recs_m_q=False,
                          msini_rec_dir_to_make_mtrue,
                          m_dir_to_make_q,
                          recoveries_m_unit='earth'):
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

    #import pdb; pdb.set_trace()

    if 'true' in tier1_dir: # If tier1_dir is mtrue or qtrue, then convert msini to mtrue
        for starname in star_df.star_name:
            recoveries_file = os.path.join(msini_rec_dir_to_make_mtrue, starname+'_recoveries.csv')
            mtrue_recoveries_save_file = os.path.join(tier1_dir, 'mtrue_recoveries/', starname+'_recoveries.csv')
            
            cu.recs_msini_converter(recoveries_file, mtrue_recoveries_save_file)
            
    else:
        os.makedirs('msini/msini_recoveries/', exist_ok=True)
        keep_cols = ['inj_msini', 'inj_au', 'inj_e', 'recovered']
        for starname in star_df.star_name:
            recoveries_file = os.path.join(msini_rec_dir_to_make_mtrue, starname+'_recoveries.csv')
            msini_recoveries_save_file = os.path.join(tier1_dir, 'msini_recoveries/', starname+'_recoveries.csv')
            
            rec_file = pd.read_csv(recoveries_file)[keep_cols]
            rec_file.to_csv(msini_recoveries_save_file, index=False)
            #import pdb; pdb.set_trace()
    
    #if convert_recs_m_q:
    if 'q' in tier1_dir: # If tier1_dir is qsini or qtrue, then convert m to q
        for i in range(len(star_df)):
            row = star_df.iloc[i]
            starname = row.star_name
            mstar = row.Mstar
            
            #dirname = 'qtrue_recoveries' if 'inj_mtrue' in pd.read_csv(recoveries_file) else 'qsini_recoveries'
            dirname = tier1_dir+'_recoveries'
            
            q_recoveries_save_file = os.path.join(tier1_dir, dirname, starname+'_recoveries.csv')
            recoveries_file = os.path.join(m_dir_to_make_q, starname+'_recoveries.csv')
            
            cu.recs_mass_ratio_converter(recoveries_file, q_recoveries_save_file, mstar, m_unit=recoveries_m_unit)
            
    return

def prep_maps(tier1_dir,
              star_df,
              path_to_recoveries,
              m_unit='earth'):
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
    
    ncores = 30#mp.cpu_count()
    args_list = [
        (row, path_to_recoveries, maps_save_path, maps_ycol, m_unit)
        for _, row in star_df.iterrows()
        ]

    with mp.Pool(ncores) as pool:
        pool.map(_process_single_star, args_list)
    
    #if make_interps:
    ## Make and save idividual interp functions for sensitivity maps
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
                    saved_maps_dir=None, m_unit='earth',
                    fig_title='Catalog Posteriors'):

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
    """
        
    #### CUSTOMIZE your own sampler to match the posterior format ####
    ## The output of custom sampler should be a dict whose keys are companion names
    ## and whose values are 2xN arrays, where the first/second sub-array is SMA/mass samples
    post_sample_dict = su.post_sampler2(comp_post_dir, star_df, num_samples=500, m_unit=m_unit) # First sample posteriors 
    
    ## If using mass ratio, convert masses to q
    #if "qtrue" in saved_maps_dir or "qsini" in saved_maps_dir:
    if 'q' in tier1_dir:

        for comp_name in post_sample_dict.keys():
            row = star_df[star_df['comp_list'].apply(lambda name_list: comp_name in name_list)]
            mstar = row.Mstar.values
                
            old_samples = post_sample_dict[comp_name]
            #import pdb; pdb.set_trace()
            new_samples = np.array([old_samples[0], old_samples[1]/mstar]) # Same a samples; m-->q
            post_sample_dict[comp_name] = new_samples
                
    #import pdb; pdb.set_trace()
    post_prior_sample_dict = su.interim_prior(post_sample_dict, prior_type='loguniform') # Then calculate prior at each draw. Each value is a 3xN array of SMA samples, mass samples, and prior values
    #import pdb; pdb.set_trace()
    # Now add on completeness values
    sampled_post_with_compls = su.include_post_completeness(post_prior_sample_dict,
                                                            star_df,
                                                            tier1_dir, tier2_dir)
                                                                
    ## Saves dict with companion names as key names
    ## Each value is a 6xN array of:
    ## [a_list, m_list, avg_compls, single_star_compls,
    ##  compl_over_prior_avg, compl_over_prior_single]
    ## Probably the only compl array I'll use is compl_over_prior_single. compl_over_prior_avg is to test whether using avg completeness changes the answer. The two completeness arrays are for testing/sanity checks.
    saved_dict_dir = os.path.join(tier1_dir, tier2_dir, 'sampled_post_prior_compl.npz')
    np.savez(saved_dict_dir, **sampled_post_with_compls)
        
        
    return
    
    
def prep_occurrence_materials(tier1_dir, tier2_dir, tier3_dir,
                              a_edges, m_or_q_edges, star_df, 
                              compl_type='single',
                              m_unit='earth',
                              fig_title='Companions in Occurrence Region'):
    
    """
    Prepare inputs to occurrence calculation framework. Specifically:
    - Make cell_dict, which contains useful grid-specific info, like
      individual cell bounds, avg completeness in each cell, and all
      cell sizes
    - Add lambda indices to companion sample catalog. That is, assign
      to every a/m pair the index of the occurrence cell it falls in
    - Make bin_lam_dict. This holds two useful ingredients for the
      likelihood: all (compl/prior) values that fall in each bin for
      each companion, AND the fraction (aka weight) for each bin/comp
      pair
    
    Arguments:
	a_edges/m_or_q_edges (np arrays): Arrays defining bin edges, like 
		                     array([0.8, 1.6, 3.2])
        star_df (dataframe): Pandas df with columns 'star_name', 'mstar',
              and 'comp_list'. comp_list is a list of the names of all 
              companions orbiting that star. mstar is in solar masses
        comp_sample_path (str): Path to a file containing a dictionary with
                                companion samples. Each key is a companion
                                name and each value is a list of 6 elements:
                                [a_list, m_list, avg_compls, single_star_compls,
                                 compl_over_prior_avg, compl_over_prior_single]
        avg_map_fn_path (str): Path to interpolation function representing the
                               average completeness map over all systems in question
        compl_type (str): Either 'avg' or 'single' to determine which completeness
                          map draws to use. 'single' is generally the most correct;
                          it associates each draw with the completeness of its
                          original map. 'avg' is more for testing/comparison
    """
    
    ## Define paths to companion samples and average map
    comp_samples_path=os.path.join(tier1_dir, tier2_dir, 'sampled_post_prior_compl.npz')
    avg_map_fn_path=os.path.join(tier1_dir, tier2_dir, 'avg_map/interp_fn.pkl')
                      
    # Make cell_dict, which contains useful cell info
    cell_dict = ou.cell_values(a_edges, m_or_q_edges, avg_map_fn_path)
    comp_samples = dict(np.load(comp_samples_path)) # Load companion samples
    total_sample_num = np.shape((comp_samples[list(comp_samples.keys())[0]]))[1] # Record sample num before arrays change len

    
    comp_ROIweights = {}
    comps_outsideROI = []
    ## In this loop, calculate the lambda index of every a/m sample for every companion
    for comp_name in comp_samples.keys():
        a_m_prior_compl = comp_samples[comp_name]
        a_list, m_list = a_m_prior_compl[:2] # First 2 sub-arrays are a/m lists
        
        lam_inds = ou.assign_cells(a_list, m_list, cell_dict['a_m_lims_pairs']) # Calculate lambda inds
        
        ROIweight = len(lam_inds[lam_inds>-0.5])/len(lam_inds) # Fraction of samples in full ROI
        comp_ROIweights[comp_name] = ROIweight # Put weights in separate dict
        
        
        if ROIweight==0: # Skip companions that fall outside the ROI. Their weights=0 anyway, but saves compute to remove
            print(f'{comp_name} falls fully outside the ROI; dropping.')
            comps_outsideROI.append(comp_name)
            
        ## For companions with at least some samples in the ROI, prune the ones outside it
        lamROI_mask = lam_inds>-0.5

        a_m_prior_compl_lam = np.vstack([a_m_prior_compl, lam_inds]) # Append lambda inds to array
        a_m_prior_compl_lam = a_m_prior_compl_lam[:, lamROI_mask] # Remove any vals outside ROI
        comp_samples[comp_name] = a_m_prior_compl_lam # New dict entry is the updated array
            
    for comp_name in comps_outsideROI:
        del comp_samples[comp_name]
    
    # import pdb; pdb.set_trace()
    print("Companions at least partially in ROI: ", len(comp_samples))
    ##################################################################
    
    # 4th element of catalog_dict is average completeness/prior (for testing)
    # 5th element of catalog_dict is single system completeness/prior (most correct)
    compl_ind = 4 if compl_type=='avg' else 5 if compl_type=='single' else 5 # Default to 5 anyway
    bin_lam_dict = {}
    for comp_name in comp_samples.keys():
        a_samples = comp_samples[comp_name][0]
        m_samples = comp_samples[comp_name][1]
        compl_over_prior = comp_samples[comp_name][compl_ind]
        lam_inds = comp_samples[comp_name][6].astype(int) # Ensure lambda inds are ints, not floats
        
        try:
            sysname = star_df[star_df["comp_list"].apply(lambda x: comp_name in x)].star_name.iloc[0]
        except:
            import pdb; pdb.set_trace()
        
        ## Check for companions that have lots of NaN completeness values
        ## Should not be a problem for real companions
        lam_nancount = np.isnan(compl_over_prior).sum()
        if lam_nancount>10:
            #import pdb; pdb.set_trace()
            if lam_nancount>999:
                print(f'main.prep_occurrence_materials: \n'
                      f'{comp_name} in system {sysname} has {lam_nancount}/{len(compl_over_prior)} sample NaNs. \n'
                      f'    Likely it is mostly outside the ROI, with the few in-ROI samples in NaN regions of the \n'
                      f'    host star map.')
            else:
                print(f'main.prep_occurrence_materials: \n'
                      f'{comp_name} in system {sysname} has {lam_nancount}/{len(compl_over_prior)} sample NaNs')
        
        
        for bin_ind in range(cell_dict['num_cells']):
            
            lam_mask = lam_inds==bin_ind # All locations where the lambda index equals the current bin
            compl_over_prior_in_cell_avg = np.nanmean(compl_over_prior[lam_mask]) # Pre-compute avg. for likelihood. Use nanmean() for now to catch samples in NaN space. Should not be a problem for real companions
            if np.isnan(compl_over_prior_in_cell_avg): # Last catch for really bad systems. Delete later.
                compl_over_prior_in_cell_avg=0
            
            weight = lam_mask.sum()/total_sample_num # (Num. of samples in cell)/(tot. # of samples)
            
            bin_lam_dict[f"{comp_name}_cell{bin_ind}_compl_over_prior_avg_and_weight"] = [compl_over_prior_in_cell_avg, weight]
    
    
    saved_dicts_dir = os.path.join(tier1_dir, tier2_dir, tier3_dir, 'saved_dicts/')
    os.makedirs(saved_dicts_dir, exist_ok=True)

    np.savez(saved_dicts_dir+'cell_dict.npz', **cell_dict) ## Cell-specific info
    np.savez(saved_dicts_dir+'sampled_post_prior_compl_lam_inROI.npz', **comp_samples) # Sample-specific info
    np.savez(saved_dicts_dir+'comp_ROIweights.npz', **comp_ROIweights) # Fraction of each comp that falls in ROI
    np.savez(saved_dicts_dir+'bin_lam_dict.npz', **bin_lam_dict) # Pre-computed info for hist likelihood

    return
    
    
    
def run_mcmc(tier1_dir, tier2_dir, tier3_dir,
             run_models, a_edges, m_edges, stack_dim,
             nstars, parallel=False,
             nwalkers=50, nsteps=5000, burnin=1000):
    """
    Entry point for MCMC occurrence calculation.
    Collects pre-computed materials to feed to MCMC.
    """
    saved_chains_dir = os.path.join(tier1_dir, tier2_dir, tier3_dir, 'saved_chains/')
    os.makedirs(saved_chains_dir, exist_ok=True)
    
    
    load_materials_dir = os.path.join(tier1_dir, tier2_dir, tier3_dir)
    

    samples_inROI_path = os.path.join(load_materials_dir, 'saved_dicts/sampled_post_prior_compl_lam_inROI.npz')
    samples_inROI = dict(np.load(samples_inROI_path))
    comp_names_inROI = list(samples_inROI.keys())
    
    ## Handle hist separately because it uses different pre-computed values
    if 'hist' in run_models:
            
        cell_dict_path = os.path.join(load_materials_dir, 'saved_dicts/cell_dict.npz')
        bin_lam_dict_path = os.path.join(load_materials_dir, 'saved_dicts/bin_lam_dict.npz')
    
        cell_dict = dict(np.load(cell_dict_path)) # Includes bin sizes and avg_cell_compls
        bin_lam_dict = dict(np.load(bin_lam_dict_path)) # Contains, for every cell and for every companion, all (compl/prior) values that fall in that cell, AND the fraction (aka weight). This is equivalent to the info. stored in sampled_post_prior_compl_lam.npz, but compressed and sorted by lambda index.
    
        mcmc_hist.mcmc(nstars, comp_names_inROI, cell_dict, bin_lam_dict,
                save_path=saved_chains_dir+'chains_hist.npz', parallel=parallel,
                nwalkers=nwalkers, nsteps=nsteps, burnin=burnin)
    
    ## Handle piecewise power models together because they follow the same pattern
    pp_model_names = [model_name for model_name in run_models if 'hist' not in model_name]
    for model_name in pp_model_names:
        # pp_chain_save_path = saved_chains_dir+f'chains_{model_name}.npz'
        # if os.path.exists(pp_chain_save_path):
        #     continue
        
        ROIweights_path = os.path.join(load_materials_dir, 'saved_dicts/comp_ROIweights.npz')
        ROIweights_dict = dict(np.load(ROIweights_path))
        alims = a_edges[0], a_edges[-1]
        mlims = m_edges[0], m_edges[-1]
        interp_fn_avg_path = os.path.join(tier1_dir, tier2_dir, 'avg_map/interp_fn.pkl')
        interp_fn_avg = pickle.load(open(interp_fn_avg_path, 'rb'))
        

        mcmc_power.mcmc(nstars, comp_names_inROI, model_name,
                           samples_inROI, ROIweights_dict,
                           alims, mlims, stack_dim, interp_fn_avg,
                           save_path=saved_chains_dir+f'chains_{model_name}.npz', parallel=parallel,
                           nwalkers=nwalkers, nsteps=nsteps, burnin=burnin)
        
    return
    
def summary_stats(tier1_dir, tier2_dir, tier3_dir, verbose=False):
    """
    Load MCMC chains and make diagnostic
    plots of the results
    """
    
    load_save_dir = os.path.join(tier1_dir, tier2_dir, tier3_dir)
    
    cell_dict_path = os.path.join(load_save_dir, 'saved_dicts/cell_dict.npz')
    bin_lam_dict_path = os.path.join(load_save_dir, 'saved_dicts/bin_lam_dict.npz')
    path_to_chains = os.path.join(load_save_dir, 'saved_chains/chains_hist.npz')
    save_summary_dict_path = os.path.join(load_save_dir, 'saved_dicts/summary_dict.npz')
    
    
    cell_dict = dict(np.load(cell_dict_path)) # Includes bin sizes and avg_cell_compls
    bin_lam_dict = dict(np.load(bin_lam_dict_path)) # Contains, for every cell and for every companion, all (compl/prior) values that fall in that cell, AND the fraction (aka weight). This is equivalent to the info. stored in sampled_post_prior_compl_lam.npz, but compressed and sorted by lambda index.
    summary_dict = ou.summary_stats(path_to_chains, cell_dict, bin_lam_dict, verbose=verbose)
    np.savez(save_summary_dict_path, **summary_dict) ## Cell-specific info
    
    return
    
def make_results_plots(tier1_dir, tier2_dir, tier3_dir,
                       nstars, stack_dim, plot_model,
                       m_unit='earth', 
                       hist_title=''):
    """
    Load MCMC chains and make diagnostic
    plots of the results
    """
    
    
    load_save_dir = os.path.join(tier1_dir, tier2_dir, tier3_dir)
    plot_save_dir = os.path.join(load_save_dir, 'plots/')
    

    ## Make corner and completeness plot from histogram  ##
    ########################################################
    path_to_cell_dict = os.path.join(load_save_dir, 'saved_dicts/cell_dict.npz')
    path_to_hist_chains = os.path.join(load_save_dir, 'saved_chains/chains_hist.npz') 
    path_to_summary = os.path.join(load_save_dir, 'saved_dicts/summary_dict.npz')

    ## Start with corner plot for histogram model ##
    cell_dict = dict(np.load(path_to_cell_dict)) # Includes bin sizes and avg_cell_compls
    save_corner_dir = os.path.join(plot_save_dir, 'corner_hist.png')
    pu.plot_corner_from_file(path_to_hist_chains, plot_model='hist', outpath=save_corner_dir,
                             thin=10, max_samples=50000)



    ## Now plot completeness map + catalog + derived occurrence rates + eff pl. + avg. compl.
    avg_comp_path = os.path.join(tier1_dir, tier2_dir, 'avg_map/')
    xgrid = np.load(avg_comp_path+"parent_xgrid.npy")
    ygrid = np.load(avg_comp_path+"parent_ygrid.npy")
    zgrid = np.load(avg_comp_path+"parent_zgrid.npy")

    summary_dict = dict(np.load(path_to_summary))


    ## Plot completeness map with grid cells and occurrence rate annotations
    y_col = 'inj_'+tier1_dir # e.g. inj_mtrue
    #import pdb; pdb.set_trace()
    pu.completeness_plotter(xgrid, ygrid, zgrid, plot_save_dir+"ROI_with_occurrence.png", 
                            title=f'Occurrence Summary for {nstars} Stars', save_plot=True, 
                            a_m_lims_pairs=cell_dict['a_m_lims_pairs'], summary_dict=summary_dict,
                            ycol=y_col, m_unit=m_unit)
    ############################################################
    ############################################################

    
    ## Plot occurrence histogram in OR and ORD
    pu.plot_occurrence_hist(summary_dict, stack_dim=stack_dim, m_unit=m_unit, mtype=tier1_dir,
                            rate_type='OR', title=hist_title,
                            savepath=plot_save_dir+f'occurrence_OR.png', figsize=(6, 4))
                            
    if plot_model is None:
        pu.plot_occurrence_hist(summary_dict, stack_dim=stack_dim, m_unit=m_unit, mtype=tier1_dir,
                                rate_type='ORD', title=hist_title,
                                savepath=plot_save_dir+f'occurrence_ORD.png', figsize=(6, 4))
        
    else:
        
        ## Plot corner for power law model
        # plot_dim = 'a' if stack_dim=='m' else 'm' if stack_dim=='a' else None # dim to label corner correctly
        path_to_power_chains = os.path.join(load_save_dir, f'saved_chains/chains_{plot_model}.npz')
        save_corner_dir = os.path.join(plot_save_dir, f'corner_{plot_model}.png')
        pu.plot_corner_from_file(path_to_power_chains, plot_model=plot_model, outpath=save_corner_dir,
                                 thin=10, max_samples=50000)
        
        fig, ax = pu.plot_occurrence_hist(summary_dict, stack_dim=stack_dim, m_unit=m_unit, mtype=tier1_dir,
                                          rate_type='ORD', title=hist_title, return_fig_ax=True,
                                          savepath=None, figsize=(6, 4))
        pu.plot_power(fig, ax, plot_model, save_path=plot_save_dir+f'occurrence_ORD.png')
    
    return
    
    
    
    
    
    
    
