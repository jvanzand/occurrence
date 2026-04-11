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

from occurrence.completeness_utils import _process_single_star

def prep_recoveries_files(convert_recs_msini_mtrue=False,
                          convert_recs_m_q=False,
                          msini_rec_dir_to_make_mtrue=None,
                          m_dir_to_make_q=None,
                          star_df=None,
                          m_unit='earth',
                          parent_dir='results/'):
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
                          
    if convert_recs_msini_mtrue:
        for starname in star_df.star_name:
            recoveries_file = os.path.join(msini_rec_dir_to_make_mtrue, starname+'_recoveries.csv')
            mtrue_recoveries_save_file = os.path.join(parent_dir, 'mtrue_recoveries/', starname+'_recoveries.csv')
            
            cu.recs_msini_converter(recoveries_file, mtrue_recoveries_save_file)
    
    if convert_recs_m_q:
        for i in range(len(star_df)):
            row = star_df.iloc[i]
            starname = row.star_name
            mstar = row.mstar
            recoveries_file = os.path.join(m_dir_to_make_q, starname+'_recoveries.csv')
            dirname = 'qtrue_recoveries' if 'inj_mtrue' in pd.read_csv(recoveries_file) else 'qsini_recoveries'
            q_recoveries_save_file = os.path.join(parent_dir, dirname, starname+'_recoveries.csv')
            
            cu.recs_mass_ratio_converter(recoveries_file, q_recoveries_save_file, mstar, m_unit=m_unit)
            
    return

def prep_maps(make_maps=False,
              make_interps=False,
              path_to_recoveries=None,
              maps_ycol=None,
              m_unit='earth',
              star_df=None,
              parent_dir='results/'):
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
    
                  
    maps_save_path = 'saved_maps_'+maps_ycol.split('_')[1] # e.g. inj_mtrue --> saved_maps_mtrue
    maps_save_path = os.path.join(parent_dir, maps_save_path)
    
    if make_maps:

        ncores = mp.cpu_count()
        args_list = [
            (row, path_to_recoveries, maps_save_path, maps_ycol, m_unit)
            for _, row in star_df.iterrows()
        ]

        with mp.Pool(ncores) as pool:
            pool.map(_process_single_star, args_list)
    
    if make_interps:
        ## Makes and saves both idividual and average interp functions for sensitivity maps
        avg_map_dir = os.path.join(maps_save_path, 'avg_map')
        cu.map_interpolator(maps_save_path, avg_map_dir, star_df.star_name.to_list(), ycol=maps_ycol, m_unit=m_unit)
    
    return
    
    
def prep_post_draws(sample_posts=False,
                    star_df=None, comp_post_dir=None,
                    saved_maps_dir=None, m_unit='earth',
                    a_edges=None, m_edges=None,
                    parent_dir='results/'):

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
    saved_dicts_dir = os.path.join(parent_dir, 'saved_dicts/')
    os.makedirs(saved_dicts_dir, exist_ok=True)
    
    if sample_posts:
        
        #### CUSTOMIZE your own sampler to match the posterior format ####
        ## The output of custom sampler should be a dict whose keys are companion names
        ## and whose values are 2xN arrays, where the first/second sub-array is SMA/mass samples
        post_sample_dict = su.post_sampler2(comp_post_dir, star_df, num_samples=1000, m_unit=m_unit) # First sample posteriors

        ## If using mass ratio, convert masses to q
        if "qtrue" in saved_maps_dir or "qsini" in saved_maps_dir:
            Ms2Mj = (c.M_sun/c.M_jup).value # 1 Msun --> 1047 Mjup
            Ms2Me = (c.M_sun/c.M_earth).value # 1 Msun --> ~300k Mearth
            m_conversion = Ms2Mj if m_unit=='jupiter' else Ms2Me if m_unit=='earth' else None
            for comp_name in post_sample_dict.keys():
                row = star_df[star_df['comp_list'].apply(lambda name_list: comp_name in name_list)]
                mstar = row.mstar.values*m_conversion
                
                old_samples = post_sample_dict[comp_name]
                #import pdb; pdb.set_trace()
                new_samples = np.array([old_samples[0], old_samples[1]/mstar])
                post_sample_dict[comp_name] = new_samples
                
        #import pdb; pdb.set_trace()
        post_prior_sample_dict = su.interim_prior(post_sample_dict, prior_type='loguniform') # Then calculate prior at each draw. Each value is a 3xN array of SMA samples, mass samples, and prior values
        #import pdb; pdb.set_trace()
        # Now add on completeness values
        sampled_post_with_compls = su.include_post_completeness(post_prior_sample_dict,
                                                                star_df,
                                                                saved_maps_dir)
                                                                
        ## Saves dict with companion names as key names
        ## Each value is a 6xN array of:
        ## [a_list, m_list, avg_compls, single_star_compls,
        ##  compl_over_prior_avg, compl_over_prior_single]
        ## Probably the only compl array I'll use is compl_over_prior_single. compl_over_prior_avg is to test whether using avg completeness changes the answer. The two completeness arrays are for testing/sanity checks.
        np.savez(saved_dicts_dir+'sampled_post_prior_compl.npz', **sampled_post_with_compls)
        
        pu.plot_catalog(saved_maps_dir, saved_dicts_dir+'sampled_post_prior_compl.npz', 
                        a_edges, m_edges, star_df, m_unit=m_unit,
                        fig_savename=parent_dir+'plots/catalog_and_completeness.png')
        
    return
    
    
def prep_occurrence_materials(a_edges, m_edges, star_df, 
                              saved_maps_dir,
                              compl_type='single',
                              m_unit='earth',
                              parent_dir='results/'):
    
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
		a_edges/m_edges (np arrays): Arrays defining bin edges, like 
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
    saved_dicts_dir = os.path.join(parent_dir, 'saved_dicts/')
    os.makedirs(saved_dicts_dir, exist_ok=True)
    
    ## Define paths to companion samples and average map
    comp_samples_path=os.path.join(saved_dicts_dir, 'sampled_post_prior_compl.npz')
    avg_map_fn_path=os.path.join(saved_maps_dir, 'avg_map/interp_fn.pkl')
                      
    # Make cell_dict, which contains useful cell info
    cell_dict = ou.cell_values(a_edges, m_edges, avg_map_fn_path)
    comp_samples = dict(np.load(comp_samples_path)) # Load companion samples
    
    ## In this loop, calculate the lambda index of every a/m sample for every companion
    for comp_name in comp_samples.keys():
        a_m_prior_compl = comp_samples[comp_name]
        a_list, m_list = a_m_prior_compl[:2] # First 2 sub-arrays are a/m lists
        
        lam_inds = ou.assign_cells(a_list, m_list, cell_dict['a_m_lims_pairs']) # Calculate lambda inds
        
        #if '8765' in comp_name:
        #    import pdb; pdb.set_trace()
        a_m_prior_compl_lam = np.vstack([a_m_prior_compl, lam_inds]) # Append lambda inds to array
        # import pdb; pdb.set_trace()
        
        comp_samples[comp_name] = a_m_prior_compl_lam # Put updated array back into dictionary
    #import pdb; pdb.set_trace()
    
    ## Drop companions that do not fall at least partially in the ROI
    print("Total companions: ", len(comp_samples))
    for comp_name in list(comp_samples.keys()):
        samples_in_ROI = len(np.where(comp_samples[comp_name][-1]>-0.5)[0])
        if samples_in_ROI==0:
            print(f'{comp_name} falls fully outside the ROI; dropping.')
            del comp_samples[comp_name]
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
        
        sysname = star_df[star_df["comp_list"].apply(lambda x: comp_name in x)].star_name.iloc[0]
        
        ## Check for companions that have lots of NaN completeness values
        ## Should not be a problem for real companions
        lam_nancount = np.isnan(compl_over_prior).sum()
        if lam_nancount>10:
            #import pdb; pdb.set_trace()
            print(f'main.prep_occurrence_materials: \n'
                  f'{comp_name} in system {sysname} has {lam_nancount}/{len(compl_over_prior)} sample NaNs')
            #import pdb; pdb.set_trace()
        
        # Frac. of inds that fall in at least 1 cell. This will be used
        # in MCMC prep. function to exclude comps outside ROI for efficiency
        total_weight = (lam_inds>-1).sum()/len(lam_inds)
        bin_lam_dict[f"{comp_name}_weight_total"] = total_weight
        
        #if '8765' in comp_name:
        #    import pdb; pdb.set_trace()
        
        for bin_ind in range(cell_dict['num_cells']):
            
            lam_mask = lam_inds==bin_ind # All locations where the lambda index equals the current bin
            compl_over_prior_in_cell_avg = np.nanmean(compl_over_prior[lam_mask]) # Pre-compute avg. for likelihood. Use nanmean() for now to catch samples in NaN space. Should not be a problem for real companions
            if np.isnan(compl_over_prior_in_cell_avg): # Last catch for really bad systems. Delete later.
                compl_over_prior_in_cell_avg=0
            
            weight = lam_mask.sum()/len(lam_inds) # (Num. of samples in cell)/(tot. # of samples)
            
            compl_over_prior_in_cell_avg_and_weight = [compl_over_prior_in_cell_avg, weight]
            bin_lam_dict[f"{comp_name}_cell{bin_ind}_compl_over_prior_avg_and_weight"] = compl_over_prior_in_cell_avg_and_weight
    
    np.savez(saved_dicts_dir+'cell_dict.npz', **cell_dict) ## Cell-specific info
    np.savez(saved_dicts_dir+'sampled_post_prior_compl_lam_inROI.npz', **comp_samples) # Sample-specific info
    np.savez(saved_dicts_dir+'bin_lam_dict.npz', **bin_lam_dict) # Pre-computed info for likelihood func.
    #import pdb; pdb.set_trace()
    
    
    pu.plot_catalog(saved_maps_dir, saved_dicts_dir+'sampled_post_prior_compl_lam_inROI.npz', 
                    a_edges, m_edges, star_df, m_unit=m_unit,
                    fig_savename=parent_dir+'plots/catalog_inROI_and_completeness.png')

    return
    
    
    
def run_mcmc(nstars, parallel=False,
             nwalkers=50, nsteps=5000, burnin=1000,
             parent_dir='results/'):
    """
    Entry point for MCMC occurrence calculation.
    Collects pre-computed materials to feed to MCMC.
    
    Note: this function doesn't *technically* need to
    exist (I could just run the code right in ou.run_mcmc),
    but it's nice to have all my primary functions in 
    the main.py module.
    """
    saved_chains_dir = os.path.join(parent_dir, 'saved_chains/')
    os.makedirs(saved_chains_dir, exist_ok=True)
    
    samples_inROI_path = os.path.join(parent_dir, 'saved_dicts/sampled_post_prior_compl_lam_inROI.npz')
    samples_inROI = dict(np.load(samples_inROI_path))
    comp_names_inROI = list(samples_inROI.keys())
    
    cell_dict_path = os.path.join(parent_dir, 'saved_dicts/cell_dict.npz')
    bin_lam_dict_path = os.path.join(parent_dir, 'saved_dicts/bin_lam_dict.npz')
    
    cell_dict = dict(np.load(cell_dict_path)) # Includes bin sizes and avg_cell_compls
    bin_lam_dict = dict(np.load(bin_lam_dict_path)) # Contains, for every cell and for every companion, all (compl/prior) values that fall in that cell, AND the fraction (aka weight). This is equivalent to the info. stored in sampled_post_prior_compl_lam.npz, but compressed and sorted by lambda index.
    
    ou.mcmc(nstars, comp_names_inROI, cell_dict, bin_lam_dict,
            save_path=saved_chains_dir+'chains.npz', parallel=parallel,
            nwalkers=nwalkers, nsteps=nsteps, burnin=burnin)
    
    return
    
def summary_stats(parent_dir='results/'):
    """
    Load MCMC chains and make diagnostic
    plots of the results
    """
    
    cell_dict_path = os.path.join(parent_dir, 'saved_dicts/cell_dict.npz')
    bin_lam_dict_path = os.path.join(parent_dir, 'saved_dicts/bin_lam_dict.npz')
    path_to_chains = os.path.join(parent_dir, 'saved_chains/chains.npz')
    save_summary_dict_path = os.path.join(parent_dir, 'saved_dicts/summary_dict.npz')
    
    
    cell_dict = dict(np.load(cell_dict_path)) # Includes bin sizes and avg_cell_compls
    bin_lam_dict = dict(np.load(bin_lam_dict_path)) # Contains, for every cell and for every companion, all (compl/prior) values that fall in that cell, AND the fraction (aka weight). This is equivalent to the info. stored in sampled_post_prior_compl_lam.npz, but compressed and sorted by lambda index.
    summary_dict = ou.summary_stats(path_to_chains, cell_dict, bin_lam_dict)
    np.savez(save_summary_dict_path, **summary_dict) ## Cell-specific info
    
    return
    
def make_results_plots(completeness_dir, stack_dim='m', 
                       m_unit='earth', parent_dir='results/'):
    """
    Load MCMC chains and make diagnostic
    plots of the results
    """
    plots_dir = os.path.join(parent_dir, 'plots/')
    
    
    path_to_cell_dict = os.path.join(parent_dir, 'saved_dicts/cell_dict.npz')
    path_to_chains = os.path.join(parent_dir, 'saved_chains/chains.npz') 
    path_to_summary = os.path.join(parent_dir, 'saved_dicts/summary_dict.npz') 
    
    ## Start with corner plot to gauge MCMC results ##
    cell_dict = dict(np.load(path_to_cell_dict)) # Includes bin sizes and avg_cell_compls
    pu.plot_corner_from_file(path_to_chains, cell_dict, outpath=plots_dir+"corner.png",
                             param_names=None, thin=10, max_samples=50000)
    
    
    ## Now plot completeness map + catalog + derived occurrence rates + eff pl. + avg. compl.
    xgrid = np.load(f"{completeness_dir}/avg_map/parent_xgrid.npy")
    ygrid = np.load(f"{completeness_dir}/avg_map/parent_ygrid.npy")
    zgrid = np.load(f"{completeness_dir}/avg_map/parent_zgrid.npy")
    
    summary_dict = dict(np.load(path_to_summary))

    y_type = completeness_dir.split('_')[-1] # e.g. 'mass_ratio/saved_maps_qtrue' --> qtrue
    pu.plot_occurrence_hist(summary_dict, stack_dim=stack_dim, m_unit=m_unit, ytype=y_type,
                            savepath=plots_dir+'occurrence.png', figsize=(6, 4), dpi=300)
    
    return
    
    
    
    
    
    
    
