## Utility functions related to sampling posteriors distributions 
## and adding info like priors and completeness values
import numpy as np
import pandas as pd
import os
import pickle
import h5py

from astropy import constants as c
Mj2Me = (c.M_jup/c.M_earth).value
Ms2Me = (c.M_sun/c.M_earth).value
Ms2Mj = (c.M_sun/c.M_jup).value

def post_sampler1(companion_post_dir, star_df, num_samples=1000, m_unit='earth'):
    """
    Customizable function to sample from a set of companion posteriors.
    """
    post_sample_dict = {}

    for i in range(len(star_df)):
        comp_name_list = star_df.iloc[i]['comp_list']
        
        for comp_name in comp_name_list:
            post_file = os.path.join(companion_post_dir, f"{comp_name}_post.csv")
            post = pd.read_csv(post_file)

            sampled_a = np.array(post.sma_au.sample(num_samples, replace=True))
            sampled_m = np.array(post.mass_mearth.sample(num_samples, replace=True))
            
            if m_unit=='jupiter':
                sampled_m = sampled_m/Mj2Me # Convert M_earth to M_jupiter

            post_sample_dict[comp_name] = np.array([sampled_a, sampled_m])
        
    
    return post_sample_dict
    
    
def post_sampler2(companion_post_dir, star_df, num_samples=1000, m_unit='earth'):
    """
    Function to sample from orvara posteriors in particular
    Note that chains for individual companions are saved under
    the host system, so we need the system names.
    """

    # If using q instead of m, use m_conversion=1 (ie, leave m_samples in M_sun)
    m_conversion = Ms2Me if m_unit=='earth' else Ms2Mj if m_unit=='jupiter' else 1
    
    post_sample_dict = {}
    
    # CLS systems (in the pool of 128 that also have a planet/BD) with a trend
    # Do not sample from these trend posteriors because I don't consider them very reliable
    CLSI_trend_systems = ['HD145934', 'HD1326',
	                      'HD183263', 'HD34445',
	                      'HD195019', 'HD24040', 
	                      'HD45184',  'HIP57050']
    
    sysname_list = star_df.star_name.to_list()
    cls_rename_fn = lambda name: 'HD'+name.upper() if name[0].isdigit() else name.upper()

    for sys_name_lowercase in sysname_list:
        sys_name = cls_rename_fn(sys_name_lowercase)
        chain_file = os.path.join(companion_post_dir, sys_name+'.h5')
        if not os.path.exists(chain_file):
            continue
        # import pdb; pdb.set_trace()
        with h5py.File(chain_file, 'r') as f:
            #burned_and_checked.append(sys_name)
            cols = f["chains"].attrs["param_names"] # Use f['chains'].attrs.keys() to see that param_names is a key
            last_chars = [s[-1] for s in cols] # Last char of every col name
            max_comp_ind = max([int(lc) for lc in last_chars if lc.isdigit()]) # Determine last comp


            # For trend systems, remove the trend comp so it's not sampled
            if any([tr_name in sys_name for tr_name in CLSI_trend_systems]):
                max_comp_ind -= 1


            # Load chains
            chain = f["chains"][:]  # Use f.keys() to see that "chains" is a key of f
            nsteps, nwalkers, ndims = np.shape(chain)
            new_nsteps = nsteps*nwalkers
            chain_flat = np.reshape(chain, newshape=(new_nsteps, ndims)) # New shape

            chain_dict = {cols[i]:chain_flat[:,i] for i in range(ndims)} # Assoc. each col name to its chain

            for comp_ind in range(max_comp_ind+1):
        
                rand_inds = np.random.randint(0, new_nsteps, size=num_samples) # Inds to take random draws
                a_chain = chain_dict[f'sau{comp_ind}'][rand_inds]
                m_chain = chain_dict[f'msec{comp_ind}'][rand_inds]*m_conversion # Convert M_sun to Me or Mj

                comp_name = sys_name_lowercase+'_'+str(comp_ind)
                post_sample_dict[comp_name] = [a_chain, m_chain]
    #unchecked = [sysname for sysname in burned_files if sysname not in burned_and_checked]
    #import pdb; pdb.set_trace()
    return post_sample_dict
    
    
    
def interim_prior(post_sample_dict, prior_type='loguniform'):
    """
    Given a set of posterior samples,
    calculate the interim prior associated with each
    and update the input dictionary.
    
    Assumes the input dictionary has columns of
    'comp_name', 'a_list', and 'm_list'
    """

    if prior_type=='loguniform':
        
        for comp_name in post_sample_dict.keys():
            a_m_samples = post_sample_dict[comp_name]
            prior_array = 1/a_m_samples[0] * 1/a_m_samples[1]
            
            post_sample_dict[comp_name] = np.vstack([a_m_samples, prior_array])

    return post_sample_dict


def include_post_completeness(sampled_post_dict, star_df,
                              tier1_dir, tier2_dir):
    """
    Given a dictionary with companion posterior
    samples, calculate the completeness at each
    sample. 
    Calculate two completeness values:
    the value for the system the sample is from
    and the value for the average over all
    stars in star_comp_dict
    """

    ## Load average interpolation function
    avg_compl_interp_str = os.path.join(tier1_dir, tier2_dir, 'avg_map/interp_fn.pkl')
    avg_compl_interp = pickle.load(open(avg_compl_interp_str, 'rb'))
    saved_maps_dir = os.path.join(tier1_dir, f"saved_maps_{tier1_dir}")
    # import pdb; pdb.set_trace()
    for star_name in star_df.star_name:
        
        ## Load single-system interpolation function
        single_compl_interp_str = os.path.join(saved_maps_dir, star_name, 'interp_fn.pkl')
        single_compl_interp = pickle.load(open(single_compl_interp_str, 'rb'))
        
        comp_list = star_df.query(f"star_name=='{star_name}'").comp_list.iloc[0] # List of comp names
        ## For every companion in the system, calculate the average compl over all stars AND the single-system compl
        #import pdb; pdb.set_trace()
        for comp_name in comp_list:
            
            a_m_prior = sampled_post_dict[comp_name] # Already-saved values, which we will append to
            
            ## Compute average and single-system completeness
            avg_compls = avg_compl_interp((a_m_prior[0], a_m_prior[1])) # interp_fn((a_list, m_list))
            single_star_compls = single_compl_interp((a_m_prior[0], a_m_prior[1]))
            
            ## The likelihood later requires completeness/single_system_prior. So compute that now.
            compl_over_prior_avg = avg_compls/a_m_prior[2]
            compl_over_prior_single = single_star_compls/a_m_prior[2]
            
            nan_mask = (~np.isnan(compl_over_prior_avg)) & (~np.isnan(compl_over_prior_single))
            
            # Updated array includes a_samples, m_samples, average completenesses, single star completenesses, compl_over_prior_avg, compl_over_prior_single.
            # Probably the only compl array I'll use is compl_over_prior_single. compl_over_prior_avg is to test whether using avg completeness changes the answer. The two completeness arrays are for testing/sanity checks.
            new_array = [a_m_prior[:2], avg_compls, single_star_compls,
                         compl_over_prior_avg, compl_over_prior_single]
            masked_array = np.vstack(new_array)[:,nan_mask]
            sampled_post_dict[comp_name] = masked_array

            
            #import pdb; pdb.set_trace()
    
    return sampled_post_dict
    
    
    
    
