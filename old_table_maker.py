## Module to collect and format histogram fitting results
import os
import sys
import numpy as np
import pandas as pd
import re
np.seterr(invalid='ignore') # ignore warning from optimize routine

github_path = '/data/user/judahvz/GitHub/'
occurrence_path = os.path.join(github_path, 'occurrence/')
sys.path.insert(0, github_path)
from occurrence import occurrence_utils as ou
from occurrence import mcmc_powerlaw as mcmc_power
from occurrence import analysis_utils as au


model_dict = {'flat':['FlatLine',
                     1, 
                     ['C'],
                     ],
               'logG':['log_gaussian', 
                      3,
                     ['A', '$\mu$', '$\sigma$'],
                     ],
              'escarpment':['escarpment',
                            4, 
                            ['C1', 'C2', '$\ln (bp_1)$', '$\ln (bp_2)$'],
                            ],}
                     
           

def make_variables_file(tier1_list, tier2_types, tier3_list, stack_dim, m_unit):
    """
    Programmatically generate variables.tex for paper.
    Should contain integrated occurrence rates, model parameters,
    and significance levels between compared parameter values.
    """
    
    #float_format_fn = lambda x: f"{x:.2f}" if abs(x)>=0.01 else f"{x:.1e}"
    float_format_fn = lambda x, prec: (
                          f"{x:.{prec}f}"
                          if abs(x) >= 0.01
                          else rf"${float(f'{x:.1e}'.split('e')[0]):g}\times10^{{{int(f'{x:.1e}'.split('e')[1])}}}$"
                                   )
    row_list = []
    for t1_val in tier1_list:
        for t2_type in tier2_types:
            t2_val1 = 'high'+t2_type
            t2_val2 = 'low'+t2_type
            for t3_val in tier3_list:
            
                row = {'tier1_val': t1_val,
                       'tier2_type':t2_type}
                #import pdb; pdb.set_trace()
                ######## For 'allstars' only ########
                ######################################
                if t2_type=='allstars': # No comparison needed. Just get integrated OR and params
                    #single_dir = os.path.join(t1_val, t2_type, f'{t3_val}_single_cell/')
                    #if os.path.exists(single_dir):
                    #    hist_dict_single = dict(np.load(os.path.join(single_dir, 'saved_dicts/summary_dict.npz')))
                    #    mode_OR = hist_dict_single['mode_OR'][0]
                    #    OR_lower_err = mode_OR - hist_dict_single['hdi_low_OR'][0]
                    #    OR_upper_err = hist_dict_single['hdi_high_OR'][0] - mode_OR
                    #    single_cell_OR_str = ou.latex_formatter(mode_OR, OR_lower_err, OR_upper_err)
                    #    Neff_str = '{:.2f}'.format(np.sum(hist_dict_single['cell_weights']))
                        
                    #    row[f'single_cell_OR_str'] = single_cell_OR_str
                    #    row[f'Neff'] = Neff_str

                    #else:
                    #    print(f"table_maker.py: No integrated occurrence available for {single_dir}")
                        
                    #import pdb; pdb.set_trace()
                    tier123_dir = os.path.join(t1_val, t2_type, t3_val)
                    hist_dict = dict(np.load(os.path.join(tier123_dir, 'saved_dicts/summary_dict.npz')))
                    
                    mode_OR_single = hist_dict['mode_OR_single'][0]
                    OR_lower_err_single = mode_OR_single-hist_dict['hdi_low_OR_single'][0]
                    OR_upper_err_single = hist_dict['hdi_high_OR_single'][0]-mode_OR_single
                    OR_str_single = ou.latex_formatter(mode_OR_single, OR_lower_err_single, OR_upper_err_single)
                    Neff_str = '{:.2f}'.format(np.sum(hist_dict['cell_weights']))
                                        
                    row[f'single_cell_OR_str'] = OR_str_single
                    row[f'Neff'] = Neff_str
                    
                    

                    n_abins = hist_dict[f'n_abins']
                    n_stack_bins = hist_dict[f'n_{stack_dim}bins']
                    Neff_array = hist_dict['cell_weights'].reshape(-1, n_abins) # Same shape as OR grid: (n_m x n_a)
                    
                    if stack_dim=='a':
                        stack_edges = hist_dict['a_m_lims_pairs'][:n_abins, 0, :]
                        nonstack_edges = hist_dict['a_m_lims_pairs'][::n_abins, 1, :]
                        n_nonstack_bins = hist_dict[f'n_mbins']
                        Neff_summed = Neff_array.sum(axis=0) # Sum along m
                    elif stack_dim=='m':
                        stack_edges = hist_dict['a_m_lims_pairs'][::n_abins, 1, :]
                        nonstack_edges = hist_dict['a_m_lims_pairs'][:n_abins, 0, :]
                        n_nonstack_bins = hist_dict[f'n_abins']
                        Neff_summed = Neff_array.sum(axis=1) # Sum along a
                    else:
                        raise ValueError(f'Unknown stack_dim={stack_dim}')
                
                    nonstack_bin_centers = (nonstack_edges[:,1]*nonstack_edges[:,0])**0.5
                    for stack_ind in range(n_stack_bins):
                    
                        stack_ind_str = number_to_words(stack_ind)
                    
                        #import pdb; pdb.set_trace()
                        row[f'Neff_Bin{stack_dim}{stack_ind_str}'] = '{:.1f}'.format(Neff_summed[stack_ind])
                    
                        ## Items needed for BIC calculation
                        ORD_vals = hist_dict['mode_ORD'].reshape(n_nonstack_bins,-1)[:,stack_ind]
                        ORD_errs_high = hist_dict['hdi_high_ORD'].reshape(n_nonstack_bins,-1)[:,stack_ind] - ORD_vals
                        ORD_errs_low = ORD_vals - hist_dict['hdi_low_ORD'].reshape(n_nonstack_bins,-1)[:,stack_ind]

                        bic_hist_dict = {'bin_centers':nonstack_bin_centers,
                                     'lims':(nonstack_edges[0][0], nonstack_edges[-1][-1]),
                                     'ORD_vals':ORD_vals,
                                     'ORD_errs_high':ORD_errs_high,
                                     'ORD_errs_low':ORD_errs_low,
                                     }
                     
                        _, flat_bic, _, _ = mcmc_power.calculate_bic(tier123_dir, 'flat', 
                                                    bic_hist_dict, model_dict, 
                                                    stack_ind, init_type='guess',
                                                    verbose=False)
                                                    
                                                    
                        for model in ['logG', 'escarpment']:
                            chain_path = os.path.join(tier123_dir, f'saved_chains/chains_{model}_bin{stack_ind}.npz')
                            model_chain = np.load(chain_path)
                        
                            chains = model_chain["chains"]  # (nsteps, nwalkers, ndim)
                            model_samples = chains.reshape(-1, chains.shape[-1])
                         
                            full_len = model_samples.shape[0]
                            if full_len>20000: # Don't need huge chains to get summary stats. Take the end.
                                model_samples = model_samples[full_len-10000:,:] # 'burn in' until only 10k samples left left
                     
                            rate_type='model'
                            hdi_frac=0.68
                            gs=2000
                            chain_summary_dict = ou.summarize_chains(model_samples, rate_type, hdi_frac, gs)

                         
                            for i in range(len(chain_summary_dict['mode_model'])):
                                new_key = f'{model}_param{i}_Bin{stack_dim}{stack_ind_str}'
                            
                                mode = chain_summary_dict['mode_model'][i]
                                low_err = mode-chain_summary_dict['hdi_low_model'][i]
                                high_err = chain_summary_dict['hdi_high_model'][i]-mode
                                new_val = ou.latex_formatter(mode, low_err, high_err)
                                #if t1_val=='qtrue':
                                #    import pdb; pdb.set_trace()

                                row[new_key] = new_val
                            
                            ## Calculate model Delta_BIC ##
                            _, bic, _, _ = mcmc_power.calculate_bic(tier123_dir, model, 
                                                                bic_hist_dict, model_dict, 
                                                                stack_ind, init_type='chains',
                                                                verbose=False)

                            if np.isinf(bic):
                                _, bic, _, _ = mcmc_power.calculate_bic(tier123_dir, model, 
                                                            bic_hist_dict, model_dict, 
                                                            stack_ind, init_type='guess',
                                                            verbose=False)
                            row[f'{model}_DBIC_Bin{stack_dim}{stack_ind_str}'] = float_format_fn(flat_bic-bic,1)
                                                    
                    row_list.append(row)                                    
                    continue
                ###########################
                ###########################
                ## For the rest of t2_types other than allstars:

                tier123_dir1 = os.path.join(t1_val, t2_val1, t3_val)
                tier123_dir2 = os.path.join(t1_val, t2_val2, t3_val)
                
                
                ## Collect hist information for BIC calculation
                hist_dict1 = dict(np.load(os.path.join(tier123_dir1, 'saved_dicts/summary_dict.npz')))
                hist_dict2 = dict(np.load(os.path.join(tier123_dir2, 'saved_dicts/summary_dict.npz')))
                
                ## Just use hist_dict1 for values common to hist_dict1 and hist_dict2
                n_abins = hist_dict1[f'n_abins']
                n_stack_bins = hist_dict1[f'n_{stack_dim}bins']
                #import pdb; pdb.set_trace()
                
                Neff_array1 = hist_dict1['cell_weights'].reshape(-1, n_abins) # shape: (n_m x n_a)
                Neff_array2 = hist_dict2['cell_weights'].reshape(-1, n_abins) # shape: (n_m x n_a)

                if stack_dim=='a':
                    stack_edges = hist_dict1['a_m_lims_pairs'][:n_abins, 0, :]
                    nonstack_edges = hist_dict1['a_m_lims_pairs'][::n_abins, 1, :]
                    n_nonstack_bins = hist_dict1[f'n_mbins']
                    Neff_summed1 = Neff_array1.sum(axis=0) # Sum along m
                    Neff_summed2 = Neff_array2.sum(axis=0) # Sum along m
                elif stack_dim=='m':
                    stack_edges = hist_dict1['a_m_lims_pairs'][::n_abins, 1, :]
                    nonstack_edges = hist_dict1['a_m_lims_pairs'][:n_abins, 0, :]
                    n_nonstack_bins = hist_dict1[f'n_abins']
                    Neff_summed1 = Neff_array1.sum(axis=1) # Sum along a
                    Neff_summed2 = Neff_array2.sum(axis=1) # Sum along a
                else:
                    raise ValueError(f'Unknown stack_dim={stack_dim}')
                
                nonstack_bin_centers = (nonstack_edges[:,1]*nonstack_edges[:,0])**0.5

                 
                #import pdb; pdb.set_trace()
                ## Step 1: collect integrated occurrence rates

                mode_OR_single1 = hist_dict1['mode_OR_single'][0]
                OR_lower_err_single1 = mode_OR_single1-hist_dict1['hdi_low_OR_single'][0]
                OR_upper_err_single1 = hist_dict1['hdi_high_OR_single'][0]-mode_OR_single1
                OR_str_single1 = ou.latex_formatter(mode_OR_single1, OR_lower_err_single1, OR_upper_err_single1)
                Neff_str1 = '{:.1f}'.format(np.sum(hist_dict1['cell_weights']))
                
                mode_OR_single2 = hist_dict2['mode_OR_single'][0]
                OR_lower_err_single2 = mode_OR_single2-hist_dict2['hdi_low_OR_single'][0]
                OR_upper_err_single2 = hist_dict2['hdi_high_OR_single'][0]-mode_OR_single2
                OR_str_single2 = ou.latex_formatter(mode_OR_single2, OR_lower_err_single2, OR_upper_err_single2)
                Neff_str2 = '{:.1f}'.format(np.sum(hist_dict2['cell_weights']))
                                       
                row['single_cell_OR_str_high'] = OR_str_single1
                row['single_cell_OR_str_low'] = OR_str_single2
                row['OR_ratio_high_low'] = '{:.1f}'.format(mode_OR_single1/mode_OR_single2)
                row['Neff_high'] = Neff_str1
                row['Neff_low'] = Neff_str2
                        

                ## Step 3: Collect model parameters for both experiments to compare
                for stack_ind in range(n_stack_bins):
                
                    stack_ind_str = number_to_words(stack_ind)
                
                    row[f'Neff_high_Bin{stack_dim}{stack_ind_str}'] = '{:.1f}'.format(Neff_summed1[stack_ind])
                    row[f'Neff_low_Bin{stack_dim}{stack_ind_str}'] = '{:.1f}'.format(Neff_summed2[stack_ind])
                
                    ## Step 1: collect exceedance fractions and z scores
                    output=au.hist_dist(tier123_dir1=tier123_dir1,
                                        tier123_dir2=tier123_dir2,
                                        label1=None,
                                        label2=None,
                                        stack_dim=stack_dim,
                                        stack_ind=stack_ind,
                                        m_unit=m_unit,
                                        make_plot=False)
                
                    #import pdb; pdb.set_trace()
                    row[f'diff_exceedance_frac_Bin{stack_dim}{stack_ind_str}'] = float_format_fn(output[0],2)
                    row[f'shape_exceedance_frac_Bin{stack_dim}{stack_ind_str}'] = float_format_fn(output[1],2)
                    row[f'diff_zscore_Bin{stack_dim}{stack_ind_str}'] = '{:.1f}'.format(output[2])
                    row[f'shape_zscore_Bin{stack_dim}{stack_ind_str}'] = '{:.1f}'.format(output[3])
                    row[f'avg_ratio_Bin{stack_dim}{stack_ind_str}'] = float_format_fn(output[4],2)
                ###############################################
                
                
                    ## Items needed for BIC calculation
                    ORD_vals1 = hist_dict1['mode_ORD'].reshape(n_nonstack_bins,-1)[:,stack_ind]
                    ORD_errs_high1 = hist_dict1['hdi_high_ORD'].reshape(n_nonstack_bins,-1)[:,stack_ind] - ORD_vals1
                    ORD_errs_low1 = ORD_vals1 - hist_dict1['hdi_low_ORD'].reshape(n_nonstack_bins,-1)[:,stack_ind]

                    bic_hist_dict1 = {'bin_centers':nonstack_bin_centers,
                                     'lims':(nonstack_edges[0][0], nonstack_edges[-1][-1]),
                                     'ORD_vals':ORD_vals1,
                                     'ORD_errs_high':ORD_errs_high1,
                                     'ORD_errs_low':ORD_errs_low1,
                                     }
                     
                    _, flat_bic_high, _, _ = mcmc_power.calculate_bic(tier123_dir1, 'flat', 
                                                    bic_hist_dict1, model_dict, 
                                                    stack_ind, init_type='guess',
                                                    verbose=False)
                                                    
                    ORD_vals2 = hist_dict2['mode_ORD'].reshape(n_nonstack_bins,-1)[:,stack_ind]
                    ORD_errs_high2 = hist_dict2['hdi_high_ORD'].reshape(n_nonstack_bins,-1)[:,stack_ind] - ORD_vals2
                    ORD_errs_low2 = ORD_vals2 - hist_dict2['hdi_low_ORD'].reshape(n_nonstack_bins,-1)[:,stack_ind]

                    bic_hist_dict2 = {'bin_centers':nonstack_bin_centers,
                                     'lims':(nonstack_edges[0][0], nonstack_edges[-1][-1]),
                                     'ORD_vals':ORD_vals2,
                                     'ORD_errs_high':ORD_errs_high2,
                                     'ORD_errs_low':ORD_errs_low2,
                                     }                       
                    _, flat_bic_low, _, _ = mcmc_power.calculate_bic(tier123_dir2, 'flat', 
                                                    bic_hist_dict2, model_dict, 
                                                    stack_ind, init_type='guess',
                                                    verbose=False)
                
                
                    for model in ['logG', 'escarpment']:
                        chain_path1 = os.path.join(tier123_dir1, f'saved_chains/chains_{model}_bin{stack_ind}.npz')
                        chain_path2 = os.path.join(tier123_dir2, f'saved_chains/chains_{model}_bin{stack_ind}.npz')
                        model_chain1 = np.load(chain_path1)
                        model_chain2 = np.load(chain_path2)
                        
                        
                        chains1 = model_chain1["chains"]  # (nsteps, nwalkers, ndim)
                        chains2 = model_chain2["chains"]  # (nsteps, nwalkers, ndim)
                        model_samples1 = chains1.reshape(-1, chains1.shape[-1])
                        model_samples2 = chains2.reshape(-1, chains2.shape[-1])
                         
                        full_len1 = model_samples1.shape[0]
                        if full_len1>20000: # Don't need huge chains to get summary stats. Take the end.
                            model_samples1 = model_samples1[full_len1-10000:,:] # 'burn in' until only 10k samples left
                            
                        full_len2 = model_samples2.shape[0]
                        if full_len2>20000:
                            model_samples2 = model_samples2[full_len2-10000:,:] # 'burn in' until only 10k samples left
                            

                        #import pdb; pdb.set_trace()
                        rate_type='model'
                        hdi_frac=0.68
                        gs=2000
                        #try:
                        chain_summary_dict1 = ou.summarize_chains(model_samples1, rate_type, hdi_frac, gs)
                        chain_summary_dict2 = ou.summarize_chains(model_samples2, rate_type, hdi_frac, gs)
                        chain_summary_dict_diff = ou.summarize_chains(model_samples1-model_samples2, rate_type, hdi_frac, gs)
                        #except:
                        #    import pdb; pdb.set_trace()
                         
                        for i in range(len(chain_summary_dict1['mode_model'])):
                            new_key1 = f'{model}_param{i}_high_Bin{stack_dim}{stack_ind_str}'
                            new_key2 = f'{model}_param{i}_low_Bin{stack_dim}{stack_ind_str}'
                            new_key_diff = f'{model}_param{i}_diff_Bin{stack_dim}{stack_ind_str}'
                            new_key_Zscore = f'{model}_param{i}_zscore_Bin{stack_dim}{stack_ind_str}'
                            
                            mode1 = chain_summary_dict1['mode_model'][i]
                            low_err1 = mode1-chain_summary_dict1['hdi_low_model'][i]
                            high_err1 = chain_summary_dict1['hdi_high_model'][i]-mode1
                            new_val1 = ou.latex_formatter(mode1, low_err1, high_err1)
                        
                            mode2 = chain_summary_dict2['mode_model'][i]
                            low_err2 = mode2-chain_summary_dict2['hdi_low_model'][i]
                            high_err2 = chain_summary_dict2['hdi_high_model'][i]-mode2
                            new_val2 = ou.latex_formatter(mode2, low_err2, high_err2)
                            
                            mode_diff = chain_summary_dict_diff['mode_model'][i]
                            low_err_diff = mode_diff-chain_summary_dict_diff['hdi_low_model'][i]
                            high_err_diff = chain_summary_dict_diff['hdi_high_model'][i]-mode_diff
                            new_val_diff = ou.latex_formatter(mode_diff, low_err_diff, high_err_diff)
                            new_val_Zscore = '{:.1f}'.format(abs(mode_diff) / np.mean([low_err_diff, high_err_diff]))
                            
                             
                            row[new_key1] = new_val1
                            row[new_key2] = new_val2
                            row[new_key_diff] = new_val_diff
                            row[new_key_Zscore] = new_val_Zscore
                            
                        ## Calculate model Delta_BIC ##
                        _, bic_high, _, _ = mcmc_power.calculate_bic(tier123_dir1, model, 
                                                                bic_hist_dict1, model_dict, 
                                                                stack_ind, init_type='chains',
                                                                verbose=False)
                        _, bic_low, _, _ = mcmc_power.calculate_bic(tier123_dir2, model, 
                                                                bic_hist_dict2, model_dict, 
                                                                stack_ind, init_type='chains',
                                                                verbose=False)

                        if np.isinf(bic_high) or np.isinf(bic_low):
                            _, bic_high, _, _ = mcmc_power.calculate_bic(tier123_dir1, model, 
                                                            bic_hist_dict1, model_dict, 
                                                            stack_ind, init_type='guess',
                                                            verbose=False)
                            _, bic_low, _, _ = mcmc_power.calculate_bic(tier123_dir2, model, 
                                                            bic_hist_dict2, model_dict, 
                                                            stack_ind, init_type='guess',
                                                            verbose=False)
                        row[f'{model}_DBIC_high_Bin{stack_dim}{stack_ind_str}'] = float_format_fn(flat_bic_high-bic_high,1)
                        row[f'{model}_DBIC_low_Bin{stack_dim}{stack_ind_str}'] = float_format_fn(flat_bic_low-bic_low,1)
                        
                row_list.append(row)
                #import pdb; pdb.set_trace()
                
    master_df = pd.DataFrame(row_list)
    #import pdb; pdb.set_trace()
    
    
    print('table_maker.py: Omitting some master_df cols from paper tables. Check master_df cols to restore')
    #### Make appendix table ####
    df_row_list = []
    for i in range(len(master_df)):
        row = master_df.iloc[i]
        if row['tier2_type']=='allstars':
            #import pdb; pdb.set_trace()
            
            col_list = ['tier1_val', 'tier2_type', 'single_cell_OR_str']
            Neff_col_list = [f'Neff_Bin{stack_dim}{number_to_words(j)}' for j in range(n_stack_bins)]
            logG_DBIC_col_list = [f'logG_DBIC_Bin{stack_dim}{number_to_words(j)}' for j in range(n_stack_bins)]
            escarpment_DBIC_col_list = [f'escarpment_DBIC_Bin{stack_dim}{number_to_words(j)}' for j in range(n_stack_bins)]
            logG_col_list = [f'logG_param{i}_Bin{stack_dim}{number_to_words(j)}' for i in range(3) for j in range(n_stack_bins)]
            esc_col_list = [f'escarpment_param{i}_Bin{stack_dim}{number_to_words(j)}' for i in range(4) for j in range(n_stack_bins)]
            #import pdb; pdb.set_trace()
            col_list = col_list\
                      +Neff_col_list\
                      +logG_DBIC_col_list\
                      +escarpment_DBIC_col_list\
                      +logG_col_list\
                      +esc_col_list

            row_all = row[col_list]
            df_row_list.append(row_all.to_frame().T)
            
        else: ## For other rows, break up into two rows for table. Modify high/low colnames to match each other

            col_list_high = ['tier1_val', 'tier2_type']+[name for name in master_df.columns if 'high' in name]
            row_high = row[col_list_high]
            row_high.index = [col.replace('_high', '') if 'high_low' not in col else col for col in row_high.index]
            row_high['tier2_type'] = 'high'+row_high['tier2_type']
            

            col_list_low = ['tier1_val', 'tier2_type']+[name for name in master_df.columns if 'low' in name]
            row_low = row[col_list_low]
            row_low.index = [col.replace('_low', '') if 'high_low' not in col else col for col in row_low.index]
            row_low['tier2_type'] = 'low'+row_low['tier2_type']
            
            df_row_list.append(row_high.to_frame().T)
            df_row_list.append(row_low.to_frame().T)
            
    ## Set col order using one of the high/low rows
    col_order = df_row_list[-1].columns
    param_df = pd.concat(df_row_list)[col_order]
    try:
        param_df = param_df.drop(columns=['Neff', 'OR_ratio_high_low'])
    except:
        pass
    param_df.to_latex('paper_plots_and_tables/param_table_appendix.tex', index=False, escape=False)
    
    ## Make smaller table w/ param values for allstars fits to go in main text ##
    param_df_allstars = param_df.query("tier2_type=='allstars'")
    #import pdb; pdb.set_trace()
    drop_cols = ['tier1_val', 'tier2_type']+[col for col in param_df_allstars if 'DBIC' in col]
    param_df_allstars = param_df_allstars.drop(columns=drop_cols)
    col_rename_dict = {'logG_param0':'A', 'logG_param1':'$\mu$', 'logG_param2':'$\sigma$', 
                       'escarpment_param0':'C1', 'escarpment_param1':'C2',
                       'escarpment_param2':'$\log_{10} (x_{t,1})$',
                       'escarpment_param3':'$\log_{10} (x_{t,2})$'}
                       
    param_df_allstars = param_df_allstars.T.rename(index=col_rename_dict)
    
    param_df_allstars = param_df_allstars.rename(
                         index=lambda s: next(
                                (v for k, v in col_rename_dict.items() if k in s),
                                 s
                          ))
    param_df_allstars.to_latex('paper_plots_and_tables/param_table.tex', index=True, escape=False)
    
    
    
    ### Make variables.tex file ###
    #import pdb; pdb.set_trace()
    write_variables_tex(master_df)
    import pdb; pdb.set_trace()
    
    return
            
def make_3param_table():
    """
    Make table containing occurrence stats
    for 8 subsamples, split by stellar mass,
    metallicity, and activity.
    """
    
    tier1_dir = 'mtrue'
    tier2_dir_list = ["highMstarhighFeHhighAct", "highMstarhighFeHlowAct",
                      "highMstarlowFeHhighAct", "highMstarlowFeHlowAct",
                      "lowMstarhighFeHhighAct", "lowMstarhighFeHlowAct",
                      "lowMstarlowFeHhighAct", "lowMstarlowFeHlowAct"]
    tier3_dir='stellar3params'
    
    root_dir = '/data/user/judahvz/postdoc_projects/bd_desert_rv/mtrue/'
    dict_dir = 'stellar3params/saved_dicts/summary_dict.npz'
    
    row_list = []
    for t2dir in tier2_dir_list:
    
        levels = re.findall(r'(high|low)(?=[A-Z])', t2dir) # Isolate e.g. ['high', 'low', 'low']
        
        summary_dict_path = os.path.join(root_dir, t2dir, dict_dir)
        summary_dict = dict(np.load(summary_dict_path))
        
        ## Get basic stats right from the dict
        neff = summary_dict['cell_weights'][0]
        nstars = summary_dict['nstars']
        compl = summary_dict['cell_compls'][0]
        
        ## Convert OR stats into strings
        mode_OR = summary_dict['mode_OR'][0]
        OR_err_low = mode_OR - summary_dict['hdi_low_OR'][0]
        OR_err_high = summary_dict['hdi_high_OR'][0] - mode_OR
        OR_str = ou.latex_formatter(mode_OR, OR_err_low, OR_err_high)
        
        age = 'old' if levels[2]=='low' else 'young' if levels[2]=='high' else None
        row_df = pd.DataFrame([{'Mass':levels[0],
                                'FeH':levels[1],
                                'Age': age,
                                'Neff':np.round(neff, 1),
                                'Nstars':nstars,
                                'compl':np.round(compl,2),
                                'OR':OR_str}])
        row_list.append(row_df)
    
    table_df = pd.concat(row_list)
    table_df.to_latex('paper_plots_and_tables/three_param_OR_table.tex', escape=False, index=False)
    
    param_list = ['Mass', 'FeH', 'Age']
    for table_param in param_list:
        idx_cols = [p for p in param_list if p!=table_param]

        sub_table = table_df.pivot(index=idx_cols, columns=table_param, values="OR")\
                            .reset_index()\
                            .rename_axis(None, axis=1)
        if table_param != 'Age':
            col_order = [*idx_cols, 'low', 'high']
        else:
            col_order = [*idx_cols, 'young', 'old']
        
        sub_table = sub_table[col_order]
        sub_table.to_latex(f'paper_plots_and_tables/sub_table_{table_param}.tex', escape=False, index=False)
                        
    import pdb; pdb.set_trace()
    
    return
            
            
            

def write_variables_tex(df, output_file="variables.tex"):
    """
    Convert a dataframe of comparison statistics into a LaTeX
    variables.tex file containing \\newcommand definitions.
    """

    model_param_name_dict = {
        'Neff':'Neff',
        'single_cell_OR_str': 'IntOcc',
        'logG_param0': 'LogGParamA',
        'logG_param1': 'LogGParamMu',
        'logG_param2': 'LogGParamSigma',
        'logG_DBIC': 'LogGDbic',
        'escarpment_param0': 'EscarpmentParamCOne',
        'escarpment_param1': 'EscarpmentParamCTwo',
        'escarpment_param2': 'EscarpmentParamBPOne',
        'escarpment_param3': 'EscarpmentParamBPTwo',
        'escarpment_DBIC': 'EscarpmentDbic',
        'diff_exceedance_frac': 'DiffExceedanceFrac',
        'shape_exceedance_frac': 'ShapeExceedanceFrac',
        'diff_zscore': 'DiffZscore',
        'shape_zscore': 'ShapeZscore',
        'avg_ratio': 'AvgRatio',
        'OR_ratio': 'ORRatio'
    }

    tier1_prefix_dict = {
        'mtrue': 'Mc',
        'qtrue': 'Q'
    }

    def format_scientific(val):
        """
        Convert a float to LaTeX scientific notation.
        """
        mantissa, exponent = f"{val:.2e}".split("e")
        exponent = int(exponent)

        mantissa = mantissa.rstrip("0").rstrip(".")

        return rf"{mantissa} \cdot 10^{{{exponent}}}"

    def latexify_value(val):
        """
        Convert value to string suitable for \ensuremath{}.
        """

        # Strings are assumed already LaTeX-formatted
        if isinstance(val, str):
            val = val.strip("$")
            return val

        if isinstance(val, (float, np.floating, int, np.integer)):

            if np.isnan(val):
                return None

            if val != 0 and abs(val) < 1e-3:
                return format_scientific(val)

            return str(val)

        return str(val)

    def get_macro_name(prefix, col):
        """
        Convert dataframe column name into macro name.
        """

        # Find longest matching dictionary key
        base_key = max(
                (k for k in model_param_name_dict if col.startswith(k)),
                key=len
            )

        macro_name = model_param_name_dict[base_key]

        suffix = col[len(base_key):]

        if suffix:

            # "_high_low" -> ["high","low"]
            suffix_parts = [s for s in suffix.split("_") if s]

            suffix_parts = [
                part[0].upper() + part[1:]
                for part in suffix_parts
            ]

            macro_name += "".join(suffix_parts)

        return prefix + macro_name

    lines = []

    for _, row in df.iterrows():

        prefix = (
            tier1_prefix_dict[row["tier1_val"]]
            + row["tier2_type"]
        )
        #import pdb; pdb.set_trace()
        for col, val in row.items():

            if col in ["tier1_val", "tier2_type"]:
                continue

            if pd.isna(val):
                continue

            macro_name = get_macro_name(prefix, col)

            latex_val = latexify_value(val)

            if latex_val is None:
                continue

            lines.append(
                rf"\newcommand{{\{macro_name}}}{{\ensuremath{{{latex_val}}}}}"
            )

    with open(output_file, "w") as f:
        f.write("% Auto-generated file\n\n")
        f.write("\n".join(lines))

    return
                              

def number_to_words(n):
    ones = ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten", 
            "Eleven", "Twelve", "Thirteen", "Fourteen", "Fifteen", "Sixteen", "Seventeen", "Eighteen", "Nineteen"]
    tens = ["", "", "Twenty", "Thirty", "Forty", "Fifty", "Sixty", "Seventy", "Eighty", "Ninety"]

    if n < 0:
        return "minus " + number_to_words(abs(n))
    
    if n < 20:
        return ones[n]
    
    if n < 100:
        return tens[n // 10] + ("-" + ones[n % 10] if (n % 10 != 0) else "")
    
    if n < 1000:
        return ones[n // 100] + "Hundred" + (number_to_words(n % 100) if (n % 100 != 0) else "")
    
    #for power, unit in [(10**6, "million"), (10**3, "thousand")]:
    #    if n >= power:
    #        return number_to_words(n // power) + f" {unit}" + (" " + number_to_words(n % power) if (n % power != 0) else "")


if __name__=="__main__":
    
    make_variables_file(tier1_list=['mtrue', 'qtrue'],
                          tier2_types=['allstars', 'Mstar', 'FeH', 'Act'],
                          tier3_list=['paper_bounds'],
                          stack_dim='a',
                          m_unit='jupiter')
    
    #make_3param_table()






