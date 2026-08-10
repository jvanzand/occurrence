

import os
import numpy as np
from scipy.stats import chi2, norm
from scipy.optimize import minimize_scalar
from scipy.special import erfinv
import matplotlib.pyplot as plt


from occurrence import plotting_utils as pu
from occurrence import occurrence_utils as ou



def hist_dist(tier123_dir1, tier123_dir2, label1, label2, stack_dim, stack_ind=0, m_unit=None, make_plot=True):
    """
    Estimate the difference between two histograms
    """
    
    ## First, verify that the specified histograms have the same length
    dict_path1 = os.path.join(tier123_dir1, 'saved_dicts/summary_dict.npz')
    dict_path2 = os.path.join(tier123_dir2, 'saved_dicts/summary_dict.npz')
    dict1 = dict(np.load(dict_path1))
    dict2 = dict(np.load(dict_path2))
    
    nonstack_dim = 'm' if stack_dim=='a' else 'a'
    n_stack_bins1, n_nonstack_bins1 = dict1[f'n_{stack_dim}bins'], dict1[f'n_{nonstack_dim}bins']
    n_stack_bins2, n_nonstack_bins2 = dict2[f'n_{stack_dim}bins'], dict2[f'n_{nonstack_dim}bins']
    
    #import pdb; pdb.set_trace()
    if n_nonstack_bins1 != n_nonstack_bins2: # Hists should be of equal length
        raise Exception("hist_dist: Histograms must have same number of bins")    
    
    
    ## Now load and analyze chains
    chain_path1 = os.path.join(tier123_dir1, 'saved_chains/chains_hist.npz')
    chain_path2 = os.path.join(tier123_dir2, 'saved_chains/chains_hist.npz')
    
    chain1 = np.load(chain_path1)
    chain2 = np.load(chain_path2)
    # Assume flat chains
    #if "flat_chains" in chain1:
    ORD_samples1 = chain1["flat_chains"]
    ORD_samples2 = chain2["flat_chains"]
    
    full_len1 = ORD_samples1.shape[0]
    if full_len1>20000: # Don't need huge chains to get summary stats. Take the end.
        ORD_samples1 = ORD_samples1[full_len1-10000:,:] # 'burn in' until only 10k samples left
    full_len2 = ORD_samples2.shape[0]
    if full_len2>20000: # Don't need huge chains to get summary stats. Take the end.
        ORD_samples2 = ORD_samples2[full_len2-10000:,:] # 'burn in' until only 10k samples left

    
    ## Reshape samples to (nsteps, n_nonstack, n_stack), then take the stack_th histogram
    ORD_samples1 = ORD_samples1.reshape(-1, n_nonstack_bins1, n_stack_bins1)[:,:,stack_ind]
    ORD_samples2 = ORD_samples2.reshape(-1, n_nonstack_bins2, n_stack_bins2)[:,:,stack_ind]
        
    #import pdb; pdb.set_trace()
    
    resample_num=10000
    resampled_inds1 = np.random.choice(ORD_samples1.shape[0], size=resample_num, replace=False)
    resampled_inds2 = np.random.choice(ORD_samples2.shape[0], size=resample_num, replace=False)
    
    
    ORD_resamples1 = ORD_samples1[resampled_inds1]
    ORD_resamples2 = ORD_samples2[resampled_inds2]
    
    
    var1 = np.var(ORD_resamples1, axis=0, ddof=1)
    var2 = np.var(ORD_resamples2, axis=0, ddof=1)
    
    #import pdb; pdb.set_trace()
    ## Now get the null comparison by comparing a distribution to itself
    #null_pool = np.vstack([ORD_resamples1, ORD_resamples2]) # The null dist should include both dists to avoid bias
    
    #null_resample_inds1 = np.random.choice(null_pool.shape[0], size=resample_num, replace=False)
    #null_resample_inds2 = np.random.choice(null_pool.shape[0], size=resample_num, replace=False)
    
    #null_resamples1 = null_pool[null_resample_inds1]
    #null_resamples2 = null_pool[null_resample_inds2]
    #null_var1 = np.var(null_resamples1, axis=0, ddof=1)
    #null_var2 = np.var(null_resamples2, axis=0, ddof=1)
    


    nbins = ORD_resamples1.shape[1]
    dof = nbins
    
    ## TEST 1: Are the histograms DIFFERENT?
    ## To calculate a single metric for the difference between histograms: 
    ##     Calculate a chi square statistic and compare to null.
    #####################################################################
    
    ## Reduced (note: np.mean, ie, divide by Nbins=Ndata) chi_sq statistic array
    #diff_chi_sq_array = (1/nbins)*np.sum((ORD_resamples1-ORD_resamples2)**2/(var1+var2), axis=1)

    ## We expect for the null case that the reduced (note: np.mean) chi_sq will be centered near 1
    #null_diff_chi_sq_array = (1/nbins)*np.sum((null_resamples1-null_resamples2)**2/(null_var1+null_var2), axis=1)
    
    ## Calculate the frac of chi_sq vals that are larger than their randomly drawn counterparts from the null distr.
    ## In other words, it's the confidence with which we can say that the two distributions are different
    #diff_chi_sq_diff_array = diff_chi_sq_array - null_diff_chi_sq_array
    #diff_chi_sq_exceedance_frac = np.sum(diff_chi_sq_diff_array>0)/diff_chi_sq_diff_array.shape[0]
    
    ## New approach: a single chisq value using mean values
    diff_chi_sq = np.sum((ORD_resamples1.mean(axis=0)-ORD_resamples2.mean(axis=0))**2/(var1+var2))
    diff_exceedance_frac = chi2.sf(diff_chi_sq, dof)
    #import pdb; pdb.set_trace()
    #diff_zscore = norm.isf(diff_exceedance_frac)
    
    
    ## TEST 2: Do the histograms have the same SHAPE?
    ## Idea: draw a realization from A and one from B, calculate their CDF and normalize 
    ##       Now you have two CDFs that each capture the shape of their histogram. Compute a distance metric.
    #####################################################################
    
    #import pdb; pdb.set_trace()
    
    ## To get CDFs, divide samples by sum to normalize, cumsum along hist axis, and omit last entry bc it always =1
    ## (in other words, the 7-bin CDF has only 6 deg of freedom)
    ## Transpose so shape is (Nbins-1 x Nsamples)
    #normalized_cdfs1 = np.cumsum(ORD_resamples1/ORD_resamples1.sum(axis=1)[:,None], axis=1)[:,:-1].T
    #normalized_cdfs2 = np.cumsum(ORD_resamples2/ORD_resamples2.sum(axis=1)[:,None], axis=1)[:,:-1].T
    
    #mean_difference_vector = (normalized_cdfs1-normalized_cdfs2).mean(axis=1)[:,None] # Average over sample dimension
    #cov = np.cov(normalized_cdfs1) + np.cov(normalized_cdfs2) # Multivariate analog of var1+var2
    
    #shape_chi_sq = np.linalg.multi_dot([mean_difference_vector.T, np.linalg.inv(cov), mean_difference_vector])
    #shape_chi_sq = float(shape_chi_sq) # Convert from 1x1 'matrix' to float
    #shape_exceedance_frac = chi2.sf(shape_chi_sq, dof-1) # One fewer DOF bc last bin of CDF is not free
    ##shape_zscore = norm.isf(shape_exceedance_frac)
    ###########################################################

    #import pdb; pdb.set_trace()
    
    # Take ratio between samples
    
    #avg_ratio_array = np.array([minimize_scalar(chisq, args=(ORD_resamples1[i,:], ORD_resamples2[i,:], var1, var2)).x for i in range(resample_num)])[:,None]
    #ratio_chi_sq_array = (1/(nbins-1))*np.sum((ORD_resamples1 - avg_ratio_array*ORD_resamples2)**2/(var1+avg_ratio_array**2*var2), axis=1)
    
    #import pdb; pdb.set_trace()
    ## Update: calculating single chi_sq value using mean values (assumes gaussian)
    avg_ratio = minimize_scalar(chisq, args=(ORD_resamples1.mean(axis=0), ORD_resamples2.mean(axis=0), var1, var2)).x

    shape_chi_sq = np.sum((ORD_resamples1.mean(axis=0) - avg_ratio*ORD_resamples2.mean(axis=0))**2/(var1+avg_ratio**2*var2))
    
    ## Null ratio array
    #null_avg_ratio_array = np.array([minimize_scalar(chisq, args=(null_resamples1[i,:], null_resamples2[i,:], null_var1, null_var2)).x for i in range(resample_num)])[:,None]
    #null_ratio_chi_sq_array = (1/(nbins-1))*np.sum((null_resamples1 - null_avg_ratio_array*null_resamples2)**2/(null_var1+null_avg_ratio_array*null_var2), axis=1)
    
    #ratio_chi_sq_diff_array = ratio_chi_sq_array - null_ratio_chi_sq_array
    #ratio_chi_sq_exceedance_frac = np.sum(ratio_chi_sq_diff_array>0)/ratio_chi_sq_diff_array.shape[0]
    
    shape_exceedance_frac = chi2.sf(shape_chi_sq, dof-1)
    

    ## Convert exceedance fractions into Z scores
    diff_zscore = np.sqrt(2)*erfinv(1-diff_exceedance_frac)
    shape_zscore = np.sqrt(2)*erfinv(1-shape_exceedance_frac)


    if make_plot:

        ## Retrieve histogram
        hist_dict = dict(np.load(os.path.join(tier123_dir1, 'saved_dicts/summary_dict.npz')))
    
        ## Make labels
        chain1_label = chain_path1.split('saved_chains')[0].replace('/', '_')
        chain2_label = chain_path2.split('saved_chains')[0].replace('/', '_')
        plot_title = f"{label1} vs. {label2}"
    
        ## Save locations
        save_dir = 'hist_comparisons/'
    
        subdir_part1 = tier123_dir1.split('/')[0]
        subdir_part2 = tier123_dir1.split('/')[1]+'_'+tier123_dir2.split('/')[1]
        save_subdir = os.path.join(save_dir, subdir_part1+'_'+subdir_part2)
        os.makedirs(save_subdir, exist_ok=True)
    
        hist_compare_label = tier123_dir1.split('/')[2]
        save_path_hist_compare = os.path.join(save_subdir, f'hist_diff_{hist_compare_label}.png')
    
        if 'qtrue' in tier123_dir1 or 'qsini' in tier123_dir1:
            mtype = 'q'
            m_unit = None
        else:
            mtype = 'm'
    
        pu.plot_comparison_violins(ORD_resamples1, ORD_resamples2,
                                   label1, label2,
                                   hist_dict, 
                                   diff_exceedance_frac, diff_zscore,
                                   shape_exceedance_frac, shape_zscore,
                                   avg_ratio, stack_dim, plot_title, 
                                   save_path_hist_compare, mtype, m_unit
                                   )
    
        ## Also plot the comparisons of the chi-square distributions ##
        #################################################################
        #chisq_compare_sets = [[diff_chi_sq_array, null_diff_chi_sq_array, diff_chi_sq_exceedance_frac, 'diff'],
        #                       [ratio_chi_sq_array, null_ratio_chi_sq_array, ratio_chi_sq_exceedance_frac, 'scale']]
        chisq_compare_sets=[]
    
        for chisq_set in chisq_compare_sets:
            chisq_array, null_chisq_array, exceedance_frac, plot_label = chisq_set
        
            null_chisq_array[null_chisq_array==0] = 0.001 # Some 0 vals. Replace for log calculation
            plt.hist(np.log(chisq_array), bins=100, label='$\chi^2_{red}$', alpha=0.7)
            plt.hist(np.log(null_chisq_array), bins=100, label='Null $\chi^2_{red}$', alpha=0.7)
        
        
            plt.xlabel('ln($\chi^2_{red}$)')
            plt.legend()
        
            if plot_label=='ratio': ## For ratio only: plot summary stats for slope
                plt.title(f'Exceedance Fraction = {exceedance_frac:.2f}, R={avg_ratio}')
            else:
                plt.title(f'Exceedance Fraction = {exceedance_frac:.2f}')
            plt.tight_layout()
            plt.savefig(os.path.join(save_subdir, f'{plot_label}_chisq_compare.png'))
            plt.close()
        
    
        chisq_compare_sets_new = [[diff_chi_sq, diff_exceedance_frac, diff_zscore, 'diff'],
                                  [shape_chi_sq, shape_exceedance_frac, shape_zscore, 'shape']]
        for chisq_set in chisq_compare_sets_new:
            chisq_single, exceedance_frac, zscore, plot_label = chisq_set
        
            plt.hist(chi2.rvs(dof, size=10000), bins=100, label=f'$\chi^2 (\\nu={dof})$')
            plt.vlines(chisq_single, 0, plt.ylim()[1]/2, label=f'$\chi^2 {plot_label}$', linestyles='--', color='red')
        
            plt.xlabel('$\chi^2$')
            plt.legend()
            
            if plot_label=='ratio': ## For ratio only: plot summary stats for slope
                plt.title(f'Significance = {zscore:.2f}$\sigma$, c={avg_ratio:.2f}')
            else:
                plt.title(f'Significance = {zscore:.2f}$\sigma$')
        
            plt.tight_layout()
            plt.savefig(os.path.join(save_subdir, f'{plot_label}_chisq_compare_single.png'))
            plt.close()
            
    
    #print(f"These are the vals: {diff_exceedance_frac:4f}, {shape_exceedance_frac:.4f}, {diff_zscore:.4f}, {shape_zscore:.4f}, {avg_ratio:.4f}")
    return diff_exceedance_frac, shape_exceedance_frac, diff_zscore, shape_zscore, avg_ratio




def chisq(c, A, B, varA, varB):
    """
    Calculate chi_sq of A-cB, given
    a value for c
    This is the objective function for
    optimization in hist_dist() above
    
    A and B should be 1D arrays, each
    representing a histogram draw
    """
    
    return np.sum((A-c*B)**2/(varA+c**2*varB))
   





