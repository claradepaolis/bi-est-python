import time
import copy

import numpy as np
from scipy.linalg import sqrtm
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from initialization import initialize_from_labeled, init_params_from_mu
from logl import pnu_loglikelihood
import optutils

def PU_nested_em_opt(X_unlabeled, X_labeled_pos, X_labeled_neg, K, max_steps=2000,rnd_state=0, initialization='labeled_means'):
    
    # convergence conditions
#     max_steps = 2000
    tol = 1e-11
    diff_tol_count = 0  # count of steps since convergence still under tolerance
    param_tol = 1e-8  # tolerance level for relative paramater changes
    param_diff_tol_count = 0
    eps = 1e-300  # small number to replace 0 for log calculations
    min_diff_steps = 100  # number of steps under the convergence tolerance
    min_param_diff_steps = 10  
    pos_scale=neg_scale=1
    
    N_unlabeled, dim = np.shape(X_unlabeled) 
    N_labeled_pos = np.shape(X_labeled_pos)[0] # num positive labeled samples
    N_labeled_neg = np.shape(X_labeled_neg)[0] # num negative labeled samples
    N = N_unlabeled + N_labeled_neg + N_labeled_pos

    
    # initialize unlabeled means with kmeans
    kmeans = KMeans(n_clusters=2*K, random_state=rnd_state, init='k-means++').fit(X_unlabeled)
    k_mu = kmeans.cluster_centers_
    # k_mu should be shaped 2K*dim 
    
    mu, _, w_labeled = initialize_from_labeled(X_labeled_pos, X_labeled_neg, dim, K, k_mu, 
                                               initialization,rnd_state)
    
    
    alphas, w, sg = init_params_from_mu(X_unlabeled, mu);
    
    sg = optutils.check_conditioning(sg, dim, K)
    
    start_time = time.time()
    
    step = 0
    diff = 2 * tol
    
    
    
    ll,ll_unlabeled,ll_posl,ll_negl  = pnu_loglikelihood(
        X_unlabeled, X_labeled_pos, X_labeled_neg, 
        mu, sg, alphas, w, w_labeled, eps, pos_scale, neg_scale)
    lls = [ll]
    

    ## Iterate optimization
    continue_opt = True
    while continue_opt:
        #step < max_steps && (diff_tol_count < min_diff_steps || param_diff_tol_count < min_param_diff_steps)

        mu_old =  copy.deepcopy(mu)
        ll_old = copy.deepcopy(ll)

        ## E step
        posteriors = np.zeros([N_unlabeled, 2, K]) # row 1 for positive, row 2 for negative, dim 2 for component
        numerator_N = np.zeros([N_unlabeled, 2,K])
        mult=np.zeros([2,K])
        inv_sig=np.zeros([2,K,dim,dim])
        det_sig=np.zeros([2,K])

        # terms from unlabeled points
        for c in range(2):  # pos c=0 and neg c=1 components
            for k in range(K):  # components within pos or neg
                t_N = X_unlabeled - mu[c, :, k]  #   N x dim
                sig = np.reshape(sg[c, :, :, k],[dim,dim]); #select sigma for component c, subcomponent k 
                
                inv_sig[c,k,:,:] = np.linalg.inv(sig)
                det_sig[c,k] = np.linalg.det(sig)
                mult[c,k]=alphas[c]*w[c,k] / np.sqrt((2*np.pi) ** dim * det_sig[c,k]);
                
                sqrtinvsig = sqrtm(inv_sig[c,k,:,:])
                expsum = np.sum(np.matmul(t_N, sqrtinvsig)**2, axis=1)
                numerator_N[:,c,k] = mult[c,k] * np.exp(-0.5* expsum)  

        denom = np.sum(numerator_N, axis=(1,2))
        denom[denom==0] = eps
        for c in range(2):
            for k in range(K):
                posteriors[:,c,k] = numerator_N[:,c,k]/denom
            
        # terms from labeled positives points
        if N_labeled_pos > 0:
            posteriors_labeled = np.zeros([N_labeled_pos, K])
            numerator_N = np.zeros([N_labeled_pos,K]);
            mult = np.zeros(K)
            inv_sig=np.zeros([K,dim,dim])
            det_sig=np.zeros([K,1])
            c = 0 # select positive param index
            for k in range(K):  # subcomponents within pos 
                t_N = X_labeled_pos - mu[c, :, k]  # N x dim
                sig = np.reshape(sg[c, :, :, k], [dim,dim])  #select sigma for component c, subcomponent k
                inv_sig[k,:,:] = np.linalg.inv(sig);
                det_sig[k] = np.linalg.det(sig);
                mult[k]= w_labeled[c,k] / np.sqrt((2*np.pi) ** dim * det_sig[k]);
                
                sqrtinvsig = sqrtm(inv_sig[k,:,:])
                expsum = np.sum(np.matmul(t_N, sqrtinvsig)**2, axis=1)
                numerator_N[:,k] = mult[k] * np.exp(-0.5* expsum) 
            

            denom = np.sum(numerator_N, axis=1);
            denom[denom==0] = eps;
            for k in range(K):
                posteriors_labeled[:,k] = numerator_N[:,k]/denom
        
        #  terms from labeled negative points
        if N_labeled_neg > 0:
            posteriors_labeled_neg = np.zeros([N_labeled_neg, K])
            numerator_N = np.zeros([N_labeled_neg,K])
            mult = np.zeros(K);
            inv_sig = np.zeros([K,dim,dim]);
            det_sig = np.zeros(K);
            c = 1;  # select negative param index
            for k in range(K):  # components within neg 
                t_N = (X_labeled_neg - mu[c, :, k])  # N x dim
                sig = np.reshape(sg[c, :, :, k], [dim,dim])  # select sigma for component c
                inv_sig[k,:,:] = np.linalg.inv(sig);
                det_sig[k] = np.linalg.det(sig);
                mult[k]= w_labeled[c,k] / np.sqrt((2*np.pi) ** dim * det_sig[k])
                
                sqrtinvsig = sqrtm(inv_sig[k,:,:])
                expsum = np.sum(np.matmul(t_N, sqrtinvsig)**2, axis=1)
                numerator_N[:,k] = mult[k] * np.exp(-0.5* expsum)  

            denom = np.sum(numerator_N, axis=1)
            denom[denom==0] = eps
            for k in range(K):
                posteriors_labeled_neg[:,k] = numerator_N[:,k]/denom

        if N_labeled_pos > 0:
            full_posteriors_labeled=np.zeros([N_labeled_pos,2,K]);
            full_posteriors_labeled[:,0,:] = np.reshape(posteriors_labeled,[N_labeled_pos,K]);
            posteriors_labeled = full_posteriors_labeled
        if N_labeled_neg > 0:
            full_posteriors_labeled_neg=np.zeros([N_labeled_neg,2,K]);
            full_posteriors_labeled_neg[:,1,:] = np.reshape(posteriors_labeled_neg,[N_labeled_neg,K]);
            posteriors_labeled_neg = full_posteriors_labeled_neg

        # M step
        # given label and component posteriors, update parameters: alphas, w, mu, and sg
        #alphas and ws are estimated using only unlabeled data

        # Update parameters alpha and w 
        # get sorted sums for more accurate results
        p = np.zeros([2,K])
        p_labeled = np.zeros([2,K])
        for c in range(2):
            for k in range(K):
                #p[c,k] = np.sum(sort(posteriors(:,c,k),'ascend')); # sum over instances
                p[c,k] = np.sum(posteriors[:,c,k])
                if (N_labeled_pos > 0) and (c==0):
                    #p_labeled[c,k] = pos_scale*np.sum(sort(posteriors_labeled(:,c,k),'ascend'));
                    p_labeled[c,k] = pos_scale*np.sum(posteriors_labeled[:,c,k])

                if (N_labeled_neg > 0) and (c==1):
                    #p_labeled[c,k] = neg_scale*np.sum(sort(posteriors_labeled_neg(:,c,k),'ascend'));
                    p_labeled[c,k] = neg_scale*np.sum(posteriors_labeled_neg[:,c,k])
                
        for c in range(2):
            comp_posterior_sum = np.sum(p[c,:])  # sum over subcomponents
            alphas[c] = comp_posterior_sum/N_unlabeled
            for k in range(K):
                w[c,k] = np.sum(p[c,k])/comp_posterior_sum
                if (N_labeled_pos > 0) and (c==0):
                    lcomp_posterior_sum = np.sum(p_labeled[c,:])
                    w_labeled[c, k] = p_labeled[c,k]/lcomp_posterior_sum
                if (N_labeled_neg > 0) and (c==1):
                    lcomp_posterior_sum = np.sum(p_labeled[c,:])
                    w_labeled[c, k] = p_labeled[c,k]/lcomp_posterior_sum
                    
        # Correct mixing proportions
        # prevent later taking log(w_i) if w_i==0
        if np.sum(w==0)>0:
            w[w==0] = eps
            w[0,:] = w[0,:]/np.sum(w[0,:])
            w[1,:] = w[1,:]/np.sum(w[1,:])
        if N_labeled_pos > 0:
            w_labeled[0, w_labeled[0, :]==0]=eps;
            w_labeled[0,:] = w_labeled[0,:]/sum(w_labeled[0,:])

        if N_labeled_neg > 0:
            w_labeled[1, w_labeled[1, :]==0]=eps;
            w_labeled[1,:] = w_labeled[1,:]/sum(w_labeled[1,:])

         # Update parameters mu & sigma
        denom = np.reshape(np.sum(posteriors, axis=0), [2, K])
        if N_labeled_pos + N_labeled_neg > 0:  # any labeled points
            denom += p_labeled

        mu = np.zeros([2, dim, K])
        sg = np.zeros([2, dim, dim, K])

        for k in range(K):
            for c in range(2):
                pX = posteriors[:, c, k][:, np.newaxis] * X_unlabeled
                #mu(c, :, k) = np.sum(sort(pX));
                mu[c, :, k] = np.sum(pX)
                xmu_unlabeled = X_unlabeled - mu_old[c, :, k]
                pxmu = np.sqrt(posteriors[:, c, k])[:, np.newaxis]*xmu_unlabeled
                sg[c, :, :, k] = np.reshape(np.matmul(np.transpose(pxmu), pxmu), [dim, dim])
            if N_labeled_pos > 0:
                c=0;  # select positive param index
                xmu_labeled = X_labeled_pos - mu_old[c, :, k]
                pX = posteriors_labeled[:, c, k][:, np.newaxis] * X_labeled_pos
                mu[c, :, k] = mu[c, :, k] + (pos_scale*np.sum(pX))
                pxmu = np.sqrt(posteriors_labeled[:, c, k])[:, np.newaxis]*xmu_labeled
                sg[c, :, :, k] = sg[c, :, :, k] + (pos_scale*np.reshape(np.matmul(np.transpose(pxmu), pxmu), [dim, dim]))

            if N_labeled_neg > 0:
                c=1; # select negative param index
                xmu_labeled_neg = (X_labeled_neg - mu_old[c, :, k])
                pX = posteriors_labeled_neg[:, c, k][:, np.newaxis] * X_labeled_neg
                #mu[c, :, k] = mu[c, :, k] + (neg_scale*np.sum(sort(pX)));
                mu[c, :, k] = mu[c, :, k] + (neg_scale*np.sum(pX))
                pxmu = np.sqrt(posteriors_labeled_neg[:, c, k])[:, np.newaxis]*xmu_labeled_neg
                sg[c, :, :, k] = sg[c, :, :, k] + (neg_scale*np.reshape(np.matmul(np.transpose(pxmu), pxmu), [dim, dim]))


        denom[denom==0] = eps
        for c in range(2):
            for k in range(K):
                mu[c,:,k] = mu[c,:,k] / denom[c, k];
                sg[c,:,:,k] = sg[c,:,:,k] / denom[c, k];
        
        # recondition covariance matrix if necessary
        sg = optutils.check_conditioning(sg,dim,K)

        # Compute loglikelihood
        ll,ll_unlabeled,ll_posl,ll_negl  = pnu_loglikelihood(
            X_unlabeled, X_labeled_pos, X_labeled_neg, 
            mu, sg, alphas, w, w_labeled, eps, pos_scale, neg_scale);
        
        lls.append(ll)
    
#         if ll < ll_old:  #something has gone wrong, or reconditioning destabilized the optimization
#             print('### step {} logL error {}'.format(step, ll_old-ll))

        # Termination conditions
    
        param_diff = np.sum(np.abs(mu - mu_old))/np.abs(np.sum(mu))
        diff = np.abs(ll-ll_old)/np.abs(ll_old)
        if diff < tol:
            diff_tol_count += 1
        else:
            diff_tol_count = 0  # reset the count

        if param_diff < param_tol:
            param_diff_tol_count += 1;
        else:
            param_diff_tol_count = 0
    
        step +=1
        
        if (step >= max_steps): 
            continue_opt= False 
        if (diff_tol_count > min_diff_steps) and (param_diff_tol_count > min_param_diff_steps):
            continue_opt= False 
            
            
    elapsed = time.time()-start_time
    return alphas[0], w, w_labeled, sg, mu, lls
