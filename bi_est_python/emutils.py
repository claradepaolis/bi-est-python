import copy
import numpy as np
from scipy.linalg import sqrtm
from sklearn.cluster import KMeans
from bi_est_python import optutils


def em_opt(X, K, max_step=250, rnd_state=0):
    
    N, dim =np.shape(X)
    eps = 0.000001
    
    # initialize w, mu, and sg
    w = np.ones(K)/K
    sg = np.zeros([dim, dim, K])
    for k in range(K):
        sg[:, :, k] = np.eye(dim)
    
    # use K-means to find initial clusters
    kmeans = KMeans(n_clusters=K, random_state=rnd_state, init='k-means++').fit(X)
    mu = kmeans.cluster_centers_
    mu = np.transpose(mu)
    
    step = 0;
    diff = 2 * eps  # just to ensure it is greater than eps
    
    while (diff > eps) and (step < max_step):

        w_old = copy.deepcopy(w)
        mu_old = copy.deepcopy(mu)
        sg_old = copy.deepcopy(sg)

        # E step
        posteriors = np.zeros([N, K])  
        numerator_N = np.zeros([N, K]);
        mult=np.zeros(K)
        inv_sig=np.zeros([K,dim,dim])
        det_sig=np.zeros(K)
        
        for k in range(K): # components within pos or neg
            t_N = X - np.transpose(mu[:, k])  # N x dim
            sig = np.reshape(sg[:, :, k],[dim,dim]) #select sigma for component k 
            inv_sig[k,:,:] = np.linalg.inv(sig)
            det_sig[k] = np.linalg.det(sig)
            mult[k] = w[k] / np.sqrt((2*np.pi) ** dim * det_sig[k])

            sqrtinvsig = sqrtm(inv_sig[k,:,:])
            expsum = np.sum(np.matmul(t_N, sqrtinvsig)**2, axis=1)
            numerator_N[:,k] = mult[k] * np.exp(-0.5* expsum)  
        
        
        denom = np.sum(numerator_N,axis=-1)
        denom[denom==0] = eps
        for k in range(K): 
            posteriors[:,k] = numerator_N[:,k]/denom

        # M step
        # given label and component posteriors, update parameters: alphas, w, mu, and sg

        # Update parameters alpha and w 
        # get sorted sums for more accurate results
        p = np.zeros(K);
        for k in range(K):
            # p[k] = np.sum(sort(posteriors[:,k]),'ascend'));  # sum over instances
            p[k] = np.sum(posteriors[:,k])  # sum over instances
        
        comp_posterior_sum = np.sum(p)  # sum over subcomponents
        for k in range(K):
            w[k] = np.sum(p[k])/comp_posterior_sum


        # Correct mixing proportions
        # prevent later taking log(w_i) if w_i==0
        if np.sum(w==0)>0:
            w[w==0] = eps
            w = w/np.sum(w)

        # Update parameters mu & sigma
        denom = np.sum(posteriors, axis=0)
        mu = np.zeros([dim, K]);
        sg = np.zeros([dim, dim, K]);

        for k in range(K):
            pX = posteriors[:, k][:, np.newaxis] * X
            mu[:, k] = np.sum(pX, axis=0)
            xmu = X - mu_old[:, k]
            pxmu = np.sqrt(posteriors[:, k])[:, np.newaxis] * xmu;
            sg[:, :, k] = np.reshape(np.matmul(np.transpose(pxmu), pxmu), [dim, dim])

        denom[denom==0] = eps
        for k in range(K):
            mu[:,k] = mu[:,k] / denom[k]
            sg[:,:,k] = sg[:,:,k] / denom[k]


        # recondition covariance matrix if necessary
        sg = optutils.check_conditioning(sg, dim, K)

        # Termination conditions
        diff = np.sum(np.abs(w - w_old)) + np.sum(np.sum(np.abs(mu - mu_old))) + np.sum(np.sum(np.sum(np.abs(sg - sg_old))))
        step += 1
                                  
                              
    return w, mu, sg
