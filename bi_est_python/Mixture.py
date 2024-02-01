import scipy
import numpy as np
from scipy.stats import multivariate_normal
from scipy.stats import norm 
from bi_est_python import mixtureUtils

class NMixture:
    """
    Gaussian mixture model 
    Component means and covariancs specified my mu and sigma,
    mu: numpy array size num_components * dimensions
    sigma: numpy array size num_components * dimensions * dimensions
    ps: component proportions (should sum to 1 or will be normalized such that they do). 
    numpy array size num_componets
    """
    
    def __init__(self, mu, sigma, ps):
        
        self.ps = ps
        # normalize component mixture proportions if necessary
        if sum(ps)!=1:
            self.ps = ps/sum(ps)            
            
        self.num_comps = len(ps)
        
        # ensure correct sizes
        assert mu.shape[0]==self.num_comps
        assert len(sigma.shape)==3
        assert mu.shape[1]==sigma.shape[1]==sigma.shape[2]
        

        self.dim = mu.shape[1]
        self.mu=mu
        self.sigma=sigma
        
        if self.dim > 1:
            self.density = [multivariate_normal(mean=mu[k,:], cov=sigma[k,:]) for k in range(self.num_comps)]
        else:
            self.density = [norm(loc=mu[k].item(), scale=sigma[k].item()) for k in range(self.num_comps)]
    
    def __repr__(self):
        return f'NormalMixture({self.num_comps} components, \nmu={self.mu}, \nsigma={self.sigma}, \nps={self.ps})'
    
    def component_assingment(self,N):
        """get component assinment for a set of N points. 
        Corresponds to points that would be sampled from method sample_points"""
        return np.vstack([i*np.ones([n,1]) for i,n in enumerate(np.round(N * self.ps).astype(int))])
    
    def sample_points(self, N):
        """sample N points from mixture distribution"""
    
        if self.num_comps == 1:
            points = self.density.rvs(N)
        else:
            Nk = np.round(N * self.ps).astype(int)
            points = []
            for k in range(self.num_comps):
                if Nk[k] > 1:
                    kpoints = self.density[k].rvs(Nk[k])
                elif Nk[k] <= 1:  # only one point in component
                    kpoints = self.density[k].rvs(Nk[k]).reshape(1,-1)
                if Nk[k]>0:
                    # make sure when dim=1, points still a 2d array
                    if len(kpoints.shape)==1:
                        npoints = kpoints.shape[0]
                        kpoints=kpoints.reshape(npoints,1)

                    points.append(kpoints)
            
        return np.vstack(points)
        
    def points_pdf(self, x):
        # compute pdf of points x 
        return np.sum((w * comp.pdf(x) for (comp, w) in zip(self.density, self.ps)), axis=1)
        

class PUMixture:
    
    def __init__(self, pos, neg, alpha):
        
        self.pos = pos
        self.neg = neg
        self.alpha = alpha

        self._auc = None
        self._auc_unweighted = None
    
    def __repr__(self):
        return f'PUMixture(pos={self.pos}, \nneg={self.neg}, \nalpha={self.alpha})'
        
    def sample_points(self, N):
        """sample N points from PU mixture distribution"""
        points  = []
        Npos = np.round(N*self.alpha).astype(int)
        Nneg = N - Npos
        points.append(self.pos.sample_points(Npos))
        points.append(self.neg.sample_points(Nneg))
        
        return np.vstack(points)
        
    def points_pdf(self, x):
        return self.alpha*self.pos.points_pdf(x) + (1-self.alpha)*self.neg.points_pdf(x)
    
    def auc(self):
        if self._auc:
            return self._auc
        else:
            self._auc = mixtureUtils.auc(self.pos, self.neg, self.alpha) 
            return self._auc
        
    def kl(self):
        if self._kl:
            return self._kl
        else:
            self._kl = mixtureUtils.kl(self.pos, self.neg) 
            return self._kl
        
    def recompute_auc(self, avg_count=1):
        # force evaluation of AUC
        self._auc = np.mean([mixtureUtils.auc(self.pos, self.neg, self.alpha) for _ in range(avg_count)])
        return self._auc

    @property
    def auc_unweighted(self):
        if self._auc_unweighted:
            return self._auc_unweighted
        else:
            pos_uw = NMixture(self.pos.mu, self.pos.sigma, np.ones(self.pos.num_comps)/self.pos.num_comps)
            neg_uw = NMixture(self.neg.mu, self.neg.sigma, np.ones(self.neg.num_comps)/self.neg.num_comps)
            self._auc_unweighted = mixtureUtils.auc(pos_uw, neg_uw)
            return self._auc_unweighted
    
    