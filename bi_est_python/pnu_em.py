import time
import copy
import numpy as np
from scipy.linalg import sqrtm
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from scipy.stats import multivariate_normal
import logging
logging.basicConfig(level=logging.DEBUG)

from bi_est_python.initialization import initialize_from_labeled, init_params_from_mu
from bi_est_python.logl import pnu_loglikelihood
from bi_est_python import optutils

class PNU_EM:
    def __init__(self, **kwargs):
        """
        Optional Arguments
        ------------------
        tol : float, default=1e-11
            Tolerance for stopping criterion.

        max_steps : int, default=2000
            Maximum number of iterations.

        rnd_state : int, default=0
            Random seed for initializing the algorithm.

        """
        self.tol = kwargs.get('tol',1e-11)
        self.max_steps = kwargs.get('max_steps', 2000)
        self.rnd_state = kwargs.get('rnd_state', 0)

    def _initialize_parameters(self, **kwargs):
        self.weights_init(**kwargs)
        self.kmeans_init(**kwargs)
        self.second_init_step(**kwargs)
        self.third_and_fourth_init_step(**kwargs)

    def weights_init(self, **kwargs):
        self.weights_pos = np.ones((self.n_labeled_pos,1))
        self.weights_neg = np.ones((self.n_labeled_neg,1))
        self.weights_unlabeled = np.ones((self.n_unlabeled,1))
        if self.weighting_scheme == "sample":
            wP = 1 / self.n_labeled_pos
            wU = 1 / self.n_unlabeled
            wN = 1 / self.n_labeled_neg
            self.weights_pos *= wP
            self.weights_neg *= wN
            self.weights_unlabeled *= wU



    def kmeans_init(self, **kwargs):
        """
        Run KMeans (separately) on positives and negatives to initialize the component weights

        Optional Arguments
        ------------------
        n_init : int, default=10
            Number of KMeans initializations to perform
        """
        self.kmeans_pos = KMeans(n_clusters=self.n_components_pos,
                                    random_state=self.rnd_state,
                                        n_init=kwargs.get("n_init", 10))
        cluster_assignments_pos = self.kmeans_pos.fit_predict(self.X_labeled_pos,
                                                                sample_weight=self.weights_pos.ravel())
        self.v_pos = np.zeros((self.n_components_pos,1))
        for ki,count in zip(*np.unique(cluster_assignments_pos, return_counts=True)):
            self.v_pos[ki] = count / self.n_labeled_pos
        self.kmeans_neg = KMeans(n_clusters=self.n_components_neg,
                                    random_state=self.rnd_state,
                                        n_init=kwargs.get("n_init", 10))
        cluster_assignments_neg = self.kmeans_neg.fit_predict(self.X_labeled_neg, sample_weight=self.weights_neg.ravel())
        self.v_neg = np.zeros((self.n_components_neg,1))
        for ki,count in zip(*np.unique(cluster_assignments_neg, return_counts=True)):
            self.v_neg[ki] = count / self.n_labeled_neg

    
    def second_init_step(self,**kwargs):
        """ Run step 2 """
        kmeans_unlabeled = KMeans(n_clusters=self.n_components_pos + self.n_components_neg,
                                    random_state=self.rnd_state, n_init=kwargs.get("n_init", 10))
        kmeans_unlabeled.fit(self.X_unlabeled, sample_weight=self.weights_unlabeled.ravel())
        cluster_distances = cdist(kmeans_unlabeled.cluster_centers_,self.kmeans_pos.cluster_centers_) # shape = (n_components_pos + n_components_neg, n_components_pos)
        cluster_order = np.argsort(cluster_distances.min(1))
        self.mu_pos = kmeans_unlabeled.cluster_centers_[cluster_order[:self.n_components_pos]]
        self.mu_neg = kmeans_unlabeled.cluster_centers_[cluster_order[self.n_components_pos:]]

    def third_and_fourth_init_step(self, **kwargs):
        """ Run step 3 """
        distances_to_pos_centers = cdist(self.X_unlabeled, self.mu_pos) # shape = (n_unlabeled, n_components_pos)
        nearest_pos_center = distances_to_pos_centers.argmin(1)
        smallest_pos_distance = distances_to_pos_centers[np.arange(self.n_unlabeled), nearest_pos_center]
        distances_to_neg_centers = cdist(self.X_unlabeled, self.mu_neg) # shape = (n_unlabeled, n_components_neg)
        nearest_neg_center = distances_to_neg_centers.argmin(1)
        smallest_neg_distance = distances_to_neg_centers[np.arange(self.n_unlabeled), nearest_neg_center]
        pos_mask = smallest_pos_distance < smallest_neg_distance
        self.alpha = np.sum(pos_mask) / self.n_unlabeled
        self.w_pos = np.zeros((self.n_components_pos, 1))
        self.cov_pos = np.zeros((self.n_components_pos, self.n_features, self.n_features))
        self.cov_neg = np.zeros((self.n_components_neg, self.n_features, self.n_features))
        for k,c in zip(*np.unique(nearest_pos_center[pos_mask], return_counts=True)):
            component_unlabeled = self.X_unlabeled[pos_mask & (nearest_pos_center == k)]
            centered = component_unlabeled - self.mu_pos[k]
            self.cov_pos[k] = centered.T @ centered / len(component_unlabeled)
            self.w_pos[k] = c / np.sum(pos_mask)
        self.w_neg = np.zeros((self.n_components_neg,1))
        for k,c in zip(*np.unique(nearest_neg_center[~pos_mask], return_counts=True)):
            component_unlabeled = self.X_unlabeled[~pos_mask & (nearest_neg_center == k)]
            centered = component_unlabeled - self.mu_neg[k]
            self.cov_neg[k] = centered.T @ centered / len(component_unlabeled)
            self.w_neg[k] = c / np.sum(~pos_mask)
        
    def log_params(self):
        print(f"alpha: {self.alpha},\n")

        print(f"mu_pos: {self.mu_pos},\n")
        print(f"cov_pos: {self.cov_pos},\n")
        print(f"w_pos: {self.w_pos},\n")
        print(f"v_pos: {self.v_pos},\n")

        print(f"mu_neg: {self.mu_neg},\n")
        print(f"cov_neg: {self.cov_neg},\n")
        print(f"w_neg: {self.w_neg},\n")
        print(f"v_neg: {self.v_neg},\n")

    def fit(self, X_labeled_pos, X_labeled_neg,X_unlabeled,**kwargs):
        """
        Optional Arguments
        ------------------
        n_components_pos : int, default=2
            Number of gaussians in the pos mixture distribution

        n_components_neg : int, defaulta=2
            Number of gaussians in the neg mixture distribution

        scores_unlabeled : array-like, shape (n_unlabeled,), default=None
            probability of pathogenicity scores for the unlabeled data to be optionally used for weighted EM

        scores_labeled_pos : array-like, shape (n_labeled_pos,), default=None
            probability of pathogenicity scores for the positive labeled data to be optionally used for weighted EM
        
        scores_labeled_neg : array-like, shape (n_labeled_neg,), default=None
            probability of pathogenicity scores for the negative labeled data to be optionally used for weighted EM

        weighting_scheme : str in \{instance, sample}, default='instance'
            whether to weigh each instance or each sample equally in the EM algorithm
        """
        self.X_unlabeled = X_unlabeled
        self.X_labeled_pos = X_labeled_pos
        self.X_labeled_neg = X_labeled_neg
        if len(X_labeled_pos.shape) == 1:
            self.X_labeled_pos = self.X_labeled_pos[...,None]
        if len(X_labeled_neg.shape) == 1:
            self.X_labeled_neg = self.X_labeled_neg[...,None]
        if len(X_unlabeled.shape) == 1:
            self.X_unlabeled = self.X_unlabeled[...,None]
        self.n_unlabeled, self.n_features = X_unlabeled.shape
        self.n_labeled_pos = X_labeled_pos.shape[0]
        self.n_labeled_neg = X_labeled_neg.shape[0]

        self.n_components_pos = kwargs.get('n_components_pos', 2)
        self.n_components_neg = kwargs.get('n_components_neg', 2)
        self.scores_unlabeled = kwargs.get("scores_unlabeled", None)
        self.scores_labeled_pos = kwargs.get("scores_labeled_pos", np.ones((self.n_labeled_pos,1)))
        self.scores_labeled_neg = kwargs.get("scores_labeled_neg", np.zeros((self.n_labeled_neg,1)))
        self.weighting_scheme = kwargs.get("weighting_scheme", "instance")
        if self.weighting_scheme not in ["instance", "sample"]:
            raise ValueError("weighting_scheme must be one of ['instance', 'sample']")
        

        self._initialize_parameters(**kwargs)

        self.log_likelihood = -np.inf
        self.converged = False
        self.step = 0
        self.start_time = time.time()
        print("Initial Parameters:\n===========================")
        self.log_params()
        self.converged = False
        while not self.converged and self.step < self.max_steps:
            self.step += 1
            updated_params = self.get_updated_params()
            # self.converged = (updated_params['log_likelihood'] - self.log_likelihood) / self.log_likelihood < self.tol
            for k,v in updated_params.items():
                setattr(self, k, v)
            print(f"Step {self.step} Parameters:\n=========================")
            self.log_params()
        self.stop_time = time.time()
        logging.info(f"EM algorithm converged after {self.step} steps in {self.stop_time - self.start_time} seconds.")

    def get_component_pdf(self,X, mu, cov):
        """Calculate the probability density function of the multivariate normal distribution
        with mean mu and covariance cov at the points X for each component."""
        n= X.shape[0]
        k = mu.shape[0]
        pdf = np.zeros((n, k))
        for i in range(k):
            pdf[:,i] = multivariate_normal.pdf(X, mean=mu[i], cov=cov[i])
        assert pdf.shape == (n, k)
        return pdf

    def get_responsibilities(self, X, pos_mixture_weight, negative_mixture_weights, score=None):
        """Calculate the responsibilities of each component for each point in X."""
        n = X.shape[0]
        if score is None:
            score = np.ones((n,1)) * self.alpha
            # score = np.concatenate((1 - score, score), axis=1)
        component_pdfs_pos = self.get_component_pdf(X, self.mu_pos, self.cov_pos)
        component_pdfs_neg = self.get_component_pdf(X, self.mu_neg, self.cov_neg)

        omega_plus = np.zeros((n, self.n_components_pos))
        omega_minus = np.zeros((n, self.n_components_neg))

        for i in range(n):
            for k in range(self.n_components_pos):
                omega_plus[i,k] = score[i] * component_pdfs_pos[i,k] * pos_mixture_weight[k]
            for k in range(self.n_components_neg):
                omega_minus[i,k] = (1 - score[i]) * component_pdfs_neg[i,k] * negative_mixture_weights[k]
            norm_i = omega_plus[i].sum() + omega_minus[i].sum()
            omega_plus[i] /= norm_i
            omega_minus[i] /= norm_i
        return omega_plus, omega_minus

    def get_updated_alpha(self, omega_plus):
        return np.sum(omega_plus) / self.n_unlabeled

    def get_updated_w(self, omega_plus, omega_minus):
        w_plus = omega_plus.sum(0) / (self.alpha * self.n_unlabeled)
        w_minus = omega_minus.sum(0) / ((1 - self.alpha) * self.n_unlabeled)
        return w_plus, w_minus
    
    def get_updated_v(self, eta_plus, eta_minus):
        return eta_plus.sum(0) / self.n_labeled_pos, eta_minus.sum(0) / self.n_labeled_neg
    
    def get_updated_mean(self, omega_plus, omega_minus, eta_plus, eta_minus):
        mu_plus = np.zeros((self.n_components_pos, self.n_features))
        mu_minus = np.zeros((self.n_components_neg, self.n_features))

        for k in range(self.n_components_pos):
            mu_plus[k] = (omega_plus[:,k][...,None] * self.X_unlabeled).sum(axis=0) + \
                            (eta_plus[:,k][...,None] * self.X_labeled_pos).sum(axis=0)
            mu_plus[k] /= (np.sum(omega_plus[:,k]) + np.sum(eta_plus[:,k]))

        for k in range(self.n_components_neg):
            mu_minus[k] = (omega_minus[:,k][...,None] * self.X_unlabeled).sum(axis=0) + \
                            (eta_minus[:,k][...,None] * self.X_labeled_neg).sum(axis=0)
            mu_minus[k] /= (np.sum(omega_minus[:,k]) + np.sum(eta_minus[:,k]))

        return mu_plus, mu_minus
    
    def get_updated_cov(self,omega_plus, omega_minus, eta_plus, eta_minus):

        cov_plus = np.zeros((self.n_components_pos, self.n_features, self.n_features))
        cov_minus = np.zeros((self.n_components_neg, self.n_features, self.n_features))

        for k in range(self.n_components_pos):
            positively_centered_unlabeled = self.X_unlabeled - self.mu_pos[k]
            centered_labeled_pos = self.X_labeled_pos - self.mu_pos[k]
            num = np.zeros((self.n_features, self.n_features))
            den = 0.0
            for i in range(self.n_unlabeled):
                num += omega_plus[i,k] * np.outer(positively_centered_unlabeled[i], positively_centered_unlabeled[i])
                den += omega_plus[i,k]
            for i in range(self.n_labeled_pos):
                num += eta_plus[i,k] * np.outer(centered_labeled_pos[i], centered_labeled_pos[i])
                den += eta_plus[i,k]
            cov_plus[k] = num / den

        for k in range(self.n_components_neg):
            negatively_centered_unlabeled = self.X_unlabeled - self.mu_neg[k]
            centered_labeled_neg = self.X_labeled_neg - self.mu_neg[k]
            num = np.zeros((self.n_features, self.n_features))
            den = 0.0
            for i in range(self.n_unlabeled):
                num += omega_minus[i,k] * np.outer(negatively_centered_unlabeled[i], negatively_centered_unlabeled[i])
                den += omega_minus[i,k]
            for i in range(self.n_labeled_neg):
                num += eta_minus[i,k] * np.outer(centered_labeled_neg[i], centered_labeled_neg[i])
                den += eta_minus[i,k]
            cov_minus[k] = num / den

        return cov_plus, cov_minus

    def get_updated_params(self):
        omega_plus, omega_minus = self.get_responsibilities(self.X_unlabeled,self.w_pos, self.w_neg)
        eta_plus, _ = self.get_responsibilities(self.X_labeled_pos,
                                                    score=self.scores_labeled_pos,
                                                    pos_mixture_weight=self.v_pos,
                                                    negative_mixture_weights=self.v_neg)
        _, eta_minus = self.get_responsibilities(self.X_labeled_neg,
                                                    score=self.scores_labeled_neg,
                                                    pos_mixture_weight=self.v_pos,
                                                    negative_mixture_weights=self.v_neg)
        new_alpha = self.get_updated_alpha(omega_plus=omega_plus)
        new_w_pos, new_w_neg = self.get_updated_w(omega_plus=omega_plus, omega_minus=omega_minus)
        new_v_pos, new_v_neg = self.get_updated_v(eta_plus=eta_plus, eta_minus=eta_minus)
        new_mu_pos, new_mu_neg = self.get_updated_mean(omega_plus=omega_plus, omega_minus=omega_minus, eta_plus=eta_plus, eta_minus=eta_minus)
        new_cov_pos, new_cov_neg = self.get_updated_cov(omega_plus=omega_plus, omega_minus=omega_minus, eta_plus=eta_plus, eta_minus=eta_minus)
        new_params = dict(alpha=new_alpha, mu_pos=new_mu_pos, cov_pos=new_cov_pos, w_pos=new_w_pos, v_pos=new_v_pos,
                            mu_neg=new_mu_neg, cov_neg=new_cov_neg, w_neg=new_w_neg, v_neg=new_v_neg)
        return new_params

    # def get_updated_params(self):
    #     if self.scores_unlabeled is None:
    #         scores_unlabeled = np.ones((self.n_unlabeled,1)) * self.alpha
    #     else:
    #         scores_unlabeled = self.scores_unlabeled
    #     print("getting unlabeled responsibilities")
    #     pos_responsibilities_unlabeled, neg_responsibilities_unlabeled = self.get_responsibilities(self.X_unlabeled,score=scores_unlabeled) # shape = (U, k+), (U, k-)
    #     print("getting pos responsibilities")
    #     pos_responsibilities_labeled_pos, neg_responsibilities_labeled_pos = self.get_responsibilities(self.X_labeled_pos,score=self.scores_labeled_pos) # shape = (L+, k+), (L+, k-)
    #     print("getting neg responsibilities")
    #     pos_responsibilities_labeled_neg, neg_responsibilities_labeled_neg = self.get_responsibilities(self.X_labeled_neg,score=self.scores_labeled_neg) # shape = (L-, k+), (L-, k-)

    #     new_alpha = (self.weights_unlabeled * pos_responsibilities_unlabeled.sum(1)[...,None]).sum() / self.weights_unlabeled.sum()
    #     assert new_alpha <= 1 and new_alpha >= 0, f"new_alpha: {new_alpha}"
    #     # new positive parameters
    #     new_mu_pos = np.zeros_like(self.mu_pos)
    #     new_cov_pos = np.zeros_like(self.cov_pos)
    #     new_w_pos = np.zeros_like(self.w_pos)
    #     new_eta_pos = np.zeros_like(self.eta_pos)
    #     # new negative parameters
    #     new_mu_neg = np.zeros_like(self.mu_neg)
    #     new_cov_neg = np.zeros_like(self.cov_neg)
    #     new_w_neg = np.zeros_like(self.w_neg)
    #     new_eta_neg = np.zeros_like(self.eta_neg)
    #     # Positive Updates
    #     for k in range(self.n_components_pos):
    #         assert self.weights_unlabeled.shape == (self.n_unlabeled,1), f"weights_unlabeled shape: {self.weights_unlabeled.shape}"
    #         assert pos_responsibilities_unlabeled[:,k][...,None].shape == (self.n_unlabeled,1), f"pos_responsibilities_unlabeled shape: {pos_responsibilities_unlabeled[:,k][...,None].shape}"
    #         assert scores_unlabeled.shape == (self.n_unlabeled,1), f"scores_unlabeled shape: {scores_unlabeled.shape}"
    #         w_u = self.weights_unlabeled * pos_responsibilities_unlabeled[:,k][...,None] * scores_unlabeled
    #         assert w_u.shape == (self.n_unlabeled,1), f"w_u shape: {w_u.shape}"
    #         w_l_pos = self.weights_pos * pos_responsibilities_labeled_pos[:,k][...,None] * self.scores_labeled_pos
    #         # logging.debug(f"w_l_pos : {w_l_pos}")
    #         assert w_l_pos.shape == (self.n_labeled_pos,1), f"w_l_pos shape: {w_l_pos.shape}"
    #         w_l_neg = self.weights_neg * neg_responsibilities_labeled_neg[:,k][...,None] * self.scores_labeled_neg
    #         assert w_l_neg.shape == (self.n_labeled_neg,1), f"w_l_neg shape: {w_l_neg.shape}"
    #         # logging.debug(f"s1 : {(w_u * self.X_unlabeled).sum().shape} s2: {(w_l_pos * self.X_labeled_pos).sum().shape} s3: {(w_l_neg * self.X_labeled_neg).sum().shape}")
    #         new_mu_pos[k] = ((w_u * self.X_unlabeled).sum(0) + (w_l_pos * self.X_labeled_pos).sum(0) + (w_l_neg * self.X_labeled_neg).sum(0)) / (np.sum(w_u) + np.sum(w_l_pos) + np.sum(w_l_neg))
    #         centered_unlabeled = self.X_unlabeled - self.mu_pos[k]
    #         centered_labeled_pos = self.X_labeled_pos - self.mu_pos[k]
    #         centered_labeled_neg = self.X_labeled_neg - self.mu_pos[k]
    #         # logging.debug(f"centered unlabeled: {centered_unlabeled.shape}, w_u : {w_u.shape}")
    #         # logging.debug(f"centered cov : {np.cov(centered_unlabeled, rowvar=False,aweights=w_u.ravel()).shape}")
    #         c1 = np.zeros((self.n_features, self.n_features))
    #         c2 = np.zeros_like(c1)
    #         c3 = np.zeros_like(c1)
    #         if w_u.sum():
    #             c1 = np.cov(centered_unlabeled, rowvar=False,aweights=w_u.ravel())
    #         if w_l_pos.sum():
    #             c2 = np.cov(centered_labeled_pos, rowvar=False,aweights=w_l_pos.ravel())
    #         if w_l_neg.sum():
    #             # logging.debug(f"w_l_neg stats: min: {w_l_neg.min()} max : {w_l_neg.max()}, sum : {w_l_neg.sum()}")
    #             c3 = np.cov(centered_labeled_neg, rowvar=False,aweights=w_l_neg.ravel())
    #         # logging.debug(f"c1: {c1}, c2: {c2}, c3: {c3}")
    #         # new_cov_pos[k] = (c1 + c2 + c3) / (np.sum(w_u) + np.sum(w_l_pos) + np.sum(w_l_neg))
            
    #         num = 0.0
    #         den = 0.0
    #         for i in range(self.n_labeled_pos):
    #             num += w_l_pos[i] * np.outer(centered_labeled_pos[i], centered_labeled_pos[i])
    #             den += w_l_pos[i]
    #         for i in range(self.n_labeled_neg):
    #             num += w_l_neg[i] * np.outer(centered_labeled_neg[i], centered_labeled_neg[i])
    #             den += w_l_neg[i]
    #         for i in range(self.n_unlabeled):
    #             num += w_u[i] * np.outer(centered_unlabeled[i], centered_unlabeled[i])
    #             den += w_u[i]
    #         new_cov_pos[k] = num / den


    #         # print(f"normalized pos responsibilities", (pos_responsibilities_unlabeled[:5,k][...,None] / pos_responsibilities_unlabeled[:5].sum(1)[...,None]), )
    #         normalized_responsibility_k = pos_responsibilities_unlabeled[:,k][...,None] / pos_responsibilities_unlabeled.sum(1)[...,None]
    #         assert normalized_responsibility_k.shape == (self.n_unlabeled,1), f"normalized_responsibility_k shape: {normalized_responsibility_k.shape}"
    #         new_w_pos[k] = (self.weights_unlabeled * normalized_responsibility_k).sum() / (self.weights_unlabeled.sum())
    #         normalized_labeled_responsibility_k = pos_responsibilities_labeled_pos[:,k][...,None] / pos_responsibilities_labeled_pos.sum(1)[...,None]
    #         new_eta_pos[k] = (self.weights_pos * normalized_labeled_responsibility_k).sum() / self.weights_pos.sum()
    #     # print("new w pos: ", new_w_pos)
    #     assert new_w_pos.shape == (self.n_components_pos,1), f"new_w_pos shape: {new_w_pos.shape}"
    #     assert np.allclose(new_w_pos.sum(),1), f"new_w_pos sum: {new_w_pos.sum()}"
    #     assert new_eta_pos.shape == (self.n_components_pos,1), f"new_eta_pos shape: {new_eta_pos.shape}"
    #     assert np.allclose(new_eta_pos.sum(),1), f"new_eta_pos sum: {new_eta_pos.sum()}"
    #     # Negative Updates
    #     for k in range(self.n_components_neg):
    #         w_u = self.weights_unlabeled * neg_responsibilities_unlabeled[:,k][...,None] * (1 - scores_unlabeled)
    #         w_l_pos = self.weights_pos * neg_responsibilities_labeled_pos[:,k][...,None] * (1 - self.scores_labeled_pos)
    #         w_l_neg = self.weights_neg * neg_responsibilities_labeled_neg[:,k][...,None] * (1 - self.scores_labeled_neg)
    #         new_mu_neg[k] = ((w_u * self.X_unlabeled).sum(0) + \
    #                             (w_l_pos * self.X_labeled_pos).sum(0) + \
    #                                 (w_l_neg * self.X_labeled_neg).sum(0)) / (np.sum(w_u) + np.sum(w_l_pos) + np.sum(w_l_neg))
    #         centered_unlabeled = self.X_unlabeled - self.mu_neg[k]
    #         centered_labeled_pos = self.X_labeled_pos - self.mu_neg[k]
    #         centered_labeled_neg = self.X_labeled_neg - self.mu_neg[k]
    #         c1 = np.zeros((self.n_features, self.n_features))
    #         c2 = np.zeros_like(c1)
    #         c3 = np.zeros_like(c1)
    #         if w_u.sum():
    #             c1 = np.cov(centered_unlabeled, rowvar=False,aweights=w_u.ravel())
    #         if w_l_pos.sum():
    #             c2 = np.cov(centered_labeled_pos, rowvar=False,aweights=w_l_pos.ravel())
    #         if w_l_neg.sum():
    #             # logging.debug(f"w_l_neg stats: min: {w_l_neg.min()} max : {w_l_neg.max()}, sum : {w_l_neg.sum()}")
    #             c3 = np.cov(centered_labeled_neg, rowvar=False,aweights=w_l_neg.ravel())
    #         # logging.debug(f"c1 : {c1} c2 : {c2} c3 : {c3}")

    #         # new_cov_neg[k] = (c1 + c2 + c3) / (np.sum(w_u) + np.sum(w_l_pos) + np.sum(w_l_neg))
            
    #         num = 0.0
    #         den = 0.0
    #         for i in range(self.n_labeled_pos):
    #             num += w_l_pos[i] * np.outer(centered_labeled_pos[i], centered_labeled_pos[i])
    #             den += w_l_pos[i]
    #         for i in range(self.n_labeled_neg):
    #             num += w_l_neg[i] * np.outer(centered_labeled_neg[i], centered_labeled_neg[i])
    #             den += w_l_neg[i]
    #         for i in range(self.n_unlabeled):
    #             num += w_u[i] * np.outer(centered_unlabeled[i], centered_unlabeled[i])
    #             den += w_u[i]
    #         new_cov_neg[k] = num / den

    #         new_w_neg[k] = (self.weights_unlabeled * (neg_responsibilities_unlabeled[:,k][...,None] / neg_responsibilities_unlabeled.sum(1)[...,None])).sum() / (self.weights_unlabeled.sum())
    #         new_eta_neg[k] = (self.weights_neg * (neg_responsibilities_labeled_neg[:,k][...,None] / neg_responsibilities_labeled_neg.sum(1)[...,None])).sum() / self.weights_neg.sum()
    #     assert np.allclose(new_w_neg.sum(),1), f"new_w_neg sum: {new_w_neg.sum()}"
    #     assert np.allclose(new_eta_neg.sum(),1), f"new_eta_neg sum: {new_eta_neg.sum()}"
    #     if self.scores_unlabeled is None:
    #         new_scores_unlabeled = np.ones(self.n_unlabeled) * self.alpha
    #     else:
    #         new_scores_unlabeled = self.scores_unlabeled
    #     for k in range(self.n_components_pos):
    #         new_cov_pos[k] = self.covariance_reconditioning(new_cov_pos[k])
    #     for k in range(self.n_components_neg):
    #         new_cov_neg[k] = self.covariance_reconditioning(new_cov_neg[k])
        
    #     new_params = dict(alpha=new_alpha, mu_pos=new_mu_pos, cov_pos=new_cov_pos, w_pos=new_w_pos, eta_pos=new_eta_pos,
    #                         mu_neg=new_mu_neg, cov_neg=new_cov_neg, w_neg=new_w_neg, eta_neg=new_eta_neg, scores_unlabeled=new_scores_unlabeled)
    #     # logging.debug(f"New Parameters Step {self.step}:\n")
    #     # for k,v in new_params.items():
    #     #     logging.debug(f"{k}: {v}")


    #     # llu = np.log(new_scores_unlabeled * w_u @ self.get_component_pdf(self.X_unlabeled, new_mu_pos, new_cov_pos)  + \
    #     #                             (1 - new_scores_unlabeled) * (1 - w_u)  @ self.get_component_pdf(self.X_unlabeled, new_mu_neg, new_cov_neg)).sum()

    #     # llp = np.log(self.scores_labeled_pos * w_l_pos @ self.get_component_pdf(self.X_labeled_pos, new_mu_pos, new_cov_pos) + \
    #     #                                     (1 - self.scores_labeled_pos) * (1 - w_l_pos) @ self.get_component_pdf(self.X_labeled_pos, new_mu_neg, new_cov_neg)).sum()
    #     # lln = np.log(self.scores_labeled_neg * w_l_neg @ self.get_component_pdf(self.X_labeled_neg, new_mu_pos, new_cov_pos) + \
    #     #                                             (1 - self.scores_labeled_neg) * (1 - w_l_neg) @ self.get_component_pdf(self.X_labeled_neg, new_mu_neg, new_cov_neg)).sum()
    #     # new_log_likelihood = llu + llp + lln
    #     return dict(alpha=new_alpha, mu_pos=new_mu_pos, cov_pos=new_cov_pos, w_pos=new_w_pos, eta_pos=new_eta_pos,
    #                     mu_neg=new_mu_neg, cov_neg=new_cov_neg, w_neg=new_w_neg, eta_neg=new_eta_neg)

    

    def covariance_reconditioning(self,cov,**kwargs):
        cond_min = kwargs.get('cond_min',1000)
        eps=kwargs.get('cond_exp',0.001)
        s_cond = np.linalg.cond(cov)

        if s_cond > cond_min:
            cov = optutils.recondition_sig(cov, cond_min, self.n_features, eps)
        return cov

if __name__ == "__main__":
    test_pnu_em()