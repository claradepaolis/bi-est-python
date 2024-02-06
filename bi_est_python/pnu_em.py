import time
import copy
import numpy as np
from scipy.linalg import sqrtm
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from scipy.stats import multivariate_normal
import logging
logging.basicConfig(level=logging.DEBUG)

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
        print(f"alpha: {np.round(self.alpha, 4)},\n")

        print(f"mu_pos: {np.round(self.mu_pos, 4)},\n")
        print(f"cov_pos: {np.round(self.cov_pos, 4)},\n")
        print(f"w_pos: {np.round(self.w_pos, 4)},\n")
        print(f"v_pos: {np.round(self.v_pos, 4)},\n")

        print(f"mu_neg: {np.round(self.mu_neg, 4)},\n")
        print(f"cov_neg: {np.round(self.cov_neg, 4)},\n")
        print(f"w_neg: {np.round(self.w_neg, 4)},\n")
        print(f"v_neg: {np.round(self.v_neg, 4)},\n")

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
        self.original_scores_unlabeled = kwargs.get("scores_unlabeled", None)
        if self.original_scores_unlabeled is not None:
            self.original_scores_unlabeled = self.original_scores_unlabeled.reshape((-1,1))
        self.scores_labeled_pos = kwargs.get("scores_labeled_pos", np.ones((self.n_labeled_pos,1))).reshape((-1,1))
        self.scores_labeled_neg = kwargs.get("scores_labeled_neg", np.zeros((self.n_labeled_neg,1))).reshape((-1,1))
        self.weighting_scheme = kwargs.get("weighting_scheme", "instance")
        if self.weighting_scheme not in ["instance", "sample"]:
            raise ValueError("weighting_scheme must be one of ['instance', 'sample']")
        

        self._initialize_parameters(**kwargs)

        self.log_likelihood = -1 * 1e10
        self.converged = False
        self.step = 0
        self.start_time = time.time()
        print("Initial Parameters:\n===========================")
        self.log_params()
        self.converged = False
        while not self.converged and self.step < self.max_steps:
            if self.original_scores_unlabeled is not None:
                self.scores_unlabeled = self.original_scores_unlabeled
            else:
                self.scores_unlabeled = np.ones((self.n_unlabeled,1)) * self.alpha
            self.step += 1
            updated_params = self.get_updated_params()
            # self.converged = (updated_params['log_likelihood'] - self.log_likelihood) / self.log_likelihood < self.tol
            for k,v in updated_params.items():
                setattr(self, k, v)
            log_likelihood = self.get_log_likelihood()
            print(f"Step {self.step} log-likelihood: {log_likelihood}\n=========================")
            if not self.step % 10:
                print(f"Step {self.step} Params\n=========================")
                self.log_params()
            if (log_likelihood - self.log_likelihood) / self.log_likelihood < self.tol:
                self.converged = True
            self.log_likelihood = log_likelihood
        self.stop_time = time.time()
        logging.info(f"EM algorithm converged after {self.step} steps in {self.stop_time - self.start_time} seconds.")
        self.log_params()

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
        component_pdfs_pos = self.get_component_pdf(X, self.mu_pos, self.cov_pos)
        component_pdfs_neg = self.get_component_pdf(X, self.mu_neg, self.cov_neg)

        positive_responsibilities = np.zeros((n, self.n_components_pos))
        negative_responsibilities = np.zeros((n, self.n_components_neg))

        for i in range(n):
            for k in range(self.n_components_pos):
                positive_responsibilities[i,k] = score[i] * component_pdfs_pos[i,k] * pos_mixture_weight[k]
            for k in range(self.n_components_neg):
                negative_responsibilities[i,k] = (1 - score[i]) * component_pdfs_neg[i,k] * negative_mixture_weights[k]
            norm_i = positive_responsibilities[i].sum() + negative_responsibilities[i].sum()
            positive_responsibilities[i] /= norm_i
            negative_responsibilities[i] /= norm_i
        return positive_responsibilities, negative_responsibilities

    def get_updated_alpha(self, omega_plus):
        return (np.sum(self.weights_unlabeled * omega_plus)) / self.weights_unlabeled.sum()

    def get_updated_w(self, omega_plus, omega_minus):
        w_plus = (omega_plus.T @ (self.scores_unlabeled * self.weights_unlabeled)) / (self.alpha * (self.scores_unlabeled * self.weights_unlabeled).sum())
        w_minus = (omega_minus.T @ ((1 - self.scores_unlabeled) * self.weights_unlabeled)) / ((1 - self.alpha) * ((1 - self.scores_unlabeled) * self.weights_unlabeled).sum())
        return w_plus, w_minus
    
    def get_updated_v(self, eta_plus, eta_minus):
        return eta_plus.T @ (self.scores_labeled_pos * self.weights_pos) / (self.scores_labeled_pos * self.weights_pos).sum(), eta_minus.T @ ((1 - self.scores_labeled_neg) * self.weights_neg) / ((1 - self.scores_labeled_neg) * self.weights_neg).sum()

    def get_updated_mean(self, omega_plus, omega_minus, eta_plus, positive_eta_minus, negative_eta_plus, eta_minus):
        mu_plus = np.zeros((self.n_components_pos, self.n_features))
        mu_minus = np.zeros((self.n_components_neg, self.n_features))

        for k in range(self.n_components_pos):
            mu_plus[k] = ((self.weights_unlabeled * self.scores_unlabeled * omega_plus[:,k][...,None]) * self.X_unlabeled).sum(axis=0) + \
                            ((self.weights_pos * self.scores_labeled_pos * eta_plus[:,k][...,None]) * self.X_labeled_pos).sum(axis=0) + \
                                ((self.weights_neg * self.scores_labeled_neg * negative_eta_plus[:,k][...,None]) * self.X_labeled_neg).sum(axis=0)
            mu_plus[k] /= (np.sum(self.weights_unlabeled * self.scores_unlabeled * omega_plus[:,k][...,None]) + \
                            np.sum(self.weights_pos * self.scores_labeled_pos * eta_plus[:,k][...,None]) + \
                                np.sum(self.weights_neg * self.scores_labeled_neg * negative_eta_plus[:,k][...,None]))

        for k in range(self.n_components_neg):
            mu_minus[k] = ((self.weights_unlabeled * (1 - self.scores_unlabeled) * omega_minus[:,k][...,None]) * self.X_unlabeled).sum(axis=0) + \
                            (self.weights_neg * (1 - self.scores_labeled_neg) * eta_minus[:,k][...,None] * self.X_labeled_neg).sum(axis=0) + \
                                (self.weights_pos * (1 - self.scores_labeled_pos) * positive_eta_minus[:,k][...,None] * self.X_labeled_pos).sum(axis=0)
            mu_minus[k] /= (np.sum(self.weights_unlabeled * (1 - self.scores_unlabeled) * omega_minus[:,k][...,None]) + \
                            np.sum(self.weights_neg * (1 - self.scores_labeled_neg) * eta_minus[:,k][...,None]) + \
                                np.sum(self.weights_pos * (1 - self.scores_labeled_pos) * positive_eta_minus[:,k][...,None]))

        return mu_plus, mu_minus
    
    def get_updated_cov(self,omega_plus, omega_minus, eta_plus, positive_eta_minus, negative_eta_plus, eta_minus):

        cov_plus = np.zeros((self.n_components_pos, self.n_features, self.n_features))
        cov_minus = np.zeros((self.n_components_neg, self.n_features, self.n_features))

        for k in range(self.n_components_pos):
            positively_centered_unlabeled = self.X_unlabeled - self.mu_pos[k]
            centered_labeled_pos = self.X_labeled_pos - self.mu_pos[k]
            centered_labeled_neg = self.X_labeled_neg - self.mu_pos[k]
            num = np.zeros((self.n_features, self.n_features))
            den = 0.0
            for i in range(self.n_unlabeled):
                num += (self.scores_unlabeled[i] * self.weights_unlabeled[i] * omega_plus[i,k]) * np.outer(positively_centered_unlabeled[i], positively_centered_unlabeled[i])
                den += (self.scores_unlabeled[i] * self.weights_unlabeled[i] * omega_plus[i,k])
            for i in range(self.n_labeled_pos):
                num += (self.scores_labeled_pos[i] * self.weights_pos[i] * eta_plus[i,k]) * np.outer(centered_labeled_pos[i], centered_labeled_pos[i])
                den += (self.scores_labeled_pos[i] * self.weights_pos[i] * eta_plus[i,k])
            for i in range(self.n_labeled_neg):
                num += (self.scores_labeled_neg[i] * self.weights_neg[i] * negative_eta_plus[i,k]) * np.outer(centered_labeled_neg[i], centered_labeled_neg[i])
                den += (self.scores_labeled_neg[i] * self.weights_neg[i] * negative_eta_plus[i,k])
            cov_plus[k] = num / den

        for k in range(self.n_components_neg):
            negatively_centered_unlabeled = self.X_unlabeled - self.mu_neg[k]
            centered_labeled_neg = self.X_labeled_neg - self.mu_neg[k]
            centered_labeled_pos = self.X_labeled_pos - self.mu_neg[k]
            num = np.zeros((self.n_features, self.n_features))
            den = 0.0
            for i in range(self.n_unlabeled):
                num += ((1 - self.scores_unlabeled[i]) * self.weights_unlabeled[i] * omega_minus[i,k]) * np.outer(negatively_centered_unlabeled[i], negatively_centered_unlabeled[i])
                den += ((1 - self.scores_unlabeled[i]) * self.weights_unlabeled[i] * omega_minus[i,k])
            for i in range(self.n_labeled_neg):
                num += ((1 - self.scores_labeled_neg[i]) * self.weights_neg[i] * eta_minus[i,k]) * np.outer(centered_labeled_neg[i], centered_labeled_neg[i])
                den += ((1 - self.scores_labeled_neg[i]) * self.weights_neg[i] * eta_minus[i,k])
            for i in range(self.n_labeled_pos):
                num += ((1 - self.scores_labeled_pos[i]) * self.weights_pos[i] * positive_eta_minus[i,k]) * np.outer(centered_labeled_pos[i], centered_labeled_pos[i])
                den += ((1 - self.scores_labeled_pos[i]) * self.weights_pos[i] * positive_eta_minus[i,k])
            cov_minus[k] = num / den

        return cov_plus, cov_minus

    def get_updated_params(self):
        omega_plus, omega_minus = self.get_responsibilities(self.X_unlabeled,self.w_pos, self.w_neg)
        eta_plus, labeled_pos_eta_minus = self.get_responsibilities(self.X_labeled_pos,
                                                    score=self.scores_labeled_pos,
                                                    pos_mixture_weight=self.v_pos,
                                                    negative_mixture_weights=self.v_neg)
        labeled_neg_eta_plus, eta_minus = self.get_responsibilities(self.X_labeled_neg,
                                                    score=self.scores_labeled_neg,
                                                    pos_mixture_weight=self.v_pos,
                                                    negative_mixture_weights=self.v_neg)
        new_alpha = self.get_updated_alpha(omega_plus=omega_plus)
        new_w_pos, new_w_neg = self.get_updated_w(omega_plus=omega_plus, omega_minus=omega_minus)
        new_v_pos, new_v_neg = self.get_updated_v(eta_plus=eta_plus, eta_minus=eta_minus)
        new_mu_pos, new_mu_neg = self.get_updated_mean(omega_plus=omega_plus, omega_minus=omega_minus,
                                                        eta_plus=eta_plus, positive_eta_minus=labeled_pos_eta_minus,
                                                        negative_eta_plus=labeled_neg_eta_plus, eta_minus=eta_minus)
        new_cov_pos, new_cov_neg = self.get_updated_cov(omega_plus=omega_plus, omega_minus=omega_minus,
                                                        eta_plus=eta_plus, positive_eta_minus=labeled_pos_eta_minus,
                                                        negative_eta_plus=labeled_neg_eta_plus, eta_minus=eta_minus)
        new_params = dict(alpha=new_alpha, mu_pos=new_mu_pos, cov_pos=new_cov_pos, w_pos=new_w_pos, v_pos=new_v_pos,
                            mu_neg=new_mu_neg, cov_neg=new_cov_neg, w_neg=new_w_neg, v_neg=new_v_neg)
        return new_params

    def get_log_likelihood(self,):
        labeled_pos_log_likelihood = np.log(self.get_component_pdf(self.X_labeled_pos, self.mu_pos, self.cov_pos).sum(1)).sum()
        labeled_neg_log_likelihood = np.log(self.get_component_pdf(self.X_labeled_neg, self.mu_neg, self.cov_neg).sum(1)).sum()
        unlabeled_pos_likelihoods = self.get_component_pdf(self.X_unlabeled, self.mu_pos, self.cov_pos).sum(1)
        unlabeled_neg_likelihoods = self.get_component_pdf(self.X_unlabeled, self.mu_neg, self.cov_neg).sum(1)
        unlabeled_log_likelihood = np.log(self.alpha * unlabeled_pos_likelihoods + (1 - self.alpha) * unlabeled_neg_likelihoods).sum()

        return labeled_pos_log_likelihood + labeled_neg_log_likelihood + unlabeled_log_likelihood


if __name__ == "__main__":
    test_pnu_em()