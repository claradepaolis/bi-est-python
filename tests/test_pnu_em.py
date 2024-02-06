import numpy as np
from collections import namedtuple
import seaborn as sns
import matplotlib.pyplot as plt

from bi_est_python.pnu_em import PNU_EM
from bi_est_python.Mixture import NMixture, PUMixture

def get_data():
    # Create a dummy dataset
    rng = np.random.default_rng()
    num_components = 2
    dim = 2
    N_labeled_pos = N_labeled_neg = 5000
    N_unlabeled = 10000
    Params = namedtuple('params', ['mu_pos', 'mu_neg', 'sigma', 'w_pos', 'w_neg', 'alpha', 'v_pos', 'v_neg'])
    params = Params(alpha = 0.3,
                    mu_pos = np.array([[-4] * dim,
                                        [-8] * dim]),
                    mu_neg = np.array([[4] * dim,
                                        [8] * dim]),
                    sigma = np.tile(np.eye(dim), (num_components,1,1)),
                    w_pos = np.array([0.75, 0.25]),
                    v_pos=np.array([0.2, 0.8]),
                    w_neg = np.array([0.35, 0.65]),
                    v_neg=np.array([0.5, 0.5]))

    # positive Gaussian mixtures

    pos = NMixture(mu=params.mu_pos, sigma=params.sigma, 
                ps=params.w_pos)
    neg = NMixture(mu=params.mu_neg, sigma=params.sigma, 
                ps=params.w_neg)

    unlabeled = PUMixture(pos,neg, alpha=params.alpha)

    pos_bias = NMixture(mu=params.mu_pos, sigma=params.sigma, 
                        ps=params.v_pos)
    neg_bias = NMixture(mu=params.mu_neg, sigma=params.sigma,
                        ps=params.v_neg)

    # Sample points from distributions
    labeled_pos = pos_bias.sample_points(N_labeled_pos)
    labeled_neg = neg_bias.sample_points(N_labeled_neg)
    unlabeled_data = unlabeled.sample_points(N_unlabeled)
    return labeled_pos, labeled_neg, unlabeled_data, unlabeled, pos_bias, neg_bias

def test_pnu_em():
    labeled_pos, labeled_neg, unlabeled_data, unlabeled, pos_bias, neg_bias = get_data()
    noisy_labeled_pos = np.concatenate((labeled_pos, labeled_neg[:3]))
    scores_labeled_pos = np.concatenate((np.ones(len(labeled_pos)), np.zeros(3)))
    noisy_labeled_neg = np.concatenate((labeled_neg, labeled_pos[:3]))
    scores_labeled_neg = np.concatenate((np.zeros(len(labeled_neg)), np.ones(3)))
    em = PNU_EM(n_components=2,max_steps=500)
    em.fit(noisy_labeled_pos, noisy_labeled_neg, unlabeled_data,scores_labeled_pos=scores_labeled_pos, scores_labeled_neg=scores_labeled_neg)
    return em

if __name__ == "__main__":
    test_pnu_em()