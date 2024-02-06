import numpy as np
from collections import namedtuple
import seaborn as sns
import matplotlib.pyplot as plt

from bi_est_python.pnu_em import PNU_EM
from bi_est_python.empnu import PU_nested_em_opt
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
    em = PNU_EM(n_components=2,max_steps=500)
    em.fit(labeled_pos, labeled_neg, unlabeled_data)
    return em

def test_empnu():
    labeled_pos, labeled_neg, unlabeled_data, unlabeled, pos_bias, neg_bias = get_data()
    alpha, w, w_labeled, sg, mu, lls = PU_nested_em_opt(unlabeled_data, labeled_pos, labeled_neg, 2)
    # print(alpha, w, w_labeled, sg, mu, lls)
    print(f"alpha: {alpha}")
    print(f"w: {w}")
    print(f"w_labeled: {w_labeled}")
    print(f"sg: {sg}")
    print(f"mu: {mu}")
    # print(f"lls: {lls}")


def plot_data(pos_bias, neg_bias, unlabeled):
    # plotting

    xs=np.arange(-10,15,0.1)
    sns.lineplot(x=xs, y=pos_bias.points_pdf(xs), linewidth=2, label='Positive')
    sns.lineplot(x=xs, y=neg_bias.points_pdf(xs), linewidth=2, label='Negative')

    sns.lineplot(x=xs, y=unlabeled.points_pdf(xs), linewidth=2, label='Unlabeled')

    sns.lineplot(x=xs, y=unlabeled.alpha*unlabeled.pos.points_pdf(xs), 
                linewidth=2, linestyle=':', label='true positive in unlabeled')
    sns.lineplot(x=xs, y=(1-unlabeled.alpha)*unlabeled.neg.points_pdf(xs), 
                linewidth=2, linestyle=':', label='true negative in unlabeled')
    return plt.gcf()

if __name__ == "__main__":
    test_pnu_em()
    # test_empnu()