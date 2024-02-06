import scipy
import numpy as np
from sklearn import metrics
from scipy.special import rel_entr


def auc(mix1, mix2, alpha=0.5, N=1e5):
    "AUC between two distributions that are mixtures of Gaussians (Mixture objects)"
    
    n_comps1 = mix1.num_comps
    n_comps2 = mix2.num_comps
    
    N1 = int(alpha*N)
    N2 = int((1-alpha)*N)
    if n_comps1 > 1:
        N1k = np.round(N1*mix1.ps).astype(int)
    else:
        N1k = np.array([N1])
    if n_comps2 > 1:
        N2k = np.round(N2*mix2.ps).astype(int)
    else:
        N2k = np.array([N2])

    points1 = mix1.sample_points(N1)
    points2 = mix2.sample_points(N2)
    
    true_labels = np.vstack([np.ones([points1.shape[0],1]), np.zeros([points2.shape[0],1])])
    
    X = np.concatenate((points1, points2), axis=0)   
    
    pdf1 = mix1.points_pdf(X) 
    pdf2 = mix2.points_pdf(X)

    posteriors = 0.5*pdf1/(0.5*pdf1 + 0.5*pdf2)
    roc_auc = metrics.roc_auc_score(true_labels, posteriors)
    
    return roc_auc


def kl(mix1, mix2, alpha=0.5, N=1e5):
    "KL div between two distributions that are mixtures of Gaussians (Mixture objects)"
    
    n_comps1 = mix1.num_comps
    n_comps2 = mix2.num_comps
    
    N1 = int(alpha*N)
    N2 = int((1-alpha)*N)
    if n_comps1 > 1:
        N1k = np.round(N1*mix1.ps).astype(int)
    else:
        N1k = np.array([N1])
    if n_comps2 > 1:
        N2k = np.round(N2*mix2.ps).astype(int)
    else:
        N2k = np.array([N2])

    points1 = mix1.sample_points(N1)
    points2 = mix2.sample_points(N2)
    
    #KL(P || Q)=
    return sum(rel_entr(points1, points2))
    

def distrib_overlap(mix1, mix2, p=1, normalized=True, N=1e5):
    
    def modeldiff(x, modelA, modelB):
        # evaluate pdfs of both models on the same points x
        p = mix1.points_pdf(x)
        q = mix2.points_pdf(x)

        pq = p-q

        t1 = np.array(pq)
        t1[t1<0] = 0
        t2 = np.array(-pq)
        t2[t2<0] = 0
        t3 = np.maximum(p,q)
        return t1, t2, t3, p, q
    
    pointsA = mix1.sample_points(N)
    pointsB = mix2.sample_points(N)
    
    (evalA_t1, evalA_t2, evalA_t3, evalA_p, evalA_q) = modeldiff(pointsA, mix1, mix2)
    (evalB_t1, evalB_t2, evalB_t3, evalB_p, evalB_q) = modeldiff(pointsB, mix1, mix2)
    
    # Estimate the integrals in the numerator via monte-carlo
    # (p-q)_+ term  is better approximated on a sample from p
    num1 = sum(evalA_t1/evalA_p)/N;
    #(q-p)_+ term is better approximated on a sample from q
    num2 = sum(evalB_t2/evalB_q)/N;
    # Estimate the denominator via monte-carlo using a mixture sample from p and q
    m1 = (0.5*evalA_p) + (0.5*evalA_q)
    m2 = (0.5*evalB_p) + (0.5*evalB_q)
    denom = (sum(evalA_t3/m1) + sum(evalB_t3/m2))/(2*N)
    
    if normalized:  
        # normalized distance
        d =(((num1**p) + (num2**p))**(1/p))/denom
    else:
        d = (((num1**p) + (num2**p))**(1/p))/2
    
    return d
