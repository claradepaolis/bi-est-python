# Bias in Semi-Supervised Setting
This is a python implementation for the Expectation-Maximization optimization for parameters of a nested mixture of Gaussians 
(a mixture of Gaussian mixtures) presented in our paper [An Approach to Identifying and Quantifying Bias in Biomedical Data](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9782737/). 
If labeled data for for the classes are avialable, they are used in the optimization.

The notebooks [demo.ipynb](demo.ipynb) and [demo-2D.ipynb](demo-2D.ipynb) have examples for learning the parameters in 1 and 2 dimensions.



## Background 
Often data features might be available, but ground truth labels for the data may be insufficient. 
Thus, the semi-supervised setting, where most data do not have accompanying labels, is common in many applications. 

In addition to working in the domain of semi-supervision, the labeled data that is available may not be representative of 
the true underlying population of the classes represented by the labels.

**Observed distributions ≠ True distributions**

<img width="1109" alt="image" src="https://github.com/claradepaolis/bi-est-python/assets/19443235/566d3e18-85ea-46c4-b3e0-d3fb5fb54311">
We’re interested in detecting and quantifying the level of bias in these labeled data in the semi-supervised learning setting.

More precisely, let’s consider a classification problem where we are interested in two classes, let’s call them the positive and the negative classes. We may have a large trove unlabeled data, represented on the left by the black distribution, but we don’t know the underlying class labels of that data. 

Ideally, we would have labels for each class that are drawn from the same underlying population, but this may not the be case.

Instead, what is available to us is some sample of labeled data, but due to feasibility of labeling, sampling bias, the availability of only some subpopulations, or other mechanisms, the distribution within each label class is not representative of the corresponding class distribution. 

We call this systematic discrepancy between the labeled and unlabeled class distributions the bias in the labels. And we are interested in detecting and quantifying this bias without knowing the true unlabeled class distributions. Although this scenario is common, perhaps especially in biomedical data, it has rarely been measured.

## Modeling Bias
Quantifying the disagreement between true class distribution and those in the labeled set in general is difficult because the parameters of both the labeled and unlabeled distributions are unknown.

We must make some assumptions in order to jointly model all of the unknowns in the problem. 

We assume that each true underlying class distribution can be represented as a mixture of K Gaussians
And the corresponding labeled class distribution can be represented as a mixture of those same K Gaussians but with different mixing proportions. 

<img width="873" alt="image" src="https://github.com/claradepaolis/bi-est-python/assets/19443235/51e5a141-05ce-46a5-b78f-71da5afa9a68">


