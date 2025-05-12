import numpy as np

def map_adapt(gmm, features, max_iterations=10, likelihood_threshold=1e-20, relevance_factor=16):
    N = features.shape[0]
    D = features.shape[1]
    K = gmm.n_components  # Use n_components instead of C

    mu_new = np.zeros((K, D))
    n_k = np.zeros((K, 1))
    mu_k = gmm.means_  # Access means from sklearn's GaussianMixture
    cov_k = gmm.covariances_  # Access covariances from sklearn's GaussianMixture
    pi_k = gmm.weights_  # Access mixing coefficients from sklearn's GaussianMixture

    old_likelihood = gmm.score(features)
    new_likelihood = 0
    iterations = 0

    while (abs(old_likelihood - new_likelihood) > likelihood_threshold and iterations < max_iterations):
        iterations += 1
        old_likelihood = new_likelihood
        
        z_n_k = gmm.predict_proba(features)  # Get responsibilities
        n_k = np.sum(z_n_k, axis=0)

        for i in range(K):
            temp = np.zeros((1, D))
            for n in range(N):
                temp += z_n_k[n][i] * features[n, :]
            mu_new[i] = (1 / n_k[i]) * temp if n_k[i] > 0 else mu_k[i]

        adaptation_coefficient = n_k / (n_k + relevance_factor)

        for k in range(K):
            mu_k[k] = (adaptation_coefficient[k] * mu_new[k]) + ((1 - adaptation_coefficient[k]) * mu_k[k])

        gmm.means_ = mu_k  # Update means in the GMM

        log_likelihood = gmm.score(features)  # Calculate new log likelihood
        new_likelihood = log_likelihood

    return gmm