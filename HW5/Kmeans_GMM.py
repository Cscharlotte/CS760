import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt


# kmeans
def kmeanspp(X, k):
    centroids = [X[np.random.choice(X.shape[0])]]
    for i in range(1, k):
        dist = np.array([min([np.linalg.norm(c - x) ** 2 for c in centroids]) for x in X])
        probabilities = dist / dist.sum()
        centroids.append(X[np.random.choice(X.shape[0], p=probabilities)])
    return np.array(centroids)


def kmeans(X, centroids, k):
    prev_centroids = centroids.copy()
    while True:
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        cluster = np.argmin(distances, axis=1)
        new_centroids = np.zeros((k, X.shape[1]))
        for j in range(k):
            cluster_data = X[cluster == j]
            if len(cluster_data) > 0:
                new_centroids[j] = np.mean(cluster_data, axis=0)
        if np.array_equal(new_centroids, centroids):
            break
        centroids, prev_centroids = new_centroids, centroids
    return centroids, cluster


def kmeans_objective(X, centroids, cluster):
    objective = 0
    for i in range(len(X)):
        cluster_idx = cluster[i]
        centroid = centroids[cluster_idx]
        distance = np.linalg.norm(X[i] - centroid)**2
        objective += distance
    return objective


def kmeans_accuracy(X, cluster):
    bad_guess = []
    for index, label in enumerate(cluster):
        if index < 100:
            if label != np.bincount(cluster[:100]).argmax():
                bad_guess.append((index, label, 'a'))
        elif index >= 100 and index < 200:
            if label != np.bincount(cluster[100:200]).argmax():
                bad_guess.append((index, label, 'b'))
        else:
            if label != np.bincount(cluster[200:]).argmax():
                bad_guess.append((index, label, 'c'))
    k_acc = 1 - (len(bad_guess) / len(X))
    return k_acc



# Gaussian mixture model
def initialize_gmm(X, k):
    n, m = X.shape
    phi = np.full(shape=k, fill_value=1/k)
    weights = np.full(shape=(n, k), fill_value=1/k)
    random_row = np.random.randint(low=0, high=n, size=k)
    mu = [X[row_index, :] for row_index in random_row]
    sigma = [np.cov(X.T) for _ in range(k)]
    return phi, weights, mu, sigma


def e_step(X, mu, sigma, phi):
    n, k = len(X), len(mu)
    likelihood = np.zeros((n, k))
    for i in range(k):
        distribution = multivariate_normal(mean=mu[i], cov=sigma[i])
        likelihood[:, i] = distribution.pdf(X)
    numerator = likelihood * phi
    denominator = numerator.sum(axis=1)[:, np.newaxis]
    weights = numerator / denominator
    return weights


def m_step(X, weights, mu, sigma):
    k = len(mu)
    for i in range(k):
        weight = weights[:, [i]]
        total_weight = weight.sum()
        mu[i] = (X * weight).sum(axis=0) / total_weight
        sigma[i] = np.cov(X.T, aweights=(weight/total_weight).flatten(), bias=True)
    phi = weights.mean(axis=0)
    return mu, sigma, phi


def GMM(X, k, tol=1e-6, max_iter=500):
    phi, weights, mu, sigma = initialize_gmm(X, k)
    prev_mu = np.array(mu)
    for iteration in range(max_iter):
        weights = e_step(X, mu, sigma, phi)
        mu, sigma, phi = m_step(X, weights, mu, sigma)
        if np.linalg.norm(np.array(mu) - prev_mu) < tol:
            break
        prev_mu = np.array(mu)
    return phi, mu, sigma


def gmm_objective(X, mu, sigma, phi):
    n, k = len(X), len(mu)
    likelihood = np.zeros((n, k))
    for i in range(k):
        distribution = multivariate_normal(mean=mu[i], cov=sigma[i])
        likelihood[:, i] = distribution.pdf(X)
    weighted_likelihood = likelihood * phi
    log_likelihood = np.log(weighted_likelihood.sum(axis=1))
    gmm_objective = -np.sum(log_likelihood)
    return gmm_objective


def gmm_accuracy(X, true_labels, phi, mu, sigma):
    n, k = len(X), len(mu)
    weights = e_step(X, mu, sigma, phi)
    cluster = np.argmax(weights, axis=1) + 1
    cluster_to_label_mapping = {}
    for i in range(k):
        distribution = multivariate_normal(mean=mu[i], cov=sigma[i])
        cluster_mean = distribution.mean
        closest_label = min(set(true_labels),
                            key=lambda label: np.linalg.norm(cluster_mean - np.mean(X[true_labels == label], axis=0)))
        cluster_to_label_mapping[i + 1] = closest_label
    predicted_labels = np.array([cluster_to_label_mapping[cluster_id] for cluster_id in cluster])
    accuracy = np.mean(predicted_labels == true_labels)
    return accuracy


mean_a = [-1, -1]
cov_a = [[2, 0.5], [0.5, 1]]
mean_b = [1, -1]
cov_b = [[1, -0.5], [-0.5, 2]]
mean_c = [0, 1]
cov_c = [[1, 0], [0, 2]]
alpha = [0.5, 1, 2, 4, 8]
kmeans_objs = []
kmeans_acc = []
gmm_objs = []
gmm_acc = []
for a in alpha:
    np.random.seed(760)
    sample_a = np.random.multivariate_normal(mean_a, np.dot(a,cov_a), 100)
    sample_b = np.random.multivariate_normal(mean_b, np.dot(a,cov_b), 100)
    sample_c = np.random.multivariate_normal(mean_c, np.dot(a,cov_c), 100)
    sample = np.concatenate((sample_a, sample_b, sample_c))
    true_labels = np.array(['a'] * 100 + ['b'] * 100 + ['c'] * 100)
    centroids, cluster = kmeans(sample, kmeanspp(sample, 3), 3)
    kmeans_objs.append(kmeans_objective(sample, centroids, cluster))
    kmeans_acc.append(kmeans_accuracy(sample, cluster))
    phi, mu, sigma = GMM(sample, 3, tol=1e-6, max_iter=500)
    gmm_objs.append(gmm_objective(sample, mu, sigma, phi))
    gmm_acc.append(gmm_accuracy(sample, true_labels, phi, mu, sigma))


plt.plot(alpha, kmeans_objs, label='K-means objective', marker='o')
plt.plot(alpha, gmm_objs, label='GMM objective', marker='o')
plt.xlabel('Alpha')
plt.ylabel('Clustering objective')
plt.legend()
plt.show()


plt.plot(alpha, kmeans_acc, label='K-means accuracy', marker='o')
plt.plot(alpha, gmm_acc, label='GMM accuracy', marker='o')
plt.xlabel('Alpha')
plt.ylabel('Clustering accuracy')
plt.legend()
plt.show()
