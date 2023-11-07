import pandas as pd
from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt


def load_dataset(filename):
    sample_data = pd.read_csv(filename, header=None)
    X = np.array(sample_data)
    return X


def buggy_pca(X, d):
    U, D, VT = np.linalg.svd(X)
    pc = VT.T[:, :d]
    data_pca = np.dot(X, pc)
    data_reconstructed = np.dot(data_pca, pc.T)
    error = np.mean(np.sum((X - data_reconstructed)**2, axis=1))
    return data_pca, pc, data_reconstructed, error


def demeaned_pca(X, d):
    mean = np.mean(X, axis=0)
    X_demeaned = X - mean
    U, D, VT = np.linalg.svd(X_demeaned)
    pc = VT.T[:, :d]
    data_pca = np.dot(X_demeaned, pc)
    data_reconstructed = np.dot(data_pca, pc.T) + mean
    error = np.mean(np.sum((X - data_reconstructed)**2, axis=1))
    return data_pca, pc, data_reconstructed, error


def normalized_pca(X, d):
    mean = np.mean(X, axis=0)
    std_dev = np.std(X, axis=0)
    X_normalized = (X-mean) / std_dev
    U, D, VT = np.linalg.svd(X_normalized)
    pc = VT.T[:, :d]
    data_pca = np.dot(X_normalized, pc)
    data_reconstructed = np.dot(data_pca, pc.T) * std_dev + mean
    error = np.mean(np.sum((X - data_reconstructed)**2, axis=1))
    return data_pca, pc, data_reconstructed, error


def DRO(X, d):
    n, D = X.shape
    mean = np.mean(X, axis=0)
    X_demeaned = X - mean
    U, D, VT = np.linalg.svd(X_demeaned, full_matrices=False)
    A = VT[:d].T
    Z = np.dot(X, A)
    Z_mean = np.mean(Z, axis=0)
    Z_demeaned = Z - Z_mean
    b = mean - np.dot(Z_mean, A.T)
    data_reconstructed = np.dot(Z_demeaned, A.T) + mean
    error = np.mean(np.sum((X - data_reconstructed) ** 2, axis=1))
    return Z, A, b, data_reconstructed, error


def plot_DR(original_data, reconstructed_data):
    plt.figure(figsize=(8, 6))
    plt.scatter(original_data[:, 0], original_data[:, 1], marker='o', facecolors='none', edgecolors='b', alpha=0.7)
    plt.scatter(reconstructed_data[:, 0], reconstructed_data[:, 1], c='r', marker='x',alpha=0.7)
    plt.title('DRO')
    plt.xlim(0, 10)
    plt.ylim(0, 10)
    plt.grid()
    plt.show()


file = 'data\\data2D.csv'
file = 'data\\data1000D.csv'

n, D = X.shape
mean = np.mean(X, axis=0)
X_demeaned = X - mean
U, D, VT = np.linalg.svd(X_demeaned, full_matrices=False)
plt.plot(D)
plt.grid()
plt.savefig('knee point choice.pdf')
plt.show()

X = load_dataset(file)
pca_buggy, pc_buggy, reconstructed_buggy, error_buggy = buggy_pca(X,30)
plot_DR(X, reconstructed_buggy)
pca_demeaned, pc_demeaned, reconstructed_demeaned, error_demeaned = demeaned_pca(X,30)
plot_DR(X, reconstructed_demeaned)
pca_normalized, pc_normalized, reconstructed_normalized, error_normalized = normalized_pca(X,30)
plot_DR(X, reconstructed_normalized)
Z, A, b, reconstructed_DRO, error_DRO = DRO(X,30)
plot_DR(X, reconstructed_DRO)
