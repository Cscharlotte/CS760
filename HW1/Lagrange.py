import numpy as np
import math
from scipy.interpolate import lagrange
import pandas as pd
import seaborn as sns

np.random.seed(19)
X_train = np.random.uniform(low=-5, high=5, size=100)
y_train = np.array([math.sin(x) for x in X_train])
X_test = np.random.uniform(low=-5, high=5, size=30)
y_test = np.array([math.sin(x) for x in X_test])
poly = lagrange(X_train, y_train)
y_pred_train = poly(X_train)
y_pred_test = poly(X_test)
lmse_train = np.log(np.mean((y_train - y_pred_train) ** 2))
lmse_test = np.log(np.mean((y_test - y_pred_test) ** 2))

train_err = []
test_err = []
sigmas = list(np.arange(0,20,0.5))
for sigma in range(len(sigmas)):
    np.random.seed(19)
    epsilon = np.random.normal(0,sigma,100)
    X_train_updated = X_train + epsilon
    y_train_updated = np.array([math.sin(x) for x in X_train_updated])
    poly_updated = lagrange(X_train_updated, y_train_updated)
    y_pred_train_updated = poly_updated(X_train_updated)
    y_pred_test_updated = poly_updated(X_test)
    err_train = np.log(np.mean((y_train_updated - y_pred_train_updated) ** 2))
    err_test = np.log(np.mean((y_test - y_pred_test_updated) ** 2))
    train_err.append(err_train)
    test_err.append(err_test)

data_preproc = pd.DataFrame({
    'Sigma': sigmas,
    'Training':train_err,
    'Test': test_err})

ax = sns.lineplot(x='Sigma', y='value', hue='variable',
             data=pd.melt(data_preproc, ['Sigma']))
ax.set(ylabel='log(MSE)')