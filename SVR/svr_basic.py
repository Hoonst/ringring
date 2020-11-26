import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import pandas as pd

# 참조: https://scikit-learn.org/stable/auto_examples/svm/plot_svm_regression.html#sphx-glr-auto-examples-svm-plot-svm-regression-py

# #############################################################################
# Generate sample data
X = pd.read_csv('./팀플 데이터/in_total.csv')
x = X['count'].to_numpy()
#scale x
x = (x - x.min(axis=0)) / (x.max(axis=0) - x.min(axis=0))


y = np.sin(x).ravel()

# #############################################################################
# Add noise to targets
y[::5] += 3 * (0.5 - np.random.rand(len(y)//5 +1))

# #############################################################################
# Fit regression model
svr_rbf = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
svr_lin = SVR(kernel='linear', C=100, gamma='scale', max_iter= 1000000)
svr_poly = SVR(kernel='poly', C=100, gamma='scale', degree=3, epsilon=.1, coef0=1,max_iter= 1000000)
y_rbf = svr_rbf.fit(x.reshape(-1,1), y).predict(x.reshape(-1,1))
y_lin = svr_lin.fit(x.reshape(-1,1), y).predict(x.reshape(-1,1))
y_poly = svr_poly.fit(x.reshape(-1,1), y).predict(x.reshape(-1,1))

# #############################################################################
# Look at the results
lw = 2

svrs = [svr_rbf, svr_lin, svr_poly]
kernel_label = ['RBF', 'Linear', 'Polynomial']
model_color = ['m', 'c', 'g']

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 10), sharey=True)
for ix, svr in enumerate(svrs):
    axes[ix].plot(x, svr.fit(x.reshape(-1,1), y).predict(x.reshape(-1,1)), color=model_color[ix], lw=lw,
                  label='{} model'.format(kernel_label[ix]))
    axes[ix].scatter(x[svr.support_], y[svr.support_], facecolor="none",
                     edgecolor=model_color[ix], s=50,
                     label='{} support vectors'.format(kernel_label[ix]))
    axes[ix].scatter(x[np.setdiff1d(np.arange(len(x)), svr.support_)],
                     y[np.setdiff1d(np.arange(len(x)), svr.support_)],
                     facecolor="none", edgecolor="k", s=50,
                     label='other training data')
    axes[ix].legend(loc='upper center', bbox_to_anchor=(0.5, 1.1),
                    ncol=1, fancybox=True, shadow=True)

fig.text(0.5, 0.04, 'data', ha='center', va='center')
fig.text(0.06, 0.5, 'target', ha='center', va='center', rotation='vertical')
fig.suptitle("Support Vector Regression", fontsize=14)
plt.show()
