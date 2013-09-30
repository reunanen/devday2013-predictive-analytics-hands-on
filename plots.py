"""
Functions to help with plotting. These are quite specific to the dev day presentation. 
"""

import pandas as pd
import pylab as pl
import numpy as np

# 1d grid that spans the value range of a vector, evenly. 
def vargrid(x, n=300):
    x0, x1 = min(x), max(x)
    return np.arange(x0, x1, (x1 - x0)/float(n))

## Plot histogram of all the variables
def hcolsrows(X):
    cols = int(np.ceil(np.sqrt(len(X.columns))))
    rows = int(np.ceil(len(X.columns) / float(cols)))
    return cols, rows

def histograms(X, no_bins=50):
    cols, rows = hcolsrows(X)
    for count, vi in enumerate(X.columns.values):
        pl.subplot(rows, cols, count+1)
        pl.hist(X[vi], bins = no_bins, color = 'b', alpha=0.5, edgecolor='b', histtype="stepfilled")
        pl.title(vi)
    pl.show()

## Plot histograms of all the variables (target classes separately)
def class_histograms(y, X, no_bins=50, alpha=0.5):
    cols, rows = hcolsrows(X)
    for count, vi in enumerate(X.columns.values):
        pl.subplot(rows, cols, count+1)
        pl.hist(X[vi][y==0], bins = no_bins, color = 'g', alpha=alpha, edgecolor='g', histtype="stepfilled")
        pl.hist(X[vi][y==1], bins = no_bins, color = 'r', alpha=alpha, edgecolor='r', histtype="stepfilled")
        pl.title(vi)
    pl.show()

# Identity function for one argument; used in plotting functions.
_1 = lambda x: x

def scatter(y, X, var1, var2, f1=_1, f2=_1, alpha=0.5, xmax=None, ymax=None):
    pl.plot(f1(X[var1][y==0]), f2(X[var2][y==0]), 'g^', markeredgecolor='g', alpha=alpha)
    pl.plot(f1(X[var1][y==1]), f2(X[var2][y==1]), 'r*', markeredgecolor='r', alpha=alpha)
    pl.xlabel(var1)
    pl.ylabel(var2)
    pl.show()

## Visualize the result
def decision_surface(y, X, mfit, var1, var2, alpha=0.25, nlmap=lambda x, v1, v2: x):
    # Set up a grid of values of the original variables 
    x1g, x2g = np.meshgrid(vargrid(X[var1]), vargrid(X[var2]))
    # Make a data frame out of the values on the grid.
    Xg = pd.DataFrame({var1 : np.ravel(x1g), var2 : np.ravel(x2g), 'intercept' : 1.0})
    # Expand the data frame by computing all the nonlinearities.
    Xg = nlmap(Xg[X.columns], var1, var2) # The index thing fixes column order. 
    # Compute predictions, and plot them.
    p = mfit.predict(Xg)
    h = pl.contourf(x1g, x2g, np.reshape(p, x1g.shape), 300, cmap=pl.cm.gist_yarg)
    cbar = pl.colorbar()
    cbar.set_label('probability for y=1')
    # Plot data on top of predictions.
    pl.plot(X[var1][y==0], X[var2][y==0], 'go', markeredgecolor='g', alpha=alpha)
    pl.plot(X[var1][y==1], X[var2][y==1], 'ro', markeredgecolor='r', alpha=alpha)
    pl.xlabel(var1); pl.ylabel(var2)
    pl.show()

