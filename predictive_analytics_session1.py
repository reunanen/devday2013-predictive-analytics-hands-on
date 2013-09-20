import os
import pandas as pd
import statsmodels.api as sm
import pylab as pl
import numpy as np

from plots import *

# Read the data in (edit the path if necessary)
# Notice: The last option is needed since the data file do not include row indices.
df = pd.read_csv(os.environ["HOME"] + "/wine/wine_quality_data.csv", index_col=False)


# Take a look at the dataset
print df.head()


# Names of the columns in the data set
print df.columns


## Explanatory variables, i.e. predictors
vars = ['fixed_acidity', 'volatile_acidity', 'citric_acid',
        'residual_sugar', 'chlorides', 'free_sulfur_dioxide',
        'total_sulfur_dioxide', 'density', 'pH', 'sulphates',
        'alcohol']
X = df[vars]


## Target variable
y = np.array(df['wine_color']=="red", int)

# Summarize the data
print X.describe()

plot_histograms(X)
plot_class_hists(y, X, alpha=.3)


## Calculate some statistics per classes

# Proportions of target classes (pandas functionality)
df['y'] = y
print df.groupby('y').size()/float(len(df))

# Means of variables in each target class
print df.groupby('y').mean()

# Standard deviation of variables in each target class
print df.groupby('y').std()


## Select two predictors/variables based on above graphs and statistics
## that separates the classes from each other
var1, var2 = 'fixed_acidity', 'chlorides'

## Plot scatter plot of the two most important variables
# Leave the possibility to transform variables, set initially to identity function.

plot_scatter(y, X, var1, var2, alpha=0.2, ymax=0.4)
plot_scatter(y, X, var1, var2, np.log, np.log, alpha=0.2, ymax=0.4)

## Fit a logistic regression model using selected two variables
# Note: tuple does not work here (because this is used for indexing later)
vars2d = [var1, var2, 'intercept']
X['intercept'] = 1.0                    # Why is this needed? :)

# Create and fit the model
model = sm.Logit(y, X[vars2d])
mfit = model.fit()

# Evaluate the prediction accuracy
def prediction_accuracy(y, p):
    return sum(y == (p>0.5)) / float(len(y))

p = mfit.predict(X[vars2d])
print prediction_accuracy(y, p)

# 1d grid that spans the values, evenly. 
def vargrid(x, n=300):
	x0, x1 = min(x), max(x); return np.arange(x0, x1, (x1 - x0)/float(n))


visualize_result_2d_linear(y, X, mfit, var1, var2)
# In figure, there is a zoom tool that you may use to change axis limits.


## Construct a more complicated model using nonlinear transformations of the original variables
def add_mterm(X, vars):
    X = X.copy()
    X['*'.join(vars)] = np.multiply.reduce(X[list(vars)].values, 1)
    return X

nonlins = [(var1, var1), (var2, var2), (var1, var1, var1), (var2, var2, var2), (var1, var2)]
Xn = reduce(add_mterm, nonlins, X[vars2d])

## Fit the model using nonlinear transformations
m_nl = sm.Logit(y,Xn)
mfit_nl = m_nl.fit()

# Evaluate the prediction accuracy
p_nl = mfit_nl.predict(Xn)
print prediction_accuracy(y, p_nl)

## Visualize the result (nonlinear decision boundary)
x1g, x2g = np.meshgrid(vargrid(X[var1]), vargrid(X[var2]))
Xg = pd.DataFrame({var1 : np.ravel(x1g)})
Xg[var2] = np.ravel(x2g)
Xg['intercept'] = 1.0
Xg = reduce(add_mterm, nonlins, Xg)
pg = mfit_nl.predict(Xg[Xn.columns])

#h = pl.contourf(x1, x2, np.reshape(pg, x1g.shape),500, alpha=0.85)
h = pl.contourf(x1g, x2g, np.reshape(pg, x1g.shape), 300, cmap=pl.cm.gist_yarg)
cbar = pl.colorbar()
cbar.set_label('probability for y=1')
pl.plot(X[var1][y==0], X[var2][y==0], 'g^', markeredgecolor='g', alpha=0.5)
pl.plot(X[var1][y==1], X[var2][y==1], 'r*', markeredgecolor='r', alpha=0.5)
pl.xlabel(var1); pl.ylabel(var2)
pl.show()

# In figure, there is a zoom tool that you may use to change axis limits.


