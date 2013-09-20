import pandas as pd
import statsmodels.api as sm
import pylab as pl
import numpy as np

from plots import *

# Read the data (edit the path if necessary)
# The data file has no row indices/names, hence index_col=F.
df = pd.read_csv("wine_quality_data.csv", index_col=False)

# Take a look at the first rows.
print df.head()

# What columns/variables are there?
print df.columns

# Let's define what will be the explanatory variables (also called predictors).
vars = ['fixed_acidity', 'volatile_acidity', 'citric_acid',
        'residual_sugar', 'chlorides', 'free_sulfur_dioxide',
        'total_sulfur_dioxide', 'density', 'pH', 'sulphates',
        'alcohol']
# ... and make a new data frame having only the predictors. 
X = df[vars]
# Let's look at what we have now.
print X.describe()

# Histograms of all vars.
plot_histograms(X)

# This is the target, what the model tries to predict. Make it a boolean.
y = np.array(df['wine_color']=="red", int)

# Histograms by values of the target.
# Hint: This is very useful for variable selection and possible transformations
# (for modelling, later)
plot_class_hists(y, X, alpha=.3)

# Pandas data frames are handy for calculating class-wise summary statistics.
# Some examples...
# Proportions of target classes in the data
df['y'] = y # FIXME?
print df.groupby('y').size()/float(len(df))
# Means of variables in each target class
print df.groupby('y').mean()
# Standard deviation of variables in each target class
print df.groupby('y').std()

# In a model we can have all the predictors.
# But to keep things simple, let's start with two variables only.
# Things are easier to plot if you have only two dimensions. 
# Choose variables that separate the target classes (wine type) well.
# For example...
var1, var2 = 'fixed_acidity', 'chlorides'

# Plot scatter plot of the chosen vars.
plot_scatter(y, X, var1, var2, alpha=0.2, ymax=0.4)
# Our function allows transformations, see the code in plots.py
plot_scatter(y, X, var1, var2, np.log, np.log, alpha=0.2, ymax=0.4)

# Fit a logistic regression model using the selected two variables
# Note: tuple does not work here (because this is used for indexing later)
vars2d = [var1, var2, 'intercept']
# Add a third predictor that is constant! (You may try to find out why this is necessary...)
X['intercept'] = 1.0
model = sm.Logit(y, X[vars2d]) 		# Create the model object
mfit = model.fit()			# Optimize its parameters
print mfit.summary()			# It did something!

# To see how the model performs with the training data, see how it... [FIXME]
# Prediction accuracy, in percents.
def pacc(y, p): return 100*sum(y == (p>0.5)) / float(len(y))

# Predictions: what the model "remembers" about the wines in the training data,
# are the wines red or white (probabilities for the wine being red)
p = mfit.predict(X[vars2d])
# How those probabilities compare to what the wines actually were?
# There are many ways to evaluate the predictions, here we just
# see how many hits are correct if probabilities over 0.5 are guessed as red.
print pacc(y, p)


# From the model's point of view, all possible wines are defined by all possible values
# of the predictors. Let's look at what the model thinks about different possible wines.
# Note the zoom tool in the fig!
visualize_result_2d_linear(y, X, mfit, var1, var2)

# Note the linear decision boundary above. This comes from the linearity of the
# logistic regression. 

# If you want a more flexible decision boundary, you can add nonlinear terms
# to the model.
# Then the model is still linear with respect to variables it sees, but not
# with respect to the original data!
def add_mterm(X, vars):
    """Add a multiplicative term to the model."""
    X = X.copy()
    X['*'.join(vars)] = np.multiply.reduce(X[list(vars)].values, 1)
    return X

# Here are some multiplicative terms.
# For example, (var1, var1) will add the term var1*var1 to the model,
# so the model would be var1 + var2 + var1*var2 + intercept
nonlins = [(var1, var1), (var2, var2), (var1, var1, var1), (var2, var2, var2), (var1, var2)]
# Let's make a data with all the nonlinear terms above!
Xn = reduce(add_mterm, nonlins, X[vars2d])

# Fit the model with all the new nonlinear terms. 
m_nl = sm.Logit(y, Xn) 			# Model object with the expanded set of predictors. 
mfit_nl = m_nl.fit()			# Fit it. 
print mfit_nl.summary()			# See how it is

# Probabilities the new model gives to wines being red
# Hint: you may plot p_nl against the predictions of the linear model (p).
p_nl = mfit_nl.predict(Xn)
# How many it gets right?
print pacc(y, p_nl)

# Visualize the probabilities (decision boundary) of the nonlinear model.
# Set up a grid of values of the original variables 
x1g, x2g = np.meshgrid(vargrid(X[var1]), vargrid(X[var2]))
# Make a data frame out of the values on the grid. 
Xg = pd.DataFrame({var1 : np.ravel(x1g)})
# FIXME: the order is not important, collapse into a pdf.DataFrame() call.
Xg[var2] = np.ravel(x2g)
Xg['intercept'] = 1.0
# Expand the data frame by computing all the nonlinearities.
Xg = reduce(add_mterm, nonlins, Xg)
# Make sure the order of the terms is as in the original data
Xg = Xg[Xn.columns]
# Compute predictions
pg = mfit_nl.predict(Xg[Xn.columns])

# Plot them
h = pl.contourf(x1g, x2g, np.reshape(pg, x1g.shape), 300, cmap=pl.cm.gist_yarg)
cbar = pl.colorbar()
cbar.set_label('probability for y=1')
# Overlay the data on top of predictions
pl.plot(X[var1][y==0], X[var2][y==0], 'g^', markeredgecolor='g', alpha=0.5)
pl.plot(X[var1][y==1], X[var2][y==1], 'r*', markeredgecolor='r', alpha=0.5)
pl.xlabel(var1); pl.ylabel(var2)
pl.show()
# Again, note the zoom tool in the plot window.




