import pandas as pd
import statsmodels.api as sm
import pylab as pl
import numpy as np

# Our own plotting functions, somewhat specific to this hands-on session. 
from plots import *

# Read the data (edit the path if necessary)
# The data file has no row indices/names, hence index_col=F.
df = pd.read_csv("wine_quality_data.csv", index_col=False)

# How the data looks like... first rows.
print df.head()

# What columns/variables are there?
print df.columns

# Variables that could be used to predict the wine type.
def chemvars(df):
	return df[['fixed_acidity', 'volatile_acidity', 'citric_acid',
		  'residual_sugar', 'chlorides', 'free_sulfur_dioxide',
		  'total_sulfur_dioxide', 'density', 'pH', 'sulphates',
		  'alcohol']]
# Yet another way to look at the data.
print chemvars(df).describe()

# Histograms of all predictors.
plot_histograms(chemvars(df))

# The wine type is a string. Make a boolean type indicator.
def is_red(df): return np.array(df['wine_color']=="red", int)

# Calling the thing to be predicted 'y' is a convention.
y = is_red(df)
 
# Histograms by the wine type.
# Hint: This is very useful when you select or transform variables for a model
# later. 
plot_class_hists(y, chemvars(df), alpha=.3)

# Pandas data frames are handy for calculating class-wise summary statistics.
# Some examples...
# - Percentages of wine types in the data
print 100*df.groupby(y).size()/float(len(df))
# - Means (averages) by wine type
print df.groupby(y).mean()
# - Standard deviations by wine type
print df.groupby(y).std()

# We could use all the predictors in a model. 
# But to keep things simple, let's start with two variables only.
# (Things are easier to plot if you have only two dimensions.)
# Choose variables that separate the target classes (wine type) well.
# For example...
# Hint: Try other predictors. Look at the histograms by wine type.

# Plot scatter plot of the chosen vars, colors specify wine type.
plot_scatter(y, chemvars(df), 'fixed_acidity', 'chlorides', alpha=0.2, ymax=0.4)
# Our function allows transformations, see the code in plots.py
# Logarithm compresses high values and opens up the scale at the lower end.
plot_scatter(y, chemvars(df), 'fixed_acidity', 'chlorides', np.log, np.log, alpha=0.2, ymax=0.4)

# Modeling starts here...

# Fit a logistic regression model using two variables.
# [FIXME: add a reference somewhere, a slide or Wikipedia or something]
# In a regression model, you should have a free constant in the linear combination.
# It is called intercept. 
# (Think for a while a logistic regression with no variables, but either with an intercept
#  or without. The latter always gives p=0.5, the former adjusts to the prior class probabilities
#  of the data.)
# The intercept becomes conveniently added if you add a constant variable to the data.
# The function returns predictors for chosen two variables:
def preds2(df, v1, v2):
	X = df[[v1, v2]]
	X['intercept'] = 1.0
	return X

# For convenience, define a list of variables we will use.
# (Note: tuple does not work here because this is used for indexing later)

# The magic happens here.
X = preds2(df, 'fixed_acidity', 'chlorides')
model = sm.Logit(y, X)			# Create model object.
mfit = model.fit()			# Train the model, i.e., optimize its parameters
print mfit.summary()			# It did something!


# Predictions: what the model "remembers" about the wines in the training data,
# are the wines red or white (probabilities for the wine being red)
p = mfit.predict(X)
# How those probabilities compare to what the wines actually were?
# There are many ways to evaluate the predictions, here we just
# see how many hits are correct if probabilities over 0.5 are guessed as red.
def pacc(y, p):
	return 100*sum(y == (p>0.5)) / float(len(y))
print pacc(y, p)


# From the model's point of view, wine is what we have shown about it: vars2d.
# Because of the '2d', it is easy to plot what the model thinks about
# all possible wines, that is, all possible values of vars2d.
# (remember, 'intercept' is just a constant)
# Note the zoom tool in the fig!
visualize_result_2d_linear(y, X, mfit, 'fixed_acidity', 'chlorides')

# Shades of gray are probabilities.
# (You may want to look at the plotting code in plots.py, but it is not relevant
# to the big picture.)

# Note the linear decision boundary above. This comes from the linearity of the
# logistic regression model.

# If you want a more flexible decision boundary, you can add nonlinear terms
# to the model.
# Then the model is still linear with respect to the variables it sees, but not
# with respect to the original data!
def add_mterm(X, vars):
    """Add a multiplicative term to the model."""
    X = X.copy()
    X['*'.join(vars)] = np.multiply.reduce(X[list(vars)].values, 1)
    return X

# Here are some multiplicative terms.
# For example, (var1, var1) will add the term var1*var1 to the model,
# so the model would be var1 + var2 + var1*var2 + intercept
def nonlins(v1, v2):
	# A bunch of simple nonlinear multiplicative terms. 
	return [(v1, v1), (v2, v2), (v1, v1, v1), (v2, v2, v2), (v1, v2)]
# Let's make a data with all the nonlinear terms above!
v1, v2 = 'fixed_acidity', 'chlorides'
Xn = reduce(add_mterm, nonlins(v1, v2), preds2(df, v1, v2))
y = is_red(df)				# (just as a reminder)

# Fit the model with all the new nonlinear terms. 
m_nl = sm.Logit(y, Xn) 			# Model object with the expanded set of predictors. 
mfit_nl = m_nl.fit()			# Fit it. 
print mfit_nl.summary()			# See how it is

# Probabilities the new model gives to wines being red
# Hint: you may plot p_nl against the predictions of the linear model (p).
p_nl = mfit_nl.predict(Xn)
# How many does it get right, in percentages?
print pacc(y, p_nl)

# Visualize the probabilities (decision boundary) of the nonlinear model.
# This relies on v1, v2, and mfit defined above.
# - Set up a grid of values of the original variables 
x1g, x2g = np.meshgrid(vargrid(df[v1]), vargrid(df[v2]))
# - Make a data frame out of the values on the grid. 
Xg = pd.DataFrame({v1 : np.ravel(x1g), v2: np.ravel(x2g), 'intercept': 1.0})
# - Expand the data frame by computing all the nonlinearities.
Xg = reduce(add_mterm, nonlins(v1, v2), Xg)
# - Make sure the order of the terms is as in the original data
Xg = Xg[Xn.columns]
# - Compute predictions
pg = mfit_nl.predict(Xg)

# - Plot them
h = pl.contourf(x1g, x2g, np.reshape(pg, x1g.shape), 300, cmap=pl.cm.gist_yarg)
cbar = pl.colorbar()
cbar.set_label('probability for y=1')
# - Overlay the data on top of predictions
pl.plot(X[v1][y==0], X[v2][y==0], 'g^', markeredgecolor='g', alpha=0.5)
pl.plot(X[v1][y==1], X[v2][y==1], 'r*', markeredgecolor='r', alpha=0.5)
pl.xlabel(v1); pl.ylabel(v2)
pl.show()
# Quite a complex decision surface!
# Again, note the zoom tool in the plot window.


# (Last session)
# Evaluating models with leave-out data

# Make this a bit more functional. :)

# Split the data set randomly into two.
df_train, df_test = [i[1] for i in df.groupby(np.random.random(len(df))>.5)]

def my_preds2(df): return preds2(df, 'fixed_acidity', 'chlorides')

model = sm.Logit(is_red(df_train), my_preds2(df_train))
mfit = model.fit()
print mfit.summary()

# Predict on training data
p_train = mfit.predict(my_preds2(df_train))
print pacc(is_red(df_train), p_train)

# Predict on *test* data. The model has never seen these samples. 
p_test = mfit.predict(my_preds2(df_test))
print pacc(is_red(df_test), p_test)

# ... or both
print [pacc(is_red(df_part), mfit.predict(my_preds2(df_part))) for df_part in (df_train, df_test)]

# Hmm, how about with all variables?
def preds_all(df):
	X = chemvars(df)
	X['intercept'] = 1.0
	return X

model = sm.Logit(is_red(df_train), preds_all(df_train))
mfit = model.fit()
print [pacc(is_red(df_part), mfit.predict(preds_all(df_part))) for df_part in (df_train, df_test)]

# Make it a function
def test_model(predfun, df_train=df_train, df_test=df_test): 
	mfit = sm.Logit(is_red(df_train), predfun(df_train)).fit()
	print [pacc(is_red(df_part), mfit.predict(predfun(df_part))) for df_part in (df_train, df_test)]

print test_model(my_preds2)
print test_model(preds_all)
v1, v2 = 'fixed_acidity', 'chlorides'
print test_model(lambda df: reduce(add_mterm, nonlins(v1, v2), preds2(df, v1, v2)))
print test_model(lambda df: reduce(add_mterm, nonlins(v1, v2), preds_all(df)))

def all_pairs(x): return [(i, j) for i in x for j in x if i<=j]

Xbig = reduce(add_mterm, all_pairs(chemvars(df).columns), chemvars(df))
print Xbig.shape
# Oops, a singular matrix!
print test_model(lambda df: reduce(add_mterm, all_pairs(chemvars(df).columns), chemvars(df)))

# Adding noise helps?
df2 = df.copy()
for v in chemvars(df).columns: df2[v] += np.std(df2[v])*np.random.standard_normal(len(df2))
df_train, df_test = [i[1] for i in df2.groupby(np.random.random(len(df2))>.5)]
def test_model(predfun, df_train=df_train, df_test=df_test): 
	mfit = sm.Logit(is_red(df_train), predfun(df_train)).fit()
	print [pacc(is_red(df_part), mfit.predict(predfun(df_part))) for df_part in (df_train, df_test)]
print test_model(my_preds2)
print test_model(preds_all)
v1, v2 = 'fixed_acidity', 'chlorides'
print test_model(lambda df: reduce(add_mterm, nonlins(v1, v2), preds2(df, v1, v2)))
print test_model(lambda df: reduce(add_mterm, nonlins(v1, v2), preds_all(df)))

def all_pairs(x): return [(i, j) for i in x for j in x if i<=j]

Xbig = reduce(add_mterm, all_pairs(chemvars(df).columns), chemvars(df))
print Xbig.shape
# Oops, a singular matrix!
print test_model(lambda df: reduce(add_mterm, all_pairs(chemvars(df).columns), chemvars(df)))

