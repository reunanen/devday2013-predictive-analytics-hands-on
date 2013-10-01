import pandas as pd
import statsmodels.api as sm
import pylab as pl
import numpy as np
import sklearn.linear_model as skl
import sklearn.preprocessing as skp

# Our own plotting functions, somewhat specific to this hands-on session. 
import plots

# Read the data (edit the path if necessary).
# We us pandas DataFrame data structure. It is a 2-dimensional
# labeled data structure with columns of potentially
# different type.
# See more info: http://pandas.pydata.org/pandas-docs/dev/dsintro.html
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
plots.histograms(chemvars(df))
# The wine type is a string. Make a boolean type indicator.
def is_red(df): return np.array(df['wine_color']=="red", int)

# Calling the thing to be predicted 'y' is a convention.
y = is_red(df)
 
# Histograms by the wine type.
# Hint: This is very useful when you select or transform variables for a model
# later. 
plots.class_histograms(y, chemvars(df), alpha=.3)
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
plots.scatter(y, chemvars(df), 'fixed_acidity', 'chlorides', alpha=0.2, ymax=0.4)
# Our function allows transformations, see the code in plots.py
# Logarithm compresses high values and opens up the scale at the lower end.
plots.scatter(y, chemvars(df), 'fixed_acidity', 'chlorides', np.log, np.log, alpha=0.2, ymax=0.4)



# Notice: The second part starts here.

# Fit a logistic regression model using two variables.
# More info for logistic regression: http://en.wikipedia.org/wiki/Logistic_regression
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
plots.decision_surface(y, X, mfit.predict, 'fixed_acidity', 'chlorides')

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
def some_combinations(v1, v2):
	# A bunch of simple nonlinear multiplicative terms. 
	# Hint: You may want to edit the terms or define a more general function.
	return [(v1, v1), (v2, v2), (v1, v1, v1), (v2, v2, v2), (v1, v2)]
# Let's make a data with all the nonlinear terms above!
v1, v2 = 'fixed_acidity', 'chlorides'
# add_nonlins() is needed later for plotting.
def add_nonlins(X, v1, v2): return reduce(add_mterm, some_combinations(v1, v2), X)
Xn = add_nonlins(preds2(df, v1, v2), v1, v2)
y = is_red(df)				# (just as a reminder)

# Fit the model with all the new nonlinear terms. 
m_nl = sm.Logit(y, Xn)			# Model object with the expanded set of predictors. 
mfit_nl = m_nl.fit()			# Fit it. 
print mfit_nl.summary()			# See how it is

# Probabilities the new model gives to wines being red
# Hint: you may plot p_nl against the predictions of the linear model (p).
p_nl = mfit_nl.predict(Xn)
# How many does it get right, in percentages?
print pacc(y, p_nl)

# Visualize the probabily surface of the nonlinear model.
# Note that the plotter needs a function that maps a two-dimensional data frame
# onto the higher dimensional training data. 
plots.decision_surface(y, X, mfit_nl.predict, v1, v2, nlmap=add_nonlins)
# Quite a complex decision surface!
# Again, note the zoom tool in the plot window.



# Notice: The third part starts here.

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
print [pacc(is_red(d), mfit.predict(my_preds2(d))) for d in (df_train, df_test)]

# Hmm, how about with all variables?
def preds_all(df):
	X = chemvars(df)
	X['intercept'] = 1.0
	return X

# You may experiment with more quadratic terms
def diag_triangular(x): return [(i, j) for i in x for j in x if i<=j]
def triangular(x): return [(i, j) for i in x for j in x if i<j]

# By following command you may create quadratic terms
df_big = reduce(add_mterm, diag_triangular(chemvars(df).columns), chemvars(df))
# Note with big models: 
# If your models blow up, that is, the fitter gives an error about a singular matrix,
# try the regularization approach (below). 


# As a side step, before regularization...
# Cross validation does not waste half of the data for testing:
# It splits the data set into (e.g.) 10 pieces, trains on 9, tests on the remaining one,
# and loops this through all the ten pieces. 

from crossvalidate import cv

n_success = cv(is_red(df), my_preds2(df), 10,
	       # The function to build the model and predict
	       lambda ytr, Xtr, Xev: sm.Logit(ytr, Xtr).fit().predict(Xev), 
	       # The function to evaluate results: How many are right?
	       lambda p, y: sum(y == (p>0.5)))	
print 100*n_success/float(len(df))



# Does your model blow up, complaining about a singular matrix?
#    "numpy.linalg.linalg.LinAlgError: Singular matrix"
# This is because the maximum likelihood solution becomes unidentifiable:
# It is either not unique any more, or is numerically very unstable. 
# Then you need a better estimation method, and that is Maximum a Posteriori (MAP).
# https://en.wikipedia.org/wiki/Maximum_a_posteriori_estimation
# http://en.wikipedia.org/wiki/Regularization_(mathematics)
# https://en.wikipedia.org/wiki/Bayesian_interpretation_of_regularization

# MAP or regularization makes the solution identifiable by making
# coefficients near zero more probable. To realize this near-zero preference, 
# a new term is introduced to the cost function, with a weight 1/C.
# The smaller the C, the closer to zero the coefficients tend to be. 

# If you use L1 regularization, some coefficients will be exactly zero,
# or up to numerical (estimation) accuracy anyways. That is, L1 is a way
# to realize automatic variable selection. 

# In practice, one tries different values of C and evaluates the models
# with test data or cross validation. 

# Regularization is our prior statement about the coefficients of the model.
# As the coefficients are (inversely) related to the scales of the variables,
# we need to rescale the data to some known and standard scale beforehand.
# The convention is to make the standard deviation of all variables equal to one.

# So here we go...
# Create a scaler object. This needs to be wrapped in an object, because
# the same scaling needs to be used multiple times. 
# Notice: sklearn package does not work correctly with pandas DataFrames, 
#         but it assumes data as numpy arrays.
scaler = skp.StandardScaler().fit(chemvars(df_train).values)

# Create a model object with L1 regularization, and C=0.01.
# The data is not shown to the model yet. :)
m_l1 = skl.LogisticRegression(C=0.01, penalty='l1')

# Fit the model, i.e., optimize the parameters.
m_l1.fit(scaler.transform(chemvars(df_train).values), is_red(df_train))

# The values of parameters
print m_l1.coef_

# The number of variables (columns) having zero coefficient
print sum((abs(m_l1.coef_[0])<0.001).astype(int))

# The names of the variables with non-zero coefficients.
print chemvars(df_train).columns[(abs(m_l1.coef_[0])>0.001)]

# Compute predictions for the training data. 
# This class gives probabilities for both classes, hence [:, 1].
p_train = m_l1.predict_proba(scaler.transform(chemvars(df_train).values))[:, 1]

# Training set accuracy
print pacc(is_red(df_train), p_train)

# Predictions (probabilities) for test data
p_test = m_l1.predict_proba(scaler.transform(chemvars(df_test).values))[:,1]

# Test accuracy
print pacc(is_red(df_test), p_test)

# You may want to play with a larger model (consider trainign with df_big above), and different values of C.
# Use the test set or cross validation to evaluate the results.
# You may also try to plot the decision surface with respect to two important variables in the model.
# (For that, you need to fix the values of other variables, of course.)

# Hint: You may start by small enough value for C to make all the model coefficients to zero,
#       and then gradually increasing C, preferably using a logarithmic scale.
