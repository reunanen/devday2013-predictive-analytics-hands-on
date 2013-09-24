Dev Day 2013: Predictive Analytics Hands-on session
===================================================

If possible, please prepare for this session by downloading the virtual machine or the relevant libraries *beforehand*. (See section "Getting ready" below.)

It is a good idea to pair up for this session. If you are not familiar with Python, it is recommended that you find someone who is.

# Useful links

Practical ones:

* [Numpy tutorial](http://wiki.scipy.org/Tentative_NumPy_Tutorial): we will not use Numpy extensively, just for light data manipulation. Numpy is the [R](http://r-project.org/) or Matlab of Python.
* [A short tutorial](http://pandas.pydata.org/pandas-docs/stable/10min.html) on Pandas, the Python "data frame" library. Pandas is a higher-level structure on arrays, allowing heterogeneous columns, column names, etc.
* [Logistic regression with Python](http://blog.yhathq.com/posts/logistic-regression-and-python.html) and its statsmodels package. Logistic regression is a way to make a classifier that gives out probabilities.



More theoretical:

* [Logistic regression on Wikipedia](http://en.wikipedia.org/wiki/Logistic_regression). Easy to understand: a linear regression stacked with an S-curve to make outputs look like probabilities. But in its [full](http://fastml.com/regression-as-classification/) [glory](http://arantxa.ii.uam.es/~jmlobato/docs/slidesCambridge.pdf), a logistic regression model is far from elementary!

# Getting ready

Even though we are working with a toy example here, our aim is to provide an environment that will be useful even for some serious work in the field of predictive analytics. Obviously we would also like our environment to have a permissive license. For these reasons, the end result is bound to be somewhat involved. You basically have two options:

## Alternative 1: Using a virtual machine

To get started quickly (?), you can use a virtual machine that we have pre-built to already contain all the necessary libraries. **You can download the virtual machine from [here](http://download.reaktor.fi/PredictiveAnalyticsHandsOn/DevDay2013-PredictiveAnalyticsHandsOn.ova).** (It's running Ubuntu.)

If you already have something like VMware installed, feel free to use it. If not, then you can download for example [VirtualBox](https://www.virtualbox.org/wiki/Downloads). In VirtualBox, you can add the machine via the Import Appliance menu item.

Please note that:

* The version of VirtualBox should be 4.2.18 or later. If not, you may run into weird problems. 
* The virtual machine is installed with Finnish keyboard settings. You may want to change this. 
* The user's password is `DevDay2013`. 

Once you have the virtual machine running, update the latest instructions and code from GitHub:

	cd ~/devday2013-predictive-analytics-hands-on
	git pull

If you don't want to use the virtual machine, feel free to install the environment directly onto your computer. The necessary libraries are listed below.

## Alternative 2: Required libraries

**Note that this section is relevant only if _not_ using the provided virtual machine.**

The data and some sample code you can get from GitHub:

	git clone https://github.com/reunanen/devday2013-predictive-analytics-hands-on

### On Linux

To run these examples, you should install the following packages (Ubuntu names used here):

* g++, gfortran
* libblas-dev, liblapack-dev
* libfreetype6-dev, libpng12-dev
* ipython, python-pip, python-dev

For Python, you need the following (installation with [`pip`](http://en.wikipedia.org/wiki/Pip_(package_manager)) is recommended):

* [numpy](http://www.numpy.org/), [scipy](http://www.scipy.org/)
* [matplotlib](http://matplotlib.org/), [pandas](http://pandas.pydata.org/), [patsy](https://github.com/pydata/patsy), [statsmodels](http://statsmodels.sourceforge.net/)

### On OS X

* Make sure you have the XCode _command line tools_ installed.
* Install the GNU Fortran compiler and Freetype fonts. With [`brew`](http://brew.sh/) that would be

	    brew install gfortran freetype
	
* Tune up the Python environment. (This should work if you use the Python provided by Apple. With, e.g., the newest Python by brew, prepare for architecture problems between various C libraries and the Python binary.)

    	sudo easy_install pip
	    sudo pip install --upgrade numpy
	    sudo pip install scipy # If installing scipy fails because of some missing numpy files, you can try uninstalling and reinstalling numpy
	    sudo pip install pandas cython nose matplotlib ipython patsy
    	sudo pip install statsmodels pymc

### On Windows

You can download the required libraries from [Christoph Gohlke's repository](http://www.lfd.uci.edu/~gohlke/pythonlibs/).

You need:

* [ipython](http://www.lfd.uci.edu/~gohlke/pythonlibs/#ipython) (perhaps not stricly necessary, but very useful)
* [numpy](http://www.lfd.uci.edu/~gohlke/pythonlibs/#numpy), [scipy](http://www.lfd.uci.edu/~gohlke/pythonlibs/#scipy)
* [matplotlib](http://www.lfd.uci.edu/~gohlke/pythonlibs/#matplotlib), [pandas](http://www.lfd.uci.edu/~gohlke/pythonlibs/#pandas), [patsy](http://www.lfd.uci.edu/~gohlke/pythonlibs/#patsy), [statsmodels](http://www.lfd.uci.edu/~gohlke/pythonlibs/#statsmodels)
* [dateutil](http://www.lfd.uci.edu/~gohlke/pythonlibs/#python-dateutil), [pyparsing](http://www.lfd.uci.edu/~gohlke/pythonlibs/#pyparsing)

# Introduction

## What is predictive analytics?

Analytics produces actionable information from observations or from results of experiments ("data sets"). The results may be informative summaries of the data, suggest causal relations or new hypotheses for further testing, reveal underlying relationships, quantify uncertainty as probabilities or confidence intervals, or be very operational suggestions of actions, possibly embedded into an IT system. Statistical and other kind of _models_ are used to achieve these goals.

Models of analytics do not usually try to reproduce the exact dynamics of the system under study, as for example weather models do. Instead, they are highly simplified abstractions of the world. Their structure usually just tries to take into account the surface tendencies and dependencies in the data. The simplest model structures can be applied on almost any domain, from physics to psychology to business. Because of this simplification, the models are inherently probabilistic: Things not taken into account or impossible to predict appear as uncertainty in the model.

_Predictive models_ learn relationships and patterns from historical data to forecast future events, or outcomes of actions not yet made. Models are constructed with techniques from statistics, machine learning, and data mining. Predictive analytics is applied successfully in, e.g., marketing, telecommunications, insurance, retail, and manufacturing industries.

From Wikipedia:

> [*Analytics*](http://en.wikipedia.org/wiki/Analytics) is the discovery and communication of meaningful patterns in data. Especially valuable in areas rich with recorded information, analytics relies on the simultaneous application of [statistics](http://en.wikipedia.org/wiki/Statistics), [computer programming](http://en.wikipedia.org/wiki/Computer_programming) and [operations research](http://en.wikipedia.org/wiki/Operations_research) to quantify performance. Analytics often favors [data visualization](http://en.wikipedia.org/wiki/Data_visualization) to communicate insight.

>*[Predictive analytics](http://en.wikipedia.org/wiki/Predictive_analytics)* encompasses a variety of techniques from [statistics](http://en.wikipedia.org/wiki/Statistics), [modeling](http://en.wikipedia.org/wiki/Predictive_modelling), [machine learning](http://en.wikipedia.org/wiki/Machine_learning), and [data mining](http://en.wikipedia.org/wiki/Data_mining) that analyze current and historical facts to make [predictions](http://en.wikipedia.org/wiki/Prediction) about future, or otherwise unknown, events.

(Predictive) analytics is not reporting, [web analytics](http://en.wikipedia.org/wiki/Web_analytics) or [big data](http://en.wikipedia.org/wiki/Big_data), although one can apply models to web data or large data sets, or improve reports with inference techniques.

## The Wine data

In this hands-on session, a Wine data set containing chemical measurements from red and white variants of the Portuguese "[Vinho Verde](http://en.wikipedia.org/wiki/Vinho_Verde)" wine is analyzed. The chemical measurements include the following variables:

* fixed acidity
* volatile acidity
* citric acid
* residual sugar
* chlorides
* free sulfur dioxide
* total sulfur dioxide
* density
* pH
* sulphates
* alcohol

The variables are available for 1599 red and 4898 white wines, respectively. The objective is to predict the color of wine using only the chemical measurements listed above. Note that there are no data about grapes, brand, selling price etc. available.

## The first insights

Before modelling, it is essential to get familiar with the data. Simple statistics such as mean and standard deviation give some idea of variation of the data. More visual and comprehensive strategy is to plot histograms or scatter plots of the variables. Histogram is a kind of estimate of the (probability) distribution of the variable. Below as an example is the distribution of the pH values of wines. pH values outside the range 2.9-3.6 are uncommon. The second graph shows histograms of pH values separately for red (red bars) and white (green bars).

![Histogram](img/histogram_pH2.png)

![Histogram with classes](img/hist_classes2.png)

A scatter plot shows not only variation but _covariation_ of two variables. Other information can be included. Below colours of the dots present wine type, so one sees how well the red (red dots) and white (yellowish dots) wines are separated by these two variables.

The blue line is a _decision boundary_ that could be used to classify new wines, if only these two variables were available. In our case, the objective of modelling is to find a decision boundary such that the classification accuracy of _new, yet unseen wines_, is maximized.

![Scatter plot](img/scatter_plot2.png)

A decision boundary can also be 'soft' in the sense that near the boundary we know that the correct class could really be either. Furthermore, the decision boundary can be _non-linear_. The figure below illustrates a non-linear, 'soft' decision rule.

![Scatter plot with a non-linear decision boundary](img/scatter_nonlinear.png)

## Logistic regression

There are numerous techniques to find a good decision boundary. The basic strategy is to optimize for the existing data, but not too well. (The problem of doing too well becomes apparent later.)

To fit a model, one usually chooses a model class, then optimizes the model with respect to some _parameters_, numbers in the model with no given values. In practice the process is iterative: the model is fitted, fine tuned, fitted again, etc.

The model class used below is [logistic regression](http://en.wikipedia.org/wiki/Logistic_regression). It is a classic techique, very simple at its core but surprisingly rich in practice. The model has a linear combination of some variables with free coefficients (to be optimized), then an S-like function (inverse [logit](http://en.wikipedia.org/wiki/Logit), quite close to tanh()) on top of that to limit the results to be between 0 and 1. With a suitable optimization technique, such an outcome becomes interpretable as [probabilities](http://en.wikipedia.org/wiki/Probability).

The input variables, or explanatory variables, are here the chemical properties of the wines, or at least a subset of them.

The optimization technique is called [maximum likelihood](http://en.wikipedia.org/wiki/Maximum_likelihood). It finds coefficients of the linear combination in the model (parameters of the model) that maximizes the probability of the observed data (wine types, red or white). (The theory behind, and why this is good, is quite deep and complicated. And there are also other ways to fit a model, maximum likelihood is not the last word.)


# Actually doing something

The task is to devise a classifier for the colour of wine, given examples of existing wines and their 11 chemical properties (which are not directly related to colour). Below, a possible approach is sketched. At any point, if you think you are lost, feel free to ask!

## Import relevant packages

There is a lot of interactive work ahead, so we use acronyms for the packages. (If you are proficient with these, you could import with '*' for even more brevity.)

	import os
	import pandas as pd
	import statsmodels.api as sm
	import pylab as pl
	import numpy as np

## How to read in the data

	# In the same directory as wine data
	df = pd.read_csv("wine_quality_data.csv")

## The first insights into the data

	# Take a look at the first rows of the data set
	print df.head()

	# Names of the columns in the data set
	print df.columns

	## Select the explanatory variables
	vars = ['fixed_acidity', 'volatile_acidity', 'citric_acid',
		'residual_sugar', 'chlorides', 'free_sulfur_dioxide',
		'total_sulfur_dioxide', 'density', 'pH', 'sulphates',
		'alcohol']
	X = df[vars]


	## Select a target variable. Notice: y==1 refers to red and y==0 to white wines from now on.
	y = np.array(df['wine_color']=="red", int)

	## Summarize the data (means, standard deviations, quantiles)
	print X.describe()


	## Draw a histogram for each explanatory variable
	def hcolsrows(X):
	    cols = int(np.ceil(np.sqrt(len(X.columns))))
	    rows = int(np.ceil(len(X.columns) / float(cols)))
	    return cols, rows

	def plot_histograms(X, no_bins=50):
	    cols, rows = hcolsrows(X)
	    for count, vi in enumerate(X.columns.values):
		pl.subplot(rows, cols, count+1)
		pl.hist(X[vi], bins = no_bins, color = 'b', alpha=0.5, edgecolor='b', histtype="stepfilled")
		pl.title(vi)
	    pl.show()

	## Notice, you may test different no_bins to see how it affects shape of plots
	plot_histograms(X)

## Further insights

	## Try to get preliminary idea whether the classes are separable
	## Plot histograms of all the variables (target classes separately)
	def plot_class_hists(y, X, no_bins=50, alpha=0.5):
	    cols, rows = hcolsrows(X)
	    for count, vi in enumerate(X.columns.values):
		pl.subplot(rows, cols, count+1)
		pl.hist(X[vi][y==0], bins = no_bins, color = 'g', alpha=alpha, edgecolor='g', histtype="stepfilled")
		pl.hist(X[vi][y==1], bins = no_bins, color = 'r', alpha=alpha, edgecolor='r', histtype="stepfilled")
		pl.title(vi)
	    pl.show()

	plot_class_hists(y, X, alpha=.3)


	## Calculate some statistics per classes

	# Proportions of target classes (pandas functionality)
	df['y'] = y
	print df.groupby('y').size()/float(len(df))

	# Means of variables in each target class
	print df.groupby('y').mean()

	# Standard deviation of variables in each target class
	print df.groupby('y').std()

	## Based on above graphs and statistics: Select two variables that could separate the classes from each other
	# For example: var1, var2 = 'fixed_acidity', 'chlorides'
	var1, var2 = 'EDIT_VARIABLE1', 'EDIT_VARIABLE2'

	## Plot scatter plot of the selected two variables
	# Leave the possibility to transform variables, set initially to identity function.
	_1 = lambda x: x
	def plot_scatter(y, X, var1, var2, f1=_1, f2=_1, alpha=0.5, xmax=None, ymax=None):
	    pl.plot(f1(X[var1][y==0]), f2(X[var2][y==0]), 'g^', markeredgecolor='g', alpha=alpha)
	    pl.plot(f1(X[var1][y==1]), f2(X[var2][y==1]), 'r*', markeredgecolor='r', alpha=alpha)
	    pl.xlabel(var1)
	    pl.ylabel(var2)
	    pl.show()

	# Using the the identity function
	plot_scatter(y, X, var1, var2, alpha=0.2, ymax=0.4)

	# Using the logarithmic transformation. You may think is there any advantage with logarithmic transformation.
	plot_scatter(y, X, var1, var2, np.log, np.log, alpha=0.2, ymax=0.4)
	{code}

Some questions:
* What can be seen from the histograms? Do they give any indication which variables could be useful in classification of wines?
* How to interpret the scatter plot? If we know values of both variables of an unknown wine sample, what can we say about the color of the wine? When can we say something and when not?
* How could this be modelled automatically? (Hint: see next two sections below.)


## Find a linear decision boundary

	## Fit a logistic regression model using selected two variables
	# Note: tuple does not work here (because this is used for indexing later)
	vars2d = [var1, var2, 'intercept']
	X['intercept'] = 1.0                    # Do not remove this, but you may think why is this needed? :)

	# Create and fit the model
	model = sm.Logit(y, X[vars2d])
	mfit = model.fit()

	# The probabilities of y==1 (being red wine) for each sample
	p = mfit.predict(X[vars2d])

	## Evaluate prediction accuracy. If probability is higher than 0.5, the sample is classified as red wine (y==1)
	def prediction_accuracy(y, p):
	    return sum(y == (p>0.5)) / float(len(y))

	print prediction_accuracy(y, p)


	## Visialize the result by a scatter plot and decision surface
	# 1d grid that spans the values, evenly.
	def vargrid(x, n=300):
		x0, x1 = min(x), max(x); return np.arange(x0, x1, (x1 - x0)/float(n))

	## Draw a plot
	def visualize_result_2d_linear(y, X, ml, var1, var2, alpha=0.25):
	    # Set up a 2d grid
	    x1g, x2g = np.meshgrid(vargrid(X[var1]), vargrid(X[var2]))
	    Xg = pd.DataFrame({var1 : np.ravel(x1g), var2 : np.ravel(x2g), 'intercept' : 1.0})
	    # Draw predictions below
	    p = ml.predict(Xg[[var1, var2, 'intercept']])
	    #h = pl.contourf(x1g, x2g, np.reshape(p, x1g.shape), 500, alpha=0.2)
	    h = pl.contourf(x1g, x2g, np.reshape(p, x1g.shape), 300, cmap=pl.cm.gist_yarg)
	    cbar = pl.colorbar()
	    cbar.set_label('probability for y=1')
	    # ... and data above.
	    pl.plot(X[var1][y==0], X[var2][y==0], 'go', markeredgecolor='g', alpha=alpha)
	    pl.plot(X[var1][y==1], X[var2][y==1], 'ro', markeredgecolor='r', alpha=alpha)
	    pl.xlabel(var1); pl.ylabel(var2)
	    pl.show()

	visualize_result_2d_linear(y, X, mfit, var1, var2)
	# Notice: In figure, there is a zoom tool that you may use to change axis limits.

Some questions:
* Is this the best combination of two variables?
* Is the linear decision boundary (straight line) the best choice?
* How can we tell?
* Would a more complex boundary be better?


## Construct a more complicated model

	## Let us use nonlinear transformations of the selected variables in the logistic regression model
	def add_mterm(X, vars):
	    X = X.copy()
	    X['*'.join(vars)] = np.multiply.reduce(X[list(vars)].values, 1)
	    return X

	## Create some transformations using powers and products of original variables
	# For example: nonlins = [(var1, var1), (var2, var2), (var1, var1, var1), (var2, var2, var2), (var1, var2)]
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

	h = pl.contourf(x1g, x2g, np.reshape(pg, x1g.shape), 300, cmap=pl.cm.gist_yarg)
	cbar = pl.colorbar()
	cbar.set_label('probability for y=1')
	pl.plot(X[var1][y==0], X[var2][y==0], 'g^', markeredgecolor='g', alpha=0.5)
	pl.plot(X[var1][y==1], X[var2][y==1], 'r*', markeredgecolor='r', alpha=0.5)
	pl.xlabel(var1); pl.ylabel(var2)
	pl.show()

	# In figure, there is a zoom tool that you may use to change axis limits.

Some questions:
* Is this better than the linear decision boundary? How can we tell?
* Would it be beneficial to use more than two variables? How would that happen? How can the results be visualized?