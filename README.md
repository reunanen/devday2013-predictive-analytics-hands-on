Dev Day 2013: Predictive Analytics Hands-on session
===================================================

If possible, please prepare for this session by downloading the virtual machine or the relevant libraries *beforehand*. (See section "Getting ready" below.)

It is a good idea to pair up for this session. If you are not familiar with Python, it is recommended that you find someone who is.

# Useful links

Practical ones:

* [Numpy tutorial](http://wiki.scipy.org/Tentative_NumPy_Tutorial): we will not use Numpy extensively, just for light data manipulation. Numpy is the [R](http://r-project.org/) or Matlab of Python.
* [A short tutorial](http://pandas.pydata.org/pandas-docs/stable/10min.html) on Pandas, the Python "data frame" library. Pandas is a higher-level structure on arrays, allowing heterogeneous columns, column names, etc.
* [Logistic regression with Python](http://blog.yhathq.com/posts/logistic-regression-and-python.html) and its statsmodels package. Logistic regression is a way to make a classifier that gives out probabilities.
* [Regularized logistic regression with Python](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) from the [Scikit-learn](http://scikit-learn.org/) package. We don't introduce regularization in this hands-on. In practice it is almost mandatory if you have a large predictive problem. 


More theoretical:

* [Logistic regression on Wikipedia](http://en.wikipedia.org/wiki/Logistic_regression). Easy to understand: a linear regression stacked with an S-curve to make outputs look like probabilities. But in its [full](http://fastml.com/regression-as-classification/) [glory](http://arantxa.ii.uam.es/~jmlobato/docs/slidesCambridge.pdf), a logistic regression model is far from elementary!
* Want to go deeper, with Python? Try [Probabilistic Programming and Bayesian Methods for Hackers](https://github.com/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers), a book freely available on Github.

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

Analytics produces actionable information from observations or from results of experiments ("data sets"). The results of analytics are diverse. They may be just informative summaries of the data, with uncertainty quantified as probabilities or confidence intervals. Analytics may reveal underlying relationships, suggest causality, or new hypotheses for further testing. Sometimes the results are very operational suggestions of actions, possibly embedded into an IT system, so that humans only see the actions, not the analytics. Statistical and other kind of _models_ are used to achieve these goals.

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

![Histogram with classes](img/hists_clasess2.png)

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

        import pandas as pd
        import statsmodels.api as sm
        import pylab as pl
        import numpy as np

        # Our own plotting functions, somewhat specific to this hands-on session. 
        from plots import *

## How to read in the data

        # Read the data (edit the path if necessary).
        # We us pandas DataFrame data structure. It is a 2-dimensional
        # labeled data structure with columns of potentially
        # different type.
        # See more info: http://pandas.pydata.org/pandas-docs/dev/dsintro.html
        # The data file has no row indices/names, hence index_col=F.
        df = pd.read_csv("wine_quality_data.csv", index_col=False)

## The first part: Insights into the data

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

## The first part: Further insights

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
        {code}

Some questions:
* What can be seen from the histograms? Do they give any indication which variables could be useful in classification of wines?
* How to interpret the scatter plot? If we know values of both variables of an unknown wine sample, what can we say about the color of the wine? When can we say something and when not?
* How could this be modelled automatically? (Hint: see next two sections below.)


## The second part: Find a linear decision boundary

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
        model = sm.Logit(y, X)                          # Create model object.
        mfit = model.fit()                              # Train the model, i.e., optimize its parameters
        print mfit.summary()                            # It did something!


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

Some questions:
* Is this the best combination of two variables?
* Is the linear decision boundary (straight line) the best choice?
* How can we tell?
* Would a more complex boundary be better?


## The second part: Find A nonlinear decision boundary

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
            # Hint: You may edit the terms to test different terms.
            return [(v1, v1), (v2, v2), (v1, v1, v1), (v2, v2, v2), (v1, v2)]
        # Let's make a data with all the nonlinear terms above!
        v1, v2 = 'fixed_acidity', 'chlorides'
        Xn = reduce(add_mterm, nonlins(v1, v2), preds2(df, v1, v2))
        y = is_red(df)                                     # (just as a reminder)

        # Fit the model with all the new nonlinear terms. 
        m_nl = sm.Logit(y, Xn)           # Model object with the expanded set of predictors. 
        mfit_nl = m_nl.fit()                     # Fit it. 
        print mfit_nl.summary()                        # See how it is

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

Some questions:
* Is this better than the linear decision boundary? How can we tell?
* Would it be beneficial to use more than two variables? How would that happen? How can the results be visualized?

## The third part: Automated model selection

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

        # Cross validation does not waste half of the data for testing:
        # It splits the data set into (e.g.) 10 pieces, trains on 9, tests on the remaining one,
        # and loops this through all the ten pieces. 

        from crossvalidation import cv

        n_success = cv(is_red(df), my_preds2(df), 10,
                   # The function to build the model and predict
                   lambda ytr, Xtr, Xev: sm.Logit(ytr, Xtr).fit().predict(Xev), 
                   # The function to evaluate results: How many are right?
                   lambda p, y: sum(y == (p>0.5))) 
        print 100*n_success/float(len(df))
