import itertools
import numpy as np

def complement_splits(x, islice):
    n = 1 + np.max(islice)
    return ((x[islice != i], x[islice == i]) for i in xrange(n))

def cv(y, X, nsplit, f_predict, f_evaluate):
    """
    Cross validation, with nsplit folds. y and X are target and predictors.
    f_predict(y, X, Xnew) produces predictions for Xnew, with a model trained with (y, X).
    f_evaluate: f_evaluate(predictions, y) gives a summary statistic for a slice
    """
    s = None; np.random.seed(234)
    islice = np.random.permutation(len(y)) % nsplit
    csp = complement_splits
    for ((ytr, yev), (Xtr, Xev)) in itertools.izip(csp(y, islice), csp(X, islice)):
        r = f_evaluate(f_predict(ytr, Xtr, Xev), yev)
        s = r if s==None else s + r
    return s
