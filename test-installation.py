print "Imports"
import pandas as pd
import statsmodels.api as sm
import pylab as pl
import numpy as np
import sklearn.linear_model as skl
import sklearn.preprocessing as skp

print "Pandas, read_csv"
df = pd.read_csv("wine_quality_data.csv", index_col=False)
def is_red(df): return np.array(df['wine_color']=="red", int)
y = is_red(df)

def preds2(df, v1='alcohol', v2='free_sulfur_dioxide'):
	X = df[[v1, v2]]
	X['intercept'] = 1.0
	return X

print "Statsmodels.Logit"
model = sm.Logit(y, preds2(df))
mfit = model.fit()
assert 5125 < mfit.aic < 5126

print "Scikit-learn, linear_model and preprocessing"
scaler = skp.StandardScaler().fit(preds2(df))
m_l1 = skl.LogisticRegression(C=1, penalty='l1')
m_l1.fit(scaler.transform(preds2(df)), is_red(df))
assert -0.333 < m_l1.coef_[0,0] < -0.331

print "Everything may be ok, unless you see an error message above."
