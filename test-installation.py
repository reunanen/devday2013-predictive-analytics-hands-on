import pandas as pd
import statsmodels.api as sm
import pylab as pl
import numpy as np

df = pd.read_csv("wine_quality_data.csv", index_col=False)
def is_red(df): return np.array(df['wine_color']=="red", int)
y = is_red(df)

def preds2(df, v1, v2):
	X = df[[v1, v2]]
	X['intercept'] = 1.0
	return X

model = sm.Logit(y, preds2(df, 'alcohol', 'free_sulfur_dioxide'))
mfit = model.fit()
print mfit.summary()


print "Plotting functions and data are imported, and models do fit, unless you see an error message above."
