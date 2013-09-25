# This requires the stuff of the main script to be present

pairs = triangular(chemvars(df).columns)
y = is_red(df)
r = []
for v1, v2 in pairs:
	X = reduce(add_mterm, ((v1, v1), (v2, v2), (v1, v2)), preds2(df, v1, v2))
	try: 
		nfail = cv(y, X, 40,
			   lambda ytr, Xtr, Xev: sm.Logit(ytr, Xtr).fit().predict(Xev),
			lambda p, y: sum(y != (p>0.5)))
	except numpy.linalg.linalg.LinAlgError:
		nfail = None 
	r.append((v1, v2, nfail))

print [(v1, v2, nf) for v1, v2, nf in r if nf==None]
r.sort(lambda i, j: cmp(i[2], j[2]))
for i in r[:20]: print i
