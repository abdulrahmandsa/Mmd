import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
df = pd.read_csv('GroceryStoreDataSet.csv', names = ['products'], sep = ',')
df.head()
df.shape
data = list(df["products"].apply(lambda x:x.split(",") ))
data
from mlxtend.preprocessing import TransactionEncoder
a = TransactionEncoder()
a_data = a.fit(data).transform(data)
df = pd.DataFrame(a_data,columns=a.columns_)
df = df.replace(False,0)
df
df = apriori(df, min_support = 0.2, use_colnames = True, verbose = 1)
df
df_ar = association_rules(df, metric = "confidence", min_threshold = 0.6)
df_ar



#b
import numpy
def PageRank(A, d = 0.85, eps = 0.0005, maxIterations = 1000,
verbose = False):
# find the size of the "Internet"
N = A.shape[0]
# initialize the old and new PageRank vectors
vOld = numpy.ones([N])
vNew = numpy.ones([N])/N
# initialize a counter
i = 0
# compute the update matrix
U = d * A.T + (1 - d) / N
while numpy.linalg.norm(vOld - vNew) >= eps:
# if the verbose flag is true, print the progress at each iteration
if verbose:
print('At iteration', i, 'the error is',
numpy.round(numpy.linalg.norm(vOld - vNew), 3),
'with PageRank', numpy.round(vNew, 3))
# save the current PageRank as the old PageRank
vOld = vNew
# update the PageRank vector
vNew = numpy.dot(U, vOld)
# increment the counter
i += 1
# if it runs too long before converging, stop and notify the user
if i == maxIterations:
print('The PageRank algorithm ran for',
maxIterations, 'with error',
numpy.round(numpy.linalg.norm(vOld - vNew), 3))
# return the PageRank vectora and the
return vNew, i
# return the steady state PageRank vector and iteration number
return vNew, i
# transition probability matrix
A = numpy.array([[0, 1/4, 1/4, 1/4, 1/4],
[1/2, 0, 0, 1/2, 0],
[1/3, 0, 0, 1/3, 1/3],
[1, 0, 0, 0, 0],
[0, 0, 0, 1, 0]])
# Run the PageRank algorithm with default settings
PageRank(A, verbose = True)
