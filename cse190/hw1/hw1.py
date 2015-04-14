import numpy
import urllib
import scipy.optimize
import random

#MSE
from sklearn.metrics import mean_squared_error

def parseData(fname):
  for l in urllib.urlopen(fname):
    yield eval(l)

print "Reading data..."
data = list(parseData("http://jmcauley.ucsd.edu/cse190/data/beer/beer_50000.json"))
print "done"


##1.2


##----------------------------1.3
data2 = [d for d in data if d.has_key('beer/ABV')]

def feature(datum):
  feat = [1]
  feat.append(datum['beer/ABV'])
  return feat

X = [feature(d) for d in data2]
y = [d['review/taste'] for d in data2]
theta,residuals,rank,s = numpy.linalg.lstsq(X, y)

## 3.11521115, 0.10905507
## Rating For 0 ABV, Increase as ABV grows -- beer is rated higher, the higher the ABV

##----------------------------1.4

##split data in half
def split_list(a_list):
    half = len(a_list)/2
    return a_list[:half], a_list[half:]

data1, data2 = split_list(data)

X = [feature(d) for d in data1]
y = [d['review/taste'] for d in data1]
theta,residuals,rank,s = numpy.linalg.lstsq(X, y)

X2 = [feature(d) for d in data2]
y2 = [d['review/taste'] for d in data2]
theta,residuals,rank,s = numpy.linalg.lstsq(X2, y2)

#mean_squared_error(y_actual, y_predicted)
mean_squared_error(X, X2)

##training set
#2.99503282, 0.11690802
print theta


##2.1
bData = list(parseData("http://jmcauley.ucsd.edu/cse255/data/amazon/book_descriptions_50000.json"))
