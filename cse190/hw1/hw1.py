import numpy
import urllib
import scipy.optimize
import random

def parseData(fname):
  for l in urllib.urlopen(fname):
    yield eval(l)

print "Reading data..."
data = list(parseData("http://jmcauley.ucsd.edu/cse190/data/beer/beer_50000.json"))
print "done"


##1.2


##1.3
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

##1.4
