import numpy
import urllib
import scipy.optimize
import random
import math

#MSE
from sklearn.metrics import mean_squared_error

def parseData(fname):
  for l in urllib.urlopen(fname):
    yield eval(l)

print "Reading data..."
data = list(parseData("http://jmcauley.ucsd.edu/cse190/data/beer/beer_50000.json"))
print "done"

# Problem 1.1.1
data2 =  [d['beer/beerId'] for d in data]
uniqueBeer = len(set(data2))
print ""
print "Number of unique beer: " + str(uniqueBeer)

# Problem 1.1.2
data2 =  [d['user/profileName'] for d in data]
uniqueUser = len(set(data2))
print ""
print "Number of unique beer: " + str(uniqueUser)

# Problem 1.1.3
data1 = [d['review/appearance'] for d in data]
data2 = [d['review/palate'] for d in data]
data3 = [d['review/overall'] for d in data]
data4 = [d['review/aroma'] for d in data]
data5 = [d['review/taste'] for d in data]

print numpy.mean(data1) 
print numpy.mean(data2) 
print numpy.mean(data3) 
print numpy.mean(data4) 
print numpy.mean(data5) 

# Problem 1.1.4
data5 = [d['beer/ABV'] for d in data]

print numpy.mean(data5)

# Problem 1.2
data1 = [d['review/taste'] for d in data]

print numpy.var(data1)

## Problem 1.3
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
	
def feature2(datum):
  feat = [1]
  feat.append(datum['beer/ABV'])
  feat.append(datum['review/taste'])
  return feat
  
def formula(x, t0, t1):
  sum = 0
  for i in x:
    sum += math.pow((i[1] - (t0 + t1*i[2])),2)
  return sum/len(x)

data1, data2 = split_list(data)

X = [feature(d) for d in data1]
y = [d['review/taste'] for d in data1]
theta,residuals,rank,s = numpy.linalg.lstsq(X, y)

Xy = [feature2(d) for d in data1]
MSE = formula(Xy, theta[0], theta[1])

pred = [(theta[0] + theta[1]*d[1]) for d in X]

mean_squared_error(y, pred)

X2 = [feature(d) for d in data2]
y2 = [d['review/taste'] for d in data2]
theta,residuals,rank,s = numpy.linalg.lstsq(X2, y2)

X2y2 = [feature2(d) for d in data1]
MSE2 = formula(X2y2, theta[0], theta[1])

pred2 = [(theta[0] + theta[1]*d[1]) for d in X2]

mean_squared_error(y2, pred2)


	


#mean_squared_error(y_actual, y_predicted)
mean_squared_error(y, y2)

##training set
#2.99503282, 0.11690802
print theta


##2.1
bData = list(parseData("http://jmcauley.ucsd.edu/cse255/data/amazon/book_descriptions_50000.json"))

# p(romance)
prior = ["Romance" in b['categories'] for b in bData]
prior = sum(prior) * 1.0 / len(prior)
print prior 
#0.03198

# p(mentions love | romance)
p1 = ['love' in b['description'] for b in bData if "Romance" in b['categories']]
p1 = sum(p1) * 1.0 / len(p1)
print p1

#0.4878048

#
p2 = ['Romance' in b['categories'] for b in bData if 'love' and 'beaut' in b['description']]
p2 = sum(p2) * 1.0 / len(p2)
print p2

#bottom half
p3 = ['love' and 'beaut' in b['description'] for b in bData if 'Romance' not in b['categories']]
p3 = sum(p3) * 1.0 / len(p3) 
print p3

#