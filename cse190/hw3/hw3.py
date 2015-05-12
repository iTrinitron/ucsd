import gzip
import numpy
import urllib
import scipy.optimize
import random
from sklearn.decomposition import PCA
from collections import defaultdict

def readGz(f):
  for l in gzip.open(f):
    yield eval(l)
	
def parseData(fname):
  for l in urllib.urlopen(fname):
    yield eval(l)

data = list(parseData("train.json"))
	
##################################Problem 1#################################
#Fit a simple predictor
X = [[l['reviewerID'],l['itemID'],l['rating']] for l in data]
#Get list of reviews
y = [l[2] for l in X]
#Change to array to manipulate functions
y = numpy.array(y)
#Get mean of reviews
ymean = y.mean()  ##ANSWER

real = []
full = []
for l in open("labeled_Rating.txt"):
	u,i,r = l.strip().split(' ')
	real.append(float(r))

for l in data:
	u,i,r = l['reviewerID'],l['itemID'],l['rating']
	full.append((u,i,r))

#Grab the end
real = numpy.array(real)

#MSE
mse_data = numpy.array((ymean - real)**2)
mse = sum(mse_data)  ##ANSWER

##################################Problem 2#################################
X = [(l['reviewerID'],l['rating']) for l in data]
Y = [(l['itemID'],l['rating']) for l in data]
res = defaultdict(list)
res2 = defaultdict(list)
for v, k in X: res[v].append(k)
for a, b in Y: res2[a].append(b)

#rez has all the means
rez = res
rez2 = res2
for u in res:
	rez[u] = (sum(res[u])/len(res[u])) - ymean
for i in res2:
	rez2[i] = sum(res2[i])/len(res2[i]) - ymean

#ANSWER 2	
ans2 = ymean + rez2['I102776733'] + rez['U566105319']
#MSE

#regular
# for p in rez:
MSK = 0
for f in full:
	MSK += ((f[2] - (ymean + rez[f[0]] + rez2[f[1]]))**2) 
#ANSWER
print MSK

##################################Problem 3#################################
#U229891973
#U622491081
X = [(l['reviewerID'],l['itemID']) for l in data]
res = defaultdict(list)
for v, k in X: res[v].append(k)
res['U229891973']
res['U622491081']

#Find common elements
def jax(u1,u2):
	common = set(res[u1]) & set(res[u2])
	numCommon = len(common)
	union = set(res[u1]) | set(res[u2])
	numUnion = len(union)
	#JACCARD
	jacc = float(numCommon) / float(numUnion)
	return jacc
	
print jax('U229891973', 'U622491081')# ANSWER PART A

high = 0
jak = []
for u in res:
	jac = jax(u, 'U622491081')
	if jac > high:
		high = jac
		jak = [u]
	elif jac == high:
		jak.append(u)
print str(jak)# ANSWER PART B
		
#TEST to see if corrrect
for j in jak:
	print jax(j, 'U622491081')
		
##################################Problem 4#################################
#foreach -- sum(helpful) / len(helpful) = value <-- find the mean of that
avg =[]
for l in data:
	helpful,total = l['helpful']['nHelpful'],l['helpful']['outOf']
	avg.append(float(helpful)/float(total))
a = sum(avg)/len(avg)# ANSWER PART A

mse = 0
absolute = 0
for l in data:
	u,i,helpful,total = l['reviewerID'],l['itemID'],l['helpful']['nHelpful'],l['helpful']['outOf']
	mse += ((helpful - a*total)**2)
	absolute += abs((helpful - a*total))
#ANSWER PART B
print str(mse) 
print str(absolute)

wordCount = 0
rating = 0
for d in data:
	wordCount += len(d['reviewText'].split(' '))
	rating += 




