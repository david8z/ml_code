"""
Author: David Alarcn Sgearra
Mail: alarconsegarradavidgmail.com

Description: Implementation of the perception algorithm that generates the decision boundary of two classes.

"""
import matplotlib.pyplot as plt
import numpy as np
import random

####################
# GLOBAL VARIABLES #
####################


def gaussian_distribution(samples=1000,media=0):
	"""
	Returns 2 dimensional gaussian distribution.
	@params
	samples, number of sample (default 1000)
	"""
	x,y =[],[] 
	for i in range(samples):
		x.append(random.gauss(media,0.6))
		y.append(random.gauss(media,0.6))
	return [list(a) for a in zip(x , y)]

def linearly_separable(dist_a, dist_b):
	
	return true 

#/TODO: Function that generates random.gauss()
#/TODO: Function that checks if two datasets are linearly separable
dist_1 = np.array(gaussian_distribution(10,0))
dist_2 = np.array(gaussian_distribution(100,4))

print np.hstack((np.ones((len(dist_1,0)),dist_1)) 

"""

np.hstack((np.ones((len(dist_2,1)),dist_2)) 

print dist_1
[f.append(1) for f in dist_2]

global_dist = dist_1 + dist_2
random.shuffle(global_dist)

plt.plot(list(zip(*dist_1)[0]),list(zip(*dist_1)[1]),'o')
plt.plot(list(zip(*dist_2)[0]),list(zip(*dist_2)[1]),'o')
plt.show()

#X represent our data points, [ featur 1, feature 2, class ]
X = np.array([[0, 3, 1], [1, 1, 1],  [-1, 1, 0], [1, -1, 0]])
#Matrix of weights for a and b initialized to 0
W = [np.zeros(len(dist_1)),np.zeros(len(dist_1))]
#Learning rate
alfa = 0.5
#Margin
beta = 0.2
#Classes array
C= list(set(zip(*dist_1)[2]))+ list(set(zip(*dist_2)[2]))
#Number of good classified samples
m = 0
while m < len(X):
	#We need to reset m so that it checks again all the points this will generate
	#an infinite loop in case it isn't linearly separable.
	m=0
	#Loop to go through all the points
	for x in global_dist:
		#Store the class at i and inittiate g
		i = x[2]
		g = np.dot(W[i].reshape(1,3),np.concatenate(([1],x[:2])))[0]
		error = False
		#Go through all the classes to see if the function of any of the classes
		#is greater than the function of the actual class (g) in case of this weights
		#are updated
		for c in C:
			if c!=i:
				if np.dot(W[c].reshape(1,3),np.concatenate(([1],x[:2])))[0] + beta > g: 
					W[c] = W[c]-alfa*np.concatenate(([1],x[:2])) 
					error = True
		if error:
			W[i]=W[i]+alfa*np.concatenate(([1],x[:2]))
		else:	

			m=m+1


print 'Weight of the classes: ',W

#PLOTTING
class_1 = [list(elem[:2]) for elem  in filter(lambda x: x[2]==0, X)]
class_2 = [list(elem[:2]) for elem in  filter(lambda x: x[2]==1, X)]
plt.plot(class_1[:][0],class_1[:][1],'o')
plt.plot(map((lambda x: x[0]), class_2),map((lambda x: x[1]), class_2),'*')

#Extract the function from the weights
x = np.linspace(-20,22,3)
y =( W[0][0]-W[1][0]+x*(W[0][1]-W[1][1]))/(W[1][2]-W[0][2])
plt.plot(x,y,'-')
plt.show()
"""
