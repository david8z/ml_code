"""
Author: David Alarcn Sgearra
Mail: alarconsegarradavidgmail.com

Description: Implementation of the perception algorithm that generates the decision boundary of two classes.

"""
import matplotlib.pyplot as plt
import numpy as np

#X represent our data points, [ featur 1, feature 2, class ]
X = np.array([[0, 3, 1], [1, 1, 1],  [-1, 1, 0], [1, -1, 0]])
#Matrix of weights for a and b initialized to 0
W = [np.zeros(3),np.zeros(3)]
#Learning rate
alfa = 0.7
#Margin
beta = 0.2
#Classes array
C=[0,1]
#Number of good classified samples
m = 0

while m < len(X):
	#We need to reset m so that it checks again all the points this will generate
	#an infinite loop in case it isn't linearly separable.
	m=0
	#Loop to go through all the points
	for x in  X:
		print m, x
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
