import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from collections import defaultdict
import matplotlib.pyplot as plt

#Method to compute euclidean distance
def euclidean(centroid, datapoint):
	return np.sqrt(np.sum((datapoint-centroid)**2))


def kmeans_initialization(k,  data):
	#add column to data that specifies cluster it belongs to
	centroids = defaultdict(lambda: np.ndarray(0))
	clusters = defaultdict(lambda: np.empty((0,4)))
	#initialize k random centroids
	for c in range(1,k+1):
		#centroids are random tuples with the data dimensionality
		centroids[c]=data[np.random.randint(2,len(data))]
	for d in data:
		min_dist = 1000000
		#update centroid of the point
		for c in centroids.keys():
			aux_dist = euclidean(centroids[c],d)
			if min_dist > aux_dist:
				min_dist = aux_dist
				max_centroid = c
		clusters[max_centroid] = np.append(clusters[max_centroid],[d],axis=0)
	return kmeans_auxiliar(centroids, clusters)

def kmeans_auxiliar(centroids, clusters):
	transformed = False
	for c in centroids.keys():
		centroids[c]=sum(clusters[c])/len(clusters[c])
	for cl in clusters.keys():
		delete_list = []
		data = clusters[cl]
		for pos, d in enumerate(data):
			min_dist = 1000000
			#update centroid of the point
			for c in centroids.keys():
				aux_dist = euclidean(centroids[c],d)
				if min_dist > aux_dist:
					min_dist = aux_dist
					max_centroid = c
			if max_centroid != cl:
				transformed = True
				delete_list.append(pos)
				clusters[max_centroid] = np.append(clusters[max_centroid],[d],axis=0)
		
		clusters[cl]=np.delete(clusters[cl],delete_list,axis=0)
	if transformed:
		return kmeans_auxiliar(centroids, clusters)
	else:
		return clusters


#reading iris data set
data_frame = pd.read_csv('iris.data.csv', sep=',')
data_frame = data_frame.ix[:,:-1] 

#we make the feature vectors so that all features have same influence
data = data_frame.ix[:,:].values
standard_scalar = StandardScaler()
data_std = standard_scalar.fit_transform(data)
np.random.shuffle(data_std)

clusters = kmeans_initialization(3, data_std)

for c in range(1,4):
	if c==1:
		data = clusters[c]
		cl_data = [c]*len(clusters[c])
		print(len(clusters[c]))
	else:
		data = np.append(data,clusters[c],axis=0)
		cl_data= cl_data + ([c]*len(clusters[c]))
		print(len(clusters[c]))



#dimensionality reduction TSNE and PCA
#tsne = TSNE(n_components=2,random_state=0)
#data_2d = tsne.fit_transform(data)
pca = PCA(n_components=2)
data_2d = pca.fit_transform(data)

#plotting
color_map = {1:'red', 2:'blue',3:'lightgreen'}
for p, c in zip(data_2d, cl_data):
	plt.scatter(x=p[0],y=p[1],c=color_map[c])
plt.show()
