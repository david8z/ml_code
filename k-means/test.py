import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

data_frame = pd.read_csv('iris.data.csv', sep=',')
data_frame = data_frame.ix[:5,:-1]
data = data_frame.values
np.random.shuffle(data)
print(data)
pca = PCA(n_components=2,)#tsne = TSNE(n_components = 2)
data_2d =pca.fit_transform(data)
# = tsne.fit_transform(data)

print(sum(data_2d))
