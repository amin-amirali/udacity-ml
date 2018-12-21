# AgglomerativeClustering
from sklearn import datasets
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score
from sklearn import preprocessing

iris = datasets.load_iris()
# normalizing data
normalized_X = preprocessing.normalize(iris.data)
normalized_X[:10]

# linkage : {“ward”, “complete”, “average”, “single”}, optional (default=”ward”)
ward = AgglomerativeClustering(n_clusters=3, linkage='ward')
ward_pred = ward.fit_predict(normalized_X)

ward_ar_score = adjusted_rand_score(iris.target, ward_pred)

from scipy.cluster.hierarchy import linkage

# Specify the linkage type. Scipy accepts 'ward', 'complete', 'average', as well as other values
# Pick the one that resulted in the highest Adjusted Rand Score
linkage_type = 'ward'
linkage_matrix = linkage(normalized_X, linkage_type)

from scipy.cluster.hierarchy import dendrogram
import matplotlib.pyplot as plt
plt.figure(figsize=(22,18))

# plot using 'dendrogram()'
dendrogram(linkage_matrix)

plt.show()

