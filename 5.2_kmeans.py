
from sklearn.cluster import KMeans

model = KMeans(n_clusters=2)
predictions = model.fit_predict(X)


