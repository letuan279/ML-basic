from sklearn import cluster
import matplotlib.pyplot as plt
import numpy as np

means = [[2, 8], [8, 8], [5, 3]]
cov = [[1, 0], [0, 1]]
N = 300
X0 = np.random.multivariate_normal(means[0], cov, N)
X1 = np.random.multivariate_normal(means[1], cov, N)
X2 = np.random.multivariate_normal(means[2], cov, N)

X = np.concatenate((X0, X1, X2), axis=0)
K = 3

kmeans = cluster.KMeans(n_clusters=3, random_state=0, n_init='auto').fit(X)
centers = kmeans.cluster_centers_
print('Centers: ')
print(centers)
pred_label = kmeans.predict(X)

X0 = X[pred_label == 0, :]
X1 = X[pred_label == 1, :]
X2 = X[pred_label == 2, :]

plt.plot(X0[:, 0], X0[:, 1], 'b^', markersize=4)
plt.plot(X1[:, 0], X1[:, 1], 'go', markersize=4)
plt.plot(X2[:, 0], X2[:, 1], 'rs', markersize=4)
plt.plot(centers[:, 0], centers[:, 1], 'y^', markersize=10)
plt.show()
