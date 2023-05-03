from sklearn import neighbors, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

iris = datasets.load_iris()
iris_X = iris.data
iris_y = iris.target

X0 = iris_X[iris_y == 0,:]
X1 = iris_X[iris_y == 1,:]
X2 = iris_X[iris_y == 2,:]

X_train, X_test, y_train, y_test = train_test_split(iris_X, iris_y, test_size=50)
# print(len(y_train))

clf = neighbors.KNeighborsClassifier(n_neighbors=10, p=2, weights='distance')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# print(y_pred[20:40])
# print(y_test[20:40])
print(100 * accuracy_score(y_pred, y_test), '%')