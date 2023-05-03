from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score

X, y = make_classification(n_samples=1000, n_features=4,
                           n_informative=2, n_redundant=0, random_state=0, shuffle=False)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=100)
rf = RandomForestClassifier(
    n_estimators=100, max_depth=2, random_state=0, criterion='entropy')
rf.fit(X_train, y_train)
predictions = rf.predict(X_test)
print(100 * accuracy_score(y_test, predictions))
