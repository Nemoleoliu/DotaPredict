import sys
from feature_extract import *
# from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn import neighbors
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.utils import shuffle
from sklearn.cross_validation import cross_val_score

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(0.1, 1.0, 100, endpoint=True)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

def get_score(model, X, y):
    model.fit(X, y)
    print ("Cross validating...")
    print (np.mean(cross_val_score(model, X_train, y, scoring='roc_auc')))


def main(count):
    #["Logistic Regression", "KNeighbors Classifier", "Random Forest Classifier", "Gradient Boosting Classifier"]
    names = ["Logistic Regression"]
    classifiers = [
        # Logistic model
        #log_regr = linear_model.LogisticRegression(),
        linear_model.LogisticRegression(C=1.0)
        # knn model
        #knn_model = neighbors.KNeighborsClassifier(n_neighbors=5),
        #neighbors.KNeighborsClassifier(n_neighbors=15),
        #random forest model  RFC(max_depth=5, n_estimators=10, max_features=1)
        #random_forest = RandomForestClassifier(max_depth=5, n_estimators=15, max_features=3)
        #RandomForestClassifier(max_depth=20, n_estimators=15, max_features=3),
        #GradientBoostingClassifier()
        ]

    # get dataset
    dataset_X, dataset_y = feature_vector_extract(count)
    X_shuf, Y_shuf = shuffle(dataset_X, dataset_y)
    # cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
    for name, classifier in zip(names, classifiers):
        # plot_learning_curve(classifier, 'Learning Curve-%d (%s)' %(count,name), X_shuf, Y_shuf)
        get_score(classifier, X_shuf, Y_shuf)

if __name__ == '__main__':
    count = 1000
    if len(sys.argv) == 1:
        print('We are now using 1000 samples to train.')
        sys.argv.append(count)

    for i in sys.argv[1:]:
        count = int(i)
        main(count)
    plt.show()
