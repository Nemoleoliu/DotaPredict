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
from sklearn.cross_validation import cross_val_score
import warnings

def plot_learning_curve(estimator, title, X, y, file_name=None, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(0.1, 1.0, 100, endpoint=True)):
    fig = plt.figure()
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

    plt.savefig(file_name)
    plt.close(fig)


def get_score(model, X, y):
    # 0print "C =", model.C
    # print ("Training model...")
    model.fit(X, y)
    # print ("Cross validating...")
    print (np.mean(cross_val_score(model, X, y, scoring='roc_auc')))


def main(count, mode):
    #["Logistic Regression", "KNeighbors Classifier", "Random Forest Classifier", "Gradient Boosting Classifier"]
    names = ["Logistic Regression"]*100
    classifiers = [
        # Logistic model
        # log_regr = linear_model.LogisticRegression(),
        linear_model.LogisticRegression(C=1, n_jobs=-1),
        # knn model
        # knn_model = neighbors.KNeighborsClassifier(n_neighbors=5),
        # neighbors.KNeighborsClassifier(n_neighbors=15),
        # random forest model  RFC(max_depth=5, n_estimators=10, max_features=1)
        # RandomForestClassifier(max_depth=6, n_estimators=500, max_features=10),
        # GradientBoostingClassifier()
    ]
    file_nameX = './data/X-{0}-{1}.csv'.format(count, mode)
    file_namey = './data/y-{0}-{1}.csv'.format(count, mode)
    X = pd.read_csv(file_nameX)
    y = pd.read_csv(file_namey)
    y = y['0']
    for name, classifier in zip(names, classifiers):
        plot_learning_curve(
            classifier,
            'Learning Curve-%d (%s)' %(count, name),
            X,
            y,
            file_name='%s(%d)-(M%d)' %(name, count, mode)
        )
        get_score(classifier, X, y)

if __name__ == '__main__':
    count_list = []
    for arg in sys.argv[1:]:
        count_list.append(int(arg))
    for count in count_list:
        main(count, 1)
