import sys
from feature_extract import *
# from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn import neighbors
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.cross_validation import cross_val_score
from sklearn.neural_network import MLPClassifier
import warnings

def plot_learning_curve(estimator, title, X, y, file_name=None, ylim=None, cv=None,
                        n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 100, endpoint=True)):
    fig = plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring='accuracy')
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
    models = {}
    # models['Logistic Regression'] = linear_model.LogisticRegression(C=1, n_jobs=-1)
    # models['Random Forest'] = RandomForestClassifier(max_depth=6, n_estimators=500, max_features=10)
    # models['KNN'] = neighbors.KNeighborsClassifier(n_neighbors=5)
    models['ANN'] = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(100, 50, 10, 5), random_state=1)
    file_nameX = './data/X-{0}-{1}.csv'.format(count, mode)
    file_namey = './data/y-{0}-{1}.csv'.format(count, mode)
    X = pd.read_csv(file_nameX)
    y = pd.read_csv(file_namey)
    y = y['0']
    for name,classifier in models.iteritems():
        plot_learning_curve(
            classifier,
            'Learning Curve-%s (%d)' %(name, count),
            X,
            y,
            file_name='%s(%d)-(M%d)' %(name, count, mode)
        )
        get_score(classifier, X, y)

def test_feature_weight(mode):
    count = 18000
    file_nameX = './data/X-{0}-{1}.csv'.format(count, mode)
    file_namey = './data/y-{0}-{1}.csv'.format(count, mode)
    X = pd.read_csv(file_nameX)
    y = pd.read_csv(file_namey)
    y = y['0']
    forest = ExtraTreesClassifier(n_estimators=250,random_state=0)
    forest.fit(X, y)
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")

    for f in range(X.shape[1]):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

    # Plot the feature importances of the forest
    fig = plt.figure()
    plt.title("Feature importances")
    plt.bar(range(X.shape[1]), importances[indices],
           color="r", yerr=std[indices], align="center")
    plt.xticks(range(X.shape[1]), indices)
    plt.xlim([-1, X.shape[1]])
    plt.savefig('Feature Weight Mode {0}'.format(mode))
    plt.close(fig)


if __name__ == '__main__':
    count_list = []
    for arg in sys.argv[1:]:
        count_list.append(int(arg))
    # for count in count_list:
    #     main(count, 0)
    test_feature_weight(0)
    test_feature_weight(1)
    test_feature_weight(2)
    test_feature_weight(3)
    test_feature_weight(4)
