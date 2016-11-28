from perf_output import *
from feature_extract import *
# from sklearn.model_selection import train_test_split
from sklearn import linear_model
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
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


def main():
    count = 160000
    # init the model
    regr = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
    # get dataset
    dataset_X, dataset_y = feature_vector_extract(count)
    cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
    plot_learning_curve(regr, 'Learning Curve (Logistic Regression)', dataset_X, dataset_y, n_jobs=4, cv=cv)
    plt.show()
    # # X_train, X_test, y_train, y_test = train_test_split(dataset_X, dataset_y)
    # # # Use logisitic regression to predict
    # # regressionFunc = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
    # # train_score=regressionFunc.fit(X_train,y_train).score(X_train,y_train)
    # # train_mse = np.mean((regressionFunc.predict(X_train) - y_train) ** 2)
    #
    # # Test
    # test_score=regressionFunc.score(X_test,y_test)
    # test_mse = np.mean((regressionFunc.predict(X_test) - y_test) ** 2)
    #
    # # perf_output



if __name__ == '__main__':
    main()
