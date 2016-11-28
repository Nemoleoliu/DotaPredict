from perf_output import *
from feature_extract import *
from sklearn.model_selection import train_test_split
from sklearn import linear_model

def main():
    count = 1000
    dataset_X, dataset_y = feature_vector_extract(count)
    X_train, X_test, y_train, y_test = train_test_split(dataset_X, dataset_y)
    regr = linear_model.LinearRegression();
    # Use logisitic regression to predict
    regressionFunc = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
    train_sco=regressionFunc.fit(X_train,y_train).score(X_train,y_train)


    # Test
    test_sco=regressionFunc.score(X_test,y_test)

    # perf_output

if __name__ == '__main__':
    main()
