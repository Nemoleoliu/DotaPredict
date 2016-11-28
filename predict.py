from perf_output import *
from feature_extract import *
from sklearn.model_selection import train_test_split

def main():
    count = 1000
    dataset = feature_vector_extract(count)
    data_train, data_test = train_test_split(dataset);

    data_train_X = data_train.iloc[:, predictor];
    data_train_y = data_train['MEDV'];
    data_test_X = data_test.iloc[:, predictor];
    data_test_y = data_test['MEDV'];
    regr = linear_model.LinearRegression();
    X_train, X_test, Y_train, Y_test = ...
    # Use logisitic regression to predict

    # Test

    # perf_output

if __name__ == '__main__':
    main()
