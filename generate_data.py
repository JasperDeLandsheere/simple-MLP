import sklearn.datasets as dt
from sklearn.model_selection import train_test_split
import numpy as np

def normalize(data, mu, std):
    return (data - mu) / std

def generate_data():
    X, y = dt.make_regression(n_samples = 1000, n_features = 20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    X_mean, X_std = np.mean(X), np.std(X)
    y_mean, y_std = np.mean(y), np.std(y)
    X_train = normalize(X_train, X_mean, X_std)
    X_test = normalize(X_test, X_mean, X_std)
    y_train = normalize(y_train, y_mean, y_std)
    y_test = normalize(y_test, y_mean, y_std)
    return X_train, X_test, y_train, y_test
def generate_data_for_visualisation():
    X = np.linspace(-1, 1, 1000).reshape(-1, 1)
    y = (np.sin(6 * X) + np.cos(2 * X)).reshape(-1) + np.random.normal(0, 0.05, 1000)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    X_mean, X_std = np.mean(X), np.std(X)
    y_mean, y_std = np.mean(y), np.std(y)
    X_train = normalize(X_train, X_mean, X_std)
    X_test = normalize(X_test, X_mean, X_std)
    y_train = normalize(y_train, y_mean, y_std)
    y_test = normalize(y_test, y_mean, y_std)
    return X_train, X_test, y_train, y_test