import sys
from io import StringIO
from sklearn.linear_model import SGDRegressor
import matplotlib as plt
import numpy as np


def compute_and_plot_gradient_descent(x: np.array, y: np.array,**kwargs) -> SGDRegressor:
    '''
    Args:
        x: Numpy array with training features
        y: Numpy array with training labels
    Returns:
        sgd: Trained sgd model
    Example invocation:
        sgd = compute_and_plot_gradient_descent(x_train,y_train, max_iter=10000, eta0=0.3, verbose=1)
    '''
    old_stdout = sys.stdout
    sys.stdout = mystdout = StringIO()
    sgd = SGDRegressor(**kwargs)
    sgd.fit(x, y)
    sys.stdout = old_stdout
    loss_history = mystdout.getvalue()
    loss_list = []
    for line in loss_history.split('\n'):
        if(len(line.split("loss: ")) == 1):
            continue
        loss_list.append(float(line.split("loss: ")[-1]))
    plt.figure()
    plt.plot(np.arange(len(loss_list)), loss_list)
    plt.xlabel("Time in epochs")
    plt.ylabel("Loss")
    plt.show()
    print(f"number of iterations completed: {sgd.n_iter_}, number of weight updates: {sgd.t_}")
    return sgd


def plot_linear_regression_results(slope, intercept, x, y) -> None:
    '''
    Args:
    slope: w, output of Linear regression model
    Intercept b, ouput of linear regression model
    x: Numpy array with training features
    y: Numpy array with training labels
    '''
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.scatter(x, y, marker='*', c='b')
    plt.plot(x_vals, y_vals, '--')