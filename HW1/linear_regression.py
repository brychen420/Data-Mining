import numpy as np
import matplotlib.pyplot as plt

LR = 0.000016
EPOCH = 200000

def Mean_square_error(ys, y_pred):
    
    y_diff = ys - y_pred
    y_square = np.square(y_diff)
    mse = y_square.mean(axis=None)
    
    return mse


def gradient_descent(xs, ys, weight, bias):
    
    y_pred = np.matmul(xs, weight) + bias
    y_diff = ys - y_pred
    
    w_grad =  (-2/xs.shape[0]) * (np.matmul(xs.T, y_diff))
    b_grad = (-2/xs.shape[0]) * np.sum(y_diff)
    
    weight -= LR * w_grad
    bias -= LR * b_grad


def linear_regression(xs, ys, x_val, y_val, epoch):
    
    dim = xs.shape[1]
    weight = 4 * np.random.rand(dim, 1) - 2
    bias = 4 * np.random.rand() - 2
    
    ys = np.reshape(ys, (len(ys), 1))
    y_val_reshape = np.reshape(y_val, (len(y_val), 1))
    
    mse_list = []
    mse_val_list = []
    
    for _ in range(epoch):
        y_pred = np.matmul(xs, weight) + bias
        y_val_pred = np.matmul(x_val, weight) + bias
        
        mse = Mean_square_error(ys, y_pred)
        mse_val = Mean_square_error(y_val_reshape, y_val_pred)
        
        mse_list.append(mse)
        mse_val_list.append(mse_val)
        
        gradient_descent(xs, ys, weight, bias)
    
    return weight, bias, mse_list, mse_val_list


def plot_result(mse_list, mse_val_list, start):
    
    x = np.arange(start, len(mse_list))
    plt.plot(x, mse_list[start:], label='training')
    plt.plot(x, mse_val_list[start:], label='validation')
    plt.xlabel('iteration')
    plt.ylabel('MSE')
    plt.legend()
    plt.show()