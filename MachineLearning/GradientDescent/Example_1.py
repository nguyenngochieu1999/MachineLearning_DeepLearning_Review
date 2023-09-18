import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return 3*x**2 - 10*np.sin(x)

def grad(x):
    return 6*x - 10*np.cos(x)

def grad_descent(x0, learning_rate):
    x = [x0]
    for i in range(2000):
        x_new = x[-1] - learning_rate*grad(x[-1])
        x.append(x_new)
        if abs(grad(x_new)) < 1e-3:
            break

    return np.array(x)
      
if __name__ == '__main__':
    x_test = np.arange(-10, 10)
    y_test = f(x_test)
    plt.plot(x_test, y_test)
    
    x_init_1 = 100
    x_init_2 = -100
    learning_rate = 0.001
    
    x_result_1 = grad_descent(x_init_1, learning_rate)
    x_result_2 = grad_descent(x_init_2, learning_rate)
    y_result_1 = f(x_result_1)
    y_result_2 = f(x_result_2)

    # plt.scatter(x_result_1[-1], y_result_1[-1], 40, c=2)
    plt.scatter(x_result_2[-1], y_result_2[-1], 40, c=4)

    plt.show()