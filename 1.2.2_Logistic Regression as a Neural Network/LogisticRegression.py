import numpy as np
import matplotlib.pyplot as plt
import h5py
from PIL import Image
import time



def load_dataset():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")

    train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # your train set labels

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # your test set labels

    classes = np.array(test_dataset["list_classes"][:])  # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


def savePicture(data, i=0):
    time_start = time.time()
    for image_i in range(data.shape[0]):
        new_im = Image.fromarray(data[image_i])
        plt.imshow(new_im)
        name = f"picture/{i+image_i}.png"
        print(name)
        plt.savefig(name)
    time_end = time.time()
    print(f"{i}_time={(time_end-time_start)} s")


def savePicMult(data_set, processesNum=1):
    import multiprocessing
    time_start = time.time()
    times = data_set.shape[0] // processesNum
    print(f"data_set={data_set.shape}")
    print(f"processesNum={processesNum}")
    print(f"times={times}")
    pool = multiprocessing.Pool(processes=processesNum)
    for i in range(processesNum):
        if i == processesNum - 1:
            data = data_set[0 + i * times:, ]
            print(data.shape)
        else:
            data = data_set[0 + i * times:0 + (i + 1) * times, ]
            print(data.shape)
        pool.apply_async(func=savePicture, args=(data, i * times))
    pool.close()
    pool.join()  # 调用join之前，先调用close函数，否则会出错。执行完close后不会有新的进程加入到pool,join函数等待所有子进程结束
    print("Sub-process(es) done.")
    time_end = time.time()
    print(f"time={(time_end-time_start)}s")


def reshape4DAndstandardize(X):
    # data1D=data4D.reshape(data4D.shape[1] * data4D.shape[2] * data4D.shape[3],
    #                          data4D.shape[0]).T
    # return data1D
    X_flatten = X.reshape(X.shape[0], -1).T
    # -1 自动计算
    return X_flatten / 255
    # 将数据转为列向量


def sigmoid(z):
    """
    Compute the sigmoid of z

    Arguments:
    z -- A scalar or numpy array of any size.

    Return:
    s -- sigmoid(z)
    """

    ### START CODE HERE ### (≈ 1 line of code)
    # s = 1/(1+cmath.e**(-z))
    s = 1 / (1 + np.exp(-z))
    ### END CODE HERE ###

    return s


def initialize_with_zeros(dim):
    """
    This function creates a vector of zeros of shape (dim, 1) for w and initializes b to 0.

    Argument:
    dim -- size of the w vector we want (or number of parameters in this case)

    Returns:
    w -- initialized vector of shape (dim, 1)
    b -- initialized scalar (corresponds to the bias)
    """

    ### START CODE HERE ### (≈ 1 line of code)
    w = np.zeros((dim, 1))
    b = 0
    ### END CODE HERE ###

    assert (w.shape == (dim, 1))
    assert (isinstance(b, float) or isinstance(b, int))

    return w, b


# GRADED FUNCTION: propagate
def propagate(w, b, X, Y):
    """
    Implement the cost function and its gradient for the propagation explained above

    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat) of size (1, number of examples)

    Return:
    cost -- negative log-likelihood cost for logistic regression
    dw -- gradient of the loss with respect to w, thus same shape as w
    db -- gradient of the loss with respect to b, thus same shape as b

    Tips:
    - Write your code step by step for the propagation. np.log(), np.dot()
    """

    m = X.shape[1]
    # print(f"w.shape{w.shape}")
    # FORWARD PROPAGATION (FROM X TO COST)
    ### START CODE HERE ### (≈ 2 lines of code)
    A = sigmoid(np.dot(w.T, X) + b)  # compute activation
    cost = -np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A)) / m  # compute cost
    ### END CODE HERE ###

    # BACKWARD PROPAGATION (TO FIND GRAD)
    ### START CODE HERE ### (≈ 2 lines of code)
    dw = np.dot(X, (A - Y).T) / m
    db = np.sum(A - Y) / m
    ### END CODE HERE ###

    assert (dw.shape == w.shape)
    assert (db.dtype == float)
    cost = np.squeeze(cost)
    assert (cost.shape == ())

    grads = {"dw": dw,
             "db": db}

    return grads, cost


# GRADED FUNCTION: optimize
def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost=False):
    """
    This function optimizes w and b by running a gradient descent algorithm

    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of shape (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat), of shape (1, number of examples)
    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
    print_cost -- True to print the loss every 100 steps

    Returns:
    params -- dictionary containing the weights w and bias b
    grads -- dictionary containing the gradients of the weights and bias with respect to the cost function
    costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.

    Tips:
    You basically need to write down two steps and iterate through them:
        1) Calculate the cost and the gradient for the current parameters. Use propagate().
        2) Update the parameters using gradient descent rule for w and b.
    """

    costs = []
    grads = None

    for i in range(1,num_iterations+1):

        # Cost and gradient calculation (≈ 1-4 lines of code)
        ### START CODE HERE ###
        grads, cost = propagate(w, b, X, Y)
        ### END CODE HERE ###

        # Retrieve derivatives from grads

        # update rule (≈ 2 lines of code)
        ### START CODE HERE ###
        w -= learning_rate * grads["dw"]
        b -= learning_rate * grads["db"]
        ### END CODE HERE ###

        # Record the costs
        if i % 100 == 0:
            costs.append(cost)

        # Print the cost every 100 training iterations
        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))

    params = {"w": w,
              "b": b}

    return params, grads, costs


# GRADED FUNCTION: predict

def predict(w, b, X):
    '''
    Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)

    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)

    Returns:
    Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X
    '''

    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)  # 确保维数对应

    # Compute vector "A" predicting the probabilities of a cat being present in the picture
    ### START CODE HERE ### (≈ 1 line of code)
    A = sigmoid(np.dot(w.T, X) + b)
    ### END CODE HERE ###

    # for i in range(A.shape[1]):
    #     # Convert probabilities A[0,i] to actual predictions p[0,i]
    #     ### START CODE HERE ### (≈ 4 lines of code)
    #     pass
    #     ### END CODE HERE ###
    Y_prediction = A.round()
    assert (Y_prediction.shape == (1, m))
    return Y_prediction


# GRADED FUNCTION: model

def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):
    """
    Builds the logistic regression model by calling the function you've implemented previously

    Arguments:
    X_train -- training set represented by a numpy array of shape (num_px * num_px * 3, m_train)
    Y_train -- training labels represented by a numpy array (vector) of shape (1, m_train)
    X_test -- test set represented by a numpy array of shape (num_px * num_px * 3, m_test)
    Y_test -- test labels represented by a numpy array (vector) of shape (1, m_test)
    num_iterations -- hyperparameter representing the number of iterations to optimize the parameters
    learning_rate -- hyperparameter representing the learning rate used in the update rule of optimize()
    print_cost -- Set to true to print the cost every 100 iterations

    Returns:
    d -- dictionary containing information about the model.
    """

    ### START CODE HERE ###
    # initialize parameters with zeros (≈ 1 line of code)
    w, b = initialize_with_zeros(X_train.shape[0])

    # Gradient descent (≈ 1 line of code)
    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost=print_cost)

    # Retrieve parameters w and b from dictionary "parameters"
    w = parameters["w"]
    b = parameters["b"]

    # Predict test/train set examples (≈ 2 lines of code)
    Y_prediction_train = predict(w, b, X_train)
    Y_prediction_test = predict(w, b, X_test)
    # print(f"Y_prediction_train.shape{Y_prediction_train.shape}")
    # print(f"Y_prediction_train.shape{Y_prediction_train.shape}")

    ### END CODE HERE ###

    # Print train/test Errors
    train_accuracy=100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100
    test_ccuracy=100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100
    print(f"train accuracy: {train_accuracy} %")
    print(f"test_ccuracy: {test_ccuracy} %")

    d = {"costs": costs,
         # "Y_prediction_test": Y_prediction_test,
         # "Y_prediction_train": Y_prediction_train,
         # "w": w,
         # "b": b,
         "learning_rate": learning_rate,
         "num_iterations": num_iterations,
         "train_accuracy":train_accuracy,
         "test_ccuracy":test_ccuracy}

    return d


def WhatPicIS(index):
    print(f"index = {index} y = {str(train_set_y[:, index])} " +
          "it's a '" + classes[np.squeeze(train_set_y[:, index])].decode("utf-8") + "' picture.")  # np.squeeze去维

if __name__ == '__main__':
    # Common steps for pre-processing a new dataset are:
    # - Figure out the dimensions and shapes of the problem (m_train, m_test, num_px, ...)
    # - Reshape the datasets such that each example is now a vector of size (num_px \* num_px \* 3, 1)
    # - "Standardize" the data

    # Loading the data (cat/non-cat)
    train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()
    print(f"train_set_x_orig={train_set_x_orig.shape}")
    print(f"train_set_y={train_set_y.shape}")
    print(f"test_set_x_orig={test_set_x_orig.shape}")
    print(f"test_set_y={test_set_y.shape}")
    print(f"classes{classes.shape}")

    # 本地化所有图片
    # savePicMult(train_set_x_orig,processesNum=10)

    # 查看某一张照片是否为猫
    WhatPicIS(8)


    ### START CODE HERE ### (≈ 3 lines of code)
    m_train = train_set_x_orig.shape[0]
    m_test = test_set_x_orig.shape[0]
    num_px = train_set_x_orig.shape[1]
    ### END CODE HERE ###

    print("Number of training examples: m_train = " + str(m_train))
    print("Number of testing examples: m_test = " + str(m_test))
    print("Height/Width of each image: num_px = " + str(num_px))
    print("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
    print("train_set_x shape: " + str(train_set_x_orig.shape))
    print("train_set_y shape: " + str(train_set_y.shape))
    print("test_set_x shape: " + str(test_set_x_orig.shape))
    print("test_set_y shape: " + str(test_set_y.shape))
    # Reshape the training and test examples

    ### START CODE HERE ### (≈ 2 lines of code)
    train_set_x_flatten = reshape4DAndstandardize(train_set_x_orig)
    test_set_x_flatten = reshape4DAndstandardize(test_set_x_orig)
    ### END CODE HERE ###

    print("train_set_x_flatten shape: " + str(train_set_x_flatten.shape))
    print("train_set_y shape: " + str(train_set_y.shape))
    print("test_set_x_flatten shape: " + str(test_set_x_flatten.shape))
    print("test_set_y shape: " + str(test_set_y.shape))
    print("sanity check after reshaping: " + str(train_set_x_flatten[0:5, 0]))

    d = model(train_set_x_flatten, train_set_y, test_set_x_flatten, test_set_y, num_iterations=2000, learning_rate=0.005,
              print_cost=True)

    # 全部转为列向量并标准化

    # Key steps: In this exercise, you will carry out the following steps:
    # - Initialize the parameters of the model
    # - Learn the parameters for the model by minimizing the cost
    # - Use the learned parameters to make predictions (on the test set)
    # - Analyse the results and conclude

    # print ("sigmoid([0, 2]) = " + str(sigmoid(np.array([0,2]))))
    #
    #
    # dim = 2
    # w, b = initialize_with_zeros(dim)
    # print ("w = " + str(w))
    # print ("b = " + str(b))

    # time_start = time.time()
    # sigmoid(np.random.rand(10**8))
    # time_end = time.time()
    # print(f"time={(time_end-time_start)} s")
    # w, b, X, Y = np.array([[1.], [2.]]), 2., np.array([[1., 2., -1.], [3., 4., -3.2]]), np.array([[1, 0, 1]])

    # grads, cost = propagate(w, b, X, Y)
    # print ("dw = " + str(grads["dw"]))
    # print ("db = " + str(grads["db"]))
    # print ("cost = " + str(cost))

    # params, grads, costs = optimize(w, b, X, Y, num_iterations=1010, learning_rate=0.009, print_cost=False)
    #
    # print("w = " + str(params["w"]))
    # print("b = " + str(params["b"]))
    # print("dw = " + str(grads["dw"]))
    # print("db = " + str(grads["db"]))
    # print(costs)
    #
    # w = np.array([[0.1124579], [0.23106775]])
    # b = -0.3
    # X = np.array([[1., -1.1, -3.2], [1.2, 2., 0.1]])
    # print("predictions = " + str(predict(w, b, X)))

    # d=model(train_set_x_flatten, train_set_y, test_set_x_flatten, test_set_y, num_iterations=100, learning_rate=0.005,
    #           print_cost=True)
    # # Plot learning curve (with costs)
    # costs = np.squeeze(d['costs'])
    # plt.plot(costs)
    # plt.ylabel('cost')
    # plt.xlabel('iterations (per hundreds)')
    # plt.title("Learning rate =" + str(d["learning_rate"]))
    # plt.show()

    # ## START CODE HERE ## (PUT YOUR IMAGE NAME)
    # my_image = "my_image.jpg"  # change this to the name of your image file
    # ## END CODE HERE ##
    #
    # # We preprocess the image to fit your algorithm.
    # fname = "images/" + my_image
    # image = np.array(ndimage.imread(fname, flatten=False))
    # my_image = scipy.misc.imresize(image, size=(num_px, num_px)).reshape((1, num_px * num_px * 3)).T
    # my_predicted_image = predict(d["w"], d["b"], my_image)
    #
    # plt.imshow(image)
    # print("y = " + str(np.squeeze(my_predicted_image)) + ", your algorithm predicts a \"" + classes[
    #     int(np.squeeze(my_predicted_image)),].decode("utf-8") + "\" picture.")

    #观察过拟合

    # import multiprocessing
    # time_start = time.time()
    # result=[]
    # processesNum=4
    # pool = multiprocessing.Pool(processes=processesNum)
    # for i in range(0,2010,50):
    #     result.append(pool.apply_async(func=model, args=(train_set_x_flatten, train_set_y, test_set_x_flatten, test_set_y, i, 0.005)))
    # pool.close()
    # pool.join()  # 调用join之前，先调用close函数，否则会出错。执行完close后不会有新的进程加入到pool,join函数等待所有子进程结束
    # time_end = time.time()
    # print(f"time={(time_end-time_start)}s")
    # result = [item.get() for item in result]
    # result_sorted = sorted(result, key=lambda e:(e.__getitem__('num_iterations')))
    #
    # # "train_accuracy": train_accuracy,
    # # "": test_ccuracy}
    # train_accuracy_list=[]
    # test_ccuracy_list=[]
    # num_list=[]
    # for item in result_sorted:
    #     train_accuracy_list.append(item["train_accuracy"])
    #     test_ccuracy_list.append(item["test_ccuracy"])
    #     num_list.append(item["num_iterations"])
    #
    #
    # # 这里导入你自己的数据
    # # ......
    # # ......
    # # x_axix，train_pn_dis这些都是长度相同的list()
    #
    # # 开始画图
    # plt.title('Result Analysis')
    # plt.plot(num_list, train_accuracy_list, color='blue', label='training accuracy')
    # plt.plot(num_list, test_ccuracy_list, color='red', label='testing accuracy')
    # plt.legend()  # 显示图例
    #
    # plt.xlabel('iteration times')
    # plt.ylabel('rate%')
    # plt.show()
    # # python 一个折线图绘制多个曲线


