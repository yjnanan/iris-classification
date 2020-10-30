import numpy as np
import pandas as pd
import random
from pandas.core.frame import DataFrame
import matplotlib.pyplot as plt

#read data to a list and shuffle data
iris_data=pd.read_csv("iris.data",header=None)
labels_codes=pd.Categorical(iris_data[4]).codes
for i in range(150):
    iris_data.loc[i,4]=labels_codes[i]
datalist=iris_data.values.tolist()
print(datalist)
random.seed(9)
random.shuffle(datalist)
print(datalist)

#process input features
x_features=(DataFrame(datalist))[range(4)].values.tolist()
x0=np.ones(150)
x_features=np.mat(x_features)
x_features=np.insert(x_features,0,x0,axis=1)

#process output labels
y_labels=[]
for i in range(150):
    if datalist[i][4]==0:
        y_labels.append([1,0,0])
    elif datalist[i][4]==1:
        y_labels.append([0,1,0])
    else:
        y_labels.append([0,0,1])
y_labels=np.mat(y_labels)

#initialize weight from input layer to hidden layer
w_vec_ji=[]
for i in range(20):
    w=[]
    for j in range(5):
        w.append(random.uniform(-0.1,0.1))
    w_vec_ji.append(w)
w_vec_ji=np.mat(w_vec_ji)
print(w_vec_ji)

w_vec_kj=[]
for i in range(3):
    w=[]
    for j in range(20):
        w.append(random.uniform(-0.1,0.1))
    w_vec_kj.append(w)
w_vec_kj=np.mat(w_vec_kj)
print(w_vec_kj)

#set learning rate
learning_rate=0.001

np.random.seed(9)
theta1=[]
theta2=[]

#batch gradient descent
for i in range(5):
    # select input data
    train_set=np.vstack((x_features[0:120-30*i],x_features[150-30*i:150]))
    train_label=np.vstack((y_labels[0:120-30*i],y_labels[150-30*i:150]))
    train_data = np.hstack((train_set, train_label))
    np.random.shuffle(train_data)
    cost=[]
    iteration=[]
    w1 = w_vec_ji
    w2 = w_vec_kj
    print(w_vec_kj)
    print(w_vec_ji)
    x = train_data[:, 0:5]
    y = train_data[:, 5:8]
    # start iteration
    for j in range(1500):
        # forward
        a = np.dot(x, w1.T)
        z = np.tanh(a)
        train_y = np.dot(z, w2.T)
        # backward
        sigma_k = train_y - y
        sigma_j = np.multiply((1 - np.multiply(z, z)), np.dot(sigma_k, w2))
        der_ji = (np.dot(sigma_j.T, x))
        der_jk = (np.dot(sigma_k.T, z))
        # update weight
        w2 = w2 - learning_rate * der_jk
        w1 = w1 - learning_rate * der_ji
        # calculate error rate
        error_rate = 0.5 * (np.square(train_y - y).sum(axis=1))
        iteration.append(j)
        cost.append(np.sum(error_rate)/120)
    theta1.append(w1)
    theta2.append(w2)
    plt.plot(iteration, cost)
    plt.xlabel("Iteration")
    plt.ylabel("average error rate")
    plt.title("Batch Gradient Descent")
plt.show()

#test and calculate accuracy
accuracy=[]
for i in range(5):
    test_set = x_features[120 - 30 * i:150 - 30 * i]
    test_label = y_labels[120 - 30 * i:150 - 30 * i]
    w1=theta1[i]
    w2=theta2[i]
    a = np.dot(test_set, w1.T)
    z = np.tanh(a)
    test_y = np.dot(z, w2.T)
    right=0
    for x in range(30):
        index=np.argmax(test_y[x,:])
        test_y[x,index]=1
        for y in range(3):
            if test_y[x,y]!=1:
                test_y[x,y]=0
        if((test_y[x] == test_label[x]).all()):
            right=right+1
    accuracy.append(right/30)
    print(test_y)
    print(test_label)
print('each subset accuracy:',accuracy)
average_accuracy=np.mean(accuracy)#calculate average accuracy
v_accuracy=np.var(accuracy)#calculate variance
sd_accuracy=np.std(accuracy)#calculate standard deviation
print('average accuracy:',average_accuracy)
print('variance',v_accuracy)
print('standard deviation:', sd_accuracy)


















