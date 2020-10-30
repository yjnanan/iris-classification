import numpy as np
import pandas as pd
import random
from pandas.core.frame import DataFrame
import math
import matplotlib.pyplot as plt

#read data to a list and shuffle data
iris_data=pd.read_csv("iris.data",header=None)
labels_codes=pd.Categorical(iris_data[4]).codes
print(iris_data)
for i in range(150):
    iris_data.loc[i,4]=labels_codes[i]
datalist=iris_data.values.tolist()
print(datalist)
random.seed(17)
random.shuffle(datalist)
print(datalist)

#process input features
x_features=(DataFrame(datalist))[range(4)].values.tolist()
x0=np.ones(150)
x_features=np.mat(x_features)
x_features=np.insert(x_features,0,x0,axis=1)
#print(x_features)

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
#print(y_labels)

#initialize weight from input layer to hidden layer
w_vec_ji=[]
for i in range(20):
    w=[]
    for j in range(5):
        w.append(random.uniform(-math.sqrt(3/4),math.sqrt(3/4)))
    w_vec_ji.append(w)
w_vec_ji=np.mat(w_vec_ji)
print(w_vec_ji)

#initialize weight from hidden layer to output layer
w_vec_kj=[]
for i in range(3):
    w=[]
    for j in range(20):
        w.append(random.uniform(-math.sqrt(3/4),math.sqrt(3/4)))
    w_vec_kj.append(w)
w_vec_kj=np.mat(w_vec_kj)
print(w_vec_kj)

#set learning rate
learning_rate=0.001

#split dataset to 5 subsets
np.random.seed(17)
theta1=[]
theta2=[]
train_data=np.hstack((x_features,y_labels))
print(train_data)
train_data=np.vsplit(train_data,5)
#print(train_data)
train_data1=train_data[0].copy()
train_data2=train_data[1].copy()
train_data3=train_data[2].copy()
train_data4=train_data[3].copy()
train_data5=train_data[4].copy()

#set iterations
epoch=160

#stochastic gradient descent
for i in range(5):
    #select input data
    if i==0:
        data=np.vstack((train_data1,train_data2,train_data3,train_data4))
    elif i==1:
        data=np.vstack((train_data1,train_data2,train_data3,train_data5))
    elif i==2:
        data=np.vstack((train_data1,train_data2,train_data4,train_data5))
    elif i==3:
        data=np.vstack((train_data1,train_data3,train_data4,train_data5))
    else:
        data=np.vstack((train_data2,train_data3,train_data4,train_data5))
    cost = []
    iteration = []
    w1 = w_vec_ji
    w2 = w_vec_kj
    #start iteration
    for j in range(epoch):
        np.random.shuffle(data)
        #print(data)
        x = data[:, 0:5]
        y = data[:, 5:8]
        error_rate = 0
        for index in range(120):
            #forward
            a = np.dot(x[index], w1.T)
            z = np.tanh(a)
            train_y = np.dot(z, w2.T)
            #backward
            sigma_k = train_y - y[index]
            sigma_j = np.multiply((1 - np.multiply(z, z)), np.dot(sigma_k, w2))
            der_ji = np.dot(sigma_j.T, x[index])
            der_jk = np.dot(sigma_k.T, z)
            #update weight
            w2 = w2 - learning_rate * der_jk
            w1 = w1 - learning_rate * der_ji
            #calculate error rate
            error_rate = error_rate + 0.5 * (math.pow((train_y[0, 0] - y[index, 0]), 2) + math.pow((train_y[0, 1] - y[index, 1]),2) + math.pow((train_y[0, 2] - y[index, 2]), 2))
        iteration.append(j)
        #print(error_rate/120)
        cost.append(error_rate / 120)
    theta1.append(w1)
    theta2.append(w2)
    plt.plot(iteration, cost)
    plt.xlabel("Iteration")
    plt.ylabel("average error rate")
    plt.title("Stochastic Gradient Descent")
plt.show()

#test and calculate accuracy
accuracy=[]
for i in range(5):
    if i==0:
        data=train_data5.copy()
    elif i==1:
        data=train_data4.copy()
    elif i==2:
        data=train_data3.copy()
    elif i==3:
        data=train_data2.copy()
    else:
        data=train_data1.copy()
    test_set = data[:, 0:5]
    test_label = data[:, 5:8]
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
        if((test_y[x]==test_label[x]).all()):
            right=right+1
    accuracy.append(right/30)
print('each subset accuracy:',accuracy)
average_accuracy=np.mean(accuracy)#calculate average accuracy
v_accuracy=np.var(accuracy)#calculate variance
sd_accuracy=np.std(accuracy)#calculate standard deviation
print('average accuracy:',average_accuracy)
print('variance',v_accuracy)
print('standard deviation:', sd_accuracy)



















