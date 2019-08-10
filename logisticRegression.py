# 逻辑回归算法(决策边界为直线的情况)
# 假设函数(3阶多项式):h(x)=theta0*x0+theta1*x1+theta2*x2+theta3x1^2+...
#
# step1.加载数据并打印散点图
import numpy as np
import matplotlib.pyplot as plt
theta = np.zeros((10,1))
iter_num = 200000      #迭代次数
learningRate=0.01   #学习率
min_change = 1e-5    #梯度变化阈值

def mapFeature(X, n):
    X1 = X[:, 1].reshape(-1, 1)
    X2 = X[:, 2].reshape(-1, 1)
    map_X = np.zeros((len(X), 1))
    num = 0
    for i in range(0, n + 1):
        for j in range(0, n - i + 1):
            num += 1
            temp = X1 ** i * X2 ** j
            map_X = np.hstack((map_X, np.array(temp).reshape(-1, 1)))
    return map_X[:, 1:]
def printScatter(filename):
    # 读取数据
    data = np.loadtxt(filename, delimiter=',')
    x = data[:, 0:2]
    y = data[:, 2]
    pos = np.where(y == 1)
    neg = np.where(y == 0)
    plt.scatter(x[pos, 0], x[pos, 1], marker='s', c='r')
    plt.scatter(x[neg, 0], x[neg, 1], marker='x', c='b')
    plt.show()
printScatter("ex2data2.txt")
# step2.加载训练数据集(0,6)loadDataset(filename)
def loadDataset(filename, percent):
    data = np.loadtxt(filename, delimiter=',')
    np.random.shuffle(data)
    total = len(data)
    m = int(total * percent)
    X = data[:m, 0:2]
    Y = data[:m, -1].reshape(-1,1)
    rest_X = data[m:,0:2]
    temp = np.ones((total-m, 1)) # col 1
    rest_X = np.hstack((temp, rest_X))
    #add c 1
    rest_Y = data[m:,-1].reshape(-1,1)
    X0 = np.ones((m, 1))
    X = np.hstack((X0, X))
    n = X.shape[1]
    return X, Y, m, n,rest_X,rest_Y
X, Y, m, n,rest_X,rest_Y = loadDataset('ex2data2.txt', 0.6)
#print(X,Y)
#矩阵扩展为3阶多项式矩阵
X = mapFeature(X,3)
#print(X)
# step3.计算代价函数computeCost(theta,X,Y)
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
def computeCost(X, Y, theta):
    m = len(Y)
    z = sigmoid(np.dot(X, theta))
    z[z == 1] = 9e-18
    z[z == 0] = 1e-18
    J = - (np.dot(Y.T, np.log(z)) + np.dot((1 - Y).T, np.log(1 - z))) / m
    return J
J = computeCost(X, Y, theta)
#print("J cost:\t",J)
# step4.计算梯度computeDescent(theta,X,Y)
def computeDescent(theta, X, Y):
    # print("theta::",theta.shape)
    # print("X::", X.shape)
    # print("Y::", Y.shape)
    z = sigmoid(np.dot(X, theta))
    # print("z::", z.shape)
    gradient = np.dot(X.T,(z - Y))/m
    # print("gradient::", gradient.shape)
    return gradient
#print("computeDescent:\t",computeDescent(theta,X,Y))
# step5.梯度下降函数gradientDescent(theta,X,Y,iter_num,learningRate)
def gradientDescent(theta,X,Y,iter_num,learningRate):
    i = 0
    J_list = []
    gradient = computeDescent(theta,X,Y)
    while (not np.all(np.absolute(gradient)<=min_change)) and (i<iter_num):
        theta = theta - learningRate*gradient
        gradient = computeDescent(theta, X, Y)
        J_list.append(computeCost(X, Y, theta))
        i += 1
        print(i)
        #print("gradient",gradient)
    return theta,J_list,i
# step6.执行梯度下降函数
theta ,J_list,iter_num = gradientDescent(theta,X,Y,iter_num,learningRate)
print(theta)
# step7.(附1:评价)生成代价函数(数值)与迭代次数(iter_num)的曲线
plt.plot(np.arange(0,iter_num,1),np.array(J_list).reshape(iter_num,1),'-r')
plt.show()
# # step8.(附2:决策边界,绘图)决策边界printDecisionBoundary()
# #打印散点图
# plt.subplots(figsize=(100,100))
# data = np.loadtxt("ex2data1_small.txt", delimiter=',')
# x = data[:, 0:2]
# y = data[:, 2]
# pos = np.where(y == 1)
# neg = np.where(y == 0)
# plt.scatter(x[pos, 0], x[pos, 1], marker='s', c='r')
# plt.scatter(x[neg, 0], x[neg, 1], marker='x', c='b')
# #拟合决策边界
# x = np.linspace(-1,1,450).reshape(50,9)
# y = -np.dot(x,theta[0:9])/theta[:,9]
# plt.subplot(X,Y,color="green")
# axes = plt.gca()
# axes.legend(['decision boundary','dataset'])
# plt.show()
#  step9.计算算法的分类准确度
#取出剩余的训练数据作为输入数据
rest_X = mapFeature(rest_X,3)
predict_Y = sigmoid(np.dot(rest_X,theta))
predict_Y[predict_Y>=0.5] = 1
predict_Y[predict_Y<0.5] = 0
predict_data = predict_Y-rest_Y
Train_Accuracy = len(predict_data[predict_data==0])/len(rest_Y)
print('Train Accuracy: %.2f'%( Train_Accuracy* 100)+' %')
