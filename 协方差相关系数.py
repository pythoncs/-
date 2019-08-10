#期望    期望我们也可以简单的理解为平均数
def mean(a):
    A = sum(a) / len(a)
    return A

#方差   每一个元素减去期望的平方和除以长度
def fanxha(a):
    list_a = []
    for i in a:
        list_a.append((i-mean(a))**2)
    fc = sum(list_a)/len(list_a)
    return fc

#标准差    标准差就是方差的平方根
import math
def standard_deviation(a):
    return math.sqrt(fanxha(a))


# 计算每一项数据与均值的差
def de_mean(x):
  x_bar = mean(x)
  return [x_i - x_bar for x_i in x]
# 将x,y每一项数据与均值的差相成
def dot(v, w):
  return sum(v_i * w_i for v_i, w_i in zip(v, w))
# 协方差　　x,y每一项元素与x,y均值的差相成除以x或y的长度
def covariance(x, y):
  n = len(x)
  return dot(de_mean(x), de_mean(y)) / (n)


# 相关系数     相关系数则是由协方差除以两个变量的标准差而得
def correlation(x, y):
  stdev_x = standard_deviation(x)
  stdev_y = standard_deviation(y)
  if stdev_x > 0 and stdev_y > 0:
    return covariance(x, y) / stdev_x / stdev_y
  else:
    return 0


x = [190,160,170]
y = [183,161,169]
print('标准差',standard_deviation(x))
print('标准差',standard_deviation(y))
print('协方差',covariance(x,y))
print('相关系数',correlation(x,y))


import scipy.stats
print(scipy.stats.pearsonr(x, y))






