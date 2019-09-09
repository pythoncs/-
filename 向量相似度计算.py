import math
list1=["中国","首都"]
list2=["中国","首都","哪里"]
def computer_similary_cos(list1,list2):
    similary = 0
    #第一步：列出所有的词
    union_set = set(list1)|set(list2)
    #print(union_set)
    #第二步：计算词频并出词频向量
    list_1 = [list1.count(i) for i in union_set]
    list_2 = [list2.count(i) for i in union_set]
    #计算相似度
    # cos = 数量积/(|a||b|)
    sum = 0
    squre_a = 0
    squre_b = 0
    for i in range(len(union_set)):
        sum += list_1[i] * list_2[i]
        squre_a += list_1[i]**2
        squre_b += list_2[i]**2
    similary = sum /( math.sqrt(squre_a) *math.sqrt(squre_b))
    return similary

print(computer_similary_cos(list1,list2))
