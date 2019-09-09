
#分词程序(逆向最大匹配法)
#输入:严守一把手机关了
#输出: 严守一/把/手机/关了
#加载字典(从文件dict.txt读取)
dict_word = {}
file_dict = open("dict.txt",encoding="utf8")
for line in file_dict:
    #print(line,end="")
    line_list = line.split()
    #print(line_list[1])
    dict_word[line_list[1]] = 1
file_dict.close()
#print(len(dict_word))
#函数名称:Segment
#函数功能:分词
#输入参数:句子
#返回参数:分词
#时间:
def Segment(sentence):
    segment = ""
    while(len(sentence)>0):
        n = len(sentence)
        for i in range(0,n):
            cur = sentence[i:] #cur 当前字符串
            #print(cur)
            #判断cur是否在字典dict_word中
            if cur in dict_word.keys():
                sentence = sentence[0:i]#截取cur
                #把cur连接到字符串segment
                segment = cur +"/"+segment
            else:
                if len(cur) == 1:
                    sentence = sentence[0:i]
                    segment = cur + "/" + segment
            #print(cur,"###",sentence)

    return segment
sentence = input("输入句子")
print(Segment(sentence)) #严守一/把/手机/关了