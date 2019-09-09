'''
简单的人机对话程序
预处理(对答案进行分词,并加载到内存)
步骤1.输入问题A
步骤2.对问题A进行分词list_seg_A
步骤3.对答案依次与list_seg_A进行相似度比较
步骤4.输出相似度最大值对应的答案
'''
#预处理(对答案进行分词,并加载到字典dict_answer)
import NLP_1712
nlp = NLP_1712.NLP()
sim_min = 0.5  #相似度最小阈值
#答案库字典,比如{(沙溢,的,老婆),胡可}
dict_answer = {}
#武林/外传/男/主角/老/白/的/老婆/
def str2tuple(segment):
    return tuple(segment[:-1].split("/"))
file_answer = open("dialog.txt",encoding="utf8")
#计算相似度
#Jaccard方法
def computerSimilary(set_ask,set_question):
    sim = 0
    jiaoji_num = len(set_ask&set_question)
    bingji_num = len(set_ask|set_question)
    if bingji_num!=0:
        sim = jiaoji_num/bingji_num
    return sim

for line in file_answer:
    #print(line,end="")
    list_line = line.split("###")
    #字典dict_answer中的key,就是tuple_line
    tuple_line = str2tuple(nlp.Segment(list_line[0]))
    # 字典dict_answer中的value,就是
    anwser_line = list_line[1]
    #逐行加载到dict_answer
    dict_answer[tuple_line] = anwser_line
file_answer.close()
#步骤1.输入问题A
while(True):
    question = input("请提问")
    tuple_ask = str2tuple(nlp.Segment(question))
    #步骤3.对答案依次与list_seg_A进行相似度比较
    sim_max = 0  #相似度最大值
    finaly_answer = "我正在学习呢"  #最终答案
    for k in dict_answer:
        sim = computerSimilary(set(tuple_ask),set(k))
        if sim>sim_max:
            sim_max = sim
            finaly_answer = dict_answer[k]
    if sim_max>=sim_min:
        print("机器人-->",finaly_answer,"(",sim_max,")")
    else:
        print("机器人-->我正在学习呢")
