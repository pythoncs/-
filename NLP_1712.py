class NLP:
    dict_words = {}  #  字典
    def __init__(self):
        self.load_words()  #加载字典
    def load_words(self):
        file_dict = open("dict.txt", encoding="utf8")
        for line in file_dict:
            line_list = line.split()
            self.dict_words[line_list[1]] = 1
        file_dict.close()
    def Segment(self,sentence):
        segment = ""
        while (len(sentence) > 0):
            n = len(sentence)
            for i in range(0, n):
                cur = sentence[i:]  # cur 当前字符串
                # 判断cur是否在字典dict_word中
                if cur in self.dict_words.keys():
                    sentence = sentence[0:i]  # 截取cur
                    # 把cur连接到字符串segment
                    segment = cur + "/" + segment
                else:
                    if len(cur) == 1:
                        sentence = sentence[0:i]
                        segment = cur + "/" + segment
        return segment

    def __del__(self):
        self.dict_words.clear()  #释放内存

# nlp = NLP() #实例化
# print(nlp.Segment("严守一把手机关了"))
