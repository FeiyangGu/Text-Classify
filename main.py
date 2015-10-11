#!/usr/bin/python
import numpy as np
import sklearn
from sklearn import svm
import jieba
import jieba.posseg as pseg
import sys
import lda

reload(sys)
sys.setdefaultencoding( "utf-8" )

class Data:
    def filt(self, seg_list):
        seg = set([])
        mark = False
        categary = "0"
        for ele in seg_list:
            word = ele.word
            flag = ele.flag
            if mark == False:
                mark = True
                categary = word
                continue
            if flag.find("n") != -1 or flag.find("v")!=-1:
                seg.add(word)
        return categary, seg
                
    def processData(self, inputfile):
        stopkey=[line.strip().decode('utf-8') for line in open('stopkey.txt').readlines()]  
        fin = open(inputfile, "r")
        lines = fin.readlines()
        fin.close()

        data_str = []
        k = 0
        n = len(lines)
        
        for line in lines:
            seg_list = pseg.cut(line)
            categary, seg = self.filt(seg_list)
            string = categary +" " + " ".join(list( seg-set(stopkey) ))
            k += 1
            if k % 100 == 0:
                print k, "/", n
            data_str.append(string+"\n")
       
        dic = self.count(data_str)
        m = len(dic)
        print m, " words" 
        data = np.zeros( (n, m), dtype='int64' )
        classes= []
        ni = 0
        for line in data_str:
            temp = line.split(" ")
            if temp[0] == "1":
                classes.append(0)
            if temp[0] == "12":
                classes.append(1)
            if temp[0] == "38":
                classes.append(2)
            length = len(temp)
            for i in range(1,length):
                data[ni][ dic[temp[i]] ] += 1
            ni += 1
        return data, classes

    def count(self, data_str):
        cnt = 0
        dic = {}
        for line in data_str:
            temp = line.split(" ")
            mark = False
            for ele in temp:
                if mark ==False:
                    mark = True
                    continue
                if dic.has_key(ele) == False:
                    dic[ele] = cnt
                    cnt += 1
        return  dic

class Classify:
    def lda_pro(self, data, classes):
        model = lda.LDA(n_topics=20, n_iter=1500, random_state=1)
        model.fit(data)
        doc_topic = model.doc_topic_
        return doc_topic
    def classify_pro(self, tocTopic, classes):
        clf = svm.SVC(kernel = "rbf", C=1.0)
        n = len(classes)
        port = n*9/10
        clf.fit(doc_topic[0:port], classes[0:port])
        cnt = 0
        for i in range(port, n):
            m = clf.predict(doc_topic[i])
            if m == classes[i]:
                cnt += 1
        total = n - port
        print 1.0*cnt/total

if __name__ == '__main__':
    
    doc_topic = np.load("doc_topic.npy")
    classes = np.load("classes.npy")
    CLY = Classify()
    CLY.classify_pro(doc_topic, classes)
    sys.exit()

    DT = Data()
    data, classes = DT.processData(sys.argv[1])
    np.save("data.npy", data)
    np.save("classes.npy", classes)
    
    CLY = Classify()
    doc_topic = CLY.lda_pro(data, classes)
    np.save("doc_topic.npy", doc_topic)
    
    CLY.classify_pro(doc_topic, classes)

