from  sklearn.feature_extraction import DictVectorizer
import csv
from sklearn import preprocessing
from sklearn import tree
from sklearn.externals.six import StringIO
import  numpy

f_data = open('E:/py-ai/decisiontree/data.csv', 'r')
f_csv = csv.reader(f_data)
headers = next(f_csv)

print(headers)

featureList = []
labelList = []
for row in f_csv:
    #标记：yes / no
    labelList.append(row[len(row) - 1])
    rowDict = {}
    for i in range(1, len(row) - 1):
        rowDict[headers[i]] = row[i]

    featureList.append(rowDict)

#print(featureList)

#实例化
vec = DictVectorizer()
dummyX = vec.fit_transform(featureList).toarray()
print("dummyX:"+str(dummyX))
print(vec.get_feature_names())

# label的转化，直接用preprocessing的LabelBinarizer方法
lb = preprocessing.LabelBinarizer()
dummyY = lb.fit_transform(labelList)
print("dummyY:"+str(dummyY))
print("labelList:"+str(labelList))

#criterion是选择决策树节点的标准，这里是按照“熵”为标准，即ID3算法；默认标准是gini index，即CART算法。
#可构建好这个决策树
clf = tree.DecisionTreeClassifier(criterion = 'entropy')
clf = clf.fit(dummyX,dummyY)
print("clf:"+str(clf))


#生成dot文件
with open("allElectronicInformationGainOri.dot",'w') as f:
    f = tree.export_graphviz(clf,feature_names = vec.get_feature_names(),out_file = f)


#测试代码，取第1个实例数据，将001->100，即age：youth->middle_aged

oneRowX = dummyX[0,:]
print("oneRowX:"+str(oneRowX))
newRowX = oneRowX
#测试预测age：youth->middle_aged
newRowX[0] = 1
newRowX[2] = 0
print("newRowX:"+str(newRowX))
#结果
predictedY = clf.predict([newRowX])
print("predictedY:"+str(predictedY))