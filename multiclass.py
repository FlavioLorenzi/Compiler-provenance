import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
from sklearn import feature_extraction, model_selection, naive_bayes, metrics, svm
from sklearn.metrics import precision_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC 
import json
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split 

import utils


print("")
print("We are training the jsonl file ...")
print("The current algorithm is the SVM classifier")
print("")

'''SVM FOR MULTICLASS PREDICTION : the compiler prediction'''

# read json into a dataframe
df=utils.conv2mnemonic()
print(df)

df['instructions'] = df['instructions'].apply(lambda x: ' '.join(x))  #delete ','
#print(df)

v1 = 'compiler'  #gcc icc clang
v2 = 'instructions'
data = df


'''
#MOST COMMON WORDS USED IN COMPILER
countG = Counter(" ".join(data[data[v1]=='gcc'][v2]).split()).most_common(20)
df0 = pd.DataFrame.from_dict(countG)
df0 = df0.rename(columns={0: "words in gcc", 1 : "count"})
print(df0)
countI = Counter(" ".join(data[data[v1]=='icc'][v2]).split()).most_common(20)
df1 = pd.DataFrame.from_dict(countI)
df1 = df1.rename(columns={0: "words in icc", 1 : "count"})
print(df1)
countC = Counter(" ".join(data[data[v1]=='clang'][v2]).split()).most_common(20)
df2 = pd.DataFrame.from_dict(countC)
df2 = df2.rename(columns={0: "words in clang", 1 : "count"})
print(df2)
'''


#feature extraction
f = feature_extraction.text.CountVectorizer()
X = f.fit_transform(data[v2])
#print(f.get_feature_names())   
np.shape(X)

#take labels    and split dataset
data[v1]=data[v1].map({'gcc':0,'icc':1,'clang':2})
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, data[v1], test_size=0.3, random_state=42)
print("Train and test splitting into: ",[np.shape(X_train), np.shape(X_test)])


#LINEAR SVM FOR MULTICLASS
svm_model_linear = SVC(kernel = 'linear', C = 1).fit(X_train, y_train) 
svm_predictions = svm_model_linear.predict(X_test) 

predictions = []
for i in svm_predictions:
	if(i == 0):
		predictions.append('gcc')
	if(i == 1):
		predictions.append('icc')
	if(i == 2):
		predictions.append('clang')

# model accuracy for X_test   
accuracy = svm_model_linear.score(X_test, y_test) 
# creating a confusion matrix 
m_confusion = metrics.confusion_matrix(y_test, svm_predictions)

#print("PREDICTIONS:" ,predictions)
cm = pd.DataFrame(data = m_confusion, columns = ['Predicted G', 'Predicted I', 'Predicted C'],
            						index = ['Actual G', 'Actual I', 'Actual C' ])


#print("predictions :", predictions)
print("Total Accuracy:",accuracy)
print("The confusion matrix is :")
print(cm)
print()

#For each class we have to compute the following values  (compiler: G,I,C)
true_positiveG = m_confusion[0,0]
true_positiveI = m_confusion[1,1]
true_positiveC = m_confusion[2,2]


false_positiveG = m_confusion[1,0] + m_confusion[2,0]
false_negativeG = m_confusion[0,1] + m_confusion[0,2]

false_positiveI = m_confusion[0,0] + m_confusion[2,0]
false_negativeI = m_confusion[1,0] + m_confusion[1,2]

false_positiveC = m_confusion[1,0] + m_confusion[0,0]
false_negativeC = m_confusion[2,1] + m_confusion[2,2]


PrecisionG = true_positiveG/(true_positiveG+false_positiveG)
PrecisionI = true_positiveI/(true_positiveI+false_positiveI)
PrecisionC = true_positiveC/(true_positiveC+false_positiveC)
print("PRECISION FOR EACH CLASS gcc, icc, clang: ",PrecisionG,PrecisionI,PrecisionC)

Accuracy_G = (true_positiveG+false_negativeG)/(true_positiveG+false_positiveG+false_negativeG+true_positiveI+true_positiveC)
Accuracy_I = (true_positiveI+false_negativeI)/(true_positiveI+false_positiveI+false_negativeI+true_positiveG+true_positiveC)
Accuracy_C = (true_positiveC+false_negativeC)/(true_positiveC+false_positiveC+false_negativeC+true_positiveI+true_positiveG)
print("ACCURACY FOR EACH CLASS gcc, icc, clang: ",Accuracy_G,Accuracy_I,Accuracy_C)

Recall_G = true_positiveG/(true_positiveG+false_negativeG)
Recall_I = true_positiveI/(true_positiveI+false_negativeI)
Recall_C = true_positiveC/(true_positiveC+false_negativeC)
print("RECALL FOR EACH CLASS gcc, icc, clang: ",Recall_G,Recall_I,Recall_C)














