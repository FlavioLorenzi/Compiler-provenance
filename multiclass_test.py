#Testing the blind dataset

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
import multiclass
import utils



countG=0
countI=0
countC=0
# read json into a dataframe
df=utils.conv2mnemonic_test()
print(df)

df['instructions'] = df['instructions'].apply(lambda x: ' '.join(x))  #delete ','
#print(df)

v2 = 'instructions'
data = df




#feature extraction
X = multiclass.f.transform(data[v2])
np.shape(X)
 





#LINEAR SVM FOR MULTICLASS

svm_predictions = multiclass.svm_model_linear.predict(X) 

predictions = []
for i in svm_predictions:
	if(i == 0):
		countG += 1
		predictions.append('gcc')
	if(i == 1):
		countI += 1
		predictions.append('icc')
	if(i == 2):
		countC += 1
		predictions.append('clang')


print(predictions)
print(countG,countI,countC)










