#@autor: Flavio Lorenzi mat 1662963
""" support vector machines for opt prediction """

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import utils
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import array as arr
from sklearn import metrics
from sklearn.model_selection import train_test_split

data = utils.load_jsonl('partial.jsonl')
y = []
call = []
lea = []
xor = []
listcount = []
features = ['call','lea','xor']


dimension = 0





for current in data:
    # print(json.dumps(current, indent=2))
    [instructions, optimizer] = utils.get_instructions_optimizer(current)    #X , y

    if(optimizer == 'H'):			                                         #ground Through   Z 444 U 557
    	y.append(0)
    	
    if(optimizer == 'L'):
    	y.append(1)

    call.append(utils.count_appearances('call ', instructions))
    lea.append(utils.count_appearances('lea ', instructions))
    xor.append(utils.count_appearances('xor ', instructions))

    #conto valori associati ad ogni feature (quante ce ne sono per ogni istruzione)
    count = np.array([call[dimension-1],lea[dimension-1],xor[dimension-1]])
    listcount.append(count)
    
	
   	
	    

#print(zero,uno) 


    
X_train, X_test, y_train, y_test = train_test_split(listcount, y, test_size=0.3,random_state=109) # 70% training and 30% test
#X_train, X_test, y_train, y_test = train_test_split(len(instructions), optimizer, test_size = 0.20)
svclassifier = SVC(kernel='linear')
svclassifier.fit(X_train, y_train)  #0 se correct 1 otherwise

y_pred = svclassifier.predict(X_test)

#print(y_pred)	#251 750
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

# Model Precision: what percentage of positive tuples are labeled as such?
print("Precision:",metrics.precision_score(y_test, y_pred))

# Model Recall: what percentage of positive tuples are labelled as such?
print("Recall:",metrics.recall_score(y_test, y_pred))


m_confusion_test = metrics.confusion_matrix(y_test, y_pred)
cm = pd.DataFrame(data = m_confusion_test, columns = ['Predicted High', 'Predicted Low'],
            index = ['Actual High', 'Actual Low'])
print(cm)














