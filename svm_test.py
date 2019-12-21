#test for svm
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
import svm

countH = 0
countL = 0
lista_opt = []
call = []
lea = []
xor = []
listcount = []

test = utils.load_jsonl('test_dataset_blind.jsonl')

dimension = 0

for current in test:
    # print(json.dumps(current, indent=2))
    instructions = current['instructions']

    # print("current instructions list")
    #print(json.dumps(instructions, indent=2))
    # print(len(instructions), optimizer)
    # print("current opt ", optimizer)
    dimension += 1
    
    call.append(utils.count_appearances('call ', instructions))
    lea.append(utils.count_appearances('lea ', instructions))
    xor.append(utils.count_appearances('xor ', instructions))

    #conto valori associati ad ogni feature (quante ce ne sono per ogni istruzione)
    count = np.array([call[dimension-1],lea[dimension-1],xor[dimension-1]])
    listcount.append(count)
    

    

y_pred = svm.svclassifier.predict(listcount)
#print(y_pred)


for i in y_pred:
    if(y_pred[i]==0):
        lista_opt.append('H')
        countH += 1

    if(y_pred[i]==1):
        lista_opt.append('L')
        countL += 1


print(lista_opt)
print(countL,countH)






