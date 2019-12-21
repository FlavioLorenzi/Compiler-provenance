'''
File containing all methods needed
'''

import json
import json_lines
import pandas as pd

"""  ---------------------------- UTILS -------------------------------   """


def get_instructions_optimizer(curr):
    """ output:  instructions as LIST and optmizier as STRING """
    istrc = curr['instructions']
    opt = curr['opt']
    return [istrc, opt]


def get_opt_probability(dataset, param):
    """ output:  probability of param in dataset under opt key """
    detections = 0
    dataset_lenght = 0

    for current in dataset:
        dataset_lenght += 1
        if current['opt'] == param:
            detections += 1
    prob = detections / dataset_lenght
    return prob



# metodo che crea una lista di liste (oggetto python) trasformando il dataset in formato jsonlist

def load_jsonl(input_path) -> list:
    """
    Read list of objects from a JSON lines file.
    """
    data = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.rstrip('\n|\r')))
    print('Loaded {} records from {}'.format(len(data), input_path))
    return data


def lenght_classifier(soglia, data, lenght):
    """
        Return accuracy of a classifier based on just lenght of INSTRUCTIONS
    """
    errore = 0

    for current in data:
        [instructions, label] = get_instructions_optimizer(current)

        prediction = 'H'
        if len(instructions) > soglia:
            prediction = 'L'

        if prediction != label:
            errore += 1

    accuracy = errore / lenght

    return 1 - accuracy


def count_appearances(param, lista):
    """
    counts occurencies of param in lista
    """
    appearances = 0
    for i in lista:
        if param in i:
            appearances += 1
    return appearances


def lea_classification(soglia_lea, data, dimension):
    """
        Return accuracy of a classifier based on just occurencies of lea in ISTRUCTIONS
    """
    errore = 0
    for current in data:
        [instructions, label] = get_instructions_optimizer(current)
        prediction = 'L'

        appearances = count_appearances('lea', instructions)
        
        if appearances > soglia_lea:
            prediction = 'H'
            

        if prediction != label:
            errore += 1
    
    accuracy = errore / dimension
    return 1 - accuracy

def call_classification(soglia_call, data, dimension):
    """
        Return accuracy of a classifier based on just occurencies of call in ISTRUCTIONS
    """
    errore = 0
    for current in data:
        [instructions, label] = get_instructions_optimizer(current)
        prediction = 'L'

        appearances = count_appearances('call', instructions)
        
        if appearances > soglia_call:
            prediction = 'H'
            
        if prediction != label:
            errore += 1
    
    accuracy = errore / dimension
    return 1 - accuracy

def xor_classification(soglia_xor, data, dimension):
    """
        Return accuracy of a classifier based on just occurencies of lea in ISTRUCTIONS
    """
    errore = 0
    for current in data:
        [instructions, label] = get_instructions_optimizer(current)
        prediction = 'L'

        appearances = count_appearances('xor', instructions)
        
        if appearances > soglia_xor:
            prediction = 'H'
            

        if prediction != label:
            errore += 1
    
    accuracy = errore / dimension
    return 1 - accuracy

#-------------------------------------------

'''
#calcoliamo lea given H e lea given L
def lea_probability_counter(data,dimension):
    leaH = 0
    leaL = 0
    for current in data:
        [instructions, label] = get_instructions_optimizer(current)
        appearances = count_appearances('lea', instructions)

        if(label == 'L'):
            leaL += appearances
            #print(leaL,"lea in L trovatoi")
        if(label == 'H'):
            leaH += appearances
            #print(leaH,"lea in H trovato")

    tot_lea = leaL + leaH  #lunghezza totale

    p_leaL = leaL / tot_lea     #probabilità a posteriori di trovare lea dato Low

    return p_leaL
'''
#NB: si fa num lea in L / num totale di lea in text
#oppure si fa num lea dentro L / num totale di parole in H ????

def leaH_probability_counter(data,dim,current):
    leaL = 0
    found = 0.1
    numH = countH(data)
    numL = countL(data)
    somma = numH + numL
   

    [instructions, label] = get_instructions_optimizer(current)

    leaL = count_appearances('lea ', instructions)

    if(label == 'H'):
        if(leaL >= 5):
            found += 1

    p_leaL = (found)/(numH)  #probabilità a posteriori di trovare lea dato HIGH
    
    return p_leaL

def leaL_probability_counter(data,dim,current):
    leaL = 0
    found = 0.1
    numH = countH(data)
    numL = countL(data)
    somma = numH + numL
    

    [instructions, label] = get_instructions_optimizer(current)
    leaL = count_appearances('lea ', instructions)

    if(label == 'L'):

        if(leaL < 5):
            found += 1 

    p_leaL = (found)/(numL)    #probabilità a posteriori di trovare lea dato HIGH

    return p_leaL


def callH_probability_counter(data,dim,current):
    callL = 0
    found = 0.1
    numH = countH(data)
    numL = countL(data)
    p_callL = 0
  
    somma = numH + numL
    [instructions, label] = get_instructions_optimizer(current)
    callL = count_appearances('call ', instructions)

    if(label == 'H'):
        if(callL >= 10):
            found += 1 

    
    p_callL = (found)/(numH)    #probabilità a posteriori di trovare lea dato HIGH

    return p_callL

def callL_probability_counter(data,dim,current):
    callL = 0
    found = 0.1
    numH = countH(data)
    numL = countL(data)
    somma = numH + numL
    
    
    [instructions, label] = get_instructions_optimizer(current)
    callL = count_appearances('call ', instructions)

    if(label == 'L'):
        if(callL < 10):
            found += 1 

    p_callL = (found)/(numL)    #probabilità a posteriori di trovare lea dato HIGH

    return p_callL



#xor prob
def xorH_probability_counter(data,dim,current):
    xL = 0
    found = 0.1
    numH = countH(data)
    numL = countL(data)
    somma = numH + numL
    p_xL = 0

    [instructions, label] = get_instructions_optimizer(current)
    xL = count_appearances('xor ', instructions)

    if(label == 'H'):

        if(xL >= 4):
            found += 1 

    p_xL = (found)/(numH)    #probabilità a posteriori di trovare lea dato HIGH

    return p_xL

def xorL_probability_counter(data,dim,current):
    xL = 0
    found = 0.1
    numH = countH(data)
    numL = countL(data)
    somma = numH + numL
    [instructions, label] = get_instructions_optimizer(current)
    xL = count_appearances('xor ', instructions)
    p_xL = 0

    if(label == 'L'):

        if(xL < 4):
            found += 1 

    p_xL = (found)/(numL)    #probabilità a posteriori di trovare lea dato HIGH


    return p_xL






#metodo che calcola la probabilità che sia lunga o meno un'istruzione, conoscendo che sia low

def big_length_probability_counter(data,dim,current):
    found = 0.1
    numH = countH(data)
    numL = countL(data)
    somma = numH + numL
    [instructions, label] = get_instructions_optimizer(current)
    

    if(label == 'L'): 
        
        if(len(instructions)>80):
            found += 1      
    probL = (found)/(numL)     #prob che dato L la lunghezza delle istruzioni sia maggiore di 10
        
    return probL


#metodo che calcola la probabilità che sia lunga o meno un'istruzione, conoscendo che sia High

def small_length_probability_counter(data,dim,current):
    found = 0.1
    numH = countH(data)
    numL = countL(data)
    somma = numH + numL
    
    
    [instructions, label] = get_instructions_optimizer(current)
    
    if(label == 'H'):


        if(len(instructions)<=80):
            found += 1
 
    probH = (found)/ (numH) #prob che dato H la lunghezza delle istruzioni sia minore di 10

    return probH



def accuracy(data, y, dim):
    """
        Return accuracy of the classifier 
        true positive + true negative / true positive + true negative + false positive + false negative
    """
    errore = 0

    for current in data:
        [instructions, label] = get_instructions_optimizer(current)

        prediction = y

        if (prediction != label):   
            errore += 1

    accuracy = errore / dim

    return 1 - accuracy



def countH(dataset):
    """ output:  number of real H """
    detections = 0

    for current in dataset:
        
        if current['opt'] == 'H':
            detections += 1

    return detections

def countL(dataset):
    """ output:  number of real H """
    detections = 0

    for current in dataset:
        
        if current['opt'] == 'L':
            detections += 1

    return detections



####

#import and convert the dataset
def conv2mnemonic():
    # read json dataset of 30000 instr into a dataframe of mnemonics words
    df=pd.read_json("train_dataset.jsonl",lines=True, orient='values')
    for count in range(0,len(df['instructions'])):
        for i in range(0,len(df['instructions'][count])):
            a = df['instructions'][count][i].partition(' ')
            df['instructions'][count][i] = a[0]

    return df


#import and convert the dataset
def conv2mnemonic_test():
    # read json dataset of 30000 instr into a dataframe of mnemonics words
    df=pd.read_json("test_dataset_blind.jsonl",lines=True, orient='values')
    for count in range(0,len(df['instructions'])):
        for i in range(0,len(df['instructions'][count])):
            a = df['instructions'][count][i].partition(' ')
            df['instructions'][count][i] = a[0]

    return df
