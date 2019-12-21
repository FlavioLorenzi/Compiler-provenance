#@autor: FLAVIO LORENZI mat. 1662963

import json_lines
import json
import utils
import numpy as np

''' This script let you know how to manage JSONL and JSON items'''
# JSONL is list of JSON item

data = utils.load_jsonl('partial2.jsonl')

"""prior probability"""
prob_L = utils.get_opt_probability(data, 'L')
prob_H = 1 - prob_L
'''
print("")
print("prior probability LOW", prob_L)
print("prior probability HIGH", prob_H)
print("")
'''
dimension = 0

for current in data:
    # print(json.dumps(current, indent=2))
    [instructions, optimizer] = utils.get_instructions_optimizer(current)
    # print("current instructions list")
    # print(json.dumps(instructions, indent=2))
    # print(len(instructions), optimizer)
    # print("current opt ", optimizer)
    dimension += 1

    
    

    """ Posterior Probability """
    
    
    #l'accuracy è buona, quindi posso prendere queste tre features per l'algoritmo: 
    prob_leaH = utils.leaH_probability_counter(data,dimension,current)
    #print("posterior probability P(LEA|HIGH) : ",prob_leaH)  #prob di ottenere lea>5 dato high
    prob_leaL = utils.leaL_probability_counter(data,dimension,current)
    #print("posterior probability P(LEA|LOW) : ",prob_leaL)   #prob di ottenere lea<5 dato low

    prob_callH = utils.callH_probability_counter(data,dimension,current)
    #print("posterior probability P(CALL|HIGH) : ",prob_callH)
    prob_callL = utils.callL_probability_counter(data,dimension,current)
    #print("posterior probability P(CALL|LOW) : ",prob_callL)

    prob_xH = utils.xorH_probability_counter(data,dimension,current)
    #print("posterior probability P(XOR|HIGH) : ",prob_xH)
    prob_xL = utils.xorL_probability_counter(data,dimension,current)
    #print("posterior probability P(XOR|LOW) : ",prob_xL)
	
    #prob che instr sia lunga given Low
    prob_bigL = utils.big_length_probability_counter(data,dimension,current)
    #print("posterior probability P(BigInstruction|LOW)",prob_bigL)

    #prob che instr sia corta given High
    prob_smallH = utils.small_length_probability_counter(data,dimension,current)
    #print("posterior probability P(SmallInstruction|HIGH)",prob_smallH)
    print("")

 
    """binary NAIVE BAYES """
    #HIGH = P(HIGH)*P(LEA|HIGH)*P(CALL|HIGH)*P(LENGTH|HIGH) 
    #LOW = P(LOW)*P(LEA|LOW)*P(CALL|LOW)*P(LENGTH|LOW) 
    #argamx (HIGH, LOW)   to predict the current-used optimizzation

    high = prob_H* prob_smallH * prob_leaH * prob_callH  * prob_xH
    low = prob_L* prob_bigL * prob_leaL * prob_callL  * prob_xL
    
    
	
    res = np.array([low,high])
    vnb = np.argmax(res)  #metodo per convertirlo in valore
    #print(vnb)
    if(vnb == 0):
        y = 'L' #prediction positive
        print("Instruction",dimension,"= LOW")
    	

    if(vnb == 1):
    	y = 'H'	#prediction negative
    	print("Instruction",dimension,"= HIGH")
        

#accuracy Confusion MAtrix
accuracyAlgo = utils.accuracy(data,y,dimension) 
print("Comparing with the real values of optimizzation we reach an accuracy",accuracyAlgo)

#precision
precisionL = low / (high + low)
print("There will be a precision of ",precisionL,"% to predict LOW optimizer")
print("")


'''



#FEATURE EXTRACTION
soglia = 80
accuracy_lenght = utils.lenght_classifier(soglia, data, dimension)
print("accuracy calcolando solo LENGHT : ", accuracy_lenght)

soglia_lea = 10
accuracy_lea = utils.lea_classification(soglia_lea, data, dimension)
print("accuracy calcolando solo LEA : ", accuracy_lea)

soglia_call = 10
accuracy_call = utils.call_classification(soglia_call, data, dimension)
print("accuracy calcolando solo CALL : ", accuracy_call)

soglia_xor = 4
accuracy_xor = utils.xor_classification(soglia_xor, data, dimension)
print("accuracy calcolando solo XOR : ", accuracy_xor)
    '''





