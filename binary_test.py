import json_lines
import json
import utils
import numpy as np


test = utils.load_jsonl('test_dataset_blind.jsonl')
lista_opt = []
dimension = 0
countH = 0
countL = 0

for current in test:
    # print(json.dumps(current, indent=2))
    instructions = current['instructions']

    # print("current instructions list")
    #print(json.dumps(instructions, indent=2))
    # print(len(instructions), optimizer)
    # print("current opt ", optimizer)
    dimension += 1
    
    lea = utils.count_appearances('lea', instructions)
    call = utils.count_appearances('call', instructions)
    xor = utils.count_appearances('xor', instructions)




    #Given the good accuracy of the training we can rewrite the bounds 
    #of prediction in this way to find High or Low accuracy in the test
   
    if(lea>30 or call>30 or len(instructions)<=80):

        #add to csv high or low
        lista_opt.append('high')
        countH += 1
    	#print(dimension ,"HIGH")
    else:
        lista_opt.append('low')
        countL += 1
    	#print(dimension , "LOW")
    

print(lista_opt)
print(countL,countH)


def test():
    return lista_opt


