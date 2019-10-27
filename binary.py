#@autor: FLAVIO LORENZI mat. 1662963

import json_lines
import json
import utils
import numpy as np

''' This script let you know how to manage JSONL and JSON items'''
# JSONL is list of JSON item

data = utils.load_jsonl('train_dataset.jsonl')

dimension = 0

for current in data:
    # print(json.dumps(current, indent=2))
    [instructions, optimizer] = utils.get_instructions_optimizer(current)
    # print("current instructions list")
    # print(json.dumps(instructions, indent=2))
    # print(len(instructions), optimizer)
    # print("current opt ", optimizer)
    dimension += 1

prob_L = utils.get_opt_probability(data, 'L')
prob_H = 1 - prob_L
print("")
print("prior probability LOW", prob_L)
print("prior probability HIGH", prob_H)
print("")


""" FEATURE EXTRACTION """

soglia = 10
accuracy_lenght = utils.lenght_classifier(soglia, data, dimension)
print("accuracy calcolando solo LENGHT : ", accuracy_lenght)

soglia_lea = 5
accuracy_lea = utils.lea_classification(soglia_lea, data, dimension)
print("accuracy calcolando solo LEA : ", accuracy_lea)

soglia_call = 10
accuracy_call = utils.call_classification(soglia_call, data, dimension)
print("accuracy calcolando solo CALL : ", accuracy_call)

