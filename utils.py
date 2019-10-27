'''
File containing all methods needed
'''

import json
import json_lines


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


""" --------------------------------------------------------------------  """
