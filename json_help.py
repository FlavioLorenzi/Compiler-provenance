import json_lines
import json

''' This script let you know how to manage JSONL and JSON items'''
# JSONL is list of JSON item

with open('partial.jsonl') as file:
    dataset = json_lines.reader(file)

    for item in dataset:
        print("")
        print("Each row is a JSON file considered as: ", type(item))
        print(item)
        print("")
        print("Istructions value is a :", type(item['instructions']))
        print((item['instructions']))
        print("You can access to single element of Instructions using int-indices: ", (item['instructions'][1]))
        print("")
        print("Opt value is a :", type(item['opt']))
        print("You can access to single element of Opt using int-indices: ", (item['opt']))
        print("")
        print("Compiler value is a :", type(item['compiler']))
        print("You can access to single element of Compiler using int-indices: ", (item['compiler']))

        # command to remove keys : del item['compiler']
        # break

        # with this command here you have a better view of data
        print(json.dumps(item, indent=3))
