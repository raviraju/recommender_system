
"""Module of generic utilities"""
import json

def load_json_file(filename):
    """load Json File"""
    with open(filename, 'r') as json_file:
        data = json.load(json_file)
    return data

def dump_json_file(data, filename):
    """dump Json File"""
    with open(filename, 'w') as json_file:
        json.dump(data, fp=json_file, indent=4)

def convert_sec(no_of_secs):
    """return no_of_secs to min or hrs string"""
    if no_of_secs < 60:
        return "Time Taken : {:06.4f}    sec".format(no_of_secs)
    elif no_of_secs < 3600:
        return "Time Taken : {:06.4f}    min".format(no_of_secs/60)
    else:
        return "Time Taken : {:06.4f}    hr".format(no_of_secs/3600)
