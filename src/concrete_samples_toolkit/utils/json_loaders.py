import json
import json5

def load_config(path):
    '''
    Loads a JSON5 config file
    '''
    with open(path, "r") as f:
        return json5.load(f)

def load_json_schema(path):
    '''
    Loads a JSON schema file
    '''
    with open(path, "r") as f:
        return json.load(f)
