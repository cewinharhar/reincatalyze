import json

def prepare4APIRequest(payload : dict):
    package = {"predictionCallDict": payload}
    package["predictionCallDict"] = json.dumps(package["predictionCallDict"], 
                                               default=str, # use str method to serialize non-json data
                                               separators=(",", ":")) # remove spaces between separators
    return package