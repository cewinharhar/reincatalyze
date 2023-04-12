import requests
import json
from src.mainHelpers.prepare4APIRequest import prepare4APIRequest
import numpy as np

def embeddingRequest(seq, returnNpArray = False, embeddingUrl = "http://0.0.0.0/embedding"):
    #check if list, if not, make oen
    #if not isinstance(seq, list):
    #    seq = [seq]
    seq = ["".join(seq)]    

    payload = dict(inputSeq = seq)

    package = prepare4APIRequest(payload)

    try:
        response = requests.post(embeddingUrl, json=package).content.decode("utf-8")
        embedding = json.loads(response) 
        if returnNpArray:   
            return np.array(embedding[0])
        else:
            return embedding[0]
    
    except requests.exceptions.RequestException as e:
        errMes = "Something went wrong\n" + str(e)
        print(errMes)
        raise Exception