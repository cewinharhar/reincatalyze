import requests
import json
from src.mainHelpers.prepare4APIRequest import prepare4APIRequest

def deepMutRequest(seq, rationalMaskIdx, deepMutUrl = "http://0.0.0.0/deepMut", nrOfSequences = 1,
                   task = "rational", max_length = 512, huggingfaceID = "Rostlab/prot_t5_xl_uniref50",
                   do_sample = True, temperature = 1.5, top_k = 20):
    #INIT WITH WILDTYPE
    payload = dict(
                        inputSeq            = [x for x in seq],
                        rationalMaskIdx     = rationalMaskIdx ,
                        num_return_sequences= nrOfSequences,
                        task                = task,
                        max_length          = max_length,
                        huggingfaceID       = huggingfaceID,
                        do_sample           = do_sample,
                        temperature         = temperature,
                        top_k               = top_k
    )
    #make json
    package = prepare4APIRequest(payload)

    #get predictions
    try:
        response = requests.post(deepMutUrl, json=package).content.decode("utf-8")
        deepMutOutput = json.loads(response)
        return deepMutOutput

    except requests.exceptions.RequestException as e:
        errMes = "Something went wrong\n" + str(e)
        print(errMes)
        pass