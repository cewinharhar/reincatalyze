from fastapi import FastAPI, Request, Body
from pydantic import BaseModel

import uvicorn

import json

try:
    from app.deepMut_predictionPipe import deepMutPredictionPipe
    from app.getProteinEmbedding import getProteinEmbedding
    from app.callModel4API import callModel4API
except:
    from deepMut_predictionPipe import deepMutPredictionPipe
    from getProteinEmbedding import getProteinEmbedding
    from callModel4API import callModel4API

#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
"""
How to use me:

    cd alphaKGDGenerator/transformerAPI/docker
    uvicorn app.main:app --host "0.0.0.0" --port "9999"

    Then Access api over browser 0.0.0.0:9999/ to check if successfully deployed
"""
#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\


if "model" not in globals():
    print("Model not loaded yet: waking up benjamin!")
    model = callModel4API(huggingfaceID = "app/protT5")
    print("---------------\n Benjamin is awake and loaded. \n Ready to BOOST!\n---------------")

if "encoder" not in globals():
    print("Encoder not loaded yet: waking up benjamin Junior!")
    encoder = callModel4API(huggingfaceID = "app/protT5HalfEncoder", encoder = True)
    print("---------------\n Benjamin Junior is awake and loaded. \n Ready to BOOST!\n---------------")
     
""" class deepMutInputStructure(BaseModel):
    predictionCallDict : str """

app = FastAPI()

#set connection check root endpoint
@app.get("/")
def home():
    return {"Status": "Connected!"}

@app.post("/deepMut")
def deepMut(package = Body()):
        """
        example request:
        {'predictionCallDict': 
            "{\n  "inputSeq": 
                ["M", "S", "T", "E", "...", "T", "L", "R"],\n 
                "huggingfaceID": "Rostlab/prot_t5_xl_uniref50",\n  
                "paramSet": "",\n  
                "task": "rational",\n  
                "rationalMaskIdx": [71, 83, 246],\n  
                "tokenizer": null,\n  
                "num_return_sequences": 5,\n  
                "do_sample": true,\n  
                "temperature": 1,\n  
                "top_k": 3\n}"
        }  
        """
        #load the input data excpected to be converted to list automatically
        inputDic = json.loads(package['predictionCallDict'])
        #inputDic["model"] = model

        print("API INPUT: \n ")
        for item_ in inputDic.items():
            print(item_)

        prediction = deepMutPredictionPipe(**inputDic, model = model)
        
        print(prediction)    
        return prediction

@app.post("/embedding")
def embedding(package = Body()):
        """
        example request:
        {'predictionCallDict': 
            {["A", "B", "G"]}
        """
        #load the input data excpected to be converted to list automatically
        seq_ = json.loads(package['predictionCallDict'])
        print(seq_)
        print(seq_["inputSeq"])
        print(type(seq_["inputSeq"]))
        encoding = getProteinEmbedding(seq_ = seq_["inputSeq"], encoder = encoder)
        print(encoding)
        return encoding


""" if __name__ == "__main__":
    print("Goood morning night city!")
    config = uvicorn.Config("main:app", host = "0.0.0.0", port = 9999, log_level = "info")
    server = uvicorn.Server(config)
    server.run() """
