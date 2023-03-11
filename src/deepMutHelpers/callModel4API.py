from src.deepMutHelpers.setDevice import setDevice
from transformers import AutoModelForSeq2SeqLM

def callModelForAPI(huggingfaceID : str = "Rostlab/prot_t5_xl_uniref50"):
    if "model" in globals():
        return
    #set the device
    device = setDevice()
    #load the model
    model = AutoModelForSeq2SeqLM.from_pretrained(huggingfaceID).to(device)
    return model