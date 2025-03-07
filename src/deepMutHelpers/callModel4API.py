from src.deepMutHelpers.setDevice import setDevice
from transformers import AutoModelForSeq2SeqLM, T5EncoderModel

def callModelForAPI(huggingfaceID : str = "Rostlab/prot_t5_xl_uniref50", encoder = False):
    if "model" in globals():
        return
    #set the device
    device = setDevice()
    #load the model
    if not encoder:
        model = AutoModelForSeq2SeqLM.from_pretrained(huggingfaceID).to(device)
        return model
    else:
        encoder = T5EncoderModel.from_pretrained(huggingfaceID).to(device)
        return encoder