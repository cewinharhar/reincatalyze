from transformers import AutoModelForSeq2SeqLM, T5EncoderModel

def callModel4API(huggingfaceID : str = "Rostlab/prot_t5_xl_uniref50", encoder = False, device = "cuda:0"):
    if "model" in globals():
        return
    #set the device
    device = device
    #load the model
    if not encoder:
        model = AutoModelForSeq2SeqLM.from_pretrained(huggingfaceID).to(device)
        return model
    else:
        encoder = T5EncoderModel.from_pretrained(huggingfaceID).to(device)
        return encoder