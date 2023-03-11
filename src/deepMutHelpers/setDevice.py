from torch import device
def setDevice():
    device = device('cuda:0' if torch.cuda.is_available() else 'cpu') 
    return device