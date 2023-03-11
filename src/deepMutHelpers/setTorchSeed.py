from torch import manual_seed, cuda
#in Case of hard reproducability
def setTorchSeed(seed):
  manual_seed(seed)
  if cuda.is_available():
    cuda.manual_seed_all(seed)