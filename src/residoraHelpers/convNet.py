import torch.nn as nn
 
class convNet(nn.Module):
    """This class represents a convolutional neural network used in the ActorCritic class.
    It takes in the following parameters:
    :param activationFunction: A function used as the activation function in the convolutional neural network.
    :param out_channel: An integer representing the number of output channels from the convolutional layer.
    :param kernel_size: An integer representing the size of the kernel used in the convolutional layer.
    :param padding: An integer representing the number of padding pixels used in the convolutional layer.
    :param stride: An integer representing the stride length used in the convolutional layer.
    :param dropOutProb: A float representing the probability of dropping out a neuron in the dropout layer.
    """
    def __init__(self, activationFunction, out_channel, kernel_size : int, padding : int, stride : int, dropOutProb : float):
        super(convNet, self).__init__()
    
        self.cnn = nn.Sequential(
                            nn.Conv1d(in_channels = 1, out_channels = out_channel, kernel_size=kernel_size, padding=padding, stride = stride ), # 7x32
                            activationFunction(),
                            nn.Dropout( dropOutProb ),
                            nn.Flatten()
        )

        #get the dimen
 
    def forward(self, x1D):
        #x3D = x1D.unsqueeze(0).unsqueeze(0)
        return self.cnn(x1D)