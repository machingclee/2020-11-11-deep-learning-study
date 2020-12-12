from pyimagesearch.utils import plot_model
from pyimagesearch.nn.conv import DeeperGoogLeNet
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model


# output = DeeperGoogLeNet.conv_module(x, 64, 5, 5, (1, 1), -1, name="block_1")

model = DeeperGoogLeNet.build(64, 64, 3, 200)

"""
Just as a review, in conv layer we use convolution of depth 64, (5,5) filter, (1,1) stride, the number of parameter is 64 * ((3*5*5) + 1) = 4864, as 

    model.summary()

indicated
"""

plot_model(model, to_file="DeeperGoogleNet.png", dpi=350)
