from pyimagesearch.utils import plot_model
from pyimagesearch.nn.conv import DeeperGoogLeNet
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model


x = Input(shape=(64, 64, 3))
# output = DeeperGoogLeNet.conv_module(x, 64, 5, 5, (1, 1), -1, name="block_1")
output = DeeperGoogLeNet.inception_module(64, 96, 128, 16, 32, 32, -1, "stage-1-")(x)
model = Model(x, output, name="conv_module")
"""
Just as a review, in conv layer we use convolution of depth 64, (5,5) filter, (1,1) stride, the number of parameter is 64 * ((3*5*5) + 1) = 4864, as 

    model.summary()

indicated
"""

plot_model(model, to_file="inception_module.png")
