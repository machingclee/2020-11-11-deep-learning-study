from pyimagesearch.nn.conv.ResNet import ResNet
from pyimagesearch.utils import plot_model
from pyimagesearch.nn.conv import DeeperGoogLeNet
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model


# output = DeeperGoogLeNet.conv_module(x, 64, 5, 5, (1, 1), -1, name="block_1")

# model = DeeperGoogLeNet.build(64, 64, 3, 200)
model = ResNet.build(32, 32, 3, 10, (0, 9, 9, 9), (64, 64, 128, 256), reg=0.005)
plot_model(model, to_file="resnet_cifar10.png", dpi=96)
