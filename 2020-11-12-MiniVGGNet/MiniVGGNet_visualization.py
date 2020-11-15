from pyimagesearch.nn.conv import MiniVGGNet
from tensorflow.keras.utils import plot_model

model = MiniVGGNet.build(32, 32, 3, 10)
plot_model(model, to_file="MiniVGGNet.png", show_shapes=True)
