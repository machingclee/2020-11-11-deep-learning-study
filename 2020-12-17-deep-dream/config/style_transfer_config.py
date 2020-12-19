import os


contentLayers = ["block4_conv2"]

styleLayers = ["block1_conv1", "block2_conv1", "block3_conv1", "block4_conv1", "block5_conv1"]

styleWeight = 1.0
contentWeight = 1e4
tvWeight = 20.0

epochs = 15
stepsPerEpoch = 100

contentImage = os.path.sep.join(["inputs", "jp.jpg"])
styleImage = os.path.sep.join(["inputs", "mcescher.jpg"])
finalImage = "final.png"
intermOutputs = "intermediate_outputs"
