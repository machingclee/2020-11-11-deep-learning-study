from config import srcnn_config as config
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import argparse
import PIL
import cv2

input_image = "puppy.png"
baseline_image = "baseline.png"
output_path = "resized.png"

model = load_model(config.MODEL_PATH)
print("[INFO] generating image...")
image = cv2.imread(input_image)
(h, w) = image.shape[:2]
w = w - int(w % config.SCALE)
h = h - int(h % config.SCALE)
image = image[0:h, 0:w]
highW = int(w * (config.SCALE/1.0))
highH = int(h * (config.SCALE/1.0))


scaled = np.array(Image.fromarray(image).resize((highW, highH), resample=PIL.Image.BICUBIC))
cv2.imwrite(baseline_image, scaled)

output = np.zeros(scaled.shape)
(h, w) = output.shape[:2]

for y in range(0, h - config.INPUT_DIM + 1, config.LABEL_SIZE):
    for x in range(0, w - config.INPUT_DIM + 1, config.LABEL_SIZE):

        crop = scaled[y:
                      y + config.INPUT_DIM,
                      x:
                      x + config.INPUT_DIM].astype("float32")

        # the prediction is (21, 21, 3)-dimensional
        P = model.predict(np.expand_dims(crop, axis=0))
        P = P.reshape((config.LABEL_SIZE, config.LABEL_SIZE, 3))
        output[y + config.PAD:
               y + config.PAD + config.LABEL_SIZE,
               x + config.PAD:
               x + config.PAD + config.LABEL_SIZE] = P


output = output[config.PAD:
                h - ((h % config.INPUT_DIM) + config.PAD),
                config.PAD:
                w - ((w % config.INPUT_DIM) + config.PAD)]
output = np.clip(output, 0, 255).astype("uint8")

cv2.imwrite(output_path, output)
