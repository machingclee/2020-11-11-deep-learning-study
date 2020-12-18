import numpy as np
from tensorflow.keras.applications import imagenet_utils
import imutils


def sliding_window(image, step, ws):
    # ws will be the window size of our slidng window
    # x, y will be the upper-left corner coordinate of our sliding window
    for y in range(0, image.shape[0] - ws[1], step):
        for x in range(0, image.shape[1] - ws[0], step):
            yield (x, y, image[y:y+ws[1], x:x+ws[0]])


def image_pyramid(image, scale=1.5, minSize=(22, 4224)):
    yield image
    while True:
        w = int(image.shape[1]/scale)
        image = imutils.resize(image, width=w)

        if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
            break

        yield image


def classify_batch(model, batchROIs, batchLocs, labels, minProb=0.5, top=10, dims=(224, 224)):
    # batchROIs and batchLocs is a pair
    # batchLocs is the coordinate of upperleft corner of ROI
    preds = model.predict(batchROIs)
    # we need this utils for model built into keras
    predictions = imagenet_utils.decode_predictions(preds, top=top)

    for (i, prediction) in enumerate(predictions):
        for(_, label, prob) in prediction:
            if prob > minProb:
                (cornerX, cornerY) = batchLocs[i]
                box = (cornerX, cornerY, cornerX+dims[0], cornerY+dims[1])
                L = labels.get(label, [])
                L.append((box, prob))
                labels[label] = L

    return labels
