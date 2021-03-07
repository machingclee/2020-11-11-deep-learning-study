#! /usr/bin/env python
# coding=utf-8
# ================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : video_demo.py
#   Author      : YunYang1994
#   Created date: 2019-07-12 19:36:53
#   Description :
#
# ================================================================

import cv2
import time
import numpy as np
import core.utils as utils
import tensorflow as tf
from core.yolov3 import YOLOv3, decode
from tensorflow.keras.models import load_model
from PIL import Image


video_path = "./docs/laogao.mp4"
# video_path      = 0
num_classes = 1
input_size = 416

# input_layer = tf.keras.layers.Input([input_size, input_size, 3])
# feature_maps = YOLOv3(input_layer)

# bbox_tensors = []
# for i, fm in enumerate(feature_maps):
#     bbox_tensor = decode(fm, i)
#     bbox_tensors.append(bbox_tensor)

model = load_model("./head_detection.h5")
model.summary()
vid = cv2.VideoCapture(video_path)

video_out_resize_ratio = 0.5
width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)*video_out_resize_ratio)
height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)*video_out_resize_ratio)
fps = int(vid.get(cv2.CAP_PROP_FPS))
codec = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('./output.mp4', codec, fps, (width, height))


stop = False
while stop == False:
    return_value, frame = vid.read()

    if return_value is None:
        raise ValueError("No image!")

    frame_size = frame.shape[:2]
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_data = utils.image_preporcess(np.copy(frame), [input_size, input_size])
    image_data = image_data[np.newaxis, ...].astype(np.float32)

    prev_time = time.time()
    pred_bbox = model.predict(image_data)
    pred_sbbox, pred_mbbox, pred_lbbox = pred_bbox
    pred_bbox = []
    pred_bbox.append(pred_sbbox)
    pred_bbox.append(pred_mbbox)
    pred_bbox.append(pred_lbbox)
    curr_time = time.time()
    exec_time = curr_time - prev_time

    pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
    pred_bbox = tf.concat(pred_bbox, axis=0)
    bboxes = utils.postprocess_boxes(pred_bbox, frame_size, input_size, 0.4)
    bboxes = utils.nms(bboxes, 0.4, method='nms')
    image = utils.draw_bbox(frame, bboxes)
    # image = Image.fromarray(image)
    # image.show()
    # stop = True
    # result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    (resize_w, resize_h) = np.array(frame_size).astype(float) * video_out_resize_ratio
    info = "time: %.2f ms" % (1000*exec_time)
    cv2.putText(image, text=info, org=(50, 70), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1, color=(255, 0, 0), thickness=2)
    cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
    result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    result = cv2.resize(result, (int(resize_h), int(resize_w)), interpolation=cv2.INTER_CUBIC)
    out.write(result)
    cv2.imshow("result", result)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
