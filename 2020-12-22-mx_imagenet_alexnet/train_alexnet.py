from config import imagenet_alexnet_config as config
# from mxnet.runtime import feature_list
from deeptools.nn.mxconv import MxAlexNet
import mxnet as mx
import argparse
import logging
import json
import os

# feature_list()


ap = argparse.ArgumentParser()
ap.add_argument("-c", "--checkpoints", required=True, help="path to output checkpoint directory")
ap.add_argument("-p", "--prefix", required=True, help="name of model prefix")
ap.add_argument("-s", "--start-epoch", type=int, default=0, help="epoch to restart training at")
args = vars(ap.parse_args())


logging.basicConfig(level=logging.DEBUG, filename="training_{}.log".format(args["start_epoch"]), filemode="w")
means = json.load(open(config.DATASET_MEAN))
batchSize = config.BATCH_SIZE * config.NUM_DEVICES


trainIter = mx.io.ImageRecordIter(
    path_imgrec=config.TRAIN_MX_REC,
    data_shape=(3, 227, 227),
    batch_size=batchSize,
    rand_crop=True,
    rand_mirror=True,
    rotate=15,
    max_shear_ratio=0.1,
    mean_r=means["R"],
    mean_g=means["G"],
    mean_b=means["B"],
    preprocess_threads=config.NUM_DEVICES*2
)

valIter = mx.io.ImageRecordIter(
    path_imgrec=config.VAL_MX_REC,
    data_shape=(3, 227, 227),
    batch_size=batchSize,
    mean_r=means["R"],
    mean_g=means["G"],
    mean_b=means["B"],
)


checkpointsPath = os.path.sep.join([args["checkpoints"], args["prefix"]])
argParams = None
auxParams = None

sym = None

if args["start_epoch"] <= 0:
    print("[INFO] building network...")
    sym = MxAlexNet.build(config.NUM_CLASSES)
else:
    print("[INFO] loading epoch {}...".format(args["start_epoch"]))
    _sym, _arg_params, _aux_params = mx.model.load_checkpoint(checkpointsPath, args["start_epoch"])
    argParams = _arg_params
    auxParams = _aux_params
    sym = _sym

model = mx.mod.Module(context=[mx.gpu(0)],
                      symbol=sym)

batchEndCBs = [mx.callback.Speedometer(batchSize, 500)]
epochEndCBs = [mx.callback.do_checkpoint(checkpointsPath)]
metrics = [mx.metric.Accuracy(), mx.metric.TopKAccuracy(top_k=5), mx.metric.CrossEntropy()]

print("[INFO] start training...")
model.fit(
    trainIter,
    eval_data=valIter,
    arg_params=argParams,
    aux_params=auxParams,
    eval_metric=metrics,
    num_epoch=90,
    initializer=mx.initializer.Xavier(),
    optimizer="sgd",
    optimizer_params={"learning_rate": 1e-2, "momentum": 0.9, "wd": 0.0001, "rescale_grad": 1.0/batchSize},
    begin_epoch=args["start_epoch"],
    batch_end_callback=batchEndCBs,
    epoch_end_callback=epochEndCBs
)
