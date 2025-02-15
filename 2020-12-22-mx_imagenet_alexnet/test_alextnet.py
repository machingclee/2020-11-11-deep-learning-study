from mxnet import metric
from numpy.core.numeric import require
from config import imagenet_alexnet_config as config
import mxnet as mx
import argparse
import json
import os

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--checkpoints", required=True, help="path to output checkpoint directory")
ap.add_argument("-p", "--prefix", required=True, help="name of model prefix")
ap.add_argument("-e", "--epoch", type=int, required=True, help="epoch number to load")
args = vars(ap.parse_args())


means = json.loads(open(config.DATASET_MEAN))

testIter = mx.io.ImageRecordIter(
    path_imgrec=config.TEST_MX_REC,
    data_shape=(3, 227, 227),
    batch_size=config.BATCH_SIZE,
    mean_r=means["R"],
    mean_g=means["G"],
    mean_b=means["B"]
)

checkpointPath = os.path.sep.join([args["checkpoints"], args["prefix"]])
model = mx.model.FeedForward.load(checkpointPath, args["epoch"])
model = mx.model.FeedForward(
    ctx=[mx.gpu(0)],
    symbol=model.symbol,
    arg_params=model.arg_params,
    aux_params=model.aux_params
)

metrics = [mx.metric.Accuracy(), mx.metric.TopKAccuracy(top_k=5)]
(rank1, rank5) = model.score(testIter, eval_metric=metrics)

print("[INFO] rank-1: {:.2f}%".format(rank1*100))
print("[INFO] rank-5: {:.2f}%".format(rank5*100))
