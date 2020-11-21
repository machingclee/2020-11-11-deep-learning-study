import numpy as np

# zip(iter1, iter2) yields
# ( (iter1[0], iter2[0]),
#   (iter1[1], iter2[1]),
#   ...
#   (iter1[n], iter2[n]) )
#  where n = min(len(iter1), len(iter2)) - 1.


def rank5_accuracy(preds, labels):
    rank1 = 0
    rank5 = 0

    # gt_label is an integer encoded by sklearn.preprocessing .LabelEncoder of sklearn.
    # pred is a row of probabilities, the i-th entry represents the probability of being class-i.
    # np.argsrt(pred) returns a list of "indexes, i.e., label-i's" sorted by their proability.

    for(pred, gt_label) in zip(preds, labels):
        # arr[::-1] simply reverses arr.
        pred = np.argsort(pred)[::-1]

        if gt_label in pred[:5]:
            rank5 += 1

        if gt_label == pred[0]:
            rank1 += 1

        rank1 /= float(len(preds))
        rank5 /= float(len(preds))

        return (rank1, rank5)
