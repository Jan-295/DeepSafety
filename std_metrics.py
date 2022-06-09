from tensorflow.python.ops.confusion_matrix import confusion_matrix
from sklearn.utils.multiclass import unique_labels
import pandas as pd
import tensorflow as tf


def metrics(pred, truth):
    classes = unique_labels(truth)
    true_positive = [0 for i in range(43)]
    false_negative = [0 for i in range(43)]
    false_positive = [0 for i in range(43)]

    # create confusion matrix
    cm = confusion_matrix(truth, pred)

    # set TP (true positive) values
    for i in classes:
        true_positive[i] = cm[i, i].numpy()

    # set FP (false positive) values
    cm_fp = tf.reduce_sum(cm, 0)
    for i in classes:
        false_positive[i] = cm_fp[i].numpy() - true_positive[i]

    # set FN (false negative) values


    # extract all columns and rows according to the available Classes
    cm_reduced_rows = tf.gather(cm, unique_labels(truth), axis=0)
    cm_reduced = tf.gather(cm_reduced_rows, unique_labels(truth), axis=1)



    labels = unique_labels(truth)
    columns = [f'Predicted {label}' for label in labels]
    index = [f'Actual {label}' for label in labels]
    table = pd.DataFrame(cm_reduced, columns=columns, index=index)
    return table
