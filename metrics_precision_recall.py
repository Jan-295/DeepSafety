from tensorflow.python.ops.confusion_matrix import confusion_matrix
from sklearn.utils.multiclass import unique_labels
import tensorflow as tf
import matplotlib.pyplot as plt


# class for managing all the different steps
class MetricsPrecisionRecall:

    def __init__(self, pred, truth):
        # init all parameters needed
        self._pred = pred
        self._truth = truth
        # create confusion Matrix with predictions and truth
        self._cm = confusion_matrix(truth, pred)
        # extracting all classes the test contains
        self._classes = unique_labels(truth)
        self._tp = []
        self._fp = []
        self._fn = []

    def get_tp(self):
        true_positive = [0] * 43    # generate vector of zeros to save Data later

        # set TP (true positive) values for every class
        for i in self._classes:
            true_positive[i] = self._cm[i, i].numpy()   # extract the TP value for every class from the Batch

        self._tp = true_positive                        # safe it in the class member for later use
        return true_positive                            # return the vector

    def get_fp(self):
        false_positive = [0] * 43                       # generate vector of zeros to save Data later

        # set FP (false positive) values for every class
        cm_fp = tf.reduce_sum(self._cm, 0)              # add all the rows together
        for i in self._classes:
            false_positive[i] = cm_fp[i].numpy() - self._tp[i]  # extract the FP value for every class from the Batch

        self._fp = false_positive                       # sade it in the class member for later use
        return false_positive                           # return the vector

    def get_fn(self):
        false_negative = [0] * 43                       # generate vector of zeros to save Data later

        # set FN (false negative) values for every class
        cm_fp = tf.reduce_sum(self._cm, 1)              # add all the columns together
        for i in self._classes:
            false_negative[i] = cm_fp[i].numpy() - self._tp[i]  # extract the FN value for every class from the Batch

        self._fn = false_negative                       # sade it in the class member for later use
        return false_negative                           # return the vector

    def get_precision(self):
        tp = self.get_tp()                              # calculate the TP value and save in variable
        fp = self.get_fp()                              # calculate the FP value and save in variable
        precision = [0.0] * 43

        # calculating precision by excepting the case to not divide by 0
        for i in tp:
            if tp[i] + fp[i] != 0:
                precision[i] = tp[i] / (tp[i] + fp[i])
            else:
                precision[i] = 0

        return precision

    def get_recall(self):
        tp = self.get_tp()                              # calculate the TP value and save in variable
        fn = self.get_fn()                              # calculate the FN value and save in variable
        recall = [0.0] * 43

        # calculating recall by excepting the case to not divide by 0
        for i in tp:
            if tp[i] + fn[i] != 0:
                recall[i] = tp[i] / (tp[i] + fn[i])
            else:
                recall[i] = 0

        return recall

    def plot(self, class_of_interest):
        # printing Precision on Recall to visualize for the class selected with class_of_interest
        plt.rcParams["figure.figsize"] = [4, 4]
        plt.rcParams["figure.autolayout"] = True
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Plot from class " + str(class_of_interest))
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.grid(b=True, axis='both')
        plt.plot(self.get_recall()[class_of_interest], self.get_precision()[class_of_interest], 'bo', markersize=5)
        plt.show()

    def log(self):
        # printing all available classes with their Precision and Recall
        print("_________________________< LOG of Precision and Recall >_________________________")
        for c in self._classes:
            print("Evaluation class %s: Precision: %s, Recall: %s" % (c, self.get_precision()[c], self.get_recall()[c]))
