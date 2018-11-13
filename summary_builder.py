import tensorflow as tf
import pickle
import numpy as np
import itertools
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import os

class SummaryBuilder:
    def __init__(self, log_name, mapping_constant):
        self.log_name = log_name
        self.training = tf.summary.FileWriter(logdir='logs/' + log_name + '_train/')
        self.validation = tf.summary.FileWriter(logdir='logs/' + log_name + '_val/')
        self.test = tf.summary.FileWriter(logdir='logs/' + log_name + '_test/')

        self.mapping = mapping_constant

        def unpickle(file):
            with open(file, 'rb') as fo:
                data_dict = pickle.load(fo, encoding='bytes')
                return data_dict

        raw = unpickle('data/meta')
        self.fine_label_names = [fine_label_name.decode('utf-8') for fine_label_name in raw[b'fine_label_names']]
        self.coarse_label_names = [coarse_label_name.decode('utf-8') for coarse_label_name in raw[b'coarse_label_names']]

        if not os.path.exists('piclog/'):
            os.mkdir('piclog')

        if os.path.exists('piclog/' + log_name):
            os.rmdir('piclog/' + log_name)
        os.mkdir('piclog/' + log_name)

    def __add_topk_to_summary__(self, k, probabilities, true_labels, name, mapping=None):
        true_labels = tf.expand_dims(true_labels, axis=1)
        _, topk = tf.nn.top_k(input=probabilities, k=k, sorted=False)
        if mapping is not None:
            topk = tf.map_fn(lambda top5: tf.gather(mapping, top5), elems=topk)

        accuracy = tf.reduce_mean(tf.cast(tf.reduce_any(tf.equal(x=topk, y=true_labels)), tf.float32), name=name)
        return tf.summary.scalar(name=name, tensor=accuracy)

    def build_summary(self, probabilities, loss, true_labels):
        loss_summary = tf.summary.scalar(name='Loss', tensor=loss)
        accuracy1_summary = self.__add_topk_to_summary__(k=1, probabilities=probabilities, true_labels=true_labels,
                                                         name='Top1-Accuracy')
        accuracy5_summary = self.__add_topk_to_summary__(k=5, probabilities=probabilities, true_labels=true_labels,
                                                         name='Top5-Accuracy')
        accuracysuper1_summary = self.__add_topk_to_summary__(k=1, probabilities=probabilities, true_labels=true_labels,
                                                              name='Super-Top1-Accuracy',
                                                              mapping=self.mapping)
        accuracysuper5_summary = self.__add_topk_to_summary__(k=5, probabilities=probabilities, true_labels=true_labels,
                                                              name='Super-Top5-Accuracy',
                                                              mapping=self.mapping)

        train_summary = tf.summary.merge([loss_summary,
                                          accuracy1_summary,
                                          accuracy5_summary,
                                          accuracysuper1_summary,
                                          accuracysuper5_summary], name='TrainSummary')

        validation_summary = tf.summary.merge([accuracy1_summary,
                                               accuracy5_summary,
                                               accuracysuper1_summary,
                                               accuracysuper5_summary], name='ValidationSummary')

        test_summary = tf.summary.merge([accuracy1_summary,
                                         accuracy5_summary,
                                         accuracysuper1_summary,
                                         accuracysuper5_summary], name='TestSummary')

        return train_summary, validation_summary, test_summary

    def create_confusion_and_sample(self, probabilities, true_labels):
        _, predictions = tf.nn.top_k(input=probabilities, k=1, sorted=False)

        confusion_matrix = tf.confusion_matrix(labels=true_labels, predictions=predictions, name='Confusion')

        correct_values = tf.where(tf.equal(x=predictions, y=true_labels, name='CorrectPrediction'))

        gather_correct = tf.gather(params=correct_values,
                                   indices=tf.random_uniform(shape=(5,), minval=0, dtype=tf.int32,
                                                             maxval=tf.maximum(tf.size(input=correct_values), 1)),
                                   name='GatherCorrect')

        wrong_values = tf.where(tf.not_equal(x=predictions, y=true_labels, name='WrongPrediction'))

        gather_wrong = tf.gather(params=wrong_values,
                                 indices=tf.random_uniform(shape=(5,), minval=0, dtype=tf.int32,
                                                              maxval=tf.maximum(tf.size(input=wrong_values), 1)),
                                    name='GatherWrong')

        samples = tf.concat(values=[gather_correct, gather_wrong], axis=0, name='Sampling')

        return predictions, confusion_matrix, samples

    def validate_confusion_matrix(self, confusion_matrix, step_count):
        sum = 0
        for i in range(confusion_matrix.shape[0]):
            sum += confusion_matrix[i, i]

        confusion_acc = float(sum) / float(confusion_matrix.sum())
        print(str(step_count) + ': ' + str(confusion_acc))


    def save_confusion_matix(self, confusion_matrix, count):
        sum_across_axis = confusion_matrix.sum(axis=1)[:, np.newaxis]
        confusion_matrix_interp = confusion_matrix.astype('float') / sum_across_axis
        confusion_matrix_interp = np.nan_to_num(confusion_matrix_interp)

        plt.figure(figsize=(100, 100))
        plt.imshow(confusion_matrix_interp, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix: ' + str(count))
        plt.colorbar()
        tick_marks = np.arange(len(self.fine_label_names))
        plt.xticks(tick_marks, self.fine_label_names, rotation=45)
        plt.yticks(tick_marks, self.fine_label_names)

        fmt = 'd'
        thresh = confusion_matrix_interp.max() / 2.
        for i, j in itertools.product(range(confusion_matrix.shape[0]), range(confusion_matrix.shape[1])):
            plt.text(j, i, format(confusion_matrix[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if confusion_matrix_interp[i, j] > thresh else "black")

        plt.ylabel('True label')
        plt.xlabel('Predicted label')

        plt.savefig('piclog/' + self.log_name + '_confusion_matrix' + str(count) + '.png')
        plt.close()

    def gather(self, data, labels, predictions, sampled_indices, count):
        sampled_indices = np.unique(sampled_indices.flatten())

        def save_sample(sample):
            image = np.reshape(data[sample], [3, 32, 32])
            image = np.transpose(image, [1, 2, 0])
            plt.figure()
            plt.imshow(image)
            real_name = self.fine_label_names[labels[sample]]
            predict_name = self.fine_label_names[predictions[sample]]
            plt.xlabel('Real: ' + real_name + ', Predict: ' + predict_name)
            plt.savefig('piclog/' + self.log_name + '_' + str(count) + '_sample_' + real_name + '.png')
            plt.close()

        for sample in sampled_indices:
            save_sample(sample)


###################
# TEST ONLY
###################
"""
confusion_matrix = np.array([[2, 4], [3, 1]])
summary = SummaryBuilder('Test', None)
summary.validate_confusion_matrix(confusion_matrix, 0)
"""