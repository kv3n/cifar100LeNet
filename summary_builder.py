import tensorflow as tf
import pickle


class SummaryBuilder
    def __init__(self, log_name, mapping_constant):
        self.training = tf.summary.FileWriter(logdir=log_name + '_train/')
        self.validation = tf.summary.FileWriter(logdir=log_name + '_val/')
        self.test = tf.summary.FileWriter(logdir=log_name + '_test/')

        self.mapping = mapping_constant

        def unpickle(file):
            with open(file, 'rb') as fo:
                data_dict = pickle.load(fo, encoding='bytes')
                return data_dict

        raw = unpickle('data/meta')
        self.fine_label_names = [fine_label_name.decode('utf-8') for fine_label_name in raw[b'fine_label_names']]
        self.coarse_label_names = [coarse_label_name.decode('utf-8') for coarse_label_name in raw[b'coarse_label_names']]

    def __add_topk_to_summary__(self, k, probabilities, true_labels, name, mapping=None):
        topk = tf.nn.top_k(input=probabilities, k=k, sorted=False)
        if mapping is not None:
            topk = tf.map_fn(lambda top5: tf.gather(mapping, top5), elems=topk)

        accuracy = tf.reduce_mean(tf.cast(tf.reduce_any(tf.equal(topk, true_labels)), tf.float32), name=name)
        return tf.summary.scalar(name=name, tensor=accuracy)

    def build_summary(self, probabilities, loss, true_labels):
        loss_summary = tf.summary.scalar(name='Loss', tensor=loss)
        accuracy1_summary = self.__add_topk_to_summary__(k=1, probabilities=probabilities, true_labels=true_labels,
                                                         name='Top1 Accuracy')
        accuracy5_summary = self.__add_topk_to_summary__(k=5, probabilities=probabilities, true_labels=true_labels,
                                                         name='Top5 Accuracy')
        accuracysuper1_summary = self.__add_topk_to_summary__(k=1, probabilities=probabilities, true_labels=true_labels,
                                                              name='Super Top1 Accuracy',
                                                              mapping=self.mapping)
        accuracysuper5_summary = self.__add_topk_to_summary__(k=5, probabilities=probabilities, true_labels=true_labels,
                                                              name='Super Top5 Accuracy',
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
        predictions = tf.nn.top_k(input=probabilities, k=1, sorted=False)

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

        return confusion_matrix, samples