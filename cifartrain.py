import tensorflow as tf
import itertools
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import pickle
import argparse
import time
import random

parser = argparse.ArgumentParser(description='Tensorflow Log Name')
parser.add_argument('logname', type=str, nargs='?', help='name of logfile', default='--t')
parser.add_argument('seed', type=int, nargs='?', help='random seed. 0 if true random', default=0)

args = parser.parse_args()
log_name = args.logname
if log_name == '--t':
    log_name = str(time.time())

seed = args.seed
if seed == 0:
    seed = random.randint(0, 1 << 32)
    print('Setting seed: ' + str(seed))


class Data:
    # Train and Test Data Layout
    # Dictionary with following keys
    # 'batch_label' - Gives the batch id the training data belongs to
    # 'coarse_labels' - 50000 labels of class names (1 .. 20)
    # 'data' - 50000 image data of type
    #           <0, 1, ...., 1023> R-Channel,
    #           <1024, 1025, ..., 2047> G-Channel,
    #           <2048, 2049, ..., 3071> B-Channel
    # 'fine_labels' - 50000 labels of class names (1...100)
    # 'filenames' - 50000 filenames

    def __init__(self, file):
        def unpickle(file):
            with open(file, 'rb') as fo:
                data_dict = pickle.load(fo, encoding='bytes')
            return data_dict

        self.raw = unpickle('data/' + file)

        self.batch_label = None
        self.coarse_labels = None
        self.fine_labels = None
        self.data = None
        self.filenames = None
        self.num_data = 0

        self.reset_data()

    def reset_data(self):
        self.batch_label = self.raw[b'batch_label']
        self.coarse_labels = np.array(self.raw[b'coarse_labels'])
        self.fine_labels = np.array(self.raw[b'fine_labels'])
        self.data = self.raw[b'data']
        self.filenames = self.raw[b'filenames']
        self.num_data = len(self.data)

    def select(self, start=0, finish=-1):
        if finish == -1:
            finish = self.num_data
        self.batch_label = self.batch_label[start:finish]
        self.coarse_labels = self.coarse_labels[start:finish]
        self.fine_labels = self.fine_labels[start:finish]
        self.data = self.data[start:finish]
        self.filenames = self.filenames[start:finish]
        self.num_data = len(self.data)


class Meta:
    def __init__(self):
        def unpickle(file):
            with open(file, 'rb') as fo:
                data_dict = pickle.load(fo, encoding='bytes')
            return data_dict

        self.raw = unpickle('data/meta')
        self.fine_label_names = [fine_label_name.decode('utf-8') for fine_label_name in self.raw[b'fine_label_names']]
        self.coarse_label_names = [coarse_label_name.decode('utf-8') for coarse_label_name in self.raw[b'coarse_label_names']]
        self.fine_label_count = len(self.fine_label_names)
        self.coarse_label_count = len(self.coarse_label_names)


meta = Meta()

train = Data('train')
validation = Data('train')
test = Data('test')

fine_to_coarse_label = dict(set(zip(train.fine_labels, train.coarse_labels)))

# Step 1: Data Selection: Select 40000 examples
TRAIN_SIZE = 40000
train.select(finish=TRAIN_SIZE)
validation.select(start=TRAIN_SIZE)

# Step 1.1: Setup training constants
EPOCHS = 150
BATCH_SIZE = 64
LEARNING_RATE = 0.001
IMAGE_SIZE = 32
CROP_SIZE = 28
IMAGE_DEPTH = 3
TRAIN_SIZE = train.num_data
VALIDATIONS_PER_EPOCH = 2
NUM_BATCHES_PER_EPOCH = TRAIN_SIZE // BATCH_SIZE
VALIDATION_INTERVAL = NUM_BATCHES_PER_EPOCH // VALIDATIONS_PER_EPOCH
TESTS_PER_EPOCH = 0.1
TEST_INTERVAL = int(NUM_BATCHES_PER_EPOCH // TESTS_PER_EPOCH)

# Step 1.2: Set random seed for all randoms in tensorflow
tf.set_random_seed(seed=seed)

# Step 2: Pre-process / Normalize data
def mean_image_initializer():
    cast_data = tf.cast(train.data, tf.float32)
    normalized_data = tf.divide(cast_data, tf.constant(255.0, tf.float32))
    return tf.reduce_mean(normalized_data,
                          axis=0,
                          keepdims=True)


mean_image = tf.Variable(initial_value=mean_image_initializer(),
                         name='MeanImage',
                         trainable=False)

# Step 3: Build NN architecture
def create_conv_layer(num, inputs, filters, size=5, stride=1):
    layer_name = 'Conv' + str(num) + '-' + str(size) + 'x' + str(size) + 'x' + str(filters) + '-' + str(stride)

    return tf.layers.conv2d(inputs=inputs,
                            filters=filters,
                            kernel_size=[size, size],
                            strides=[stride, stride],
                            activation=tf.nn.relu,
                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                            bias_initializer=tf.contrib.layers.xavier_initializer(),
                            name=layer_name)


def create_pooling_layer(num, inputs, size=2, stride=2):
    layer_name = 'Pool' + str(num) + '-' + str(size) + 'x' + str(size) + '-' + str(stride)
    return tf.layers.max_pooling2d(inputs=inputs,
                                   pool_size=[size, size],
                                   strides=[stride, stride],
                                   name=layer_name)


def create_dense_layer(num, inputs, nodes, activation=tf.nn.relu):
    layer_name = 'Dense' + str(num) + '-' + str(nodes)
    if num < 0:
        layer_name = 'Output'
    return tf.layers.dense(inputs=inputs,
                           units=nodes,
                           activation=activation,
                           kernel_initializer=tf.contrib.layers.xavier_initializer(),
                           bias_initializer=tf.contrib.layers.xavier_initializer(),
                           name=layer_name)


# Step 3.1: Prep Data
TRAIN_DATA_TYPE = 1
VAL_DATA_TYPE = 2
TEST_DATA_TYPE = 3

data_type = tf.placeholder(tf.uint8, name='DataType')
train_data_type = tf.constant(value=TRAIN_DATA_TYPE, dtype=tf.uint8, name='TrainDataType')
val_data_type = tf.constant(value=VAL_DATA_TYPE, dtype=tf.uint8, name='ValDataType')
test_data_type = tf.constant(value=TEST_DATA_TYPE, dtype=tf.uint8, name='TestDataType')


# Training Data Iterator
train_raw_input = tf.data.Dataset.from_tensor_slices((train.data, train.fine_labels, train.coarse_labels))
train_dataset = train_raw_input.shuffle(buffer_size=train.num_data,
                                        reshuffle_each_iteration=True)\
                               .repeat(count=EPOCHS)\
                               .batch(batch_size=BATCH_SIZE)
train_input_iter = train_dataset.make_one_shot_iterator()

# Validation Data Iterator
validation_raw_input = tf.data.Dataset.from_tensor_slices((validation.data, validation.fine_labels, validation.coarse_labels))
validation_dataset = validation_raw_input.repeat(count=int(EPOCHS * VALIDATIONS_PER_EPOCH))\
                                         .batch(batch_size=validation.num_data)
validation_input_iter = validation_dataset.make_one_shot_iterator()

# Test Data Iterator
test_raw_input = tf.data.Dataset.from_tensor_slices((test.data, test.fine_labels, test.coarse_labels))
test_dataset = test_raw_input.repeat(count=int(EPOCHS * TESTS_PER_EPOCH))\
                              .batch(batch_size=test.num_data)
test_input_iter = test_dataset.make_one_shot_iterator()


def get_train_iter():
    return train_input_iter.get_next(name='TrainingBatch')


def get_validation_iter():
    return validation_input_iter.get_next(name='ValidationData')


def get_test_iter():
    return test_input_iter.get_next(name='TestData')


data_batch, label_batch, coarse_label_batch = tf.case(pred_fn_pairs={tf.equal(data_type, train_data_type): get_train_iter,
                                                                     tf.equal(data_type, val_data_type): get_validation_iter,
                                                                     tf.equal(data_type, test_data_type): get_test_iter},
                                                      exclusive=True,
                                                      default=get_train_iter,
                                                      name='DataSelector')

data_batch_cast = tf.cast(data_batch, tf.float32)

data_batch_cast = tf.divide(data_batch_cast, tf.constant(255.0, tf.float32))
data_batch_cast = tf.subtract(x=data_batch_cast,
                              y=mean_image,
                              name='MeanSubtraction')

"""
# This reshape isn't correct. But that said, this gave 30% results? What did I do?
input_layer = tf.reshape(tensor=data_batch_cast,
                         shape=[-1, IMAGE_SIZE, IMAGE_SIZE, IMAGE_DEPTH],
                         name='MakeImage')
"""

input_layer = tf.reshape(tensor=data_batch_cast,
                         shape=[-1, IMAGE_DEPTH, IMAGE_SIZE, IMAGE_SIZE],
                         name='MakeImage-Part1')
input_layer = tf.transpose(a=input_layer,
                           perm=[0, 2, 3, 1],
                           name='MakeImage-Part2')

# Step 3.1.1: Augment data
def augment_only_on_train():
    return input_layer
    """
    #data_to_augment = tf.map_fn(lambda img: tf.image.random_flip_left_right(img), elems=input_layer)
    data_to_augment = tf.map_fn(lambda img: tf.image.random_crop(value=img,
                                                                 size=[CROP_SIZE, CROP_SIZE, IMAGE_DEPTH]),
                                elems=input_layer)
    data_to_augment = tf.image.resize_images(images=data_to_augment,
                                             size=[IMAGE_SIZE, IMAGE_SIZE])
    
    return data_to_augment
    """


def no_augment():
    return input_layer


augmented_layer = tf.cond(pred=tf.equal(data_type, train_data_type),
                          true_fn=augment_only_on_train,
                          false_fn=no_augment,
                          name='AugmentSelect')

# Step 3.2: Stitch Layers
first_convolution_layer = create_conv_layer(num=1,
                                            inputs=augmented_layer,
                                            filters=6)

first_pooling_layer = create_pooling_layer(num=1,
                                           inputs=first_convolution_layer)

second_convolution_layer = create_conv_layer(num=2,
                                             inputs=first_pooling_layer,
                                             filters=6)

second_pooling_layer = create_pooling_layer(num=1,
                                            inputs=second_convolution_layer)

flattened_pooling_layer = tf.layers.flatten(inputs=second_pooling_layer,
                                            name='FlattenPool2')

first_dense_layer = create_dense_layer(num=1,
                                       inputs=flattened_pooling_layer,
                                       nodes=120)

second_dense_layer = create_dense_layer(num=2,
                                        inputs=first_dense_layer,
                                        nodes=84)

logits = create_dense_layer(num=3,
                            inputs=second_dense_layer,
                            nodes=100)


# Step 3.3: Optimize Loss
loss_op = tf.losses.sparse_softmax_cross_entropy(labels=label_batch,
                                                 logits=logits)
tf.summary.scalar(name='Loss',
                  tensor=loss_op)

train_step = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss_op)


# Step 4: Detail accuracy and confusion matrix scores.
top1_op = tf.nn.in_top_k(predictions=logits,
                         targets=label_batch,
                         k=1,
                         name='Top1')

accuracy_op = tf.reduce_mean(tf.cast(top1_op, tf.float32))
accuracy_summary = tf.summary.scalar(tensor=accuracy_op,
                                     name='Accuracy1')

top5_op = tf.nn.in_top_k(predictions=logits,
                         targets=label_batch,
                         k=5,
                         name='Top5')

accuracy5_op = tf.reduce_mean(tf.cast(top5_op, tf.float32))
accuracy5_summary = tf.summary.scalar(tensor=accuracy5_op,
                                      name='Accuracy5')


prediction_op = tf.argmax(input=logits,
                          axis=1,
                          name='PredictLabels')

# 4.1 Superlabel Predictions
super_label_prediction_op = tf.map_fn(lambda label: fine_to_coarse_label[label],
                                      elems=prediction_op)

super_top1_accuracy = tf.reduce_mean(tf.cast(tf.equal(x=super_label_prediction_op,
                                                      y=coarse_label_batch),
                                             dtype=tf.float32))
accuracy_super1_summary = tf.summary.scalar(tensor=super_top1_accuracy,
                                            name='SuperLabelTop1')

prediction_op_superlabel5, _ = tf.nn.top_k(input=logits,
                                           k=5)

mapped_superlabel5 = tf.map_fn(lambda top5labels: tf.map_fn(lambda label: fine_to_coarse_label[label],
                                                            elems=top5labels),
                               elems=prediction_op_superlabel5)
mapped_superlabel5 = tf.cast(mapped_superlabel5,
                             dtype=tf.int32)
real_coarse_label = tf.cast(coarse_label_batch,
                            dtype=tf.int32)
super_top5_accuracy = tf.cast(tf.reduce_any(tf.logical_not(tf.cast(tf.subtract(x=mapped_superlabel5,
                                                                               y=real_coarse_label),
                                                                   dtype=tf.bool)),
                                            axis=1),
                              dtype=tf.float32)

super_top5_accuracy = tf.reduce_mean(super_top5_accuracy)
accuracy_super5_summary = tf.summary.scalar(tensor=super_top5_accuracy,
                                            name='SuperLabelTop5')


confusion_matrix_op = tf.confusion_matrix(labels=label_batch,
                                          predictions=prediction_op,
                                          name='Confusion')

merged_summary = tf.summary.merge_all()

correct_values_op = tf.where(tf.equal(x=prediction_op,
                                      y=label_batch,
                                      name='CorrectPrediction'))

correct_values_op = tf.gather(params=correct_values_op,
                              indices=tf.random_uniform(shape=(5,),
                                                        minval=0,
                                                        maxval=tf.maximum(tf.size(input=correct_values_op), 1),
                                                        dtype=tf.int32),
                              name='GatherCorrect')

wrong_values_op = tf.where(tf.not_equal(x=prediction_op,
                                        y=label_batch,
                                        name='WrongPrediction'))

wrong_values_op = tf.gather(params=wrong_values_op,
                            indices=tf.random_uniform(shape=(5,),
                                                      minval=0,
                                                      maxval=tf.maximum(tf.size(input=wrong_values_op), 1),
                                                      dtype=tf.int32),
                            name='GatherWrong')

samples_op = tf.concat(values=[correct_values_op, wrong_values_op],
                       axis=0,
                       name='Sampling')


def save_confusion_matix(confusion_matrix, count):
    sum_across_axis = confusion_matrix.sum(axis=1)[:, np.newaxis]
    confusion_matrix_interp = confusion_matrix.astype('float') / sum_across_axis
    confusion_matrix_interp = np.nan_to_num(confusion_matrix_interp)

    plt.figure(figsize=(100, 100))
    plt.imshow(confusion_matrix_interp, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix: ' + str(count))
    plt.colorbar()
    tick_marks = np.arange(meta.fine_label_count)
    plt.xticks(tick_marks, meta.fine_label_names, rotation=45)
    plt.yticks(tick_marks, meta.fine_label_names)

    fmt = 'd'
    thresh = confusion_matrix_interp.max() / 2.
    for i, j in itertools.product(range(confusion_matrix.shape[0]), range(confusion_matrix.shape[1])):
        plt.text(j, i, format(confusion_matrix[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if confusion_matrix_interp[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    plt.savefig(log_name + '_confusion_matrix' + str(count) + '.png')
    plt.close()


def gather(data, labels, predictions, sampled_indices, count):
    sampled_indices = np.unique(sampled_indices.flatten())

    def save_sample(sample):
        image = np.reshape(data[sample], [3, 32, 32])
        image = np.transpose(image, [1, 2, 0])
        plt.figure()
        plt.imshow(image)
        real_name = meta.fine_label_names[labels[sample]]
        predict_name = meta.fine_label_names[predictions[sample]]
        plt.xlabel('Real: ' + real_name + ', Predict: ' + predict_name)
        plt.savefig(log_name + '_' + str(count) + '_sample_' + real_name + '.png')
        plt.close()

    for sample in sampled_indices:
        save_sample(sample)


# Step 5: Run Graph
with tf.Session() as sess:
    # debug writer
    log_name = 'logs/' + log_name
    train_writer = tf.summary.FileWriter(logdir=log_name + '_train/',
                                         graph=sess.graph)

    validation_writer = tf.summary.FileWriter(logdir=log_name + '_val/')

    test_writer = tf.summary.FileWriter(logdir=log_name + '_test/')

    sess.run(tf.global_variables_initializer())

    global_batch_count = 0
    half_epoch_count = 0
    test_epoch_count = 0
    while True:
        try:
            # Run mini-batch
            _, batch_summary = sess.run([train_step, merged_summary],
                                        feed_dict={data_type: 1})

            train_writer.add_summary(batch_summary,
                                     global_step=global_batch_count)

            global_batch_count += 1

            if global_batch_count % VALIDATION_INTERVAL == 0:
                acc, acc5 = sess.run([accuracy_summary, accuracy5_summary],
                                     feed_dict={data_type: 2})
                validation_writer.add_summary(acc,
                                              global_step=half_epoch_count)
                validation_writer.add_summary(acc5,
                                              global_step=half_epoch_count)

                half_epoch_count += 1
                print('Ran half epoch ' + str(half_epoch_count))

            if global_batch_count % TEST_INTERVAL == 0:
                data, labels, predictions, samples, confusion_matrix, acc, acc5 = \
                    sess.run([data_batch, label_batch, prediction_op, samples_op, confusion_matrix_op, accuracy_summary, accuracy5_summary],
                             feed_dict={data_type: 3})
                test_writer.add_summary(acc,
                                        global_step=test_epoch_count)
                test_writer.add_summary(acc5,
                                        global_step=test_epoch_count)

                test_epoch_count += 1
                print('Ran test ' + str(test_epoch_count))
                print(confusion_matrix)
                print('-----------------------------------')
                save_confusion_matix(confusion_matrix, test_epoch_count)
                print('Saved Confusion Matrix')
                gather(data, labels, predictions, samples, test_epoch_count)
                print('Sampled Results')
        except tf.errors.OutOfRangeError:
            print('End of Epochs')
            break
