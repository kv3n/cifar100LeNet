import tensorflow as tf
import itertools
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import pickle
import argparse
import time

parser = argparse.ArgumentParser(description='Tensorflow Log Name')
parser.add_argument('logname', type=str, nargs='?', help='name of logfile', default='--t')

args = parser.parse_args()
log_name = args.logname
if log_name == '--t':
    log_name = str(time.time())

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
                dict = pickle.load(fo, encoding='bytes')
            return dict

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
                dict = pickle.load(fo, encoding='bytes')
            return dict

        self.raw = unpickle('data/meta')
        self.fine_label_names = [fine_label_name.decode('utf-8') for fine_label_name in self.raw[b'fine_label_names']]
        self.coarse_label_names = [coarse_label_name.decode('utf-8') for coarse_label_name in self.raw[b'coarse_label_names']]
        self.fine_label_count = len(self.fine_label_names)
        self.coarse_label_count = len(self.coarse_label_names)


meta = Meta()

train = Data('train')
validation = Data('train')
test = Data('test')

# Step 1: Data Selection: Select 40000 examples
TRAIN_SIZE = 40000
train.select(finish=TRAIN_SIZE)
validation.select(start=TRAIN_SIZE)
#test.select(start=0, finish=500)

# Step 1.1: Setup training constants
EPOCHS = 10
BATCH_SIZE = 64
LEARNING_RATE = 0.001
IMAGE_SIZE = 32
IMAGE_DEPTH = 3
TRAIN_SIZE = train.num_data
VALIDATIONS_PER_EPOCH = 2
NUM_BATCHES_PER_EPOCH = TRAIN_SIZE // BATCH_SIZE
VALIDATION_INTERVAL = NUM_BATCHES_PER_EPOCH // VALIDATIONS_PER_EPOCH
TESTS_PER_EPOCH = 0.1
TEST_INTERVAL = int(NUM_BATCHES_PER_EPOCH // TESTS_PER_EPOCH)

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
data_type = tf.placeholder(tf.uint8, name='DataType')


# Training Data Iterator
train_raw_input = tf.data.Dataset.from_tensor_slices((train.data, train.fine_labels))
train_dataset = train_raw_input.shuffle(buffer_size=train.num_data,
                                        reshuffle_each_iteration=True)\
                               .repeat(count=EPOCHS)\
                               .batch(batch_size=BATCH_SIZE)
train_input_iter = train_dataset.make_one_shot_iterator()

# Validation Data Iterator
validation_raw_input = tf.data.Dataset.from_tensor_slices((validation.data, validation.fine_labels))
validation_dataset = validation_raw_input.repeat(count=int(EPOCHS * VALIDATIONS_PER_EPOCH))\
                                         .batch(batch_size=validation.num_data)
validation_input_iter = validation_dataset.make_one_shot_iterator()

# Test Data Iterator
test_raw_input = tf.data.Dataset.from_tensor_slices((test.data, test.fine_labels))
test_dataset = test_raw_input.repeat(count=int(EPOCHS * TESTS_PER_EPOCH))\
                              .batch(batch_size=test.num_data)
test_input_iter = test_dataset.make_one_shot_iterator()


def get_train_iter():
    return train_input_iter.get_next(name='TrainingBatch')


def get_validation_iter():
    return validation_input_iter.get_next(name='ValidationData')


def get_test_iter():
    return test_input_iter.get_next(name='TestData')


data_batch, label_batch = tf.case(pred_fn_pairs={tf.equal(data_type, tf.constant(1, tf.uint8)): get_train_iter,
                                                 tf.equal(data_type, tf.constant(2, tf.uint8)): get_validation_iter,
                                                 tf.equal(data_type, tf.constant(3, tf.uint8)): get_test_iter},
                                  exclusive=True,
                                  default=get_train_iter,
                                  name='DataSelector')

data_batch_cast = tf.cast(data_batch, tf.float32)
# TODO: 0.0 to 1.0 the data... Is this needed?
data_batch_cast = tf.divide(data_batch_cast, tf.constant(255.0, tf.float32))
data_batch_cast = tf.subtract(x=data_batch_cast,
                              y=mean_image,
                              name='MeanSubtraction')

# Step 3.2: Stitch Layers
input_layer = tf.reshape(tensor=data_batch_cast,
                         shape=[-1, IMAGE_SIZE, IMAGE_SIZE, IMAGE_DEPTH],
                         name='MakeImage')


first_convolution_layer = create_conv_layer(num=1,
                                            inputs=input_layer,
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
predictions = tf.nn.in_top_k(predictions=logits,
                             targets=label_batch,
                             k=1,
                             name='Predict')

accuracy_op = tf.reduce_mean(tf.cast(predictions, tf.float32))
accuracy_summary = tf.summary.scalar(tensor=accuracy_op,
                                     name='Accuracy')

confusion_matrix_op = tf.confusion_matrix(labels=label_batch,
                                          predictions=predictions,
                                          name='Confusion')

merged_summary = tf.summary.merge_all()


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
            _, _, _, batch_summary = sess.run([train_step, loss_op, accuracy_op, merged_summary],
                                              feed_dict={data_type: 1})

            train_writer.add_summary(batch_summary,
                                     global_step=global_batch_count)

            global_batch_count += 1

            if global_batch_count % VALIDATION_INTERVAL == 0:
                _, accuracy = sess.run([accuracy_op, accuracy_summary],
                                       feed_dict={data_type: 2})
                validation_writer.add_summary(accuracy,
                                              global_step=half_epoch_count)

                half_epoch_count += 1
                print('Ran half epoch ' + str(half_epoch_count))

            if global_batch_count % TEST_INTERVAL == 0:
                confusion_matrix, _, accuracy = sess.run([confusion_matrix_op, accuracy_op, accuracy_summary],
                                                      feed_dict={data_type: 3})
                test_writer.add_summary(accuracy,
                                        global_step=test_epoch_count)

                test_epoch_count += 1
                print('Ran test ' + str(test_epoch_count))
                print(confusion_matrix)
                print('-----------------------------------')
                save_confusion_matix(confusion_matrix, test_epoch_count)
        except tf.errors.OutOfRangeError:
            print('End of Epochs')
            break
