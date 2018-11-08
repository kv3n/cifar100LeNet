import tensorflow as tf
import numpy as np
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


train = Data('train')
validation = Data('train')
test = Data('test')

# Step 1: Data Selection: Select 40000 examples
TRAIN_SIZE = 40000
train.select(finish=TRAIN_SIZE)
validation.select(start=TRAIN_SIZE)#, finish=TRAIN_SIZE + 100)

# Step 1.1: Setup training constants
EPOCHS = 1000
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

# Step 2: Augment Data

# Step 3: Pre-process / Normalize data

# Step 4: Build NN architecture


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


# Step 4.0: Prep Data
def mean_image_initializer():
    cast_data = tf.cast(train.data, tf.float32)
    normalized_data = tf.divide(cast_data, tf.constant(255.0, tf.float32))
    return tf.reduce_mean(normalized_data,
                          axis=0,
                          keepdims=True)


mean_image = tf.Variable(initial_value=mean_image_initializer(),
                         name='MeanImage',
                         trainable=False)


is_train = tf.placeholder(tf.bool, name='IsTrain')
is_test = tf.placeholder(tf.bool, name='IsTest')


# Training Data Iterator
train_raw_input = tf.data.Dataset.from_tensor_slices((train.data, train.fine_labels))
train_dataset = train_raw_input.shuffle(buffer_size=train.num_data,
                                        reshuffle_each_iteration=True)\
                               .repeat(count=EPOCHS)\
                               .batch(batch_size=BATCH_SIZE)
train_input_iter = train_dataset.make_one_shot_iterator()

# Validation Data Iterator
validation_raw_input = tf.data.Dataset.from_tensor_slices((validation.data, validation.fine_labels))
validation_dataset = validation_raw_input.repeat(count=EPOCHS * VALIDATIONS_PER_EPOCH)\
                                         .batch(batch_size=validation.num_data)
validation_input_iter = validation_dataset.make_one_shot_iterator()

# Test Data Iterator
test_raw_input = tf.data.Dataset.from_tensor_slices((test.data, test.fine_labels))
test_dataset = test_raw_input.repeat(count=EPOCHS * TESTS_PER_EPOCH)\
                              .batch(batch_size=test.num_data)
test_input_iter = test_dataset.make_one_shot_iterator()


def get_train_iter():
    return train_input_iter.get_next(name='TrainingBatch')


def get_validation_iter():
    return validation_input_iter.get_next(name='ValidationData')


def get_test_iter():
    return test_input_iter.get_next(name='TestData')


def test_vs_valid():
    return tf.cond(pred=is_test,
                   true_fn=get_test_iter,
                   false_fn=get_validation_iter,
                   name='TestSelector')


data_batch, label_batch = tf.cond(pred=is_train,
                                  true_fn=get_train_iter,
                                  false_fn=test_vs_valid,
                                  name='TrainSelector')
data_batch_cast = tf.cast(data_batch, tf.float32)
# TODO: 0.0 to 1.0 the data... Is this needed?
data_batch_cast = tf.divide(data_batch_cast, tf.constant(255.0, tf.float32))
data_batch_cast = tf.subtract(x=data_batch_cast,
                              y=mean_image,
                              name='MeanSubtraction')

# Step 4.1: Build Architecture
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


loss_op = tf.losses.sparse_softmax_cross_entropy(labels=label_batch,
                                                 logits=logits)
tf.summary.scalar(name='Loss',
                  tensor=loss_op)

train_step = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss_op)


predictions = tf.argmax(input=logits,
                        axis=1,
                        name='Predict')

accuracy_op = tf.reduce_mean(tf.cast(x=tf.equal(predictions, label_batch), dtype=tf.float32))
accuracy_summary = tf.summary.scalar(name='Accuracy',
                                     tensor=accuracy_op)

confusion_matrix_op = tf.confusion_matrix(labels=label_batch,
                                          predictions=predictions,
                                          name='Confusion')

merged_summary = tf.summary.merge_all()

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
                                              feed_dict={is_train: True, is_test: False})
            train_writer.add_summary(batch_summary,
                                     global_step=global_batch_count)

            global_batch_count += 1

            #print('-- Batch Count ' + str(global_batch_count % (NUM_BATCHES_PER_EPOCH + 1)))

            if global_batch_count % VALIDATION_INTERVAL == 0:
                _, accuracy = sess.run([accuracy_op, accuracy_summary],
                                       feed_dict={is_train: False, is_test: False})
                validation_writer.add_summary(accuracy,
                                              global_step=global_batch_count)

                half_epoch_count += 1
                print('Ran half epoch ' + str(half_epoch_count))

            if global_batch_count % TEST_INTERVAL == 0:
                confusion_matrix, accuracy = sess.run([confusion_matrix_op, accuracy_summary],
                                                      feed_dict={is_train: False, is_test: True})
                test_writer.add_summary(accuracy,
                                        global_step=global_batch_count)

                test_epoch_count += 1
                print('Ran test ' + str(test_epoch_count))
                print(confusion_matrix)
                print('-----------------------------------')
        except tf.errors.OutOfRangeError:
            print('End of Epochs')
            break
