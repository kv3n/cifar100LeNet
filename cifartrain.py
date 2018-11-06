import tensorflow as tf
import numpy as np
import pickle

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
        self.batch_label = self.batch_label[start:finish]
        self.coarse_labels = self.coarse_labels[start:finish]
        self.fine_labels = self.fine_labels[start:finish]
        self.data = self.data[start:finish]
        self.filenames = self.filenames[start:finish]
        self.num_data = len(self.data)


train = Data('train')
test = Data('test')

# Step 1: Data Selection: Select 40000 examples
train.select(finish=40000)

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


input_layer = tf.reshape(train.data, [-1, 32, 32, 3])

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
                                            name='Flatten Pool2')

first_dense_layer = create_dense_layer(num=1,
                                       inputs=flattened_pooling_layer,
                                       nodes=120)

second_dense_layer = create_dense_layer(num=2,
                                        inputs=flattened_pooling_layer,
                                        nodes=84)

output_layer = create_dense_layer(num=-1,
                                  inputs=second_dense_layer,
                                  nodes=100,
                                  activation=tf.nn.softmax)

# Step 4.1: Build debug ops
debug_input_op = tf.summary.image(name='Reshaped Input',
                                  tensor=input_layer)

# Step 5: Run Graph
with tf.Session() as sess:
    debug_input = sess.run(debug_input_op)

    # Write debug info to board
    writer = tf.train.SummaryWriter('./logs')
    writer.add_summary(debug_input)
    writer.close()
