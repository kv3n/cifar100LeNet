import tensorflow as tf

LEARNING_RATE = 0.001


def _create_conv_layer_(name, inputs, filters, size=5, stride=1, padding='valid'):
    layer_name = 'Conv' + str(name) + '-' + str(size) + 'x' + str(size) + 'x' + str(filters) + '-' + str(stride)

    return tf.layers.conv2d(inputs=inputs,
                            filters=filters,
                            kernel_size=[size, size],
                            strides=[stride, stride],
                            activation=tf.nn.relu,
                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                            bias_initializer=tf.contrib.layers.xavier_initializer(),
                            padding=padding,
                            name=layer_name)


def _create_pooling_layer_(name, inputs, size=2, stride=2):
    layer_name = 'Pool' + str(name) + '-' + str(size) + 'x' + str(size) + '-' + str(stride)
    return tf.layers.max_pooling2d(inputs=inputs,
                                   pool_size=[size, size],
                                   strides=[stride, stride],
                                   name=layer_name)


def _create_dense_layer_(name, inputs, nodes, activation=tf.nn.relu):
    layer_name = 'Dense' + str(name) + '-' + str(nodes)

    return tf.layers.dense(inputs=inputs,
                           units=nodes,
                           activation=activation,
                           kernel_initializer=tf.contrib.layers.xavier_initializer(),
                           bias_initializer=tf.contrib.layers.xavier_initializer(),
                           name=layer_name)


def _augment_(image_batch):
    augmented_batch = tf.image.random_flip_left_right(image_batch)
    augmented_batch = tf.map_fn(lambda img: tf.random_crop(value=img, size=[28, 28, 3]), elems=augmented_batch)
    augmented_batch = tf.image.resize_images(augmented_batch, size=[32, 32])
    return augmented_batch


def __noaugment__(image_batch):
    return image_batch


def build_model(image_batch, true_labels):
    image_batch = _augment_(image_batch)

    image_batch = _create_conv_layer_(name='1', inputs=image_batch, filters=6)

    image_batch = _create_pooling_layer_(name='1', inputs=image_batch)

    image_batch = _create_conv_layer_(name='2', inputs=image_batch, filters=16)

    image_batch = _create_pooling_layer_(name='2', inputs=image_batch)

    flatten_batch = tf.layers.flatten(inputs=image_batch, name='Conv2FC')

    flatten_batch = _create_dense_layer_(name='1', inputs=flatten_batch, nodes=120)

    flatten_batch = _create_dense_layer_(name='2', inputs=flatten_batch, nodes=84)

    output = _create_dense_layer_(name='3', inputs=flatten_batch, nodes=100)

    loss = tf.losses.sparse_softmax_cross_entropy(labels=true_labels, logits=output)

    optimize = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss, name='Optimize')

    return output, optimize, loss
