import tensorflow as tf
import argparse
import time
import random
from data_feed import *
from model import *
from summary_builder import *

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


tf.set_random_seed(seed=seed)

data_type = tf.placeholder(name='DataType', dtype=tf.uint8)
data_feed = Data(datatype_placeholder=data_type)
data_mapping = tf.constant(value=[data_feed.mapping[label] for label in range(100)], dtype=tf.int32)
batch, raw, fine_labels, coarse_labels = data_feed.get_batch_feed()

output, optimize, loss = build_model_no_augment(image_batch=batch, true_labels=fine_labels)

summary_builder = SummaryBuilder(log_name, data_mapping)
train_summary, validation_summary, test_summary = summary_builder.build_summary(probabilities=output,
                                                                                loss=loss,
                                                                                fine_labels=fine_labels,
                                                                                coarse_labels=coarse_labels)

predictions, confusion_matrix, sampling = summary_builder.create_confusion_and_sample(probabilities=output,
                                                                                      true_labels=fine_labels)


with tf.Session() as sess:
    summary_builder.training.add_graph(graph=sess.graph)
    sess.run(tf.global_variables_initializer())

    global_batch_count = 0
    half_epoch_count = 0
    test_epoch_count = 0
    while True:
        try:
            # Run mini-batch
            _, _, summary = sess.run([optimize, output, train_summary], feed_dict={data_type: 1})

            summary_builder.training.add_summary(summary, global_step=data_feed.global_step)

            run_validation, run_test = data_feed.step_train()

            if run_validation:
                _, summary = sess.run([output, validation_summary], feed_dict={data_type: 2})

                summary_builder.validation.add_summary(summary, global_step=data_feed.validation_step)
                print('Ran Validation: ' + str(data_feed.validation_step))

            if run_test:
                data, labels, predict, summary, cm, samples = sess.run([raw, fine_labels, predictions,
                                                                        test_summary, confusion_matrix,
                                                                        sampling],
                                                                       feed_dict={data_type: 3})

                summary_builder.test.add_summary(summary, global_step=data_feed.test_step)
                print('Ran Test: ' + str(data_feed.test_step))

                print(cm)
                print('-----------------------------------')
                summary_builder.validate_confusion_matrix(cm, data_feed.test_step)

                summary_builder.save_confusion_matix(cm, data_feed.test_step)
                print('Saved Confusion Matrix')

                summary_builder.gather(data, labels, predict, samples, data_feed.test_step)
                print('Sampled Results')
        except tf.errors.OutOfRangeError:
            print('End of Epochs')
            break
