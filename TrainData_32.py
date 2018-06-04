from __future__ import print_function
import os
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
import scipy.io
from matplotlib import pyplot as plt
import random

# from tensorflow.python.client import device_lib
# print (device_lib.list_local_devices())

##################load data#####################

all_data = pickle.load(open('SVHN_multi_crop_normalized_32.pickle', 'rb'))
train_data = all_data['train_dataset']
test_data = all_data['test_dataset']
valid_data = all_data['valid_dataset']

train_labels = all_data['train_labels']
test_labels = all_data['test_labels']
valid_labels = all_data['valid_labels']

del all_data


# print(np.max(valid_data),np.min(valid_data))



def getOneHot(Datasetlabels):
    dataSize = Datasetlabels.shape[0]
    onehotLabels1 = np.zeros((dataSize, 7))
    onehotLabels2 = np.zeros((dataSize, 10))
    onehotLabels3 = np.zeros((dataSize, 10))
    onehotLabels4 = np.zeros((dataSize, 10))
    onehotLabels5 = np.zeros((dataSize, 10))
    onehotLabels6 = np.zeros((dataSize, 10))

    for i in range(dataSize):
        labels = Datasetlabels[i]
        num_of_digits = labels[0]
        onehotLabels1[i, num_of_digits] = 1
        counter = 0
        if counter < num_of_digits:
            onehotLabels2[i, labels[1] % 10] = 1
            # counter=counter+1
        if counter < num_of_digits:
            onehotLabels3[i, labels[2] % 10] = 1
            # counter = counter + 1
        if counter < num_of_digits:
            onehotLabels4[i, labels[3] % 10] = 1
            # counter = counter + 1
        if counter < num_of_digits:
            onehotLabels5[i, labels[4] % 10] = 1
            # counter = counter + 1
        if counter < num_of_digits:
            onehotLabels6[i, labels[5] % 10] = 1
    return [onehotLabels1, onehotLabels2, onehotLabels3, onehotLabels4, onehotLabels5, onehotLabels6]


trainOneHotLabels = getOneHot(train_labels)
testOneHotLabels = getOneHot(test_labels)
validOneHotLabels = getOneHot(valid_labels)

################################################


#################Load train and test data###################


num_channels = 1  # grayscale
image_size = 32
pixel_depth = 255.0


def reformat(dataset):
    dataset = dataset.reshape(
        (-1, image_size, image_size, num_channels)).astype(np.float32)
    return dataset


train_data = reformat(train_data)
test_data = reformat(test_data)
valid_data = reformat(valid_data)

print('train_data shape is : %s' % (train_data.shape,))
print('test_data shape is : %s' % (test_data.shape,))
print('valid_data shape is : %s' % (valid_data.shape,))

test_size = test_data.shape[0]
validation_size = valid_data.shape[0]
train_size = train_data.shape[0]

############################################################



########################Training###########################

num_classifiers = 6


def get_onehot_as_string(labels):
    all_labels = []
    batch_size = labels[0].shape[0]
    for i in range(batch_size):
        num_digits = np.argmax(labels[0][i])
        st = str(num_digits)
        for j in range(1, num_classifiers):
            if (j > num_digits):
                break
            st = st + str(np.argmax(labels[j][i]))
        all_labels.append(st)
    return all_labels


def accuracy(predictions, labels):
    batch_size = predictions[0].shape[0]
    predictions = get_onehot_as_string(predictions)
    labels = get_onehot_as_string(labels)
    equalities = np.zeros(batch_size)
    for i in range(batch_size):
        if predictions[i] == labels[i]:
            equalities[i] = 1
    sum = np.sum(equalities)
    acc = (100.0 * sum) / batch_size
    return acc, predictions


# output width=((W-F+2*P )/S)+1



num_digits_labels = 7
digits_labels = 10
batch_size = 64
test_batch_size = 457
patch_size = 5
depth1 = 16
depth2 = 32
depth3 = 64
num_hidden1 = 1024
num_hidden2 = 512
num_hidden3 = 256
# regularization_lambda=4e-4





graph = tf.Graph()

with graph.as_default():
    # Input data.
    tf_train_dataset = tf.placeholder(
        tf.float32, shape=(batch_size, image_size, image_size, num_channels))
    # tf_train_labels = [tf.placeholder(tf.float32, shape=(batch_size, num_digits_labels)),
    #                    tf.placeholder(tf.float32, shape=(batch_size, digits_labels)),
    #                    tf.placeholder(tf.float32, shape=(batch_size, digits_labels)),
    #                    tf.placeholder(tf.float32, shape=(batch_size, digits_labels)),
    #                    tf.placeholder(tf.float32, shape=(batch_size, digits_labels)),
    #                    tf.placeholder(tf.float32, shape=(batch_size, digits_labels))]

    tf_train_labels_c1 = tf.placeholder(tf.float32, shape=(batch_size, num_digits_labels))
    tf_train_labels_c2 = tf.placeholder(tf.float32, shape=(batch_size, digits_labels))
    tf_train_labels_c3 = tf.placeholder(tf.float32, shape=(batch_size, digits_labels))
    tf_train_labels_c4 = tf.placeholder(tf.float32, shape=(batch_size, digits_labels))
    tf_train_labels_c5 = tf.placeholder(tf.float32, shape=(batch_size, digits_labels))
    tf_train_labels_c6 = tf.placeholder(tf.float32, shape=(batch_size, digits_labels))

    tf_train_labels = [tf_train_labels_c1,
                       tf_train_labels_c2,
                       tf_train_labels_c3,
                       tf_train_labels_c4,
                       tf_train_labels_c5,
                       tf_train_labels_c6]

    tf_test_dataset = tf.placeholder(tf.float32, shape=(test_batch_size, image_size, image_size, num_channels))
    tf_one_input = tf.placeholder(tf.float32, shape=(1, image_size, image_size, num_channels))
    tf_validation_dataset = tf.constant(valid_data)


    def get_conv_weight(name, shape):
        return tf.get_variable(name, shape=shape,
                               initializer=tf.contrib.layers.xavier_initializer_conv2d())


    def get_bias_variable(shape):
        return tf.Variable(tf.constant(1.0, shape=shape))


    def get_fully_connected_weight(name, shape):
        weights = tf.get_variable(name, shape=shape,
                                  initializer=tf.contrib.layers.xavier_initializer())
        return weights


    # Variables.


    conv1_weights = get_conv_weight('conv1_weights', [patch_size, patch_size, num_channels, depth1])
    conv1_biases = get_bias_variable([depth1])

    conv2_weights = get_conv_weight('conv2_weights', [patch_size, patch_size, depth1, depth2])
    conv2_biases = get_bias_variable([depth2])

    conv3_weights = get_conv_weight('conv3_weights', [patch_size, patch_size, depth2, depth3])
    conv3_biases = get_bias_variable([depth3])

    # number of digits classifier

    hidden1_weights_c1 = get_fully_connected_weight('hidden1_weights_c1', [num_hidden1, num_hidden2])
    hidden1_biases_c1 = get_bias_variable([num_hidden2])

    hidden2_weights_c1 = get_fully_connected_weight('hidden2_weights_c1', [num_hidden2, num_hidden3])
    hidden2_biases_c1 = get_bias_variable([num_hidden3])

    hidden3_weights_c1 = get_fully_connected_weight('hidden3_weights_c1', [num_hidden3, num_digits_labels])
    hidden3_biases_c1 = get_bias_variable([num_digits_labels])

    # first number classifier
    hidden1_weights_c2 = get_fully_connected_weight('hidden1_weights_c2', [num_hidden1, num_hidden2])
    hidden1_biases_c2 = get_bias_variable([num_hidden2])

    hidden2_weights_c2 = get_fully_connected_weight('hidden2_weights_c2', [num_hidden2, num_hidden3])
    hidden2_biases_c2 = get_bias_variable([num_hidden3])

    hidden3_weights_c2 = get_fully_connected_weight('hidden3_weights_c2', [num_hidden3, digits_labels])
    hidden3_biases_c2 = get_bias_variable([digits_labels])

    # second number classifier
    hidden1_weights_c3 = get_fully_connected_weight('hidden1_weights_c3', [num_hidden1, num_hidden2])
    hidden1_biases_c3 = get_bias_variable([num_hidden2])

    hidden2_weights_c3 = get_fully_connected_weight('hidden2_weights_c3', [num_hidden2, num_hidden3])
    hidden2_biases_c3 = get_bias_variable([num_hidden3])

    hidden3_weights_c3 = get_fully_connected_weight('hidden3_weights_c3', [num_hidden3, digits_labels])
    hidden3_biases_c3 = get_bias_variable([digits_labels])

    # third number classifier
    hidden1_weights_c4 = get_fully_connected_weight('hidden1_weights_c4', [num_hidden1, num_hidden2])
    hidden1_biases_c4 = get_bias_variable([num_hidden2])

    hidden2_weights_c4 = get_fully_connected_weight('hidden2_weights_c4', [num_hidden2, num_hidden3])
    hidden2_biases_c4 = get_bias_variable([num_hidden3])

    hidden3_weights_c4 = get_fully_connected_weight('hidden3_weights_c4', [num_hidden3, digits_labels])
    hidden3_biases_c4 = get_bias_variable([digits_labels])

    # fourth number classifier
    hidden1_weights_c5 = get_fully_connected_weight('hidden1_weights_c5', [num_hidden1, num_hidden2])
    hidden1_biases_c5 = get_bias_variable([num_hidden2])

    hidden2_weights_c5 = get_fully_connected_weight('hidden2_weights_c5', [num_hidden2, num_hidden3])
    hidden2_biases_c5 = get_bias_variable([num_hidden3])

    hidden3_weights_c5 = get_fully_connected_weight('hidden3_weights_c5', [num_hidden3, digits_labels])
    hidden3_biases_c5 = get_bias_variable([digits_labels])

    # fifth number classifier
    hidden1_weights_c6 = get_fully_connected_weight('hidden1_weights_c6', [num_hidden1, num_hidden2])
    hidden1_biases_c6 = get_bias_variable([num_hidden2])

    hidden2_weights_c6 = get_fully_connected_weight('hidden2_weights_c6', [num_hidden2, num_hidden3])
    hidden2_biases_c6 = get_bias_variable([num_hidden3])

    hidden3_weights_c6 = get_fully_connected_weight('hidden3_weights_c6', [num_hidden3, digits_labels])
    hidden3_biases_c6 = get_bias_variable([digits_labels])


    def get_logits(image_vector, hidden1_weights, hidden1_biases, hidden2_weights, hidden2_biases, hidden3_weights,
                   hidden3_biases, keep_dropout_rate=1):
        hidden = tf.nn.relu(tf.matmul(image_vector, hidden1_weights) + hidden1_biases)
        if keep_dropout_rate < 1:
            hidden = tf.nn.dropout(hidden, keep_dropout_rate)
        hidden = tf.nn.relu(tf.matmul(hidden, hidden2_weights) + hidden2_biases)
        if keep_dropout_rate < 1:
            hidden = tf.nn.dropout(hidden, keep_dropout_rate)
        return tf.matmul(hidden, hidden3_weights) + hidden3_biases


    def run_conv_layer(input, conv_weights, conv_biases):
        conv = tf.nn.conv2d(input, conv_weights, [1, 1, 1, 1], padding='SAME')
        conv = tf.nn.max_pool(value=conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        conv = tf.nn.local_response_normalization(conv)
        return tf.nn.relu(conv + conv_biases)


    # Model.
    def model(data, keep_dropout_rate=1):

        # first conv block
        hidden = run_conv_layer(data, conv1_weights, conv1_biases)
        # second conv block
        hidden = run_conv_layer(hidden, conv2_weights, conv2_biases)
        # third conv block
        hidden = run_conv_layer(hidden, conv3_weights, conv3_biases)

        shape = hidden.get_shape().as_list()
        reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])

        # first classifier
        logits1 = get_logits(reshape, hidden1_weights_c1, hidden1_biases_c1, hidden2_weights_c1, hidden2_biases_c1,
                             hidden3_weights_c1, hidden3_biases_c1, keep_dropout_rate)

        # second classifier
        logits2 = get_logits(reshape, hidden1_weights_c2, hidden1_biases_c2, hidden2_weights_c2, hidden2_biases_c2,
                             hidden3_weights_c2, hidden3_biases_c2, keep_dropout_rate)

        # third classifier
        logits3 = get_logits(reshape, hidden1_weights_c3, hidden1_biases_c3, hidden2_weights_c3, hidden2_biases_c3,
                             hidden3_weights_c3, hidden3_biases_c3, keep_dropout_rate)

        # fourth classifier
        logits4 = get_logits(reshape, hidden1_weights_c4, hidden1_biases_c4, hidden2_weights_c4, hidden2_biases_c4,
                             hidden3_weights_c4, hidden3_biases_c4, keep_dropout_rate)

        # fifth classifier
        logits5 = get_logits(reshape, hidden1_weights_c5, hidden1_biases_c5, hidden2_weights_c5, hidden2_biases_c5,
                             hidden3_weights_c5, hidden3_biases_c5, keep_dropout_rate)

        # sixth classifier
        logits6 = get_logits(reshape, hidden1_weights_c6, hidden1_biases_c6, hidden2_weights_c6, hidden2_biases_c6,
                             hidden3_weights_c6, hidden3_biases_c6, keep_dropout_rate)

        return [logits1, logits2, logits3, logits4, logits5, logits6]


    # Training computation.
    logits = model(tf_train_dataset, 0.90)

    # regularizers=regularization_lambda*(tf.nn.l2_loss(hidden1_weights) + tf.nn.l2_loss(hidden1_biases))+regularization_lambda*(tf.nn.l2_loss(hidden2_weights) + tf.nn.l2_loss(hidden2_biases))+regularization_lambda*(tf.nn.l2_loss(hidden3_weights) + tf.nn.l2_loss(hidden3_biases))
    loss = 0.0
    for i in range(num_classifiers):
        loss = loss + tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels[i], logits=logits[i]))  # +regularizers

    # tf.train.exponential_decay(learning_rate, global_step, decay_steps, decay_rate, staircase=False, name=None)
    # decayed_learning_rate = learning_rate *decay_rate ^ (global_step / decay_steps)

    global_step = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(0.001, global_step, 1000, 0.90, staircase=True)
    # Optimizer.
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars),
                                      1.0)
    optimize = optimizer.apply_gradients(
        zip(grads, tvars),
        global_step=global_step)


    # Predictions for the training, validation, and test data.

    def stack_predictions(logits):
        prediction_c1 = tf.nn.softmax(logits[0])
        prediction_c2 = tf.nn.softmax(logits[1])
        prediction_c3 = tf.nn.softmax(logits[2])
        prediction_c4 = tf.nn.softmax(logits[3])
        prediction_c5 = tf.nn.softmax(logits[4])
        prediction_c6 = tf.nn.softmax(logits[5])
        return prediction_c1, prediction_c2, prediction_c3, prediction_c4, prediction_c5, prediction_c6


    train_prediction_c1, train_prediction_c2, train_prediction_c3, train_prediction_c4, train_prediction_c5, train_prediction_c6 = stack_predictions(
        logits)

    valid_prediction_c1, valid_prediction_c2, valid_prediction_c3, valid_prediction_c4, valid_prediction_c5, valid_prediction_c6 = stack_predictions(
        model(tf_validation_dataset))
    test_prediction_c1, test_prediction_c2, test_prediction_c3, test_prediction_c4, test_prediction_c5, test_prediction_c6 = stack_predictions(
        model(tf_test_dataset))
    one_prediction_c1, one_prediction_c2, one_prediction_c3, one_prediction_c4, one_prediction_c5, one_prediction_c6 = stack_predictions(
        model(tf_one_input))

num_steps = 501

training_loss = []
training_loss_epoch = []

train_accuracy = []
train_accuracy_epoch = []

valid_accuracy = []
valid_accuracy_epoch = []

test_prediction = []

test_accuracy=0

with tf.Session(graph=graph, config=tf.ConfigProto(log_device_placement=True)) as session:
    tf.global_variables_initializer().run()
    # `sess.graph` provides access to the graph used in a `tf.Session`.
    writer = tf.summary.FileWriter('./graph_info', session.graph)

    print('Initialized')
    for step in range(num_steps):
        offset = (step * batch_size) % (train_size - batch_size)
        batch_data = train_data[offset:(offset + batch_size), :, :, :]
        batch_labels = []
        for i in range(num_classifiers):
            batch_labels.append(trainOneHotLabels[i][offset:(offset + batch_size), :])
        feed_dict = {tf_train_dataset: batch_data, tf_train_labels_c1: batch_labels[0],
                     tf_train_labels_c2: batch_labels[1], tf_train_labels_c3: batch_labels[2],
                     tf_train_labels_c4: batch_labels[3], tf_train_labels_c5: batch_labels[4],
                     tf_train_labels_c6: batch_labels[5]}
        _, l, c1, c2, c3, c4, c5, c6, lr = session.run(
            [optimize, loss, train_prediction_c1, train_prediction_c2, train_prediction_c3, train_prediction_c4,
             train_prediction_c5, train_prediction_c6, learning_rate], feed_dict=feed_dict)
        predictions = [c1, c2, c3, c4, c5, c6]
        if (step % 50 == 0):
            print('Learning rate at step %d: %.14f' % (step, lr))
            print('Minibatch loss at step %d: %f' % (step, l))
            batch_train_accuracy, _ = accuracy(predictions, batch_labels)
            print('Minibatch accuracy: %.1f%%' % batch_train_accuracy)
            training_loss.append(l)
            training_loss_epoch.append(step)
            train_accuracy.append(batch_train_accuracy)
            train_accuracy_epoch.append(step)
            if(lr==0):
                break

        if (step % 500 == 0):
            c1, c2, c3, c4, c5, c6 = session.run(
                [valid_prediction_c1, valid_prediction_c2, valid_prediction_c3, valid_prediction_c4,
                 valid_prediction_c5, valid_prediction_c6])
            predictions = [c1, c2, c3, c4, c5, c6]
            validation_accuracy, _ = accuracy(predictions, validOneHotLabels)
            print('validation accuracy: %.1f%%' % validation_accuracy)
            valid_accuracy.append(validation_accuracy)
            valid_accuracy_epoch.append(step)

    test_pred_c1 = np.zeros((test_size, num_digits_labels))
    test_pred_c2 = np.zeros((test_size, digits_labels))
    test_pred_c3 = np.zeros((test_size, digits_labels))
    test_pred_c4 = np.zeros((test_size, digits_labels))
    test_pred_c5 = np.zeros((test_size, digits_labels))
    test_pred_c6 = np.zeros((test_size, digits_labels))

    for step in range(int(test_size / test_batch_size)):
        offset = (step * test_batch_size) % (test_size - test_batch_size)
        batch_data = test_data[offset:(offset + test_batch_size), :, :, :]
        feed_dict = {tf_test_dataset: batch_data}
        c1, c2, c3, c4, c5, c6 = session.run(
            [test_prediction_c1, test_prediction_c2, test_prediction_c3, test_prediction_c4, test_prediction_c5,
             test_prediction_c6], feed_dict=feed_dict)

        test_pred_c1[offset:offset + test_batch_size] = c1
        test_pred_c2[offset:offset + test_batch_size] = c2
        test_pred_c3[offset:offset + test_batch_size] = c3
        test_pred_c4[offset:offset + test_batch_size] = c4
        test_pred_c5[offset:offset + test_batch_size] = c5
        test_pred_c6[offset:offset + test_batch_size] = c6
    predictions = [test_pred_c1, test_pred_c2, test_pred_c3, test_pred_c4, test_pred_c5, test_pred_c6]
    test_accuracy, test_predictions = accuracy(predictions, testOneHotLabels)
    writer.close()
    saver = tf.train.Saver()
    saver.save(session, "./saved_model/model.ckpt")


            #############################################################


def plot_x_y(x, y, figure_name, x_axis_name, y_axis_name):
    plt.figure()
    plt.plot(x, y)
    plt.xlabel(x_axis_name)
    plt.ylabel(y_axis_name)
    # plt.legend([line_name],loc='upper left')
    plt.savefig('./output_images/' + figure_name)
    # plt.show()


plot_x_y(training_loss_epoch, training_loss, 'training_loss.png', 'epoch', 'training batch loss')

plot_x_y(valid_accuracy_epoch, valid_accuracy, 'training_acc.png', 'epoch', 'training batch accuracy')

plot_x_y(valid_accuracy_epoch, valid_accuracy, 'valid_acc.png', 'epoch', 'validation accuracy')


def disp_prediction_samples(predictions, dataset, num_images):
    for image_num in range(num_images):
        items = random.sample(range(dataset.shape[0]), 8)
        for i, item in enumerate(items):
            plt.subplot(2, 4, i + 1)
            plt.axis('off')
            plt.title(predictions[item][1:])
            plt.imshow(dataset[item, :, :, 0])
        plt.savefig('./output_images/' + 'predictions' + str(image_num + 1) + '.png')
        # plt.show()


disp_prediction_samples(test_predictions, test_data, 10)

print('Test accuracy: %.1f%%' % test_accuracy)