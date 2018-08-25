from __future__ import division, print_function, absolute_import
import tensorflow as tf


def conv2d(x, W, b, strides=1):
    """Conv2D wrapper, with bias and relu activation, the default stride is 1"""
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return x


def relu(x):
    """Relu activation function"""
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    """Max pooling wrapper"""
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


# Store layers weight & bias
weights = {
    # 3x3 conv, 3 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([3, 3, 3, 32])),
    # 3x3 conv, 32 inputs, 32 outputs
    'wc2': tf.Variable(tf.random_normal([3, 3, 32, 32])),
    # 3x3 conv, 32 inputs, 32 outputs
    'wc3': tf.Variable(tf.random_normal([3, 3, 32, 32]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([32])),
    'bc3': tf.Variable(tf.random_normal([32]))
}


# Create model
def conv_net(x, weights=weights, biases=biases):
    # Reshape to match picture format [Height x Width x Channel]
    # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
    x = tf.Variable(x, dtype=tf.float32)
    x = tf.reshape(x, shape=[-1, 15, 15, 3])

    # Convolution Layer
    conv1 = relu(conv2d(x, weights['wc1'], biases['bc1']))
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1)
    print(conv1.shape)

    # Convolution Layer
    conv2 = relu(conv2d(conv1, weights['wc2'], biases['bc2']))
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2)
    print(conv2.shape)

    conv3 = relu(conv2d(conv2, weights['wc3'], biases['bc3']))
    print(conv3.shape)
    out = tf.reshape(conv3, [-1, conv3.shape[1] * conv3.shape[2], 32])
    return out


if __name__ == "__main__":

    from evaluation_utils import crop_and_resize
    import cv2

    file_path = "../instruction-to-video/target/0/0.mp4"

    cap = cv2.VideoCapture(file_path)
    _, first_frame = cap.read()
    first_frame = crop_and_resize(first_frame)
    print(first_frame.shape)
    output = conv_net(first_frame, weights, biases)
    print(output.shape)