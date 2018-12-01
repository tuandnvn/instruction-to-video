import tensorflow as tf
import numpy as np

data = np.load("../instruction-to-video/data/image_train.npy")
print(data.shape)
dataset = tf.data.Dataset.from_tensor_slices(data)

print(dataset.output_shapes)
print(dataset.output_types)

iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()

with tf.Session() as sess:
    for i in range(10):
        inter = sess.run(next_element)
        assert data[i].all() == inter.all()


batched_dataset = dataset.batch(32)
iterator = batched_dataset.make_one_shot_iterator()
next_element = iterator.get_next()
with tf.Session() as sess:
    batched = sess.run(next_element)
    print(batched.shape)