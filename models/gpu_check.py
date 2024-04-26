"""
Check that Nvidia GPU card is found by TensorFlow.

"""
import tensorflow as tf

print("Num GPUs Found: ", len(tf.config.experimental.list_physical_devices('GPU')))