import time

import tensorflow as tf

print(tf.test.gpu_device_name())

while True:
    time.sleep(1)
