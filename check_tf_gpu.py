import tensorflow as tf
print("TensorFlow Version:", tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print("Physical Devices:", tf.config.list_physical_devices())
try:
    print("Built with CUDA:", tf.test.is_built_with_cuda())
except:
    print("tf.test.is_built_with_cuda() not available")
try:
    print("Built with GPU Support:", tf.test.is_built_with_gpu_support())
except:
    print("tf.test.is_built_with_gpu_support() not available")
