# Cifar10-CNN
A try for an CNN written in Keras with the Cifar10 Dataset



# ----------------------------- #
A Little Help to Setup GPU/CPU and/or MultiGPU Using :
# ----------------------------- #
# Change From GPU to another GPU or CPU :

# -------
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# The GPU id to use, usually either "0" or "1"
# Change this number to 0 if you have only 0(Zero) if you have only 1 GPU
os.environ["CUDA_VISIBLE_DEVICES"]="1"
#---


# ----------------------------- #
# To Check your GPU´s :
# use this Line in a Seperate Code or in Jupyter Notebook :

print(device_lib.list_local_devices())

# ----------------------------- #
# For Use CPU only but with MultiCore Support import this lines and Delete the os.eviron lines

# use  CPU with max from 4 Cores
num_cores = 4

# config to use CPU
config = tf.ConfigProto(intra_op_parallelism_threads=num_cores,\
        inter_op_parallelism_threads=num_cores, allow_soft_placement=True,\
        device_count = {'CPU' : 4, 'GPU' : 0})
# Use this Config
session = tf.Session(config=config)

# ----------------------------- #
# Use MultiGPU
from keras.utils import multi_gpu_model

# Replicates `model` on 8 GPUs.
# This assumes that your machine has 8 available GPUs.
parallel_model = multi_gpu_model(model, gpus=1)
parallel_model.compile(loss='categorical_crossentropy',
                       optimizer='rmsprop')

# This `fit` call will be distributed on 8 GPUs.
# Since the batch size is 256, each GPU will process 32 samples.
parallel_model.fit(x, y, epochs=20, batch_size=256)
