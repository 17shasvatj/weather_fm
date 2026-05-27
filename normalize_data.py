import numpy as np

print("Normalizing train...", flush=True)
train = np.load('train_data.npy')
mean = np.load('mean.npy')
std = np.load('std.npy')
train -= mean
train /= std
np.save('train_norm.npy', train)
del train

print("Normalizing test...", flush=True)
test = np.load('test_data.npy')
test -= mean
test /= std
np.save('test_norm.npy', test)
del test

print("Done!", flush=True)