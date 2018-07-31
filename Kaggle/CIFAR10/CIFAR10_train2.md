```{.python .input  n=58}
from mxnet import autograd
from mxnet import gluon
from mxnet import image
from mxnet import init
from mxnet import nd
from mxnet.gluon.data import vision
import numpy as np
import random
import mxnet as mx
from utils.netlib import *
ctx = mx.gpu(1)
print(ctx)
```

```{.json .output n=58}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "gpu(1)\n"
 }
]
```

```{.python .input  n=69}
"""
data loader
"""
data_dir = '/home/zp/SummerSchool/CS231n/Kaggle/data/CIFAR-10/train_valid_test/'
#data_dir = "/home/zp/SummerSchool/gluon-tutorials-zh/data/kaggle_cifar10/train_valid_test/"

def _transform_test(data, label):
    im = data.astype('float32') / 255
    auglist = image.CreateAugmenter(data_shape=(3, 32, 32), mean=np.array([0.4914, 0.4822, 0.4465]),
                                   std=np.array([0.2023, 0.1994, 0.2010]))
    for aug in auglist:
        im = aug(im)
    im = nd.transpose(im, (2, 0, 1))
    return im, nd.array([label]).astype('float32')


def data_loader(batch_size, transform_train, transform_test=None):
    if transform_train is None:
        transform_train = _transform_train
    if transform_test is None:
        transform_test = _transform_test
        
    # flag=1 mean 3 channel image
    train_ds = vision.ImageFolderDataset(data_dir + 'train', flag=1, transform=transform_train)
    valid_ds = vision.ImageFolderDataset(data_dir + 'valid', flag=1, transform=transform_test)
    train_valid_ds = vision.ImageFolderDataset(data_dir + 'train_valid', flag=1, transform=transform_train)
    test_ds = vision.ImageFolderDataset(data_dir + "test", flag=1, transform=transform_test)

    loader = gluon.data.DataLoader
    train_data = loader(train_ds, batch_size, shuffle=True, last_batch='keep')
    valid_data = loader(valid_ds, batch_size, shuffle=True, last_batch='keep')
    train_valid_data = loader(train_valid_ds, batch_size, shuffle=True, last_batch='keep')
    test_data = loader(test_ds, batch_size, shuffle=False, last_batch='keep')
    return train_data, valid_data, train_valid_data, test_data, test_ds, train_valid_ds


```

## 数据增强
DA是3中不同的数据增强的方法，

DA1就是最常用的那种padding到40,然后crop的方法，就是sherlock代码里使用的加强

DA2是先resize到一定的大小，然后crop的方法，同时设置了HSI的几个参数为0.3,PCA噪声为0.01

DA3时在DA2后，将图片的颜色clip导（0,1）之间（动机时创建更符合人感官的自然图片数据）

mixup 的代码可以参考：https://github.com/unsky/mixup 

```{.python .input  n=49}
"""
data argument
"""
def transform_train_DA1(data, label):
    im = data.asnumpy()
    im = np.pad(im, ((4, 4), (4, 4), (0, 0)), mode='constant', constant_values=0)
    im = nd.array(im, dtype='float32') / 255
    auglist = image.CreateAugmenter(data_shape=(3, 32, 32), resize=0, rand_mirror=True,
                                    rand_crop=True,
                                   mean=np.array([0.4914, 0.4822, 0.4465]),
                                   std=np.array([0.2023, 0.1994, 0.2010]))
    for aug in auglist:
        im = aug(im)
    im = nd.transpose(im, (2, 0, 1)) # channel x width x height
    return im, nd.array([label]).astype('float32')


def transform_train_DA2(data, label):
    im = data.astype(np.float32) / 255
    auglist = [image.RandomSizedCropAug(size=(32, 32), min_area=0.49, ratio=(0.5, 2))]
    _aug = image.CreateAugmenter(data_shape=(3, 32, 32), resize=0, 
                                rand_crop=False, rand_resize=False, rand_mirror=True,
                                mean=np.array([0.4914, 0.4822, 0.4465]),
                                std=np.array([0.2023, 0.1994, 0.2010]),
                                brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3,
                                pca_noise=0.01, rand_gray=0, inter_method=2)
    auglist.append(image.RandomOrderAug(_aug))
    
    for aug in auglist:
        im = aug(im)
    
    im = nd.transpose(im, (2, 0, 1))
    return (im, nd.array([label]).asscalar().astype('float32'))
    

random_clip_rate = 0.3
def transform_train_DA3(data, label):
    im = data.astype(np.float32) / 255
    auglist = [image.RandomSizedCropAug(size=(32, 32), min_area=0.49, ratio=(0.5, 2))]
    _aug = image.CreateAugmenter(data_shape=(3, 32, 32), resize=0, 
                                rand_crop=False, rand_resize=False, rand_mirror=True,
#                                mean=np.array([0.4914, 0.4822, 0.4465]),
#                                std=np.array([0.2023, 0.1994, 0.2010]),
                                brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3,
                                pca_noise=0.01, rand_gray=0, inter_method=2)
    auglist.append(image.RandomOrderAug(_aug))

    for aug in auglist:
        im = aug(im)
        
    if random.random() > random_clip_rate:
        im = im.clip(0, 1)
    _aug = image.ColorNormalizeAug(mean=np.array([0.4914, 0.4822, 0.4465]),
                   std=np.array([0.2023, 0.1994, 0.2010]),)
    im = _aug(im)
    
    im = nd.transpose(im, (2, 0, 1))
    return (im, nd.array([label]).asscalar().astype('float32'))



```

```{.python .input  n=50}
"""
train
"""
import datetime
from utils import utils
import sys

def abs_mean(W):
    return nd.mean(nd.abs(W)).asscalar()

def in_list(e, l):
    for i in l:
        if i == e:
            return True
    else:
        return False

def train(net, train_data, valid_data, num_epochs, lr, lr_period, 
          lr_decay, wd, ctx, w_key, output_file=None, verbose=False, loss_f=gluon.loss.SoftmaxCrossEntropyLoss()):
    if output_file is None:
        output_file = sys.stdout
        stdout = sys.stdout
    else:
        output_file = open(output_file, "w")
        stdout = sys.stdout
        sys.stdout = output_file
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr, 'momentum': 0.9, 'wd': wd})
    prev_time = datetime.datetime.now()
    
    if verbose:
        print (" #", utils.evaluate_accuracy(valid_data, net, ctx))
    
    i = 0
    for epoch in range(num_epochs):
        train_loss = 0.
        train_acc = 0.
        if in_list(epoch, lr_period):
            trainer.set_learning_rate(trainer.learning_rate * lr_decay)
        if epoch %50 ==1:
            #model_dir="models/shelock_densenet_orign"+str(epoch)
            #model_dir="models/resnet164_e0-255_focal_clip"+str(epoch)
            #print(model_dir)models/resnet164_e0-255_focal_clip
            net.save_parameters(model_dir)
        for data, label in train_data:
            label = label.as_in_context(ctx)
            with autograd.record():
                output = net(data.as_in_context(ctx))
                loss = loss_f(output, label)
            loss.backward()
            trainer.step(data.shape[0])
            
            _loss = nd.mean(loss).asscalar()
            _acc = utils.accuracy(output, label) 
            #_acc = utils.accuracy(output, label)
            train_loss += _loss
            train_acc += _acc
            
            if verbose:
                print (" # iter", i,)
                print ("loss %.5f" % _loss, "acc %.5f" % _acc,)
                print ("w (",)
                for k in w_key:
                    w = net.collect_params()[k]
                    print ("%.5f, " % abs_mean(w.data()),)
                print (") g (",)
                for k in w_key:
                    w = net.collect_params()[k]
                    print ("%.5f, " % abs_mean(w.grad()),)
                print (")")
                i += 1
            
        cur_time = datetime.datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = "Time %02d:%02d:%02d" % (h, m, s)
        
        train_loss /= len(train_data)
        train_acc /= len(train_data)
        
        if valid_data is not None:
            valid_acc = utils.evaluate_accuracy(valid_data, net, ctx)
            epoch_str = ("epoch %d, loss %.5f, train_acc %.4f, valid_acc %.4f" 
                         % (epoch, train_loss, train_acc, valid_acc))
        else:
            epoch_str = ("epoch %d, loss %.5f, train_acc %.4f"
                        % (epoch, train_loss, train_acc))
        prev_time = cur_time
        output_file.write(epoch_str + ", " + time_str + ",lr " + str(trainer.learning_rate) + "\n")
        output_file.flush()  # to disk only when flush or close
    if output_file != stdout:
        sys.stdout = stdout
        output_file.close()
```

### Exp1: res164_v2 + DA1: 0.9529

```{.python .input  n=9}
batch_size = 128
transform_train = transform_train_DA1
train_data, valid_data, train_valid_data, test_data, test_ds, train_valid_ds = data_loader(batch_size, transform_train)
net = ResNet164_v2(10)
loss_f = gluon.loss.SoftmaxCrossEntropyLoss()

num_epochs = 100
learning_rate = 0.1
weight_decay = 1e-4
lr_period = [90, 140]
lr_decay=0.1
log_file = None

net.collect_params().initialize(mx.init.Xavier(), ctx=ctx, force_reinit=True)
net.hybridize()
w_key = []
train(net, train_data, valid_data, num_epochs, learning_rate, 
      lr_period, lr_decay, weight_decay, ctx, w_key, log_file, False, loss_f)

net.save_parameters("models/shelock_resnet_orign2")
```

```{.json .output n=9}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "epoch 0, loss 1.50946, train_acc 0.4416, valid_acc 0.6064, Time 00:03:14,lr 0.1\nepoch 1, loss 0.98563, train_acc 0.6497, valid_acc 0.6018, Time 00:03:20,lr 0.1\nepoch 2, loss 0.78067, train_acc 0.7260, valid_acc 0.6980, Time 00:03:17,lr 0.1\nepoch 3, loss 0.65889, train_acc 0.7706, valid_acc 0.7877, Time 00:03:19,lr 0.1\nepoch 4, loss 0.57477, train_acc 0.8015, valid_acc 0.7830, Time 00:03:14,lr 0.1\nepoch 5, loss 0.52184, train_acc 0.8194, valid_acc 0.7604, Time 00:03:19,lr 0.1\nepoch 6, loss 0.48165, train_acc 0.8336, valid_acc 0.8166, Time 00:03:21,lr 0.1\nepoch 7, loss 0.44974, train_acc 0.8434, valid_acc 0.8033, Time 00:03:18,lr 0.1\nepoch 8, loss 0.41501, train_acc 0.8557, valid_acc 0.8426, Time 00:03:15,lr 0.1\nepoch 9, loss 0.39230, train_acc 0.8643, valid_acc 0.8383, Time 00:03:17,lr 0.1\nepoch 10, loss 0.37666, train_acc 0.8697, valid_acc 0.8543, Time 00:03:18,lr 0.1\nepoch 11, loss 0.35127, train_acc 0.8790, valid_acc 0.8516, Time 00:03:18,lr 0.1\nepoch 12, loss 0.33768, train_acc 0.8831, valid_acc 0.8514, Time 00:03:15,lr 0.1\nepoch 13, loss 0.32778, train_acc 0.8869, valid_acc 0.8588, Time 00:03:20,lr 0.1\nepoch 14, loss 0.31145, train_acc 0.8924, valid_acc 0.8688, Time 00:03:17,lr 0.1\nepoch 15, loss 0.29807, train_acc 0.8974, valid_acc 0.8760, Time 00:03:18,lr 0.1\nepoch 16, loss 0.29299, train_acc 0.8976, valid_acc 0.8621, Time 00:03:22,lr 0.1\nepoch 17, loss 0.28337, train_acc 0.8999, valid_acc 0.8557, Time 00:03:17,lr 0.1\nepoch 18, loss 0.26833, train_acc 0.9063, valid_acc 0.8883, Time 00:03:15,lr 0.1\nepoch 19, loss 0.26323, train_acc 0.9096, valid_acc 0.8656, Time 00:03:16,lr 0.1\nepoch 20, loss 0.25583, train_acc 0.9099, valid_acc 0.8748, Time 00:03:18,lr 0.1\nepoch 21, loss 0.24686, train_acc 0.9150, valid_acc 0.8600, Time 00:03:18,lr 0.1\nepoch 22, loss 0.23998, train_acc 0.9151, valid_acc 0.8934, Time 00:03:16,lr 0.1\nepoch 23, loss 0.23434, train_acc 0.9191, valid_acc 0.8828, Time 00:03:17,lr 0.1\nepoch 24, loss 0.23558, train_acc 0.9179, valid_acc 0.8850, Time 00:03:16,lr 0.1\nepoch 25, loss 0.22934, train_acc 0.9205, valid_acc 0.8873, Time 00:03:18,lr 0.1\nepoch 26, loss 0.21884, train_acc 0.9228, valid_acc 0.8805, Time 00:03:16,lr 0.1\nepoch 27, loss 0.22520, train_acc 0.9222, valid_acc 0.8830, Time 00:03:16,lr 0.1\nepoch 28, loss 0.21016, train_acc 0.9273, valid_acc 0.8984, Time 00:03:19,lr 0.1\nepoch 29, loss 0.21331, train_acc 0.9263, valid_acc 0.8859, Time 00:03:20,lr 0.1\nepoch 30, loss 0.21093, train_acc 0.9270, valid_acc 0.8900, Time 00:03:18,lr 0.1\nepoch 31, loss 0.20585, train_acc 0.9282, valid_acc 0.8812, Time 00:03:19,lr 0.1\nepoch 32, loss 0.20773, train_acc 0.9264, valid_acc 0.8965, Time 00:03:21,lr 0.1\nepoch 33, loss 0.20355, train_acc 0.9296, valid_acc 0.8939, Time 00:03:18,lr 0.1\nepoch 34, loss 0.20376, train_acc 0.9299, valid_acc 0.8809, Time 00:03:21,lr 0.1\nepoch 35, loss 0.19341, train_acc 0.9324, valid_acc 0.8795, Time 00:03:21,lr 0.1\nepoch 36, loss 0.20092, train_acc 0.9302, valid_acc 0.9006, Time 00:03:19,lr 0.1\nepoch 37, loss 0.19353, train_acc 0.9320, valid_acc 0.8797, Time 00:03:21,lr 0.1\nepoch 38, loss 0.18818, train_acc 0.9334, valid_acc 0.8963, Time 00:03:15,lr 0.1\nepoch 39, loss 0.19334, train_acc 0.9320, valid_acc 0.9014, Time 00:03:19,lr 0.1\nepoch 40, loss 0.19393, train_acc 0.9323, valid_acc 0.8863, Time 00:03:19,lr 0.1\nepoch 41, loss 0.18255, train_acc 0.9366, valid_acc 0.8785, Time 00:03:15,lr 0.1\nepoch 42, loss 0.19154, train_acc 0.9323, valid_acc 0.8605, Time 00:03:20,lr 0.1\nepoch 43, loss 0.18654, train_acc 0.9348, valid_acc 0.9088, Time 00:03:21,lr 0.1\nepoch 44, loss 0.17995, train_acc 0.9371, valid_acc 0.8838, Time 00:03:19,lr 0.1\nepoch 45, loss 0.17934, train_acc 0.9360, valid_acc 0.8992, Time 00:03:19,lr 0.1\nepoch 46, loss 0.17960, train_acc 0.9372, valid_acc 0.8953, Time 00:03:19,lr 0.1\nepoch 47, loss 0.18006, train_acc 0.9373, valid_acc 0.8967, Time 00:03:21,lr 0.1\nepoch 48, loss 0.18122, train_acc 0.9359, valid_acc 0.9082, Time 00:03:18,lr 0.1\nepoch 49, loss 0.17991, train_acc 0.9385, valid_acc 0.8990, Time 00:03:17,lr 0.1\nepoch 50, loss 0.17349, train_acc 0.9398, valid_acc 0.8932, Time 00:03:16,lr 0.1\nepoch 51, loss 0.17509, train_acc 0.9388, valid_acc 0.8889, Time 00:03:16,lr 0.1\nepoch 52, loss 0.17567, train_acc 0.9389, valid_acc 0.9033, Time 00:03:24,lr 0.1\nepoch 53, loss 0.17226, train_acc 0.9400, valid_acc 0.8973, Time 00:03:19,lr 0.1\nepoch 54, loss 0.16931, train_acc 0.9405, valid_acc 0.9125, Time 00:03:15,lr 0.1\nepoch 55, loss 0.17194, train_acc 0.9398, valid_acc 0.9008, Time 00:03:14,lr 0.1\nepoch 56, loss 0.17035, train_acc 0.9401, valid_acc 0.8844, Time 00:03:15,lr 0.1\nepoch 57, loss 0.17199, train_acc 0.9398, valid_acc 0.8887, Time 00:03:18,lr 0.1\nepoch 58, loss 0.16802, train_acc 0.9397, valid_acc 0.9076, Time 00:03:17,lr 0.1\nepoch 59, loss 0.16895, train_acc 0.9410, valid_acc 0.8896, Time 00:03:15,lr 0.1\nepoch 60, loss 0.16763, train_acc 0.9417, valid_acc 0.8867, Time 00:03:21,lr 0.1\nepoch 61, loss 0.16738, train_acc 0.9414, valid_acc 0.9055, Time 00:03:18,lr 0.1\nepoch 62, loss 0.16229, train_acc 0.9436, valid_acc 0.9070, Time 00:03:19,lr 0.1\nepoch 63, loss 0.16073, train_acc 0.9433, valid_acc 0.8971, Time 00:03:18,lr 0.1\nepoch 64, loss 0.17038, train_acc 0.9397, valid_acc 0.9023, Time 00:03:18,lr 0.1\nepoch 65, loss 0.16770, train_acc 0.9410, valid_acc 0.9018, Time 00:03:16,lr 0.1\nepoch 66, loss 0.16101, train_acc 0.9442, valid_acc 0.9084, Time 00:03:13,lr 0.1\nepoch 67, loss 0.16027, train_acc 0.9441, valid_acc 0.8973, Time 00:03:19,lr 0.1\nepoch 68, loss 0.16649, train_acc 0.9411, valid_acc 0.9104, Time 00:03:21,lr 0.1\nepoch 69, loss 0.16232, train_acc 0.9438, valid_acc 0.9002, Time 00:03:20,lr 0.1\nepoch 70, loss 0.15518, train_acc 0.9458, valid_acc 0.8967, Time 00:03:20,lr 0.1\nepoch 71, loss 0.15915, train_acc 0.9453, valid_acc 0.8848, Time 00:03:13,lr 0.1\nepoch 72, loss 0.15495, train_acc 0.9461, valid_acc 0.9043, Time 00:03:20,lr 0.1\nepoch 73, loss 0.15978, train_acc 0.9438, valid_acc 0.8994, Time 00:03:18,lr 0.1\nepoch 74, loss 0.15937, train_acc 0.9442, valid_acc 0.8916, Time 00:03:21,lr 0.1\nepoch 75, loss 0.15149, train_acc 0.9461, valid_acc 0.8986, Time 00:03:15,lr 0.1\nepoch 76, loss 0.15160, train_acc 0.9471, valid_acc 0.8994, Time 00:03:18,lr 0.1\nepoch 77, loss 0.15902, train_acc 0.9449, valid_acc 0.8951, Time 00:03:14,lr 0.1\nepoch 78, loss 0.15454, train_acc 0.9464, valid_acc 0.8865, Time 00:03:19,lr 0.1\nepoch 79, loss 0.15584, train_acc 0.9454, valid_acc 0.8928, Time 00:03:13,lr 0.1\nepoch 80, loss 0.15105, train_acc 0.9479, valid_acc 0.8930, Time 00:03:14,lr 0.1\nepoch 81, loss 0.14515, train_acc 0.9496, valid_acc 0.9148, Time 00:03:16,lr 0.1\nepoch 82, loss 0.15359, train_acc 0.9462, valid_acc 0.9102, Time 00:03:19,lr 0.1\nepoch 83, loss 0.14415, train_acc 0.9498, valid_acc 0.8875, Time 00:03:20,lr 0.1\nepoch 84, loss 0.15090, train_acc 0.9479, valid_acc 0.9094, Time 00:03:15,lr 0.1\nepoch 85, loss 0.15029, train_acc 0.9479, valid_acc 0.9006, Time 00:03:18,lr 0.1\nepoch 86, loss 0.15177, train_acc 0.9477, valid_acc 0.8936, Time 00:03:17,lr 0.1\nepoch 87, loss 0.15112, train_acc 0.9470, valid_acc 0.9045, Time 00:03:14,lr 0.1\nepoch 88, loss 0.14859, train_acc 0.9476, valid_acc 0.8994, Time 00:03:18,lr 0.1\nepoch 89, loss 0.14403, train_acc 0.9485, valid_acc 0.8930, Time 00:03:22,lr 0.1\nepoch 90, loss 0.08680, train_acc 0.9710, valid_acc 0.9441, Time 00:03:17,lr 0.010000000000000002\nepoch 91, loss 0.03895, train_acc 0.9889, valid_acc 0.9521, Time 00:03:21,lr 0.010000000000000002\nepoch 92, loss 0.02951, train_acc 0.9922, valid_acc 0.9523, Time 00:03:18,lr 0.010000000000000002\nepoch 93, loss 0.02477, train_acc 0.9931, valid_acc 0.9549, Time 00:03:19,lr 0.010000000000000002\nepoch 94, loss 0.02035, train_acc 0.9952, valid_acc 0.9535, Time 00:03:18,lr 0.010000000000000002\nepoch 95, loss 0.01859, train_acc 0.9954, valid_acc 0.9561, Time 00:03:17,lr 0.010000000000000002\nepoch 96, loss 0.01565, train_acc 0.9963, valid_acc 0.9547, Time 00:03:17,lr 0.010000000000000002\nepoch 97, loss 0.01355, train_acc 0.9971, valid_acc 0.9553, Time 00:03:16,lr 0.010000000000000002\nepoch 98, loss 0.01291, train_acc 0.9971, valid_acc 0.9553, Time 00:03:18,lr 0.010000000000000002\nepoch 99, loss 0.01217, train_acc 0.9972, valid_acc 0.9555, Time 00:03:22,lr 0.010000000000000002\n"
 }
]
```

### Exp2:res164_v2 + DA2: 0.9527

```{.python .input}
batch_size = 128
transform_train2 = transform_train_DA2
train_data, valid_data, train_valid_data, test_data, test_ds, train_valid_ds = data_loader(batch_size, transform_train2)
net = ResNet164_v2(10)
loss_f = gluon.loss.SoftmaxCrossEntropyLoss()

num_epochs = 300
learning_rate = 0.1
weight_decay = 1e-4
lr_period = [150, 225]
lr_decay=0.1
log_file = None

net.collect_params().initialize(mx.init.Xavier(), ctx=ctx, force_reinit=True)
net.hybridize()
w_key = []
train(net, train_data, valid_data, num_epochs, learning_rate, 
      lr_period, lr_decay, weight_decay, ctx, w_key, log_file, False, loss_f)
net.save_params("models/resnet164_e300")
```

```{.json .output n=None}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "epoch 0, loss 1.89015, train_acc 0.3027, valid_acc 0.4654, Time 00:03:40,lr 0.1\nepoch 1, loss 1.38734, train_acc 0.5030, valid_acc 0.5604, Time 00:03:48,lr 0.1\nepoch 2, loss 1.09968, train_acc 0.6094, valid_acc 0.6428, Time 00:03:50,lr 0.1\nepoch 3, loss 0.96060, train_acc 0.6616, valid_acc 0.7293, Time 00:03:54,lr 0.1\nepoch 4, loss 0.85939, train_acc 0.6999, valid_acc 0.6543, Time 00:03:49,lr 0.1\nepoch 5, loss 0.78569, train_acc 0.7285, valid_acc 0.7666, Time 00:03:48,lr 0.1\nepoch 6, loss 0.72972, train_acc 0.7469, valid_acc 0.7443, Time 00:03:56,lr 0.1\nepoch 7, loss 0.69017, train_acc 0.7607, valid_acc 0.7852, Time 00:03:52,lr 0.1\nepoch 8, loss 0.66356, train_acc 0.7698, valid_acc 0.7975, Time 00:03:52,lr 0.1\nepoch 9, loss 0.62146, train_acc 0.7848, valid_acc 0.7969, Time 00:03:51,lr 0.1\nepoch 10, loss 0.59913, train_acc 0.7937, valid_acc 0.8203, Time 00:03:51,lr 0.1\nepoch 11, loss 0.58157, train_acc 0.7977, valid_acc 0.8355, Time 00:03:50,lr 0.1\nepoch 12, loss 0.55767, train_acc 0.8054, valid_acc 0.8322, Time 00:03:49,lr 0.1\nepoch 13, loss 0.53884, train_acc 0.8136, valid_acc 0.8463, Time 00:03:47,lr 0.1\nepoch 14, loss 0.52245, train_acc 0.8187, valid_acc 0.8125, Time 00:03:49,lr 0.1\nepoch 15, loss 0.50163, train_acc 0.8279, valid_acc 0.8363, Time 00:03:51,lr 0.1\nepoch 16, loss 0.49000, train_acc 0.8295, valid_acc 0.8611, Time 00:03:51,lr 0.1\nepoch 17, loss 0.48424, train_acc 0.8339, valid_acc 0.8660, Time 00:03:54,lr 0.1\nepoch 18, loss 0.47506, train_acc 0.8352, valid_acc 0.8596, Time 00:03:48,lr 0.1\nepoch 19, loss 0.46440, train_acc 0.8386, valid_acc 0.8656, Time 00:03:49,lr 0.1\nepoch 20, loss 0.45834, train_acc 0.8419, valid_acc 0.8670, Time 00:03:46,lr 0.1\nepoch 21, loss 0.44546, train_acc 0.8457, valid_acc 0.8678, Time 00:03:46,lr 0.1\nepoch 22, loss 0.43533, train_acc 0.8506, valid_acc 0.8643, Time 00:03:52,lr 0.1\nepoch 23, loss 0.43173, train_acc 0.8506, valid_acc 0.8676, Time 00:03:51,lr 0.1\nepoch 24, loss 0.43031, train_acc 0.8520, valid_acc 0.8678, Time 00:03:50,lr 0.1\nepoch 25, loss 0.42356, train_acc 0.8547, valid_acc 0.8629, Time 00:03:51,lr 0.1\nepoch 26, loss 0.41451, train_acc 0.8559, valid_acc 0.8797, Time 00:03:45,lr 0.1\nepoch 27, loss 0.40819, train_acc 0.8604, valid_acc 0.8691, Time 00:03:49,lr 0.1\nepoch 28, loss 0.41092, train_acc 0.8587, valid_acc 0.8758, Time 00:03:50,lr 0.1\nepoch 29, loss 0.40495, train_acc 0.8588, valid_acc 0.8885, Time 00:03:51,lr 0.1\nepoch 30, loss 0.40309, train_acc 0.8608, valid_acc 0.8789, Time 00:03:51,lr 0.1\nepoch 31, loss 0.39981, train_acc 0.8615, valid_acc 0.8789, Time 00:03:48,lr 0.1\nepoch 32, loss 0.39790, train_acc 0.8625, valid_acc 0.8875, Time 00:03:51,lr 0.1\nepoch 33, loss 0.39716, train_acc 0.8612, valid_acc 0.8910, Time 00:03:48,lr 0.1\nepoch 34, loss 0.38664, train_acc 0.8681, valid_acc 0.8748, Time 00:03:49,lr 0.1\nepoch 35, loss 0.38574, train_acc 0.8676, valid_acc 0.8721, Time 00:03:49,lr 0.1\nepoch 36, loss 0.38157, train_acc 0.8665, valid_acc 0.8869, Time 00:03:49,lr 0.1\nepoch 37, loss 0.37563, train_acc 0.8697, valid_acc 0.8891, Time 00:03:48,lr 0.1\nepoch 38, loss 0.38284, train_acc 0.8691, valid_acc 0.8926, Time 00:03:51,lr 0.1\nepoch 39, loss 0.37483, train_acc 0.8705, valid_acc 0.8947, Time 00:03:50,lr 0.1\nepoch 40, loss 0.37445, train_acc 0.8695, valid_acc 0.8895, Time 00:03:49,lr 0.1\nepoch 41, loss 0.36943, train_acc 0.8726, valid_acc 0.8857, Time 00:03:50,lr 0.1\nepoch 42, loss 0.36946, train_acc 0.8729, valid_acc 0.8775, Time 00:03:49,lr 0.1\nepoch 43, loss 0.36858, train_acc 0.8717, valid_acc 0.8943, Time 00:03:50,lr 0.1\nepoch 44, loss 0.36751, train_acc 0.8728, valid_acc 0.8934, Time 00:03:53,lr 0.1\nepoch 45, loss 0.36178, train_acc 0.8749, valid_acc 0.8902, Time 00:03:46,lr 0.1\nepoch 46, loss 0.35415, train_acc 0.8768, valid_acc 0.9000, Time 00:03:50,lr 0.1\nepoch 47, loss 0.36180, train_acc 0.8749, valid_acc 0.8855, Time 00:03:50,lr 0.1\nepoch 48, loss 0.36097, train_acc 0.8758, valid_acc 0.9000, Time 00:03:55,lr 0.1\nepoch 49, loss 0.36107, train_acc 0.8743, valid_acc 0.9037, Time 00:03:47,lr 0.1\nepoch 50, loss 0.36254, train_acc 0.8738, valid_acc 0.8918, Time 00:03:53,lr 0.1\nepoch 51, loss 0.35022, train_acc 0.8773, valid_acc 0.8955, Time 00:03:46,lr 0.1\nepoch 52, loss 0.35218, train_acc 0.8778, valid_acc 0.8930, Time 00:03:45,lr 0.1\nepoch 53, loss 0.35807, train_acc 0.8750, valid_acc 0.8951, Time 00:03:49,lr 0.1\nepoch 54, loss 0.35104, train_acc 0.8784, valid_acc 0.9059, Time 00:03:50,lr 0.1\nepoch 55, loss 0.35160, train_acc 0.8786, valid_acc 0.8992, Time 00:03:51,lr 0.1\nepoch 56, loss 0.34810, train_acc 0.8795, valid_acc 0.8971, Time 00:03:48,lr 0.1\nepoch 57, loss 0.34523, train_acc 0.8810, valid_acc 0.8988, Time 00:03:51,lr 0.1\nepoch 58, loss 0.34518, train_acc 0.8803, valid_acc 0.9029, Time 00:03:50,lr 0.1\nepoch 59, loss 0.33567, train_acc 0.8844, valid_acc 0.8859, Time 00:03:53,lr 0.1\nepoch 60, loss 0.34320, train_acc 0.8814, valid_acc 0.8830, Time 00:03:51,lr 0.1\nepoch 61, loss 0.34866, train_acc 0.8793, valid_acc 0.9027, Time 00:03:50,lr 0.1\nepoch 62, loss 0.34174, train_acc 0.8822, valid_acc 0.9000, Time 00:03:48,lr 0.1\nepoch 63, loss 0.33358, train_acc 0.8843, valid_acc 0.9039, Time 00:03:50,lr 0.1\nepoch 64, loss 0.33695, train_acc 0.8845, valid_acc 0.9020, Time 00:03:49,lr 0.1\nepoch 65, loss 0.33849, train_acc 0.8833, valid_acc 0.9055, Time 00:03:49,lr 0.1\nepoch 66, loss 0.33794, train_acc 0.8833, valid_acc 0.9035, Time 00:03:47,lr 0.1\nepoch 67, loss 0.33684, train_acc 0.8831, valid_acc 0.8961, Time 00:03:51,lr 0.1\nepoch 68, loss 0.33411, train_acc 0.8851, valid_acc 0.9027, Time 00:03:51,lr 0.1\nepoch 69, loss 0.33389, train_acc 0.8833, valid_acc 0.8912, Time 00:04:13,lr 0.1\nepoch 70, loss 0.33056, train_acc 0.8858, valid_acc 0.8799, Time 00:04:20,lr 0.1\nepoch 71, loss 0.33634, train_acc 0.8834, valid_acc 0.8852, Time 00:04:20,lr 0.1\nepoch 72, loss 0.32652, train_acc 0.8880, valid_acc 0.8887, Time 00:04:17,lr 0.1\nepoch 73, loss 0.32906, train_acc 0.8864, valid_acc 0.8959, Time 00:03:57,lr 0.1\nepoch 74, loss 0.33288, train_acc 0.8837, valid_acc 0.9045, Time 00:03:51,lr 0.1\nepoch 75, loss 0.32943, train_acc 0.8854, valid_acc 0.8965, Time 00:04:13,lr 0.1\nepoch 76, loss 0.32016, train_acc 0.8888, valid_acc 0.8939, Time 00:04:18,lr 0.1\n"
 }
]
```

### Exp3: res164_v2 + focal loss + DA3: 0.9540

```{.python .input  n=13}
batch_size = 128
tranform_train = transform_train_DA3
train_data, valid_data, train_valid_data, test_data, test_ds, train_valid_ds = data_loader(batch_size,tranform_train)
net = ResNet164_v2(10)
loss_f = FocalLoss()

num_epochs = 200
learning_rate = 0.1
weight_decay = 1e-4
lr_period = [150, 225]
lr_decay=0.1
log_file = None
ctx= mx.gpu(1)

net.collect_params().initialize(mx.init.Xavier(), ctx=ctx, force_reinit=True)
net.hybridize()
w_key = []
train(net, train_valid_data, None, num_epochs, learning_rate, 
      lr_period, lr_decay, weight_decay, ctx, w_key, log_file, False, loss_f)
net.save_parameters("models/res164__2_e255_focal_clip_all_data")
```

```{.json .output n=13}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "epoch 0, loss 1.50549, train_acc 0.2664, Time 00:04:21,lr 0.1\nmodels/shelock_densenet_orign1\nepoch 1, loss 1.05536, train_acc 0.4481, Time 00:04:17,lr 0.1\nepoch 2, loss 0.81992, train_acc 0.5514, Time 00:04:20,lr 0.1\nepoch 3, loss 0.70060, train_acc 0.6099, Time 00:04:09,lr 0.1\nepoch 4, loss 0.62588, train_acc 0.6471, Time 00:04:04,lr 0.1\nepoch 5, loss 0.57771, train_acc 0.6686, Time 00:04:03,lr 0.1\nepoch 6, loss 0.54043, train_acc 0.6907, Time 00:04:02,lr 0.1\nepoch 7, loss 0.50689, train_acc 0.7063, Time 00:03:59,lr 0.1\nepoch 8, loss 0.48360, train_acc 0.7167, Time 00:03:58,lr 0.1\nepoch 9, loss 0.46275, train_acc 0.7266, Time 00:03:59,lr 0.1\nepoch 10, loss 0.44417, train_acc 0.7360, Time 00:04:00,lr 0.1\nepoch 11, loss 0.42278, train_acc 0.7488, Time 00:04:00,lr 0.1\nepoch 12, loss 0.41528, train_acc 0.7521, Time 00:03:57,lr 0.1\nepoch 13, loss 0.40504, train_acc 0.7600, Time 00:03:58,lr 0.1\nepoch 14, loss 0.39116, train_acc 0.7642, Time 00:03:56,lr 0.1\nepoch 15, loss 0.38184, train_acc 0.7692, Time 00:03:57,lr 0.1\nepoch 16, loss 0.37578, train_acc 0.7728, Time 00:03:58,lr 0.1\nepoch 17, loss 0.36407, train_acc 0.7793, Time 00:03:58,lr 0.1\nepoch 18, loss 0.36198, train_acc 0.7801, Time 00:03:57,lr 0.1\nepoch 19, loss 0.35432, train_acc 0.7828, Time 00:03:58,lr 0.1\nepoch 20, loss 0.35065, train_acc 0.7895, Time 00:04:01,lr 0.1\nepoch 21, loss 0.34587, train_acc 0.7880, Time 00:04:02,lr 0.1\nepoch 22, loss 0.34055, train_acc 0.7951, Time 00:03:58,lr 0.1\nepoch 23, loss 0.33877, train_acc 0.7940, Time 00:04:00,lr 0.1\nepoch 24, loss 0.33372, train_acc 0.7968, Time 00:03:59,lr 0.1\nepoch 25, loss 0.32142, train_acc 0.8012, Time 00:03:57,lr 0.1\nepoch 26, loss 0.31780, train_acc 0.8042, Time 00:03:58,lr 0.1\nepoch 27, loss 0.31816, train_acc 0.8042, Time 00:03:56,lr 0.1\nepoch 28, loss 0.31538, train_acc 0.8070, Time 00:03:57,lr 0.1\nepoch 29, loss 0.31197, train_acc 0.8072, Time 00:04:00,lr 0.1\nepoch 30, loss 0.31165, train_acc 0.8074, Time 00:03:56,lr 0.1\nepoch 31, loss 0.31206, train_acc 0.8101, Time 00:03:58,lr 0.1\nepoch 32, loss 0.30607, train_acc 0.8116, Time 00:03:58,lr 0.1\nepoch 33, loss 0.30726, train_acc 0.8112, Time 00:03:56,lr 0.1\nepoch 34, loss 0.30504, train_acc 0.8100, Time 00:03:58,lr 0.1\nepoch 35, loss 0.30266, train_acc 0.8132, Time 00:03:56,lr 0.1\nepoch 36, loss 0.29770, train_acc 0.8167, Time 00:03:59,lr 0.1\nepoch 37, loss 0.30405, train_acc 0.8123, Time 00:04:04,lr 0.1\nepoch 38, loss 0.29383, train_acc 0.8181, Time 00:03:58,lr 0.1\nepoch 39, loss 0.29997, train_acc 0.8123, Time 00:03:58,lr 0.1\nepoch 40, loss 0.29237, train_acc 0.8164, Time 00:03:56,lr 0.1\nepoch 41, loss 0.29690, train_acc 0.8160, Time 00:03:59,lr 0.1\nepoch 42, loss 0.29573, train_acc 0.8152, Time 00:03:54,lr 0.1\nepoch 43, loss 0.29602, train_acc 0.8174, Time 00:03:57,lr 0.1\nepoch 44, loss 0.29591, train_acc 0.8142, Time 00:04:00,lr 0.1\nepoch 45, loss 0.28788, train_acc 0.8204, Time 00:03:58,lr 0.1\nepoch 46, loss 0.28550, train_acc 0.8225, Time 00:03:59,lr 0.1\nepoch 47, loss 0.29095, train_acc 0.8186, Time 00:03:59,lr 0.1\nepoch 48, loss 0.28992, train_acc 0.8205, Time 00:03:59,lr 0.1\nepoch 49, loss 0.28712, train_acc 0.8223, Time 00:03:58,lr 0.1\nepoch 50, loss 0.28749, train_acc 0.8223, Time 00:03:58,lr 0.1\nmodels/shelock_densenet_orign51\nepoch 51, loss 0.28808, train_acc 0.8227, Time 00:03:57,lr 0.1\nepoch 52, loss 0.28356, train_acc 0.8240, Time 00:03:57,lr 0.1\nepoch 53, loss 0.28286, train_acc 0.8238, Time 00:03:57,lr 0.1\nepoch 54, loss 0.27864, train_acc 0.8278, Time 00:03:58,lr 0.1\nepoch 55, loss 0.27927, train_acc 0.8265, Time 00:03:56,lr 0.1\nepoch 56, loss 0.27622, train_acc 0.8293, Time 00:04:02,lr 0.1\nepoch 57, loss 0.27674, train_acc 0.8291, Time 00:03:57,lr 0.1\nepoch 58, loss 0.27503, train_acc 0.8303, Time 00:03:56,lr 0.1\nepoch 59, loss 0.27208, train_acc 0.8296, Time 00:03:58,lr 0.1\nepoch 60, loss 0.27845, train_acc 0.8288, Time 00:04:00,lr 0.1\nepoch 61, loss 0.27438, train_acc 0.8278, Time 00:03:58,lr 0.1\nepoch 62, loss 0.27048, train_acc 0.8328, Time 00:03:58,lr 0.1\nepoch 63, loss 0.27313, train_acc 0.8304, Time 00:03:59,lr 0.1\nepoch 64, loss 0.27241, train_acc 0.8314, Time 00:03:57,lr 0.1\nepoch 65, loss 0.26768, train_acc 0.8333, Time 00:03:59,lr 0.1\nepoch 66, loss 0.27006, train_acc 0.8328, Time 00:03:58,lr 0.1\nepoch 67, loss 0.27079, train_acc 0.8318, Time 00:03:59,lr 0.1\nepoch 68, loss 0.26759, train_acc 0.8311, Time 00:04:00,lr 0.1\nepoch 69, loss 0.26252, train_acc 0.8342, Time 00:03:58,lr 0.1\nepoch 70, loss 0.26640, train_acc 0.8339, Time 00:03:58,lr 0.1\nepoch 71, loss 0.26803, train_acc 0.8329, Time 00:03:58,lr 0.1\nepoch 72, loss 0.26560, train_acc 0.8351, Time 00:04:48,lr 0.1\nepoch 73, loss 0.26515, train_acc 0.8331, Time 00:04:35,lr 0.1\nepoch 74, loss 0.26540, train_acc 0.8327, Time 00:03:57,lr 0.1\nepoch 75, loss 0.27457, train_acc 0.8293, Time 00:03:59,lr 0.1\nepoch 76, loss 0.26235, train_acc 0.8351, Time 00:04:16,lr 0.1\nepoch 77, loss 0.26237, train_acc 0.8375, Time 00:04:02,lr 0.1\nepoch 78, loss 0.26134, train_acc 0.8372, Time 00:03:56,lr 0.1\nepoch 79, loss 0.25719, train_acc 0.8397, Time 00:03:56,lr 0.1\nepoch 80, loss 0.25957, train_acc 0.8356, Time 00:03:57,lr 0.1\nepoch 81, loss 0.25857, train_acc 0.8382, Time 00:03:57,lr 0.1\nepoch 82, loss 0.25849, train_acc 0.8373, Time 00:03:56,lr 0.1\nepoch 83, loss 0.26300, train_acc 0.8349, Time 00:03:56,lr 0.1\nepoch 84, loss 0.25777, train_acc 0.8395, Time 00:03:57,lr 0.1\nepoch 85, loss 0.25336, train_acc 0.8427, Time 00:03:58,lr 0.1\nepoch 86, loss 0.25736, train_acc 0.8388, Time 00:03:56,lr 0.1\nepoch 87, loss 0.25568, train_acc 0.8393, Time 00:03:57,lr 0.1\nepoch 88, loss 0.25646, train_acc 0.8397, Time 00:03:56,lr 0.1\nepoch 89, loss 0.26010, train_acc 0.8359, Time 00:03:58,lr 0.1\nepoch 90, loss 0.25929, train_acc 0.8368, Time 00:03:56,lr 0.1\nepoch 91, loss 0.25197, train_acc 0.8428, Time 00:03:59,lr 0.1\nepoch 92, loss 0.25586, train_acc 0.8402, Time 00:03:58,lr 0.1\nepoch 93, loss 0.24830, train_acc 0.8442, Time 00:03:56,lr 0.1\nepoch 94, loss 0.25309, train_acc 0.8402, Time 00:03:56,lr 0.1\nepoch 95, loss 0.25162, train_acc 0.8390, Time 00:03:57,lr 0.1\nepoch 96, loss 0.25006, train_acc 0.8425, Time 00:03:58,lr 0.1\nepoch 97, loss 0.24804, train_acc 0.8458, Time 00:03:57,lr 0.1\nepoch 98, loss 0.24626, train_acc 0.8446, Time 00:03:59,lr 0.1\nepoch 99, loss 0.25464, train_acc 0.8406, Time 00:03:57,lr 0.1\nepoch 100, loss 0.24001, train_acc 0.8473, Time 00:03:58,lr 0.1\nmodels/shelock_densenet_orign101\nepoch 101, loss 0.24975, train_acc 0.8431, Time 00:03:59,lr 0.1\nepoch 102, loss 0.24502, train_acc 0.8445, Time 00:04:00,lr 0.1\nepoch 103, loss 0.25431, train_acc 0.8411, Time 00:04:01,lr 0.1\nepoch 104, loss 0.25022, train_acc 0.8430, Time 00:03:58,lr 0.1\nepoch 105, loss 0.24318, train_acc 0.8479, Time 00:03:58,lr 0.1\nepoch 106, loss 0.24848, train_acc 0.8459, Time 00:03:59,lr 0.1\nepoch 107, loss 0.24125, train_acc 0.8479, Time 00:04:00,lr 0.1\nepoch 108, loss 0.24429, train_acc 0.8466, Time 00:03:58,lr 0.1\nepoch 109, loss 0.24719, train_acc 0.8449, Time 00:04:00,lr 0.1\nepoch 110, loss 0.24518, train_acc 0.8460, Time 00:04:00,lr 0.1\nepoch 111, loss 0.23983, train_acc 0.8495, Time 00:04:02,lr 0.1\nepoch 112, loss 0.24413, train_acc 0.8475, Time 00:03:59,lr 0.1\nepoch 113, loss 0.24279, train_acc 0.8452, Time 00:04:00,lr 0.1\nepoch 114, loss 0.24142, train_acc 0.8471, Time 00:03:59,lr 0.1\nepoch 115, loss 0.24431, train_acc 0.8461, Time 00:03:59,lr 0.1\nepoch 116, loss 0.23769, train_acc 0.8494, Time 00:04:01,lr 0.1\nepoch 117, loss 0.23802, train_acc 0.8497, Time 00:03:58,lr 0.1\nepoch 118, loss 0.23374, train_acc 0.8518, Time 00:03:58,lr 0.1\nepoch 119, loss 0.23965, train_acc 0.8483, Time 00:03:58,lr 0.1\nepoch 120, loss 0.23849, train_acc 0.8489, Time 00:03:58,lr 0.1\nepoch 121, loss 0.23931, train_acc 0.8497, Time 00:03:59,lr 0.1\nepoch 122, loss 0.24194, train_acc 0.8501, Time 00:03:58,lr 0.1\nepoch 123, loss 0.23839, train_acc 0.8491, Time 00:03:55,lr 0.1\nepoch 124, loss 0.23429, train_acc 0.8527, Time 00:03:59,lr 0.1\nepoch 125, loss 0.23997, train_acc 0.8478, Time 00:03:57,lr 0.1\nepoch 126, loss 0.23747, train_acc 0.8508, Time 00:03:59,lr 0.1\nepoch 127, loss 0.23445, train_acc 0.8530, Time 00:03:59,lr 0.1\nepoch 128, loss 0.23246, train_acc 0.8526, Time 00:03:57,lr 0.1\nepoch 129, loss 0.23825, train_acc 0.8504, Time 00:03:58,lr 0.1\nepoch 130, loss 0.23432, train_acc 0.8524, Time 00:03:58,lr 0.1\nepoch 131, loss 0.23903, train_acc 0.8482, Time 00:03:58,lr 0.1\nepoch 132, loss 0.23360, train_acc 0.8514, Time 00:04:00,lr 0.1\nepoch 133, loss 0.23696, train_acc 0.8514, Time 00:03:58,lr 0.1\nepoch 134, loss 0.23234, train_acc 0.8531, Time 00:03:59,lr 0.1\nepoch 135, loss 0.23745, train_acc 0.8521, Time 00:03:59,lr 0.1\nepoch 136, loss 0.23732, train_acc 0.8495, Time 00:03:59,lr 0.1\nepoch 137, loss 0.23825, train_acc 0.8489, Time 00:03:59,lr 0.1\nepoch 138, loss 0.22974, train_acc 0.8552, Time 00:04:00,lr 0.1\nepoch 139, loss 0.23787, train_acc 0.8505, Time 00:04:00,lr 0.1\nepoch 140, loss 0.23055, train_acc 0.8530, Time 00:03:59,lr 0.1\nepoch 141, loss 0.23041, train_acc 0.8549, Time 00:03:59,lr 0.1\nepoch 142, loss 0.23694, train_acc 0.8505, Time 00:04:00,lr 0.1\nepoch 143, loss 0.23215, train_acc 0.8521, Time 00:04:00,lr 0.1\nepoch 144, loss 0.23219, train_acc 0.8540, Time 00:04:00,lr 0.1\nepoch 145, loss 0.23538, train_acc 0.8521, Time 00:04:00,lr 0.1\nepoch 146, loss 0.22995, train_acc 0.8528, Time 00:03:58,lr 0.1\nepoch 147, loss 0.23267, train_acc 0.8525, Time 00:04:00,lr 0.1\nepoch 148, loss 0.22749, train_acc 0.8565, Time 00:03:59,lr 0.1\nepoch 149, loss 0.23205, train_acc 0.8525, Time 00:03:59,lr 0.1\nepoch 150, loss 0.17252, train_acc 0.8903, Time 00:03:57,lr 0.010000000000000002\nmodels/shelock_densenet_orign151\nepoch 151, loss 0.13495, train_acc 0.9108, Time 00:04:00,lr 0.010000000000000002\nepoch 152, loss 0.12564, train_acc 0.9158, Time 00:04:00,lr 0.010000000000000002\nepoch 153, loss 0.12410, train_acc 0.9183, Time 00:03:59,lr 0.010000000000000002\nepoch 154, loss 0.11648, train_acc 0.9213, Time 00:04:02,lr 0.010000000000000002\nepoch 155, loss 0.11121, train_acc 0.9260, Time 00:04:00,lr 0.010000000000000002\nepoch 156, loss 0.10886, train_acc 0.9266, Time 00:03:57,lr 0.010000000000000002\nepoch 157, loss 0.11100, train_acc 0.9261, Time 00:04:00,lr 0.010000000000000002\nepoch 158, loss 0.10630, train_acc 0.9288, Time 00:03:59,lr 0.010000000000000002\nepoch 159, loss 0.10605, train_acc 0.9295, Time 00:03:57,lr 0.010000000000000002\nepoch 160, loss 0.10185, train_acc 0.9318, Time 00:03:58,lr 0.010000000000000002\nepoch 161, loss 0.10371, train_acc 0.9307, Time 00:04:00,lr 0.010000000000000002\nepoch 162, loss 0.10266, train_acc 0.9319, Time 00:04:00,lr 0.010000000000000002\nepoch 163, loss 0.10166, train_acc 0.9318, Time 00:04:00,lr 0.010000000000000002\nepoch 164, loss 0.10044, train_acc 0.9331, Time 00:03:59,lr 0.010000000000000002\nepoch 165, loss 0.09711, train_acc 0.9344, Time 00:03:59,lr 0.010000000000000002\nepoch 166, loss 0.09819, train_acc 0.9344, Time 00:03:58,lr 0.010000000000000002\nepoch 167, loss 0.09324, train_acc 0.9377, Time 00:03:59,lr 0.010000000000000002\nepoch 168, loss 0.09548, train_acc 0.9355, Time 00:03:58,lr 0.010000000000000002\nepoch 169, loss 0.09285, train_acc 0.9377, Time 00:03:58,lr 0.010000000000000002\nepoch 170, loss 0.09269, train_acc 0.9371, Time 00:03:58,lr 0.010000000000000002\nepoch 171, loss 0.09137, train_acc 0.9395, Time 00:03:59,lr 0.010000000000000002\nepoch 172, loss 0.09480, train_acc 0.9374, Time 00:03:57,lr 0.010000000000000002\nepoch 173, loss 0.09042, train_acc 0.9394, Time 00:03:56,lr 0.010000000000000002\nepoch 174, loss 0.09056, train_acc 0.9394, Time 00:03:59,lr 0.010000000000000002\nepoch 175, loss 0.08906, train_acc 0.9412, Time 00:03:59,lr 0.010000000000000002\nepoch 176, loss 0.09352, train_acc 0.9384, Time 00:03:57,lr 0.010000000000000002\nepoch 177, loss 0.08986, train_acc 0.9403, Time 00:03:58,lr 0.010000000000000002\nepoch 178, loss 0.08689, train_acc 0.9423, Time 00:03:59,lr 0.010000000000000002\nepoch 179, loss 0.08711, train_acc 0.9409, Time 00:04:01,lr 0.010000000000000002\nepoch 180, loss 0.08903, train_acc 0.9410, Time 00:04:01,lr 0.010000000000000002\nepoch 181, loss 0.08956, train_acc 0.9410, Time 00:04:00,lr 0.010000000000000002\nepoch 182, loss 0.08589, train_acc 0.9433, Time 00:03:59,lr 0.010000000000000002\nepoch 183, loss 0.08777, train_acc 0.9420, Time 00:04:00,lr 0.010000000000000002\nepoch 184, loss 0.08550, train_acc 0.9437, Time 00:04:00,lr 0.010000000000000002\nepoch 185, loss 0.08663, train_acc 0.9436, Time 00:04:00,lr 0.010000000000000002\nepoch 186, loss 0.08674, train_acc 0.9423, Time 00:03:58,lr 0.010000000000000002\nepoch 187, loss 0.08430, train_acc 0.9446, Time 00:04:00,lr 0.010000000000000002\nepoch 188, loss 0.08407, train_acc 0.9444, Time 00:03:58,lr 0.010000000000000002\nepoch 189, loss 0.08106, train_acc 0.9457, Time 00:03:58,lr 0.010000000000000002\nepoch 190, loss 0.08351, train_acc 0.9441, Time 00:04:01,lr 0.010000000000000002\nepoch 191, loss 0.08417, train_acc 0.9442, Time 00:04:00,lr 0.010000000000000002\nepoch 192, loss 0.08336, train_acc 0.9451, Time 00:03:59,lr 0.010000000000000002\nepoch 193, loss 0.08094, train_acc 0.9470, Time 00:04:03,lr 0.010000000000000002\nepoch 194, loss 0.08097, train_acc 0.9460, Time 00:04:00,lr 0.010000000000000002\nepoch 195, loss 0.08231, train_acc 0.9452, Time 00:04:00,lr 0.010000000000000002\nepoch 196, loss 0.08249, train_acc 0.9440, Time 00:03:59,lr 0.010000000000000002\nepoch 197, loss 0.08304, train_acc 0.9454, Time 00:03:59,lr 0.010000000000000002\nepoch 198, loss 0.08094, train_acc 0.9462, Time 00:03:58,lr 0.010000000000000002\nepoch 199, loss 0.08366, train_acc 0.9459, Time 00:03:59,lr 0.010000000000000002\n"
 },
 {
  "name": "stderr",
  "output_type": "stream",
  "text": "/home/zp/.conda/envs/gluon/lib/python3.6/site-packages/mxnet/gluon/block.py:338: UserWarning: save_params is deprecated. Please use save_parameters. Note that if you want to load from SymbolBlock later, please use export instead. For details, see https://mxnet.incubator.apache.org/tutorials/gluon/save_load_params.html\n  warnings.warn(\"save_params is deprecated. Please use save_parameters. \"\n"
 }
]
```

```{.python .input  n=14}
net.save_parameters("models/res164__2_e255_focal_clip_all_data")
```

### Exp4: res164_v2 + focal loss + DA3 + only train_data: 0.9506

```{.python .input  n=6}
batch_size = 128
transform_train = transform_train_DA3
train_data, valid_data, train_valid_data, test_data, test_ds, train_valid_ds = data_loader(batch_size, transform_train)
net = ResNet164_v2(10)
loss_f = FocalLoss()

num_epochs = 255
learning_rate = 0.1
weight_decay = 1e-4
lr_period = [100, 225]
lr_decay=0.1
log_file = None

net.collect_params().initialize(mx.init.Xavier(), ctx=ctx, force_reinit=True)
net.hybridize()
w_key = []
train(net, train_data, valid_data, num_epochs, learning_rate, 
      lr_period, lr_decay, weight_decay, ctx, w_key, log_file, False, loss_f)
net.save_parameters("models/resnet164_e0-255_focal_clip")
```

```{.json .output n=6}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "epoch 0, loss 1.52169, train_acc 0.2560, valid_acc 0.3824, Time 00:03:45,lr 0.1\nepoch 1, loss 1.09974, train_acc 0.4345, valid_acc 0.5721, Time 00:03:54,lr 0.1\nepoch 2, loss 0.85517, train_acc 0.5399, valid_acc 0.6461, Time 00:04:27,lr 0.1\nepoch 3, loss 0.72859, train_acc 0.5964, valid_acc 0.6318, Time 00:04:28,lr 0.1\nepoch 4, loss 0.65308, train_acc 0.6330, valid_acc 0.7100, Time 00:04:29,lr 0.1\nepoch 5, loss 0.60325, train_acc 0.6554, valid_acc 0.7367, Time 00:04:29,lr 0.1\nepoch 6, loss 0.55962, train_acc 0.6814, valid_acc 0.7232, Time 00:04:27,lr 0.1\nepoch 7, loss 0.52926, train_acc 0.6959, valid_acc 0.7486, Time 00:04:29,lr 0.1\nepoch 8, loss 0.50037, train_acc 0.7092, valid_acc 0.7168, Time 00:04:29,lr 0.1\nepoch 9, loss 0.48178, train_acc 0.7169, valid_acc 0.7963, Time 00:04:28,lr 0.1\nepoch 10, loss 0.46255, train_acc 0.7272, valid_acc 0.7367, Time 00:04:29,lr 0.1\nepoch 11, loss 0.43893, train_acc 0.7421, valid_acc 0.7740, Time 00:04:28,lr 0.1\nepoch 12, loss 0.42713, train_acc 0.7468, valid_acc 0.7799, Time 00:04:29,lr 0.1\nepoch 13, loss 0.42398, train_acc 0.7461, valid_acc 0.8086, Time 00:04:29,lr 0.1\nepoch 14, loss 0.40835, train_acc 0.7573, valid_acc 0.8219, Time 00:04:28,lr 0.1\nepoch 15, loss 0.40280, train_acc 0.7574, valid_acc 0.7947, Time 00:04:29,lr 0.1\nepoch 16, loss 0.38705, train_acc 0.7666, valid_acc 0.8408, Time 00:04:29,lr 0.1\nepoch 17, loss 0.38231, train_acc 0.7694, valid_acc 0.8234, Time 00:04:30,lr 0.1\nepoch 18, loss 0.37120, train_acc 0.7764, valid_acc 0.8410, Time 00:04:29,lr 0.1\nepoch 19, loss 0.36736, train_acc 0.7770, valid_acc 0.8404, Time 00:04:28,lr 0.1\nepoch 20, loss 0.36580, train_acc 0.7783, valid_acc 0.8576, Time 00:04:29,lr 0.1\nepoch 21, loss 0.35222, train_acc 0.7853, valid_acc 0.8582, Time 00:04:30,lr 0.1\nepoch 22, loss 0.34803, train_acc 0.7902, valid_acc 0.8494, Time 00:04:29,lr 0.1\nepoch 23, loss 0.34644, train_acc 0.7883, valid_acc 0.8453, Time 00:04:26,lr 0.1\nepoch 24, loss 0.34584, train_acc 0.7901, valid_acc 0.8385, Time 00:04:27,lr 0.1\nepoch 25, loss 0.33112, train_acc 0.7966, valid_acc 0.8732, Time 00:04:27,lr 0.1\nepoch 26, loss 0.33616, train_acc 0.7934, valid_acc 0.8365, Time 00:04:26,lr 0.1\nepoch 27, loss 0.32923, train_acc 0.7967, valid_acc 0.8609, Time 00:04:25,lr 0.1\nepoch 28, loss 0.32749, train_acc 0.8007, valid_acc 0.8654, Time 00:04:38,lr 0.1\nepoch 29, loss 0.32115, train_acc 0.8029, valid_acc 0.8559, Time 00:10:08,lr 0.1\nepoch 30, loss 0.31533, train_acc 0.8077, valid_acc 0.8639, Time 00:05:48,lr 0.1\nepoch 31, loss 0.31503, train_acc 0.8051, valid_acc 0.8680, Time 00:04:25,lr 0.1\nepoch 32, loss 0.31591, train_acc 0.8040, valid_acc 0.8561, Time 00:04:25,lr 0.1\nepoch 33, loss 0.31608, train_acc 0.8045, valid_acc 0.8422, Time 00:04:25,lr 0.1\nepoch 34, loss 0.31017, train_acc 0.8063, valid_acc 0.8531, Time 00:04:26,lr 0.1\nepoch 35, loss 0.30713, train_acc 0.8114, valid_acc 0.8318, Time 00:04:25,lr 0.1\nepoch 36, loss 0.30959, train_acc 0.8080, valid_acc 0.8602, Time 00:04:25,lr 0.1\nepoch 37, loss 0.30886, train_acc 0.8102, valid_acc 0.8693, Time 00:04:26,lr 0.1\nepoch 38, loss 0.30393, train_acc 0.8143, valid_acc 0.8750, Time 00:04:24,lr 0.1\nepoch 39, loss 0.30832, train_acc 0.8101, valid_acc 0.8668, Time 00:04:26,lr 0.1\nepoch 40, loss 0.29253, train_acc 0.8175, valid_acc 0.8814, Time 00:04:25,lr 0.1\nepoch 41, loss 0.29734, train_acc 0.8136, valid_acc 0.8699, Time 00:04:25,lr 0.1\nepoch 42, loss 0.29611, train_acc 0.8179, valid_acc 0.8809, Time 00:04:25,lr 0.1\nepoch 43, loss 0.29685, train_acc 0.8167, valid_acc 0.8834, Time 00:04:25,lr 0.1\nepoch 44, loss 0.29592, train_acc 0.8162, valid_acc 0.8764, Time 00:04:25,lr 0.1\nepoch 45, loss 0.29076, train_acc 0.8189, valid_acc 0.8881, Time 00:04:24,lr 0.1\nepoch 46, loss 0.29008, train_acc 0.8184, valid_acc 0.8699, Time 00:04:25,lr 0.1\nepoch 47, loss 0.28987, train_acc 0.8183, valid_acc 0.8703, Time 00:04:26,lr 0.1\nepoch 48, loss 0.28778, train_acc 0.8222, valid_acc 0.8891, Time 00:04:25,lr 0.1\nepoch 49, loss 0.28752, train_acc 0.8228, valid_acc 0.8881, Time 00:04:25,lr 0.1\nepoch 50, loss 0.28483, train_acc 0.8211, valid_acc 0.8795, Time 00:04:25,lr 0.1\nepoch 51, loss 0.28946, train_acc 0.8220, valid_acc 0.8654, Time 00:04:25,lr 0.1\nepoch 52, loss 0.28810, train_acc 0.8215, valid_acc 0.8713, Time 00:04:26,lr 0.1\nepoch 53, loss 0.28324, train_acc 0.8244, valid_acc 0.8789, Time 00:04:25,lr 0.1\nepoch 54, loss 0.28662, train_acc 0.8230, valid_acc 0.8857, Time 00:04:26,lr 0.1\nepoch 55, loss 0.28004, train_acc 0.8239, valid_acc 0.8826, Time 00:04:25,lr 0.1\nepoch 56, loss 0.28355, train_acc 0.8244, valid_acc 0.8797, Time 00:04:24,lr 0.1\nepoch 57, loss 0.28042, train_acc 0.8238, valid_acc 0.8559, Time 00:04:26,lr 0.1\nepoch 58, loss 0.28146, train_acc 0.8253, valid_acc 0.8814, Time 00:04:25,lr 0.1\nepoch 59, loss 0.27837, train_acc 0.8279, valid_acc 0.8775, Time 00:04:26,lr 0.1\nepoch 60, loss 0.27466, train_acc 0.8279, valid_acc 0.8611, Time 00:04:26,lr 0.1\nepoch 61, loss 0.27299, train_acc 0.8308, valid_acc 0.8861, Time 00:04:25,lr 0.1\nepoch 62, loss 0.27889, train_acc 0.8271, valid_acc 0.8727, Time 00:04:25,lr 0.1\nepoch 63, loss 0.27759, train_acc 0.8297, valid_acc 0.8830, Time 00:04:25,lr 0.1\nepoch 64, loss 0.27852, train_acc 0.8261, valid_acc 0.8930, Time 00:04:25,lr 0.1\nepoch 65, loss 0.27285, train_acc 0.8282, valid_acc 0.8910, Time 00:04:26,lr 0.1\nepoch 66, loss 0.27370, train_acc 0.8286, valid_acc 0.8709, Time 00:04:25,lr 0.1\nepoch 67, loss 0.27587, train_acc 0.8259, valid_acc 0.8916, Time 00:04:26,lr 0.1\nepoch 68, loss 0.27126, train_acc 0.8283, valid_acc 0.8822, Time 00:04:25,lr 0.1\nepoch 69, loss 0.27500, train_acc 0.8282, valid_acc 0.8924, Time 00:04:24,lr 0.1\nepoch 70, loss 0.27410, train_acc 0.8290, valid_acc 0.8771, Time 00:04:26,lr 0.1\nepoch 71, loss 0.27215, train_acc 0.8320, valid_acc 0.8998, Time 00:04:26,lr 0.1\nepoch 72, loss 0.26916, train_acc 0.8318, valid_acc 0.8771, Time 00:04:25,lr 0.1\nepoch 73, loss 0.27103, train_acc 0.8324, valid_acc 0.8658, Time 00:04:25,lr 0.1\nepoch 74, loss 0.26722, train_acc 0.8330, valid_acc 0.8701, Time 00:04:25,lr 0.1\nepoch 75, loss 0.26963, train_acc 0.8345, valid_acc 0.8678, Time 00:04:26,lr 0.1\nepoch 76, loss 0.26497, train_acc 0.8339, valid_acc 0.8906, Time 00:04:09,lr 0.1\nepoch 77, loss 0.26161, train_acc 0.8349, valid_acc 0.8908, Time 00:03:50,lr 0.1\nepoch 78, loss 0.26612, train_acc 0.8340, valid_acc 0.8861, Time 00:03:46,lr 0.1\nepoch 79, loss 0.26208, train_acc 0.8350, valid_acc 0.8895, Time 00:03:45,lr 0.1\nepoch 80, loss 0.25990, train_acc 0.8363, valid_acc 0.8883, Time 00:03:46,lr 0.1\nepoch 81, loss 0.26103, train_acc 0.8353, valid_acc 0.8912, Time 00:03:46,lr 0.1\nepoch 82, loss 0.26309, train_acc 0.8348, valid_acc 0.8562, Time 00:03:45,lr 0.1\nepoch 83, loss 0.25896, train_acc 0.8380, valid_acc 0.8828, Time 00:03:45,lr 0.1\nepoch 84, loss 0.26370, train_acc 0.8352, valid_acc 0.8945, Time 00:04:09,lr 0.1\nepoch 85, loss 0.26128, train_acc 0.8370, valid_acc 0.8838, Time 00:04:24,lr 0.1\nepoch 86, loss 0.25732, train_acc 0.8388, valid_acc 0.8752, Time 00:04:25,lr 0.1\nepoch 87, loss 0.25629, train_acc 0.8394, valid_acc 0.8861, Time 00:04:25,lr 0.1\nepoch 88, loss 0.25832, train_acc 0.8375, valid_acc 0.8904, Time 00:04:25,lr 0.1\nepoch 89, loss 0.25597, train_acc 0.8391, valid_acc 0.8914, Time 00:03:54,lr 0.1\nepoch 90, loss 0.25237, train_acc 0.8422, valid_acc 0.8936, Time 00:03:48,lr 0.1\nepoch 91, loss 0.25665, train_acc 0.8397, valid_acc 0.8984, Time 00:03:47,lr 0.1\nepoch 92, loss 0.25616, train_acc 0.8381, valid_acc 0.8771, Time 00:03:51,lr 0.1\nepoch 93, loss 0.25509, train_acc 0.8385, valid_acc 0.8932, Time 00:03:47,lr 0.1\nepoch 94, loss 0.25224, train_acc 0.8424, valid_acc 0.8906, Time 00:03:49,lr 0.1\nepoch 95, loss 0.25737, train_acc 0.8399, valid_acc 0.8945, Time 00:03:47,lr 0.1\nepoch 96, loss 0.25664, train_acc 0.8395, valid_acc 0.8863, Time 00:03:47,lr 0.1\nepoch 97, loss 0.25029, train_acc 0.8422, valid_acc 0.8961, Time 00:03:44,lr 0.1\nepoch 98, loss 0.24687, train_acc 0.8455, valid_acc 0.8939, Time 00:03:46,lr 0.1\nepoch 99, loss 0.25163, train_acc 0.8409, valid_acc 0.8922, Time 00:03:46,lr 0.1\nepoch 100, loss 0.19390, train_acc 0.8750, valid_acc 0.9297, Time 00:03:45,lr 0.010000000000000002\nepoch 101, loss 0.15161, train_acc 0.9003, valid_acc 0.9354, Time 00:03:48,lr 0.010000000000000002\nepoch 102, loss 0.14152, train_acc 0.9071, valid_acc 0.9389, Time 00:03:42,lr 0.010000000000000002\nepoch 103, loss 0.13084, train_acc 0.9116, valid_acc 0.9359, Time 00:03:46,lr 0.010000000000000002\nepoch 104, loss 0.13418, train_acc 0.9125, valid_acc 0.9389, Time 00:03:47,lr 0.010000000000000002\nepoch 105, loss 0.12306, train_acc 0.9175, valid_acc 0.9441, Time 00:03:47,lr 0.010000000000000002\nepoch 106, loss 0.12529, train_acc 0.9180, valid_acc 0.9432, Time 00:03:44,lr 0.010000000000000002\nepoch 107, loss 0.12250, train_acc 0.9191, valid_acc 0.9447, Time 00:03:45,lr 0.010000000000000002\nepoch 108, loss 0.11934, train_acc 0.9202, valid_acc 0.9420, Time 00:03:46,lr 0.010000000000000002\nepoch 109, loss 0.11951, train_acc 0.9199, valid_acc 0.9445, Time 00:03:46,lr 0.010000000000000002\nepoch 110, loss 0.11364, train_acc 0.9223, valid_acc 0.9436, Time 00:03:43,lr 0.010000000000000002\nepoch 111, loss 0.11092, train_acc 0.9254, valid_acc 0.9479, Time 00:03:48,lr 0.010000000000000002\nepoch 112, loss 0.11171, train_acc 0.9254, valid_acc 0.9449, Time 00:03:44,lr 0.010000000000000002\nepoch 113, loss 0.11050, train_acc 0.9266, valid_acc 0.9463, Time 00:03:43,lr 0.010000000000000002\nepoch 114, loss 0.11017, train_acc 0.9272, valid_acc 0.9455, Time 00:03:46,lr 0.010000000000000002\nepoch 115, loss 0.10683, train_acc 0.9278, valid_acc 0.9430, Time 00:03:46,lr 0.010000000000000002\nepoch 116, loss 0.10851, train_acc 0.9262, valid_acc 0.9467, Time 00:03:45,lr 0.010000000000000002\nepoch 117, loss 0.10537, train_acc 0.9292, valid_acc 0.9461, Time 00:03:45,lr 0.010000000000000002\nepoch 118, loss 0.10547, train_acc 0.9299, valid_acc 0.9469, Time 00:03:45,lr 0.010000000000000002\nepoch 119, loss 0.10458, train_acc 0.9303, valid_acc 0.9467, Time 00:03:46,lr 0.010000000000000002\nepoch 120, loss 0.10503, train_acc 0.9295, valid_acc 0.9379, Time 00:03:44,lr 0.010000000000000002\nepoch 121, loss 0.10269, train_acc 0.9313, valid_acc 0.9418, Time 00:03:43,lr 0.010000000000000002\nepoch 122, loss 0.10280, train_acc 0.9308, valid_acc 0.9467, Time 00:03:44,lr 0.010000000000000002\nepoch 123, loss 0.10060, train_acc 0.9308, valid_acc 0.9441, Time 00:03:46,lr 0.010000000000000002\nepoch 124, loss 0.10076, train_acc 0.9332, valid_acc 0.9432, Time 00:03:46,lr 0.010000000000000002\nepoch 125, loss 0.10247, train_acc 0.9324, valid_acc 0.9459, Time 00:03:46,lr 0.010000000000000002\nepoch 126, loss 0.09905, train_acc 0.9343, valid_acc 0.9459, Time 00:03:45,lr 0.010000000000000002\nepoch 127, loss 0.09643, train_acc 0.9354, valid_acc 0.9455, Time 00:03:46,lr 0.010000000000000002\nepoch 128, loss 0.10035, train_acc 0.9333, valid_acc 0.9512, Time 00:03:43,lr 0.010000000000000002\nepoch 129, loss 0.09577, train_acc 0.9345, valid_acc 0.9496, Time 00:03:44,lr 0.010000000000000002\nepoch 130, loss 0.09859, train_acc 0.9341, valid_acc 0.9459, Time 00:03:45,lr 0.010000000000000002\nepoch 131, loss 0.09767, train_acc 0.9344, valid_acc 0.9475, Time 00:03:46,lr 0.010000000000000002\nepoch 132, loss 0.09572, train_acc 0.9356, valid_acc 0.9437, Time 00:03:44,lr 0.010000000000000002\nepoch 133, loss 0.09414, train_acc 0.9368, valid_acc 0.9518, Time 00:03:46,lr 0.010000000000000002\nepoch 134, loss 0.09356, train_acc 0.9387, valid_acc 0.9498, Time 00:03:42,lr 0.010000000000000002\nepoch 135, loss 0.09298, train_acc 0.9367, valid_acc 0.9449, Time 00:03:44,lr 0.010000000000000002\nepoch 136, loss 0.09406, train_acc 0.9373, valid_acc 0.9439, Time 00:03:42,lr 0.010000000000000002\nepoch 137, loss 0.09354, train_acc 0.9378, valid_acc 0.9463, Time 00:03:44,lr 0.010000000000000002\nepoch 138, loss 0.09358, train_acc 0.9383, valid_acc 0.9494, Time 00:03:42,lr 0.010000000000000002\nepoch 139, loss 0.09200, train_acc 0.9378, valid_acc 0.9514, Time 00:03:46,lr 0.010000000000000002\nepoch 140, loss 0.09426, train_acc 0.9377, valid_acc 0.9498, Time 00:03:43,lr 0.010000000000000002\nepoch 141, loss 0.09058, train_acc 0.9390, valid_acc 0.9494, Time 00:03:43,lr 0.010000000000000002\nepoch 142, loss 0.08857, train_acc 0.9414, valid_acc 0.9492, Time 00:03:46,lr 0.010000000000000002\nepoch 143, loss 0.08851, train_acc 0.9407, valid_acc 0.9471, Time 00:03:44,lr 0.010000000000000002\nepoch 144, loss 0.09223, train_acc 0.9394, valid_acc 0.9523, Time 00:03:43,lr 0.010000000000000002\nepoch 145, loss 0.09489, train_acc 0.9373, valid_acc 0.9461, Time 00:03:44,lr 0.010000000000000002\nepoch 146, loss 0.08688, train_acc 0.9413, valid_acc 0.9488, Time 00:03:43,lr 0.010000000000000002\nepoch 147, loss 0.09089, train_acc 0.9405, valid_acc 0.9467, Time 00:03:42,lr 0.010000000000000002\nepoch 148, loss 0.08784, train_acc 0.9422, valid_acc 0.9488, Time 00:03:42,lr 0.010000000000000002\nepoch 149, loss 0.09030, train_acc 0.9412, valid_acc 0.9506, Time 00:03:42,lr 0.010000000000000002\nepoch 150, loss 0.08687, train_acc 0.9415, valid_acc 0.9492, Time 00:03:44,lr 0.010000000000000002\nepoch 151, loss 0.09051, train_acc 0.9392, valid_acc 0.9428, Time 00:03:46,lr 0.010000000000000002\nepoch 152, loss 0.08850, train_acc 0.9426, valid_acc 0.9449, Time 00:03:47,lr 0.010000000000000002\nepoch 153, loss 0.08774, train_acc 0.9419, valid_acc 0.9523, Time 00:03:45,lr 0.010000000000000002\nepoch 154, loss 0.08653, train_acc 0.9422, valid_acc 0.9498, Time 00:03:43,lr 0.010000000000000002\nepoch 155, loss 0.08328, train_acc 0.9436, valid_acc 0.9490, Time 00:03:45,lr 0.010000000000000002\nepoch 156, loss 0.08355, train_acc 0.9448, valid_acc 0.9500, Time 00:03:46,lr 0.010000000000000002\nepoch 157, loss 0.08707, train_acc 0.9429, valid_acc 0.9471, Time 00:03:46,lr 0.010000000000000002\nepoch 158, loss 0.08739, train_acc 0.9417, valid_acc 0.9510, Time 00:03:45,lr 0.010000000000000002\nepoch 159, loss 0.08569, train_acc 0.9431, valid_acc 0.9443, Time 00:03:44,lr 0.010000000000000002\nepoch 160, loss 0.08597, train_acc 0.9437, valid_acc 0.9504, Time 00:03:45,lr 0.010000000000000002\nepoch 161, loss 0.08600, train_acc 0.9444, valid_acc 0.9428, Time 00:03:45,lr 0.010000000000000002\nepoch 162, loss 0.08593, train_acc 0.9438, valid_acc 0.9490, Time 00:03:42,lr 0.010000000000000002\nepoch 163, loss 0.08594, train_acc 0.9438, valid_acc 0.9426, Time 00:03:45,lr 0.010000000000000002\nepoch 164, loss 0.08362, train_acc 0.9453, valid_acc 0.9465, Time 00:03:43,lr 0.010000000000000002\nepoch 165, loss 0.08293, train_acc 0.9451, valid_acc 0.9502, Time 00:03:43,lr 0.010000000000000002\nepoch 166, loss 0.08099, train_acc 0.9466, valid_acc 0.9508, Time 00:03:44,lr 0.010000000000000002\nepoch 167, loss 0.08601, train_acc 0.9437, valid_acc 0.9430, Time 00:03:44,lr 0.010000000000000002\nepoch 168, loss 0.08154, train_acc 0.9466, valid_acc 0.9477, Time 00:03:43,lr 0.010000000000000002\nepoch 169, loss 0.08548, train_acc 0.9435, valid_acc 0.9467, Time 00:03:45,lr 0.010000000000000002\nepoch 170, loss 0.08474, train_acc 0.9446, valid_acc 0.9488, Time 00:03:43,lr 0.010000000000000002\nepoch 171, loss 0.08388, train_acc 0.9452, valid_acc 0.9477, Time 00:03:45,lr 0.010000000000000002\nepoch 172, loss 0.08489, train_acc 0.9442, valid_acc 0.9465, Time 00:03:44,lr 0.010000000000000002\nepoch 173, loss 0.08431, train_acc 0.9446, valid_acc 0.9490, Time 00:03:44,lr 0.010000000000000002\nepoch 174, loss 0.08035, train_acc 0.9463, valid_acc 0.9461, Time 00:03:45,lr 0.010000000000000002\nepoch 175, loss 0.08508, train_acc 0.9451, valid_acc 0.9480, Time 00:03:44,lr 0.010000000000000002\nepoch 176, loss 0.08188, train_acc 0.9470, valid_acc 0.9471, Time 00:03:41,lr 0.010000000000000002\nepoch 177, loss 0.08183, train_acc 0.9461, valid_acc 0.9514, Time 00:03:45,lr 0.010000000000000002\nepoch 178, loss 0.08227, train_acc 0.9461, valid_acc 0.9479, Time 00:03:45,lr 0.010000000000000002\nepoch 179, loss 0.08543, train_acc 0.9435, valid_acc 0.9480, Time 00:03:44,lr 0.010000000000000002\nepoch 180, loss 0.08321, train_acc 0.9463, valid_acc 0.9461, Time 00:03:44,lr 0.010000000000000002\nepoch 181, loss 0.08168, train_acc 0.9463, valid_acc 0.9502, Time 00:03:43,lr 0.010000000000000002\nepoch 182, loss 0.08448, train_acc 0.9439, valid_acc 0.9504, Time 00:03:42,lr 0.010000000000000002\nepoch 183, loss 0.08212, train_acc 0.9464, valid_acc 0.9492, Time 00:03:43,lr 0.010000000000000002\nepoch 184, loss 0.08275, train_acc 0.9441, valid_acc 0.9488, Time 00:03:44,lr 0.010000000000000002\nepoch 185, loss 0.08448, train_acc 0.9451, valid_acc 0.9529, Time 00:03:45,lr 0.010000000000000002\nepoch 186, loss 0.08494, train_acc 0.9452, valid_acc 0.9455, Time 00:03:43,lr 0.010000000000000002\nepoch 187, loss 0.08455, train_acc 0.9439, valid_acc 0.9512, Time 00:03:42,lr 0.010000000000000002\nepoch 188, loss 0.08119, train_acc 0.9462, valid_acc 0.9437, Time 00:03:42,lr 0.010000000000000002\nepoch 189, loss 0.08296, train_acc 0.9457, valid_acc 0.9521, Time 00:03:44,lr 0.010000000000000002\nepoch 190, loss 0.08264, train_acc 0.9460, valid_acc 0.9477, Time 00:03:45,lr 0.010000000000000002\nepoch 191, loss 0.08169, train_acc 0.9465, valid_acc 0.9527, Time 00:03:45,lr 0.010000000000000002\nepoch 192, loss 0.08242, train_acc 0.9471, valid_acc 0.9512, Time 00:03:42,lr 0.010000000000000002\nepoch 193, loss 0.07926, train_acc 0.9494, valid_acc 0.9480, Time 00:03:43,lr 0.010000000000000002\nepoch 194, loss 0.08525, train_acc 0.9459, valid_acc 0.9469, Time 00:03:45,lr 0.010000000000000002\nepoch 195, loss 0.08035, train_acc 0.9471, valid_acc 0.9469, Time 00:03:45,lr 0.010000000000000002\nepoch 196, loss 0.08103, train_acc 0.9467, valid_acc 0.9502, Time 00:03:42,lr 0.010000000000000002\nepoch 197, loss 0.08116, train_acc 0.9463, valid_acc 0.9482, Time 00:03:44,lr 0.010000000000000002\nepoch 198, loss 0.07939, train_acc 0.9479, valid_acc 0.9510, Time 00:03:44,lr 0.010000000000000002\nepoch 199, loss 0.08048, train_acc 0.9479, valid_acc 0.9492, Time 00:03:44,lr 0.010000000000000002\nepoch 200, loss 0.07921, train_acc 0.9489, valid_acc 0.9508, Time 00:03:46,lr 0.010000000000000002\nepoch 201, loss 0.08178, train_acc 0.9472, valid_acc 0.9521, Time 00:03:47,lr 0.010000000000000002\nepoch 202, loss 0.08225, train_acc 0.9452, valid_acc 0.9492, Time 00:03:40,lr 0.010000000000000002\nepoch 203, loss 0.08363, train_acc 0.9452, valid_acc 0.9488, Time 00:03:44,lr 0.010000000000000002\nepoch 204, loss 0.08130, train_acc 0.9463, valid_acc 0.9500, Time 00:03:44,lr 0.010000000000000002\nepoch 205, loss 0.08193, train_acc 0.9465, valid_acc 0.9545, Time 00:03:42,lr 0.010000000000000002\nepoch 206, loss 0.08494, train_acc 0.9445, valid_acc 0.9527, Time 00:03:43,lr 0.010000000000000002\nepoch 207, loss 0.08030, train_acc 0.9481, valid_acc 0.9506, Time 00:03:43,lr 0.010000000000000002\nepoch 208, loss 0.08229, train_acc 0.9466, valid_acc 0.9508, Time 00:03:44,lr 0.010000000000000002\nepoch 209, loss 0.07848, train_acc 0.9464, valid_acc 0.9504, Time 00:03:45,lr 0.010000000000000002\nepoch 210, loss 0.08095, train_acc 0.9474, valid_acc 0.9512, Time 00:03:47,lr 0.010000000000000002\nepoch 211, loss 0.07890, train_acc 0.9480, valid_acc 0.9453, Time 00:03:44,lr 0.010000000000000002\nepoch 212, loss 0.07980, train_acc 0.9479, valid_acc 0.9455, Time 00:03:46,lr 0.010000000000000002\nepoch 213, loss 0.08324, train_acc 0.9464, valid_acc 0.9471, Time 00:03:45,lr 0.010000000000000002\nepoch 214, loss 0.08343, train_acc 0.9452, valid_acc 0.9475, Time 00:03:45,lr 0.010000000000000002\nepoch 215, loss 0.08090, train_acc 0.9480, valid_acc 0.9469, Time 00:03:45,lr 0.010000000000000002\nepoch 216, loss 0.08272, train_acc 0.9464, valid_acc 0.9443, Time 00:03:45,lr 0.010000000000000002\nepoch 217, loss 0.08309, train_acc 0.9460, valid_acc 0.9486, Time 00:03:47,lr 0.010000000000000002\nepoch 218, loss 0.07939, train_acc 0.9487, valid_acc 0.9457, Time 00:03:45,lr 0.010000000000000002\nepoch 219, loss 0.08031, train_acc 0.9476, valid_acc 0.9434, Time 00:03:47,lr 0.010000000000000002\nepoch 220, loss 0.08317, train_acc 0.9466, valid_acc 0.9453, Time 00:03:45,lr 0.010000000000000002\nepoch 221, loss 0.07977, train_acc 0.9482, valid_acc 0.9477, Time 00:03:45,lr 0.010000000000000002\nepoch 222, loss 0.08137, train_acc 0.9475, valid_acc 0.9441, Time 00:03:45,lr 0.010000000000000002\nepoch 223, loss 0.08047, train_acc 0.9479, valid_acc 0.9459, Time 00:03:44,lr 0.010000000000000002\nepoch 224, loss 0.07949, train_acc 0.9489, valid_acc 0.9490, Time 00:03:43,lr 0.010000000000000002\nepoch 225, loss 0.07514, train_acc 0.9519, valid_acc 0.9529, Time 00:03:42,lr 0.0010000000000000002\nepoch 226, loss 0.06933, train_acc 0.9554, valid_acc 0.9537, Time 00:03:44,lr 0.0010000000000000002\nepoch 227, loss 0.06452, train_acc 0.9585, valid_acc 0.9570, Time 00:03:44,lr 0.0010000000000000002\nepoch 228, loss 0.06569, train_acc 0.9586, valid_acc 0.9570, Time 00:03:44,lr 0.0010000000000000002\nepoch 229, loss 0.06019, train_acc 0.9616, valid_acc 0.9547, Time 00:03:43,lr 0.0010000000000000002\nepoch 230, loss 0.06263, train_acc 0.9606, valid_acc 0.9572, Time 00:03:45,lr 0.0010000000000000002\nepoch 231, loss 0.06135, train_acc 0.9611, valid_acc 0.9580, Time 00:03:46,lr 0.0010000000000000002\nepoch 232, loss 0.06512, train_acc 0.9599, valid_acc 0.9564, Time 00:03:45,lr 0.0010000000000000002\nepoch 233, loss 0.06147, train_acc 0.9612, valid_acc 0.9521, Time 00:03:45,lr 0.0010000000000000002\nepoch 234, loss 0.06180, train_acc 0.9616, valid_acc 0.9574, Time 00:03:46,lr 0.0010000000000000002\nepoch 235, loss 0.06161, train_acc 0.9610, valid_acc 0.9561, Time 00:03:46,lr 0.0010000000000000002\nepoch 236, loss 0.05796, train_acc 0.9628, valid_acc 0.9576, Time 00:03:45,lr 0.0010000000000000002\nepoch 237, loss 0.06214, train_acc 0.9607, valid_acc 0.9541, Time 00:03:46,lr 0.0010000000000000002\nepoch 238, loss 0.06377, train_acc 0.9611, valid_acc 0.9564, Time 00:03:45,lr 0.0010000000000000002\nepoch 239, loss 0.05700, train_acc 0.9638, valid_acc 0.9516, Time 00:03:43,lr 0.0010000000000000002\nepoch 240, loss 0.05919, train_acc 0.9633, valid_acc 0.9566, Time 00:03:47,lr 0.0010000000000000002\nepoch 241, loss 0.05998, train_acc 0.9629, valid_acc 0.9525, Time 00:03:43,lr 0.0010000000000000002\nepoch 242, loss 0.05857, train_acc 0.9642, valid_acc 0.9563, Time 00:03:45,lr 0.0010000000000000002\nepoch 243, loss 0.06102, train_acc 0.9620, valid_acc 0.9563, Time 00:03:43,lr 0.0010000000000000002\nepoch 244, loss 0.06100, train_acc 0.9619, valid_acc 0.9570, Time 00:03:43,lr 0.0010000000000000002\nepoch 245, loss 0.05877, train_acc 0.9634, valid_acc 0.9566, Time 00:03:43,lr 0.0010000000000000002\nepoch 246, loss 0.05890, train_acc 0.9633, valid_acc 0.9561, Time 00:03:47,lr 0.0010000000000000002\nepoch 247, loss 0.05718, train_acc 0.9651, valid_acc 0.9537, Time 00:03:45,lr 0.0010000000000000002\nepoch 248, loss 0.05794, train_acc 0.9640, valid_acc 0.9537, Time 00:03:45,lr 0.0010000000000000002\nepoch 249, loss 0.05806, train_acc 0.9642, valid_acc 0.9576, Time 00:03:42,lr 0.0010000000000000002\nepoch 250, loss 0.05728, train_acc 0.9649, valid_acc 0.9527, Time 00:03:46,lr 0.0010000000000000002\nepoch 251, loss 0.05863, train_acc 0.9638, valid_acc 0.9545, Time 00:03:44,lr 0.0010000000000000002\nepoch 252, loss 0.05889, train_acc 0.9630, valid_acc 0.9576, Time 00:03:42,lr 0.0010000000000000002\nepoch 253, loss 0.05654, train_acc 0.9638, valid_acc 0.9570, Time 00:03:44,lr 0.0010000000000000002\nepoch 254, loss 0.05695, train_acc 0.9646, valid_acc 0.9570, Time 00:03:48,lr 0.0010000000000000002\n"
 },
 {
  "name": "stderr",
  "output_type": "stream",
  "text": "/home/zp/.conda/envs/gluon/lib/python3.6/site-packages/mxnet/gluon/block.py:338: UserWarning: save_params is deprecated. Please use save_parameters. Note that if you want to load from SymbolBlock later, please use export instead. For details, see https://mxnet.incubator.apache.org/tutorials/gluon/save_load_params.html\n  warnings.warn(\"save_params is deprecated. Please use save_parameters. \"\n"
 }
]
```

```{.python .input  n=8}
net.save_parameters("./models/resnet164_e0-255_focal_clip")
```

### Exp5: sherlock_densenet: 0.9539

```{.python .input  n=5}
batch_size = 128
transform_train = transform_train_DA1
train_data, valid_data, train_valid_data, test_data, test_ds, train_valid_ds = data_loader(batch_size, transform_train)
net = DenseNet(growthRate=12, depth=100, reduction=0.5, bottleneck=True, nClasses=10)
loss_f = gluon.loss.SoftmaxCrossEntropyLoss()

num_epochs = 200
learning_rate = 0.1
weight_decay = 1e-4
lr_period = [90, 140]
lr_decay=0.1
log_file = None

net.hybridize()
net.initialize(ctx=ctx)
w_key = []
train(net, train_data, valid_data, num_epochs, learning_rate, lr_period, lr_decay, weight_decay, ctx, w_key, log_file, False, loss_f)
net.save_parameters("models/shelock_densenet_orign")
```

```{.json .output n=5}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "epoch 0, loss 1.51915, train_acc 0.4410, valid_acc 0.5588, Time 00:02:42,lr 0.1\nepoch 1, loss 1.00430, train_acc 0.6432, valid_acc 0.6750, Time 00:04:31,lr 0.1\nepoch 2, loss 0.80806, train_acc 0.7152, valid_acc 0.6072, Time 00:04:32,lr 0.1\nepoch 3, loss 0.68012, train_acc 0.7639, valid_acc 0.7219, Time 00:04:32,lr 0.1\nepoch 4, loss 0.59042, train_acc 0.7927, valid_acc 0.8016, Time 00:04:32,lr 0.1\nepoch 5, loss 0.52628, train_acc 0.8197, valid_acc 0.7812, Time 00:04:32,lr 0.1\nepoch 6, loss 0.48291, train_acc 0.8346, valid_acc 0.8014, Time 00:04:32,lr 0.1\nepoch 7, loss 0.44734, train_acc 0.8449, valid_acc 0.7133, Time 00:04:31,lr 0.1\nepoch 8, loss 0.41750, train_acc 0.8550, valid_acc 0.8496, Time 00:04:33,lr 0.1\nepoch 9, loss 0.39624, train_acc 0.8634, valid_acc 0.8029, Time 00:04:36,lr 0.1\nepoch 10, loss 0.37908, train_acc 0.8692, valid_acc 0.8480, Time 00:04:35,lr 0.1\nepoch 11, loss 0.35383, train_acc 0.8782, valid_acc 0.8309, Time 00:04:38,lr 0.1\nepoch 12, loss 0.33964, train_acc 0.8829, valid_acc 0.8439, Time 00:04:30,lr 0.1\nepoch 13, loss 0.32488, train_acc 0.8864, valid_acc 0.8613, Time 00:04:23,lr 0.1\nepoch 14, loss 0.31375, train_acc 0.8907, valid_acc 0.8432, Time 00:04:34,lr 0.1\nepoch 15, loss 0.30322, train_acc 0.8944, valid_acc 0.8715, Time 00:04:31,lr 0.1\nepoch 16, loss 0.29298, train_acc 0.8978, valid_acc 0.8664, Time 00:04:31,lr 0.1\nepoch 17, loss 0.28418, train_acc 0.9015, valid_acc 0.8309, Time 00:04:20,lr 0.1\nepoch 18, loss 0.26711, train_acc 0.9071, valid_acc 0.8514, Time 00:04:22,lr 0.1\nepoch 19, loss 0.27074, train_acc 0.9065, valid_acc 0.8648, Time 00:04:18,lr 0.1\nepoch 20, loss 0.25336, train_acc 0.9120, valid_acc 0.8633, Time 00:04:24,lr 0.1\nepoch 21, loss 0.25170, train_acc 0.9119, valid_acc 0.8711, Time 00:04:25,lr 0.1\nepoch 22, loss 0.24528, train_acc 0.9136, valid_acc 0.8332, Time 00:04:19,lr 0.1\nepoch 23, loss 0.23746, train_acc 0.9174, valid_acc 0.8771, Time 00:04:22,lr 0.1\nepoch 24, loss 0.23308, train_acc 0.9200, valid_acc 0.8705, Time 00:04:17,lr 0.1\nepoch 25, loss 0.23445, train_acc 0.9181, valid_acc 0.8477, Time 00:04:22,lr 0.1\nepoch 26, loss 0.22130, train_acc 0.9229, valid_acc 0.8658, Time 00:04:15,lr 0.1\nepoch 27, loss 0.21926, train_acc 0.9239, valid_acc 0.8775, Time 00:04:26,lr 0.1\nepoch 28, loss 0.22179, train_acc 0.9231, valid_acc 0.8617, Time 00:04:24,lr 0.1\nepoch 29, loss 0.21536, train_acc 0.9255, valid_acc 0.8463, Time 00:04:23,lr 0.1\nepoch 30, loss 0.21296, train_acc 0.9255, valid_acc 0.8725, Time 00:04:16,lr 0.1\nepoch 31, loss 0.20821, train_acc 0.9277, valid_acc 0.8674, Time 00:04:13,lr 0.1\nepoch 32, loss 0.20501, train_acc 0.9286, valid_acc 0.8428, Time 00:04:10,lr 0.1\nepoch 33, loss 0.20473, train_acc 0.9274, valid_acc 0.8855, Time 00:04:14,lr 0.1\nepoch 34, loss 0.20228, train_acc 0.9292, valid_acc 0.8779, Time 00:04:20,lr 0.1\nepoch 35, loss 0.19656, train_acc 0.9313, valid_acc 0.9016, Time 00:04:32,lr 0.1\nepoch 36, loss 0.19452, train_acc 0.9315, valid_acc 0.8811, Time 00:04:23,lr 0.1\nepoch 37, loss 0.19326, train_acc 0.9325, valid_acc 0.8900, Time 00:04:18,lr 0.1\nepoch 38, loss 0.19072, train_acc 0.9328, valid_acc 0.8697, Time 00:04:12,lr 0.1\nepoch 39, loss 0.18912, train_acc 0.9346, valid_acc 0.8738, Time 00:04:22,lr 0.1\nepoch 40, loss 0.18793, train_acc 0.9346, valid_acc 0.8646, Time 00:04:12,lr 0.1\nepoch 41, loss 0.18637, train_acc 0.9339, valid_acc 0.8580, Time 00:04:10,lr 0.1\nepoch 42, loss 0.18884, train_acc 0.9329, valid_acc 0.8611, Time 00:04:18,lr 0.1\nepoch 43, loss 0.18011, train_acc 0.9358, valid_acc 0.8801, Time 00:04:24,lr 0.1\nepoch 44, loss 0.17479, train_acc 0.9389, valid_acc 0.8971, Time 00:04:18,lr 0.1\nepoch 45, loss 0.17873, train_acc 0.9386, valid_acc 0.8910, Time 00:04:24,lr 0.1\nepoch 46, loss 0.17834, train_acc 0.9379, valid_acc 0.8992, Time 00:04:20,lr 0.1\nepoch 47, loss 0.17494, train_acc 0.9388, valid_acc 0.8939, Time 00:04:18,lr 0.1\nepoch 48, loss 0.17593, train_acc 0.9379, valid_acc 0.8852, Time 00:04:13,lr 0.1\nepoch 49, loss 0.16677, train_acc 0.9415, valid_acc 0.8879, Time 00:04:20,lr 0.1\nepoch 50, loss 0.17181, train_acc 0.9398, valid_acc 0.8861, Time 00:04:19,lr 0.1\nepoch 51, loss 0.16614, train_acc 0.9422, valid_acc 0.8971, Time 00:04:24,lr 0.1\nepoch 52, loss 0.16685, train_acc 0.9410, valid_acc 0.8793, Time 00:04:27,lr 0.1\nepoch 53, loss 0.16757, train_acc 0.9413, valid_acc 0.8812, Time 00:04:18,lr 0.1\nepoch 54, loss 0.16090, train_acc 0.9433, valid_acc 0.9016, Time 00:04:21,lr 0.1\nepoch 55, loss 0.15790, train_acc 0.9454, valid_acc 0.8770, Time 00:04:16,lr 0.1\nepoch 56, loss 0.16715, train_acc 0.9429, valid_acc 0.9031, Time 00:04:14,lr 0.1\nepoch 57, loss 0.16158, train_acc 0.9432, valid_acc 0.8842, Time 00:04:21,lr 0.1\nepoch 58, loss 0.16277, train_acc 0.9431, valid_acc 0.8719, Time 00:04:13,lr 0.1\nepoch 59, loss 0.15815, train_acc 0.9442, valid_acc 0.8639, Time 00:04:07,lr 0.1\nepoch 60, loss 0.16018, train_acc 0.9437, valid_acc 0.8971, Time 00:04:16,lr 0.1\nepoch 61, loss 0.15317, train_acc 0.9468, valid_acc 0.8941, Time 00:04:23,lr 0.1\nepoch 62, loss 0.15735, train_acc 0.9441, valid_acc 0.8836, Time 00:04:20,lr 0.1\nepoch 63, loss 0.14888, train_acc 0.9474, valid_acc 0.9068, Time 00:04:24,lr 0.1\nepoch 64, loss 0.16062, train_acc 0.9428, valid_acc 0.9051, Time 00:04:14,lr 0.1\nepoch 65, loss 0.14617, train_acc 0.9488, valid_acc 0.9006, Time 00:04:16,lr 0.1\nepoch 66, loss 0.15017, train_acc 0.9475, valid_acc 0.9018, Time 00:04:16,lr 0.1\nepoch 67, loss 0.14891, train_acc 0.9484, valid_acc 0.8973, Time 00:04:17,lr 0.1\nepoch 68, loss 0.14531, train_acc 0.9476, valid_acc 0.8748, Time 00:04:14,lr 0.1\nepoch 69, loss 0.15018, train_acc 0.9485, valid_acc 0.9055, Time 00:04:17,lr 0.1\nepoch 70, loss 0.14250, train_acc 0.9509, valid_acc 0.8842, Time 00:04:23,lr 0.1\nepoch 71, loss 0.14431, train_acc 0.9489, valid_acc 0.8859, Time 00:04:19,lr 0.1\nepoch 72, loss 0.14626, train_acc 0.9480, valid_acc 0.8830, Time 00:04:17,lr 0.1\nepoch 73, loss 0.14740, train_acc 0.9488, valid_acc 0.8793, Time 00:04:12,lr 0.1\nepoch 74, loss 0.14400, train_acc 0.9495, valid_acc 0.8895, Time 00:04:18,lr 0.1\nepoch 75, loss 0.14682, train_acc 0.9479, valid_acc 0.8902, Time 00:04:25,lr 0.1\n"
 },
 {
  "ename": "MXNetError",
  "evalue": "[14:29:45] src/storage/./pooled_storage_manager.h:108: cudaMalloc failed: out of memory\n\nStack trace returned 10 entries:\n[bt] (0) /home/zp/.conda/envs/gluon/lib/python3.6/site-packages/mxnet/libmxnet.so(+0x30756a) [0x7f5a7a63256a]\n[bt] (1) /home/zp/.conda/envs/gluon/lib/python3.6/site-packages/mxnet/libmxnet.so(+0x307b91) [0x7f5a7a632b91]\n[bt] (2) /home/zp/.conda/envs/gluon/lib/python3.6/site-packages/mxnet/libmxnet.so(+0x29d67b3) [0x7f5a7cd017b3]\n[bt] (3) /home/zp/.conda/envs/gluon/lib/python3.6/site-packages/mxnet/libmxnet.so(+0x29db065) [0x7f5a7cd06065]\n[bt] (4) /home/zp/.conda/envs/gluon/lib/python3.6/site-packages/mxnet/libmxnet.so(+0x440e64) [0x7f5a7a76be64]\n[bt] (5) /home/zp/.conda/envs/gluon/lib/python3.6/site-packages/mxnet/libmxnet.so(+0x24f085d) [0x7f5a7c81b85d]\n[bt] (6) /home/zp/.conda/envs/gluon/lib/python3.6/site-packages/mxnet/libmxnet.so(+0x24f0d33) [0x7f5a7c81bd33]\n[bt] (7) /home/zp/.conda/envs/gluon/lib/python3.6/site-packages/mxnet/libmxnet.so(+0x2468d0d) [0x7f5a7c793d0d]\n[bt] (8) /home/zp/.conda/envs/gluon/lib/python3.6/site-packages/mxnet/libmxnet.so(+0x2468cf7) [0x7f5a7c793cf7]\n[bt] (9) /home/zp/.conda/envs/gluon/lib/python3.6/site-packages/mxnet/libmxnet.so(+0x2468cf7) [0x7f5a7c793cf7]\n\n",
  "output_type": "error",
  "traceback": [
   "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
   "\u001b[0;31mMXNetError\u001b[0m                                Traceback (most recent call last)",
   "\u001b[0;32m<ipython-input-5-84ea9f448f42>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m()\u001b[0m\n\u001b[1;32m     15\u001b[0m \u001b[0mnet\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0minitialize\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mctx\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mctx\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     16\u001b[0m \u001b[0mw_key\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0;34m[\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m---> 17\u001b[0;31m \u001b[0mtrain\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mnet\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mtrain_data\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mvalid_data\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mnum_epochs\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mlearning_rate\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mlr_period\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mlr_decay\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mweight_decay\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mctx\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mw_key\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mlog_file\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;32mFalse\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mloss_f\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m     18\u001b[0m \u001b[0mnet\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0msave_params\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m\"models/shelock_densenet_orign\"\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
   "\u001b[0;32m<ipython-input-4-45adbe072699>\u001b[0m in \u001b[0;36mtrain\u001b[0;34m(net, train_data, valid_data, num_epochs, lr, lr_period, lr_decay, wd, ctx, w_key, output_file, verbose, loss_f)\u001b[0m\n\u001b[1;32m     45\u001b[0m             \u001b[0mtrainer\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mstep\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mdata\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mshape\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;36m0\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     46\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m---> 47\u001b[0;31m             \u001b[0m_loss\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mnd\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mmean\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mloss\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0masscalar\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m     48\u001b[0m             \u001b[0m_acc\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mutils\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0maccuracy\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0moutput\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mlabel\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     49\u001b[0m             \u001b[0;31m#_acc = utils.accuracy(output, label)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
   "\u001b[0;32m~/.conda/envs/gluon/lib/python3.6/site-packages/mxnet/ndarray/ndarray.py\u001b[0m in \u001b[0;36masscalar\u001b[0;34m(self)\u001b[0m\n\u001b[1;32m   1892\u001b[0m         \u001b[0;32mif\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mshape\u001b[0m \u001b[0;34m!=\u001b[0m \u001b[0;34m(\u001b[0m\u001b[0;36m1\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1893\u001b[0m             \u001b[0;32mraise\u001b[0m \u001b[0mValueError\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m\"The current array is not a scalar\"\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m-> 1894\u001b[0;31m         \u001b[0;32mreturn\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0masnumpy\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;36m0\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m   1895\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1896\u001b[0m     \u001b[0;32mdef\u001b[0m \u001b[0mastype\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mdtype\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mcopy\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;32mTrue\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
   "\u001b[0;32m~/.conda/envs/gluon/lib/python3.6/site-packages/mxnet/ndarray/ndarray.py\u001b[0m in \u001b[0;36masnumpy\u001b[0;34m(self)\u001b[0m\n\u001b[1;32m   1874\u001b[0m             \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mhandle\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1875\u001b[0m             \u001b[0mdata\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mctypes\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mdata_as\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mctypes\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mc_void_p\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m-> 1876\u001b[0;31m             ctypes.c_size_t(data.size)))\n\u001b[0m\u001b[1;32m   1877\u001b[0m         \u001b[0;32mreturn\u001b[0m \u001b[0mdata\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1878\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
   "\u001b[0;32m~/.conda/envs/gluon/lib/python3.6/site-packages/mxnet/base.py\u001b[0m in \u001b[0;36mcheck_call\u001b[0;34m(ret)\u001b[0m\n\u001b[1;32m    147\u001b[0m     \"\"\"\n\u001b[1;32m    148\u001b[0m     \u001b[0;32mif\u001b[0m \u001b[0mret\u001b[0m \u001b[0;34m!=\u001b[0m \u001b[0;36m0\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 149\u001b[0;31m         \u001b[0;32mraise\u001b[0m \u001b[0mMXNetError\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mpy_str\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0m_LIB\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mMXGetLastError\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    150\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    151\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
   "\u001b[0;31mMXNetError\u001b[0m: [14:29:45] src/storage/./pooled_storage_manager.h:108: cudaMalloc failed: out of memory\n\nStack trace returned 10 entries:\n[bt] (0) /home/zp/.conda/envs/gluon/lib/python3.6/site-packages/mxnet/libmxnet.so(+0x30756a) [0x7f5a7a63256a]\n[bt] (1) /home/zp/.conda/envs/gluon/lib/python3.6/site-packages/mxnet/libmxnet.so(+0x307b91) [0x7f5a7a632b91]\n[bt] (2) /home/zp/.conda/envs/gluon/lib/python3.6/site-packages/mxnet/libmxnet.so(+0x29d67b3) [0x7f5a7cd017b3]\n[bt] (3) /home/zp/.conda/envs/gluon/lib/python3.6/site-packages/mxnet/libmxnet.so(+0x29db065) [0x7f5a7cd06065]\n[bt] (4) /home/zp/.conda/envs/gluon/lib/python3.6/site-packages/mxnet/libmxnet.so(+0x440e64) [0x7f5a7a76be64]\n[bt] (5) /home/zp/.conda/envs/gluon/lib/python3.6/site-packages/mxnet/libmxnet.so(+0x24f085d) [0x7f5a7c81b85d]\n[bt] (6) /home/zp/.conda/envs/gluon/lib/python3.6/site-packages/mxnet/libmxnet.so(+0x24f0d33) [0x7f5a7c81bd33]\n[bt] (7) /home/zp/.conda/envs/gluon/lib/python3.6/site-packages/mxnet/libmxnet.so(+0x2468d0d) [0x7f5a7c793d0d]\n[bt] (8) /home/zp/.conda/envs/gluon/lib/python3.6/site-packages/mxnet/libmxnet.so(+0x2468cf7) [0x7f5a7c793cf7]\n[bt] (9) /home/zp/.conda/envs/gluon/lib/python3.6/site-packages/mxnet/libmxnet.so(+0x2468cf7) [0x7f5a7c793cf7]\n\n"
  ]
 }
]
```

# merge result

```{.python .input  n=106}
from mxnet.gluon import data as gdata, nn, loss as gloss
data_dir = '/home/zp/SummerSchool/CS231n/Kaggle/data/CIFAR-10'
train_dir = 'train'
test_dir = 'test'
batch_size = 128
input_dir = 'train_valid_test'
# 测试时，无需对图像做标准化以外的增强数据处理。
transform_test = gdata.vision.transforms.Compose([
    gdata.vision.transforms.ToTensor(),
    gdata.vision.transforms.Normalize([0.4914, 0.4822, 0.4465],
                                      [0.2023, 0.1994, 0.2010])
])


# 读取原始图像文件。flag=1 说明输入图像有三个通道（彩色）。

test_ds = gdata.vision.ImageFolderDataset(
    os.path.join(data_dir, input_dir, 'test'), flag=1)

test_data = gdata.DataLoader(test_ds.transform_first(transform_test),
                             batch_size, shuffle=False, last_batch='keep')
print(len(test_ds))
```

```{.json .output n=106}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "300000\n"
 }
]
```

class Residual(nn.Block):
    def __init__(self, channels, same_shape=True, equal=True, **kwargs):
        super(Residual, self).__init__(**kwargs)
        self.same_shape = same_shape
        self.equal = equal
        with self.name_scope():
            strides = 1 if same_shape else 2
            self.bn1 = nn.BatchNorm()
            self.conv1 = nn.Conv2D(channels, kernel_size=3, padding=1, strides=strides)
            self.bn2 = nn.BatchNorm()
            self.conv2 = nn.Conv2D(channels, kernel_size=3, padding=1)
            if (not same_shape) or (not equal):
                self.conv3 = nn.Conv2D(channels, kernel_size=1, strides=strides)
    def forward(self, x):
        out = self.conv1(nd.relu(self.bn1(x)))
        out = self.conv2(nd.relu(self.bn2(out)))
        if (not self.same_shape) or (not self.equal):
            x = self.conv3(x)
        return out + x

class wrn(nn.Block):
    def __init__(self, num_classes, **kwargs):
        super(wrn, self).__init__(**kwargs)
        with self.name_scope(): 
            net = self.net = nn.Sequential()
            net.add(nn.Conv2D(channels=16, kernel_size=3, strides=1, padding=1))
            net.add(Residual(channels=16*8, equal=False))
            net.add(Residual(channels=16*8), Residual(channels=16*8))            
            net.add(Residual(channels=32*8, same_shape=False))
            net.add(Residual(channels=32*8), Residual(channels=32*8))
            net.add(Residual(channels=64*8, same_shape=False))
            net.add(Residual(channels=64*8), Residual(channels=64*8))
            net.add(nn.BatchNorm())
            net.add(nn.Activation(activation='relu'))
            net.add(nn.AvgPool2D(pool_size=8))
            net.add(nn.Flatten())
            net.add(nn.Dense(num_classes))
    def forward(self, x):
        out = x
        for i, b in enumerate(self.net):
            out = b(out)
        return out

```{.python .input  n=107}
import os
import numpy as np
import pandas as pd



def save_net_result(net, filename, test_data, ctx):
    output = nd.zeros(shape=(300000, 10), ctx=ctx)
   
    for i, (data, label) in enumerate(test_data):
        #print(len(test_data)) #data.shape[0] 128  mxnet.gluon.data.dataloader.DataLoader 4688
        #print(net(data.as_in_context(ctx)).shape)
        #print(data.shape[0]) 128
        output[i*batch_size:i*batch_size+data.shape[0],:] = net(data.as_in_context(ctx))
        #print(output)
    #output = nd.array(output)
    nd.save(filename, output)

def test_net(data):
    return data.reshape((data.shape[0], -1))[:, :10]

def save_model_result(model_name, ctx):
    net.load_parameters("models/" + model_name, ctx=ctx)
    save_net_result(net, "result/" + model_name, test_data, ctx)

#model_list = ['resnet164_e255_focal_clip', 'res164__2_e255_focal_clip_all_data', 'resnet164_e300','resnet164_e0-255_focal_clip',
#              'res18_9', 
#              'log_shelock_densenet', 'shelock_densenet_orign',
#              'shelock_resnet_orign']
#weight_list = [0.9535, 0.9540, 0.95270, 0.95, 0.93230, 0.9346, 0.9539, 0.95]
model_list = ['resnet164_e0-255_focal_clip251', 'res164__2_e255_focal_clip_all_data','resnet164_e0-255_focal_clip',
              'resnet164_e0-255_focal_clip201', 
              'wide_resnet2',
              'shelock_resnet_orign2','shelock_densenet_orign191']
weight_list = [0.9545, 0.9459, 0.9570, 0.9521, 0.95892, 0.9555, 0.9494]

batch_size = 128


net = ResNet164_v2(10)
for model_name in model_list[:4]:
    if not os.path.exists("result/"+model_name):
        save_model_result(model_name, ctx)
         
#net = wrn(10)
#for model_name in model_list[4:5]:
#    if not os.path.exists("result/"+model_name):
#        save_model_result(model_name, ctx)
        
net = ResNet164_v2(10)
for model_name in model_list[5:6]:
    if not os.path.exists("result/"+model_name):
        save_model_result(model_name, ctx)
        

net = DenseNet(growthRate=12, depth=100, reduction=0.5, bottleneck=True, nClasses=10)
for model_name in model_list[6:]:
    if not os.path.exists("result/"+model_name):
        save_model_result(model_name, ctx)
```

```{.python .input  n=108}
"""
classfiy test set
"""
import numpy as np
import pandas as pd



def mesuare_sum(preds, weight_list=None):
    if weight_list is None:
        weight_list = [1] * len(preds)
    output = preds[0] * weight_list[0]
    for i in range(1, len(preds)):
        output = output + preds[i] * weight_list[i]
    preds = output.argmax(axis=1).astype(int).asnumpy() % 10
    return preds

def mesuare_softmax_sum(preds, weight_list=None):
    if weight_list is None:
        weight_list = [1] * len(preds)
    output = nd.softmax(preds[0], axis=1) * weight_list[0]
    for i in range(1, len(preds)):
        output = output + nd.softmax(preds[i], axis=1) * weight_list[i]
    preds = output.argmax(axis=1).astype(int).asnumpy() % 10
    return preds

def mesuare_biggest(preds, weight_list=None):
    if weight_list is not None:
        for i in range(len(preds)):
            preds[i] = preds[i] * weight_list[i]
    output = nd.concat(*preds, dim=1)
    preds = output.argmax(axis=1).astype(int).asnumpy() % 10
    return preds

#model_list = ['res164__2_e255_focal_clip_all_data', 'resnet164_e300', 'resnet164_e0-255_focal_clip',
#              'shelock_densenet_orign', 'shelock_resnet_orign']
#weight_list = [0.9540, 0.95270, 0.95, 0.9539, 0.95]
#weight_list=None
#model_list = ['resnet164_e0-255_focal_clip251', 'res164__2_e255_focal_clip_all_data','resnet164_e0-255_focal_clip',
##              'resnet164_e0-255_focal_clip201', 
#              'wide_resnet2',
#              'shelock_resnet_orign2']
# = [0.9545, 0.9459, 0.9570, 0.9521, 0.95892, 0.9555]
model_list = ['resnet164_e0-255_focal_clip251', 'res164__2_e255_focal_clip_all_data','resnet164_e0-255_focal_clip',
              'resnet164_e0-255_focal_clip201',
              'shelock_resnet_orign2','shelock_densenet_orign191']
#weight_list = [0.9545, 0.9459, 0.9570, 0.9521, 0.95892, 0.9555, 0.9494]
weight_list = [0.9545, 0.9459, 0.9570, 0.9521, 0.9555, 0.9494]
preds = []
for result_name in model_list:
    preds.append(nd.load("result/"+result_name)[0].as_in_context(ctx))
    #preds.append(nd.load("models/"+result_name)[0].as_in_context(ctx))

#preds = mesuare_biggest(preds, weight_list)
#preds = mesuare_sum(preds, weight_list)
preds = mesuare_softmax_sum(preds, weight_list)

sorted_ids = list(range(1, 300000 + 1))
sorted_ids.sort(key=lambda x: str(x))

df = pd.DataFrame({'id': sorted_ids, 'label': preds})
df['label'] = df['label'].apply(lambda x: train_valid_ds.synsets[x])
df.to_csv('submission/concat_5_softmax_sum_weight.csv', index=False)
```

```{.python .input  n=111}
preds = []
for result_name in model_list:
    preds.append(nd.load("result/"+result_name)[0].as_in_context(ctx))
#preds = mesuare_biggest(preds, weight_list)
preds = mesuare_sum(preds, weight_list)
#preds = mesuare_softmax_sum(preds, weight_list)

sorted_ids = list(range(1, 300000 + 1))
sorted_ids.sort(key=lambda x: str(x))

df = pd.DataFrame({'id': sorted_ids, 'label': preds})
df['label'] = df['label'].apply(lambda x: train_valid_ds.synsets[x])
df.to_csv('submission/concat_5_measure_sum_weight.csv', index=False)
```

```{.python .input  n=112}
preds = []
for result_name in model_list:
    preds.append(nd.load("result/"+result_name)[0].as_in_context(ctx))
preds = mesuare_biggest(preds, weight_list)
#preds = mesuare_sum(preds, weight_list)
#preds = mesuare_softmax_sum(preds, weight_list)

sorted_ids = list(range(1, 300000 + 1))
sorted_ids.sort(key=lambda x: str(x))

df = pd.DataFrame({'id': sorted_ids, 'label': preds})
df['label'] = df['label'].apply(lambda x: train_valid_ds.synsets[x])
df.to_csv('submission/concat_5_softmax_biggest_weight.csv', index=False)
```

```{.python .input}

```
