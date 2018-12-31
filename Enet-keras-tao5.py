# -*- coding: utf-8 -*-
#!unzip  drive/My\ Drive/kaggle/trainimages.zip -d drive/My\ Drive/kaggle/
"""### Slim Enet with spatial dropout and augmentation, StratifiedKFold 
Uses efficient residual pooling, squeeze 1x1 convolutions, dilations and asymmetric convolutions
Uses pipeline from:
  https://www.kaggle.com/shaojiaxin/u-net-with-simple-resnet-blocks-v2-new-loss
          Whole script designed to run on headless server
"""

import os
import shutil
import pandas as pd
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
import seaborn as sns
sns.set_style("white")
import cv2
from sklearn.model_selection import StratifiedKFold
from keras.models import Model, load_model, save_model
from keras.layers import Input, Dropout, BatchNormalization, Activation, Add, UpSampling2D, Dense
from keras.layers import Reshape, ZeroPadding3D, multiply, Permute, GlobalAveragePooling2D
from keras.layers.core import Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras import backend as K
from keras import optimizers
import keras
import tensorflow as tf
from keras.preprocessing.image import array_to_img, img_to_array, load_img#,save_img

import time
t_start = time.time()
"""Add your own data directory to the long list!"""
#datadir='drive/My Drive/kaggle/'
#datadir="./data/"
#datadir='/Users/tao/.kaggle/competitions/tgs-salt-identification-challenge/'
datadir='/home/tao/.kaggle/competitions/tgs-salt-identification-challenge/'
train_dir=datadir + 'train/'
test_dir=datadir + 'test/'
cv_total = 5
#cv_index = 1 -5
version = 2
basic_name_ori = 'Enet_v'+str(version)
save_model_name = datadir+basic_name_ori + '.model'
submission_file = datadir+basic_name_ori + '.csv'

print(save_model_name)
print(submission_file)

img_size_ori = 101
img_size_target = 101

def upsample(img):# legacy junk
    return img

# Loading of training/testing ids and depths
train_df = pd.read_csv(datadir+'train.csv', index_col="id", usecols=[0])#add train/ on sagemaker
depths_df = pd.read_csv(datadir+'depths.csv', index_col="id")#add train/ on sagemaker
train_df = train_df.join(depths_df)
#Filtering out bad images
uniques=set()
realsamples=os.listdir(train_dir+'images')
for i in realsamples:
    uniques.add(i)
for i, _ in train_df.iterrows():
    if str(i)+'.png' not in uniques:
        print(i)
        train_df=train_df.drop(i)
test_df = depths_df[~depths_df.index.isin(train_df.index)]

len(train_df)

train_df["images"] = [np.array(load_img(train_dir+"images/{}.png".format(idx), grayscale=True)) / 255 for idx in train_df.index]

train_df["masks"] = [np.array(load_img(train_dir+"masks/{}.png".format(idx), grayscale=True)) / 255 for idx in train_df.index]

train_df["coverage"] = train_df.masks.map(np.sum) / pow(img_size_ori, 2)


"""#### calculate mask type for stratify, the difficuly of training different mask type is different. 
* Reference  from Heng's discussion, search "error analysis" in the following link

https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/63984#382657****
"""
def get_mask_type(mask):
    border = 10
    outer = np.zeros((101-2*border, 101-2*border), np.float32)
    outer = cv2.copyMakeBorder(outer, border, border, border, border, borderType = cv2.BORDER_CONSTANT, value = 1)

    cover = (mask>0.5).sum()
    if cover < 8:
        return 0 # empty
    if cover == ((mask*outer) > 0.5).sum():
        return 1 #border
    if np.all(mask==mask[0]):
        return 2 #vertical

    percentage = cover/(101*101)
    if percentage < 0.15:
        return 3
    elif percentage < 0.25:
        return 4
    elif percentage < 0.50:
        return 5
    elif percentage < 0.75:
        return 6
    else:
        return 7

def histcoverage(coverage):
    histall = np.zeros((1,8))
    for c in coverage:
        histall[0,c] += 1
    return histall

train_df["coverage"] = train_df.masks.map(np.sum) / pow(img_size_target, 2)

train_df["coverage_class"] = train_df.masks.map(get_mask_type)

train_all = []
evaluate_all = []
skf = StratifiedKFold(n_splits=cv_total, random_state=1234, shuffle=True)
for train_index, evaluate_index in skf.split(train_df.index.values, train_df.coverage_class):
    train_all.append(train_index)
    evaluate_all.append(evaluate_index)
    print(train_index.shape,evaluate_index.shape) # the shape is slightly different in different cv, it's OK

def get_cv_data(cv_index):
    train_index = train_all[cv_index-1]
    evaluate_index = evaluate_all[cv_index-1]
    x_train = np.array(train_df.images[train_index].map(upsample).tolist()).reshape(-1, img_size_target, img_size_target, 1)
    y_train = np.array(train_df.masks[train_index].map(upsample).tolist()).reshape(-1, img_size_target, img_size_target, 1)
    x_valid = np.array(train_df.images[evaluate_index].map(upsample).tolist()).reshape(-1, img_size_target, img_size_target, 1)
    y_valid = np.array(train_df.masks[evaluate_index].map(upsample).tolist()).reshape(-1, img_size_target, img_size_target, 1)
    return x_train,y_train,x_valid,y_valid

"""#### Show  some examples of different mask"""

cv_index = 1
train_index = train_all[cv_index-1]
evaluate_index = evaluate_all[cv_index-1]

print(train_index.shape,evaluate_index.shape)
histall = histcoverage(train_df.coverage_class[train_index].values)
#print(f'train cv{cv_index}, number of each mask class = \n'+str(histall))
histall_test = histcoverage(train_df.coverage_class[evaluate_index].values)
#print(f'evaluate cv{cv_index}, number of each mask class = \n'+sr(histall_test))

fig, axes = plt.subplots(nrows=2, ncols=8, figsize=(24, 6), sharex=True, sharey=True)

# show mask class example
for c in range(8):
    j= 0
    for i in train_index:
        if train_df.coverage_class[i] == c:
            axes[j,c].imshow(np.array(train_df.masks[i])  )
            axes[j,c].set_axis_off()
            axes[j,c].set_title('class '+str(c))
            j += 1
            if(j>=2):
                break
plt.savefig('masks')

""" Squeeze Excitation block """
def spatial_se_block(inputs, ratio=16):
    ''' Create a squeeze-excite block
    Args:
        input: input tensor
        filters: number of output filters
        k: width factor
    Returns: a keras tensor
    '''
    init = inputs
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    filters = init._keras_shape[channel_axis]
    se_shape = (1, 1, filters)

    se = GlobalAveragePooling2D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

    if K.image_data_format() == 'channels_first':
        se = Permute((3, 1, 2))(se)

    x = multiply([init, se])
    return x
def channel_se_block(inputs):
    init = inputs
    squeezer=Conv2D(1, (1,1), activation='sigmoid')(inputs)
    output=multiply([init, squeezer])
    return output
def se_block(inputs, ratio=16):
    it=inputs
    s=spatial_se_block(it, ratio)
    c=channel_se_block(it)
    return Add()([c,s])



"""My Enet model. Uses 101x1 input and has 3|2|10|2|4 stack of squeeze blocks.
    Uses stride 1 pool for every residual connection"""
sq=12
sq_ratio=2
def input_layer(inputs, filters):
    passthroughs=inputs
    conv=Conv2D(filters-1, (3,3), padding='same')(inputs)
    cat=concatenate([conv, passthroughs])
    return cat
def bottleneck(inputs, filters, mode='regular', dilation=1, downsample=False,
              upsample=False, padadd=0, down=False, flatpool=False, dropout=0.12):
    #Picking Middle Layer
    if not down:
        down=filters//sq_ratio
    if filters<=down:
        middlefilters=down
    else:
        middlefilters=down*int(np.sqrt(filters/down))    
    #Convolution path
    squeezed=Conv2D(down, (1,1), use_bias=False)(inputs)
    if mode=='regular':
        conv=Conv2D(middlefilters, (3,3), padding='same', dilation_rate=dilation)(squeezed)
    elif mode=='asymmetric':
        conv0=Conv2D(middlefilters, (5,1),padding='same')(squeezed)
        conv=Conv2D(middlefilters, (1,5),padding='same')(conv0)
    upped=Conv2D(filters, (1,1), use_bias=False)(conv)
    batched=BatchNormalization()(upped)
    activated=keras.layers.PReLU(shared_axes=[1,2])(batched)
    
    #MaxPool pass through
    if flatpool:
        pool=MaxPooling2D((2,2),strides=(1,1), padding='same')(inputs)
    else:
        pool=inputs
    if padadd>0:
        firstpart=Lambda(lambda x: x[:,:,:,:-padadd])(activated)
        secondpart=Lambda(lambda x: x[:,:,:,-padadd:])(activated)
        smallcombine=Add()([firstpart, pool])
        combined=concatenate([smallcombine, secondpart], -1)
    elif padadd==0:
        combined=Add()([pool, activated])
    else:
        firstpart=Lambda(lambda x: x[:,:,:,:padadd])(pool)
        combined=Add()([activated, firstpart])
    if downsample==True:
        output=MaxPooling2D((2,2))(combined)
    elif upsample:
        output=UpSampling2D()(combined)
    else:
        output=combined
    if dropout is not None:
        dropped=keras.layers.Dropout(dropout)(output)
    else:
        dropped=output
    return dropped

def build_enet(inputs, sn, fm, dropout=False, flatpool=False):
    
    #Apply input layer
    initial=input_layer(inputs, sn)
    conv1_1=bottleneck(initial, fm, padadd=fm-sn, dropout=dropout)
    conv1_2=bottleneck(conv1_1, fm, dropout=dropout)
    #Downsample to 50x50
    down=bottleneck(conv1_2, 2*fm, downsample=True, padadd=fm, dropout=dropout)
    conv2_1=bottleneck(down, 2*fm, dropout=dropout)
    
    #Downsample to 25x25
    conv2_2=bottleneck(conv2_1, 4*fm, downsample=True, padadd=2*fm, dropout=dropout, flatpool=flatpool)
    conv2_3=bottleneck(conv2_2, 4*fm, mode='asymmetric', dropout=dropout)
    conv2_4=bottleneck(conv2_3, 4*fm, dilation=2, dropout=dropout, flatpool=flatpool)
    conv2_5=bottleneck(conv2_4, 4*fm, dilation=6, dropout=dropout)
    conv2_6=bottleneck(conv2_5, 4*fm, mode='asymmetric', dropout=dropout, flatpool=flatpool)
    conv2_7=bottleneck(conv2_6, 4*fm, dilation=2, dropout=dropout, flatpool=flatpool)
    conv2_8=bottleneck(conv2_7, 4*fm, dilation=6, dropout=dropout)
    conv2_9=bottleneck(conv2_8, 4*fm, dilation=2, dropout=dropout, flatpool=flatpool)
    conv2_10=bottleneck(conv2_9, 4*fm, mode='asymmetric', dropout=dropout)
    conv2_11=bottleneck(conv2_10, 4*fm, dilation=2, dropout=dropout, flatpool=flatpool)
    #Upsample to 50x50
    conv2_12=bottleneck(conv2_11, 2*fm, upsample=True, padadd=-2*fm, dropout=dropout)
    conv2_13=bottleneck(conv2_12, 2*fm, dilation=5, dropout=dropout)
    
    #Upsample to 100x100
    up=bottleneck(conv2_13, 2*fm, upsample=True, dropout=dropout)
    conv3_1=bottleneck(up, fm, padadd=-fm, dropout=dropout)
    conv3_2=bottleneck(conv3_1, fm, dropout=dropout)
    out=Conv2D(sn, (3,3), padding='same')(conv3_2)
    
    #Use one off-centered 2x2 upconvolution to transform 100x100 to 101x101
    fixedout=Conv2DTranspose(1, (2,2), padding='valid')(out)
    
    #Squash
    squashed=Activation('sigmoid')(fixedout)
    return squashed

""" Lovasz Softmax Loss """
# code download from: https://github.com/bermanmaxim/LovaszSoftmax
def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    gts = tf.reduce_sum(gt_sorted)
    intersection = gts - tf.cumsum(gt_sorted)
    union = gts + tf.cumsum(1. - gt_sorted)
    jaccard = 1. - intersection / union
    jaccard = tf.concat((jaccard[0:1], jaccard[1:] - jaccard[:-1]), 0)
    return jaccard


# --------------------------- BINARY LOSSES ---------------------------

def lovasz_hinge(logits, labels, per_image=True, ignore=None):
    """
    Binary Lovasz hinge loss
      logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
      per_image: compute the loss per image instead of per batch
      ignore: void class id
    """
    if per_image:
        def treat_image(log_lab):
            log, lab = log_lab
            log, lab = tf.expand_dims(log, 0), tf.expand_dims(lab, 0)
            log, lab = flatten_binary_scores(log, lab, ignore)
            return lovasz_hinge_flat(log, lab)
        losses = tf.map_fn(treat_image, (logits, labels), dtype=tf.float32)
        loss = tf.reduce_mean(losses)
    else:
        loss = lovasz_hinge_flat(*flatten_binary_scores(logits, labels, ignore))
    return loss


def lovasz_hinge_flat(logits, labels):
    """
    Binary Lovasz hinge loss
      logits: [P] Variable, logits at each prediction (between -\infty and +\infty)
      labels: [P] Tensor, binary ground truth labels (0 or 1)
      ignore: label to ignore
    """

    def compute_loss():
        labelsf = tf.cast(labels, logits.dtype)
        signs = 2. * labelsf - 1.
        errors = 1. - logits * tf.stop_gradient(signs)
        errors_sorted, perm = tf.nn.top_k(errors, k=tf.shape(errors)[0], name="descending_sort")
        gt_sorted = tf.gather(labelsf, perm)
        grad = lovasz_grad(gt_sorted)
        #loss = tf.tensordot(tf.nn.relu(errors_sorted), tf.stop_gradient(grad), 1, name="loss_non_void")
        loss = tf.tensordot(tf.nn.elu(errors_sorted), tf.stop_gradient(grad), 1, name="loss_non_void")
        return loss

    # deal with the void prediction case (only void pixels)
    loss = tf.cond(tf.equal(tf.shape(logits)[0], 0),
                   lambda: tf.reduce_sum(logits) * 0.,
                   compute_loss,
                   strict=True,
                   name="loss"
                   )
    return loss


def flatten_binary_scores(scores, labels, ignore=None):
    """
    Flattens predictions in the batch (binary case)
    Remove labels equal to 'ignore'
    """
    scores = tf.reshape(scores, (-1,))
    labels = tf.reshape(labels, (-1,))
    if ignore is None:
        return scores, labels
    valid = tf.not_equal(labels, ignore)
    vscores = tf.boolean_mask(scores, valid, name='valid_scores')
    vlabels = tf.boolean_mask(labels, valid, name='valid_labels')
    return vscores, vlabels

def lovasz_loss(y_true, y_pred):
    y_true, y_pred = K.cast(K.squeeze(y_true, -1), 'int32'), K.cast(K.squeeze(y_pred, -1), 'float32')
    #logits = K.log(y_pred / (1. - y_pred))
    logits = y_pred #Jiaxin
    loss = lovasz_hinge(logits, y_true, per_image = True, ignore = None)
    return loss
#IOU scoring metric
def get_iou_vector(A, B):
    A = np.squeeze(A) # new added 
    B = np.squeeze(B) # new added
    batch_size = A.shape[0]
    metric = []
    for batch in range(batch_size):
        t, p = A[batch]>0, B[batch]>0
        if np.count_nonzero(t) == 0 and np.count_nonzero(p) > 0:
            metric.append(0)
            continue
        if np.count_nonzero(t) >= 1 and np.count_nonzero(p) == 0:
            metric.append(0)
            continue
        if np.count_nonzero(t) == 0 and np.count_nonzero(p) == 0:
            metric.append(1)
            continue
        
        intersection = np.logical_and(t, p)
        union = np.logical_or(t, p)
        iou = (np.sum(intersection > 0)  )/ (np.sum(union > 0) )
        thresholds = np.arange(0.5, 1, 0.05)
        s = []
        for thresh in thresholds:
            s.append(iou > thresh)
        metric.append(np.mean(s))

    return np.mean(metric)

def my_iou_metric(lab, pred):
    return tf.py_func(get_iou_vector, [lab, pred>0.5], tf.float64)

def my_iou_metric_2(lab, pred):
    return tf.py_func(get_iou_vector, [lab, pred >0], tf.float64)

#Build model
def build_compile_model(sf=16, fm=48, lr=0.01, dropout=None):
    input_layer = Input((img_size_target, img_size_target, 1))
    output_layer = build_enet(input_layer, sf, fm, dropout=dropout)

    model1 = Model(input_layer, output_layer)

    c = optimizers.adam(lr = lr)
    model1.compile(loss="binary_crossentropy", optimizer=c, metrics=[my_iou_metric])
    return model1

def plot_history(history,metric_name):
    fig, (ax_loss, ax_score) = plt.subplots(1, 2, figsize=(15,5))
    ax_loss.plot(history.epoch, history.history["loss"], label="Train loss")
    ax_loss.plot(history.epoch, history.history["val_loss"], label="Validation loss")
    ax_loss.legend()
    ax_score.plot(history.epoch, history.history[metric_name], label="Train score")
    ax_score.plot(history.epoch, history.history["val_" + metric_name], label="Validation score")
    ax_score.legend()
    plt.savefig('historyfig')
def predict_result(model,x_test,img_size_target): # predict both orginal and reflect x
    x_test_reflect =  np.array([np.fliplr(x) for x in x_test])
    preds_test = model.predict(x_test).reshape(-1, img_size_target, img_size_target)
    preds_test2_refect = model.predict(x_test_reflect).reshape(-1, img_size_target, img_size_target)
    preds_test += np.array([ np.fliplr(x) for x in preds_test2_refect] )
    return preds_test/2

"""
model = build_compile_model(sf=10, fm=32, lr = 0.01)
model.summary()
"""
seed = 5
batch_size = 1

""" Data Augmentation """
nps1=np.random.RandomState(100)
nps2=np.random.RandomState(100)
def augmentData(x, mask, nps):
    if not mask:
        x=x+nps.normal(0, 0.01, x.shape)
        return x
    return np.float32(x > 0.5)
x_train, y_train, x_valid, y_valid =  get_cv_data(cv_index+1)

image_datagen = keras.preprocessing.image.ImageDataGenerator(horizontal_flip=True,
                     vertical_flip=True,
                     zoom_range=[1, 1.2],
                     preprocessing_function=lambda x: augmentData(x, False, nps1))
mask_datagen = keras.preprocessing.image.ImageDataGenerator(horizontal_flip=True,
                     vertical_flip=True,
                     zoom_range=[1, 1.2],
                     preprocessing_function=lambda x: augmentData(x, True, nps2))

image_datagen.fit(x_train, augment=True, seed=seed)
mask_datagen.fit(y_train, augment=True, seed=seed)

image_generator = image_datagen.flow(
    x_train,
    batch_size=batch_size,
    seed=seed)
mask_generator = mask_datagen.flow(
    y_train,
    batch_size=batch_size,
    seed=seed)
train_gen=zip(image_generator, mask_generator)
x_val_gen=image_datagen.flow(x_valid, batch_size=batch_size, seed=seed)
y_val_gen=mask_datagen.flow(y_valid, batch_size=batch_size, seed=seed)
val_gen=zip(x_val_gen, y_val_gen)
basic_name = 'Slim Enet v'+str(version)+'_cv'+str(cv_index+1)
print('############################################')
print(basic_name)
save_model_name = basic_name + '.model'
sq=32
tmodel = build_compile_model(sf=12, fm=64, lr = 0.0008, dropout=0.1)
"""
tmodel = load_model('Enet_v0',custom_objects={'my_iou_metric': my_iou_metric})
input_x = tmodel.layers[0].input

output_layer = tmodel.layers[-1].input
tmodel = Model(input_x, output_layer)
c = optimizers.adam(lr = 0.001)

# lovasz_loss need input range (-∞，+∞), so cancel the last "sigmoid" activation  
# Then the default threshod for pixel prediction is 0 instead of 0.5, as in my_iou_metric_2.
tmodel.compile(loss=lovasz_loss, optimizer=c, metrics=[my_iou_metric_2])
"""
model_checkpoint = ModelCheckpoint(basic_name_ori, monitor='val_my_iou_metric', 
                               mode = 'max', save_best_only=True, verbose=1)
model_checkpoint2 = ModelCheckpoint('epochly_'+basic_name_ori, monitor='val_my_iou_metric', 
                               mode = 'max', save_best_only=False, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_my_iou_metric', mode = 'max',
                              factor=0.5, patience=3, min_lr=0.0001, verbose=1)
"""
figure=plt.figure(figsize=(4,10), dpi=200)
num_to_save=10
for i, (testi, testm) in enumerate(train_gen):
    original_plot=plt.subplot(num_to_save,2,2*i+1)
    mask_plot=plt.subplot(num_to_save,2,2*i+1+1)
    original_plot.imshow(testi[0,:,:,0])
    mask_plot.imshow(testm[0,:,:,0])
    if i>=num_to_save-1:
        break
plt.savefig('./results/gentest.png')
"""

epochs = 100 #small number for demonstration
history = tmodel.fit_generator(train_gen,
                    validation_data=val_gen, 
                    epochs=epochs,
                    steps_per_epoch=x_train.shape[0],
                    validation_steps=x_valid.shape[0],
                    callbacks=[ model_checkpoint,reduce_lr, model_checkpoint2], 
                    verbose=1)
tmodel.summary()
"""
history=tmodel.fit(x_train, y_train,
                   validation_data=(x_valid, y_valid),
                   batch_size=4,
                   epochs=epochs,
                   callbacks=[model_checkpoint, reduce_lr])
"""
plot_history(history,'my_iou_metric')

#tmodel.load_weights(save_model_name)
resstring=' Results: Train: %s Val %s' % (min(history.history['loss']), min(history.history['val_loss']))
preds_valid = predict_result(tmodel,x_valid,img_size_target)
#ious[cv_index] = get_iou_vector(y_valid, (preds_valid > 0.5))
print(preds_valid)
num_to_save=15
val_preds=tmodel.predict(x_train[:num_to_save],
    batch_size=1)
figure=plt.figure(figsize=(4,10), dpi=200)
for i in range(num_to_save):
    original_plot=plt.subplot(num_to_save,3,3*i+1)
    pred_plot=plt.subplot(num_to_save,3,3*i+1+1)
    mask_plot=plt.subplot(num_to_save,3,3*i+2+1)
    original_plot.imshow(x_train[i,:,:,0])
    pred_plot.imshow(val_preds[i,:,:,0])
    mask_plot.imshow(y_train[i,:,:,0])
if os.path.exists('./results/plot.png'):
    nextnum=len(os.listdir('./results/old'))
    shutil.move('./results/plot.png', './results/old/plot'+str(nextnum)+'.png')
plt.title=resstring
plt.savefig('./results/plot.png')
# training

    
#model1.summary()
"""
used for converting the decoded image to rle mask
Fast compared to previous one
"""
def rle_encode(im):
    '''
    im: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = im.flatten(order = 'F')
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

x_test = np.array([(np.array(load_img(test_dir+"images/{}.png".format(idx), grayscale = True))) / 255 for idx in test_df.index].reshape(-1, img_size_target, img_size_target, 1))

# average the predictions from different folds
t1 = time.time()
preds_test = np.zeros(np.squeeze(x_test).shape)
for cv_index in range(cv_total):
    basic_name = 'Unet_resnet_v'+str(version)+'_cv'+str(cv_index+1)
    tmodel.load_weights(basic_name + '.model')
    preds_test += predict_result(tmodel,x_test,img_size_target) /cv_total
    
t2 = time.time()
print("Usedtime = "+str(t2-t1)+" s")

t1 = time.time()
threshold  = 0.5 # some value in range 0.4- 0.5 may be better 
pred_dict = {idx: rle_encode(np.round(preds_test[i]) > threshold) for i, idx in enumerate(test_df.index.values)}
t2 = time.time()

#print(f"Usedtime = {t2-t1} s")

sub = pd.DataFrame.from_dict(pred_dict,orient='index')
sub.index.names = ['id']
sub.columns = ['rle_mask']
sub.to_csv(submission_file)

t_finish = time.time()
#print(f"Kernel run time = {(t_finish-t_start)/3600} hours")

K.zeros((None, 10))

