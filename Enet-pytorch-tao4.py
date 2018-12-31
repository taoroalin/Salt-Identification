# -*- coding: utf-8 -*-
#!unzip  drive/My\ Drive/kaggle/trainimages.zip -d drive/My\ Drive/kaggle/
"""### Slim Enet with spatial dropout and augmentation, StratifiedKFold 
Uses efficient residual pooling, squeeze 1x1 convolutions, dilations and asymmetric convolutions
Uses pipeline from:
  https://www.kaggle.com/shaojiaxin/u-net-with-simple-resnet-blocks-v2-new-loss
          Whole script designed to run on headless server
"""
# Basics
import os
import time
import shutil
import pandas as pd
import numpy as np
from sys import stdout
from vat import VATLoss
# Plotting
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
import seaborn as sns
sns.set_style("white")

# Preprocessing
import cv2
from keras.preprocessing.image import  load_img, ImageDataGenerator
from sklearn.model_selection import StratifiedKFold

# Mode Library
import se_enet1 as se_enet

# PyTorch deep learning framework
import torch
import torch.nn as nn
import torch.nn.functional as F
"""Add your own data directory to the long list!"""
#datadir='drive/My Drive/kaggle/'
#datadir="./data/"
#datadir='/Users/tao/.kaggle/competitions/tgs-salt-identification-challenge/'
datadir='/home/tao/.kaggle/competitions/tgs-salt-identification-challenge/'
train_dir=datadir + 'train/'
test_dir=datadir + 'test/'
cv_total = 5
#cv_index = 1 -5
version = 3
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

""" Lovasz Softmax Loss """
# code download from: https://github.com/bermanmaxim/LovaszSoftmax
def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    gts = torch.sum(gt_sorted)
    intersection = gts - torch.cumsum(gt_sorted)
    union = gts + torch.cumsum(1. - gt_sorted)
    jaccard = 1. - intersection / union
    jaccard = torch.concat((jaccard[0:1], jaccard[1:] - jaccard[:-1]), 0)
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
        labelsf = labels.astype(logits.dtype)
        signs = 2. * labelsf - 1.
        errors = 1. - logits * signs
        errors_sorted, perm = torch.topk(errors, k=errors.size()[0])
        gt_sorted = labelsf[perm]
        grad = lovasz_grad(gt_sorted)
        #loss = tf.tensordot(tf.nn.relu(errors_sorted), tf.stop_gradient(grad), 1, name="loss_non_void")
        loss = F.elu(errors_sorted)* grad
        return loss

    # deal with the void prediction case (only void pixels)
    if logits.size()[0]==0:
        loss=torch.sum(logits)
    else:
        loss=compute_loss()
    return loss


def flatten_binary_scores(scores, labels, ignore=None):
    """
    Flattens predictions in the batch (binary case)
    Remove labels equal to 'ignore'
    """
    scores = scores.reshape(-1)
    labels = labels.reshape(-1)
    if ignore is None:
        return scores, labels
    valid = labels = ignore
    vscores = scores * valid
    vlabels = labels * valid
    return vscores, vlabels

def lovasz_loss(y_true, y_pred):
    y_true, y_pred = torch.squeeze(y_true, -1).astype(torch.int32), torch.squeeze(y_pred, -1).float()
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
    return get_iou_vector(lab, pred>0.5)

def my_iou_metric_2(lab, pred):
    return get_iou_vector(lab, pred >0)

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
    x_test=torch.from_numpy(x_test).reshape((-1, 1, img_size_target, img_size_target))
    x_test_reflect =  np.array([np.fliplr(x) for x in x_test])
    preds_test = model(x_test).reshape(-1, img_size_target, img_size_target).numpy()
    preds_test2_refect = model.predict(x_test_reflect).reshape(-1, img_size_target, img_size_target)
    preds_test += np.array([ np.fliplr(x) for x in preds_test2_refect] )
    return preds_test/2

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

"""
model = build_compile_model(sf=10, fm=32, lr = 0.01)
model.summary()
"""
seed = 10
batch_size = 10

""" Data Augmentation """
nps1=np.random.RandomState(100)
nps2=np.random.RandomState(100)
def augmentData(x, mask, nps):
    if not mask:
        #x=x+nps.normal(0, 0.01, x.shape)
        return x
    return np.float32(x > 0.5)
x_train, y_train, x_valid, y_valid =  get_cv_data(cv_index+1)

image_datagen = ImageDataGenerator(horizontal_flip=True,
                     vertical_flip=True,
                     zoom_range=[1, 1.2],
                     data_format = "channels_last",
                     preprocessing_function=lambda x: augmentData(x, False, nps1))
mask_datagen = ImageDataGenerator(horizontal_flip=True,
                     vertical_flip=True,
                     zoom_range=[1, 1.2],
                     data_format = "channels_last",
                     preprocessing_function=lambda x: augmentData(x, True, nps2))

image_datagen.fit(x_train, augment=True, seed=seed)
mask_datagen.fit(y_train, augment=True, seed=seed)

# Train  Data Generators
image_generator = image_datagen.flow(x_train, batch_size=batch_size, seed=seed)
mask_generator = mask_datagen.flow(y_train, batch_size=batch_size, seed=seed)
train_gen=zip(image_generator, mask_generator)

# Validation Data Generators
x_val_gen=image_datagen.flow(x_valid, batch_size=batch_size, seed=seed)
y_val_gen=mask_datagen.flow(y_valid, batch_size=batch_size, seed=seed)
val_gen=zip(x_val_gen, y_val_gen)

basic_name = 'PyTorch Enet v'+str(version)+'_cv'+str(cv_index+1)
print('############################################')
print(basic_name)

#tmodel = build_compile_model(sf=12, fm=64, lr = 0.0008, dropout=0.1)
epochs=200
model=se_enet.SE_Enet(1, 18, 1)
optimizer=torch.optim.Adam(model.parameters(), lr=0.001)
scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=25, verbose=1)
best_loss=0.5

cuda=True
if cuda:
    model=model.cuda()

checkpoint = torch.load('model_best3.pth.tar')
start_epoch = checkpoint['epoch']
best_loss = checkpoint['best_prec1']
best_tracking=checkpoint['tracking']
model.load_state_dict(checkpoint['state_dict'])
optimizer.load_state_dict(checkpoint['optimizer'])
#scheduler.load_state_dict(checkpoint['scheduler'])

#print("=> loaded checkpoint '{}' (epoch {})"
#      .format(start_epoch))
plt.plot(best_tracking)
print('bloss', best_loss)
print('epoch', start_epoch)
plt.savefig('tracking_plot')
if cuda:
    model=model.cuda()
"""
criterion=nn.BCELoss()
steps=x_train.shape[0]//batch_size
val_steps=x_valid.shape[0]//batch_size
valloss=0.5
tracking={'loss':[], 'val_loss':[]}
for epoch in range(epochs):
    floatloss=0.2
    for i in range(steps):
        x, y = next(train_gen)
        torchy=torch.from_numpy(y).reshape((-1, 1, 101, 101))
        torchx=torch.from_numpy(x).reshape((-1, 1, 101, 101))
        if cuda:
            torchx=torchx.cuda()
            torchy=torchy.cuda()
        optimizer.zero_grad()
        vat_loss = VATLoss(xi=0.01, eps=1.0, ip=1)
    
        # LDS should be calculated before the forward for cross entropy
        lds = vat_loss(model, torchx)
        output = model(torchx)
        pureloss=criterion(output, torchy)
        loss = pureloss + 1 * lds
        loss.backward()
        optimizer.step()
        floatloss*=0.9
        floatloss+=float(pureloss.detach().cpu().numpy())*0.1
        stdout.write("\r Epoch %d Iteration %d Train Loss %s LDS %s Val Loss %s" % (epoch, i, floatloss, lds.detach().cpu().numpy(), valloss))
        stdout.flush()
    valloss=0
    tempvalloss=0.2
    with torch.no_grad():
        for i in range(val_steps):
            x, y=next(val_gen)
            torchy=torch.from_numpy(y).reshape((-1, 1, 101, 101))
            torchx=torch.from_numpy(x).reshape((-1, 1, 101, 101))
            if cuda:
                torchx=torchx.cuda()
                torchy=torchy.cuda()
            preds=model(torchx)
            loss=criterion(preds, torchy)
            valloss+=loss.detach().cpu().numpy()
            ls=float(loss.detach().cpu().numpy())*0.1
            tempvalloss*=0.9
            tempvalloss+=0.1*ls
            stdout.write("\r Epoch %d Iteration %d Train Loss %s Val Loss %s" % (epoch, i, floatloss, tempvalloss))
    valloss /= val_steps
    is_best=False
    if valloss<best_loss:
        best_loss=valloss
        is_best=True
    save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'best_prec1': valloss,
        'tracking': tracking,
        'optimizer' : optimizer.state_dict(),
        'scheduler' : scheduler.state_dict(),
    }, is_best)
    tracking['loss'].append(floatloss)
    tracking['val_loss'].append(valloss)
    scheduler.step(valloss)
    print("\r Epoch %d Train Loss %s Val Loss %s" % (epoch, floatloss, valloss))
stdout.write("\n")

#ious[cv_index] = get_iou_vector(y_valid, (preds_valid > 0.5))
num_to_save=15
val_preds=np.zeros((num_to_save, img_size_target, img_size_target, 1))
for k in range(num_to_save):
    val_preds[i]=model(torch.from_numpy(x_train[i]).reshape((1, 1, 101, 101))).numpy()
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
plt.savefig('./results/plot.png')
# training
"""

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

testlist = [np.array(load_img(train_dir+"masks/{}.png".format(idx), grayscale=True)) / 255 for idx in train_df.index]
x_test = np.zeros((len(testlist), 1, 101, 101))
for i, asdf in enumerate(testlist()):
    x_test[i]=asdf

# average the predictions from different folds
t1 = time.time()
preds_test = np.zeros(np.squeeze(x_test).shape)
for cv_index in range(cv_total):
    basic_name = 'Unet_resnet_v'+str(version)+'_cv'+str(cv_index+1)
    model.load_weights(basic_name + '.model')
    preds_test += predict_result(model,x_test,img_size_target) /cv_total
    
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

