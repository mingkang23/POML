import os
import numpy as np
import time
import math
from importer import *

import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import average_precision_score
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.utils.class_weight import compute_class_weight
from data import DATASETS

from os.path import join


parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=0, help='random-seed ')
parser.add_argument('--ratio', type=int, default=5, help='label ratio')
parser.add_argument('--lr', type=float, default=0.01, help='Learning Rate')
parser.add_argument('--lr_limit', type=float, default=0.0001, help='Learning Rate')
parser.add_argument('--lr_ae', type=float, default=0.01, help='Learning Rate of AE')
parser.add_argument('--weight_decay', type=float, default=0.00, help='Weight Decay')
parser.add_argument('--hyper_semi',type=float, default=0.000,help='hyper of semi-loss')
parser.add_argument('--max_epoch', type=int, default=20,help='max num of epoch')
parser.add_argument('--num_class', type=int, default=312,help='num of class')
parser.add_argument('--IMG_SIZE', type=int, default=224,help='image_size')
parser.add_argument('--batch_size', type=int, default=32,help='batch_size')
parser.add_argument('--dataset_name', type=str, default='voc',help='name of dataset')
parser.add_argument('--margin',type=float, default=1.0,help='margin')
parser.add_argument('--strength',type=float, default=0.3,help='exp strength')

FLAGS = parser.parse_args()

seed = FLAGS.seed
label_ratio = FLAGS.ratio
lr = FLAGS.lr
lr_ae = FLAGS.lr_ae
lr_limit = FLAGS.lr_limit
max_epoch = FLAGS.max_epoch
num_class = FLAGS.num_class
dataset_name = FLAGS.dataset_name
weight_decay = FLAGS.weight_decay
IMG_SIZE = FLAGS.IMG_SIZE
BATCH_SIZE = FLAGS.batch_size
hyper_semi =  FLAGS.hyper_semi
margin =  FLAGS.margin
strength =  FLAGS.strength

tf.random.set_seed(seed)
os.environ['TF_DETERMINISTIC_OPS'] = '1'

# tfds.disable_progress_bar()

dataset = DATASETS['fundus']()

train_batches = dataset.train.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
valid_batches = dataset.valid.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
test_batches = dataset.test.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
index = [10,20,40,60,80,100]



if (label_ratio==1) or (label_ratio==0):
    weight_decay = 0.001


for element in test_batches.take(1):
    img_id_batch = element['img_id']
    image_batch = element['img']
    label_batch = element['label']
    attribute_batch = element['attribute']

    pass

IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

def cal_map(test_label, estimated_label):
    label=test_label
    estimation=estimated_label
    av=average_precision_score(label, estimation)
    return av

def rampup(epoch):
    if epoch < 10:
        p = max(0.0, float(epoch)) / float(10)
        p = 1.0 - p
        return math.exp(-p * p * 5.0)
    else:
        return 1.0

def fearture_extractor(inputs, model):

    regularizer_last = tf.keras.regularizers.l2(weight_decay)

    x = model(inputs)

    # Classifier
    feature_batch = tf.keras.layers.GlobalAveragePooling2D()(x)
    prediction_batch = tf.keras.layers.Dense(num_class,kernel_regularizer=regularizer_last)(feature_batch)
    output_batch = tf.keras.activations.sigmoid(prediction_batch)

    return tf.keras.Model(inputs, outputs=[feature_batch, prediction_batch, output_batch])

def ResNet_model(weights='imagenet', kmax=30, kmin=30):
    inputs = tf.keras.layers.Input(shape=[IMG_SIZE,IMG_SIZE,3])

    base_model = tf.keras.applications.ResNet101(include_top=False, weights=weights, input_tensor=None, input_shape=(IMG_SIZE, IMG_SIZE, 3), pooling=None)

    
    return fearture_extractor(inputs, base_model)

Classifier = ResNet_model()
Classifier.summary()

[feature_batch, prediction_batch, output_batch] = Classifier(image_batch,training=False)
BCE_error = tf.keras.losses.BinaryCrossentropy(from_logits=False)


def BAE_model():

    lab = tf.keras.layers.Input(shape=[num_class], name='label_vector')
    feat = tf.keras.layers.Input(shape=[2048], name='feature_vector')
    regularizer = tf.keras.regularizers.l2(weight_decay)

    lab_feat = tf.keras.layers.concatenate([lab, feat]) # (bs, 256, 256, channels*2)

    x = tf.keras.layers.Dense(2048,kernel_regularizer=regularizer)(lab_feat)
    # x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)

    x = tf.keras.layers.Dense(2048,kernel_regularizer=regularizer)(x)
    # x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)

    x = tf.keras.layers.Dense(1024,kernel_regularizer=regularizer)(x)
    # x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)

    x = tf.keras.layers.Dense(1024,kernel_regularizer=regularizer)(x)
    # x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)

    x = tf.keras.layers.Dense(1024, activation='tanh',kernel_regularizer=regularizer)(x)

    x_label = tf.keras.layers.Dense(num_class,kernel_regularizer=regularizer)(x)
    x_feature = tf.keras.layers.Dense(2048,kernel_regularizer=regularizer)(x)

    return tf.keras.Model(inputs=[lab,feat], outputs=[lab, feat, x_label, x_feature])

BAE = BAE_model()
BAE.summary()

in_label, in_feat, out_label, out_feat = BAE([attribute_batch[:,:,-1],feature_batch],training=False)

CNN_optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr, rho=0.9, momentum=0.9, epsilon=1.0)
AE_optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr_ae, rho=0.9, momentum=0.9, epsilon=1.0)
AE_optimizer_FT = tf.keras.optimizers.RMSprop(learning_rate=0.01, rho=0.9, momentum=0.9, epsilon=1.0)
###############################  Training  #####################################
EPOCHS = max_epoch

import datetime
log_dir="logs/"

summary_writer = tf.summary.create_file_writer(log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

@tf.function
def train_step(image_batch,label_batch,epoch):
  with tf.GradientTape() as cl_tape, tf.GradientTape() as ae_tape:
    [feature_batch,prediction_batch,output_batch] = Classifier(image_batch, training=False)
    decoded_batch=tf.cast(output_batch > 0.5, dtype=tf.float32)
    cnn_accuracy = tf.reduce_mean(tf.cast(tf.equal(decoded_batch, label_batch), dtype=tf.float32))
    regularization_loss = tf.math.add_n(Classifier.losses)

    labels_binary = tf.math.divide(label_batch+1,2)
    mask_batch = tf.abs(label_batch)

    mask_batch_ae = tf.abs(label_batch + 2)

    weighted_mask =  mask_batch_ae * mask_batch

    label_batch_mc = label_batch
    masked_label_batch = tf.math.multiply(label_batch_mc, mask_batch)

    mask_zero = tf.zeros_like(masked_label_batch, tf.float32)

    in_label, in_feat, out_label, out_feat= BAE([mask_zero,feature_batch], training=True)
    # out_label = tf.clip_by_value(out_label, clip_value_min=-1, clip_value_max=1)

    regularization_loss_ae = tf.math.add_n(BAE.losses)

    out_label_ae = (out_label + 1)/2

    ae_decoded_batch = tf.cast(out_label > 0.0, dtype=tf.float32)
    ae_accuracy = tf.reduce_mean(tf.cast(tf.equal(ae_decoded_batch, label_batch), dtype=tf.float32))

    batch_size = tf.shape(feature_batch)[0]


    hard_label = tf.cast(tf.reduce_sum(tf.math.multiply(labels_binary[:batch_size // 2], labels_binary[batch_size // 2:2*(batch_size // 2)]),axis=1) > 15.0, dtype=tf.float32)



    D = tf.reduce_mean(tf.square(feature_batch[:batch_size // 2] - feature_batch[batch_size // 2:2*(batch_size // 2)]), axis=1)
    D = tf.clip_by_value(D, clip_value_min=1e-16, clip_value_max=10)
    D_sq = tf.sqrt(D)

    pos = D * hard_label
    neg = (1. - hard_label) * tf.square(tf.maximum(margin- D_sq, 0))
    semi_loss = tf.reduce_mean(pos+neg)

    BCE_loss = tf.compat.v1.losses.sigmoid_cross_entropy(multi_class_labels=labels_binary, logits=prediction_batch,weights=mask_batch)
    AE_loss = tf.reduce_sum(tf.math.pow(label_batch_mc - out_label, 2) * weighted_mask)/tf.reduce_sum(mask_batch) + 0.1* tf.reduce_mean(tf.math.pow(in_feat - out_feat, 2))

  Classifier_gradients = cl_tape.gradient(BCE_loss, Classifier.trainable_variables)
  CNN_optimizer.apply_gradients(zip(Classifier_gradients, Classifier.trainable_variables))
  # BAE_gradients = ae_tape.gradient(AE_loss, BAE.trainable_variables)
  # AE_optimizer.apply_gradients(zip(BAE_gradients, BAE.trainable_variables))

  return cnn_accuracy, ae_accuracy, BCE_loss, AE_loss, output_batch,out_label_ae, semi_loss

@tf.function
def train_step_ft(image_batch,label_batch,epoch):
  with tf.GradientTape() as cl_tape, tf.GradientTape() as ae_tape:
    [feature_batch,prediction_batch,output_batch] = Classifier(image_batch, training=False)
    decoded_batch=tf.cast(output_batch > 0.5, dtype=tf.float32)
    cnn_accuracy = tf.reduce_mean(tf.cast(tf.equal(decoded_batch, label_batch), dtype=tf.float32))
    regularization_loss = tf.math.add_n(Classifier.losses)

    labels_binary = tf.math.divide(label_batch+1,2)
    mask_batch = tf.abs(label_batch)

    mask_batch_ae = tf.abs(label_batch + 2)

    weighted_mask =  mask_batch_ae * mask_batch

    label_batch_mc = label_batch
    masked_label_batch = tf.math.multiply(label_batch_mc, mask_batch)

    mask_zero = tf.zeros_like(masked_label_batch, tf.float32)

    in_label, in_feat, out_label, out_feat= BAE([mask_zero,feature_batch], training=True)
    # out_label = tf.clip_by_value(out_label, clip_value_min=-1, clip_value_max=1)

    regularization_loss_ae = tf.math.add_n(BAE.losses)

    out_label_ae = (out_label + 1)/2

    ae_decoded_batch = tf.cast(out_label > 0.0, dtype=tf.float32)
    ae_accuracy = tf.reduce_mean(tf.cast(tf.equal(ae_decoded_batch, label_batch), dtype=tf.float32))

    batch_size = tf.shape(feature_batch)[0]
    hard_label = tf.cast(tf.reduce_sum(tf.math.multiply(labels_binary[:batch_size // 2], labels_binary[batch_size // 2:2*(batch_size // 2)]),axis=1) > 15.0, dtype=tf.float32)



    D = tf.reduce_mean(tf.square(feature_batch[:batch_size // 2] - feature_batch[batch_size // 2:2*(batch_size // 2)]), axis=1)
    D = tf.clip_by_value(D, clip_value_min=1e-16, clip_value_max=10)
    D_sq = tf.sqrt(D)

    pos = D * hard_label
    neg = (1. - hard_label) * tf.square(tf.maximum(margin- D_sq, 0))
    semi_loss = tf.reduce_mean(pos+neg)

    BCE_loss = tf.compat.v1.losses.sigmoid_cross_entropy(multi_class_labels=labels_binary, logits=prediction_batch,weights=mask_batch) +regularization_loss
    AE_loss = tf.reduce_sum(tf.math.pow(label_batch_mc - out_label, 2) * weighted_mask)/tf.reduce_sum(mask_batch) + 0.1* tf.reduce_mean(tf.math.pow(in_feat - out_feat, 2)) + regularization_loss_ae

  BAE_gradients = ae_tape.gradient(AE_loss, BAE.trainable_variables)
  AE_optimizer_FT.apply_gradients(zip(BAE_gradients, BAE.trainable_variables))

  return cnn_accuracy, ae_accuracy, BCE_loss, AE_loss, output_batch,out_label_ae, semi_loss


@tf.function
def test_batch(image_batch,label_batch, epoch):
  [feature_batch, prediction_batch, output_batch] = Classifier(image_batch, training=False)
  decoded_batch=tf.cast(output_batch > 0.5, dtype=tf.float32)
  cnn_accuracy = tf.reduce_mean(tf.cast(tf.equal(decoded_batch, label_batch), dtype=tf.float32))

  labels_binary = tf.math.divide(label_batch + 1, 2)
  mask_batch = tf.abs(label_batch)

  batch_size = tf.shape(feature_batch)[0]

  label_batch_mc = label_batch
  masked_label_batch = tf.math.multiply(label_batch_mc, mask_batch)

  mask_zero = tf.zeros_like(masked_label_batch, tf.float32)

  in_label, in_feat, out_label, out_feat = BAE([mask_zero, feature_batch], training=False)
  # out_label = tf.clip_by_value(out_label, clip_value_min=-1, clip_value_max=1)

  out_label_ae = (out_label + 1) / 2

  ae_decoded_batch = tf.cast(out_label > 0.0, dtype=tf.float32)
  ae_accuracy = tf.reduce_mean(tf.cast(tf.equal(ae_decoded_batch, label_batch), dtype=tf.float32))

  hard_label = tf.cast(tf.reduce_sum(tf.math.multiply(label_batch[:batch_size // 2], label_batch[batch_size // 2:2 * (batch_size // 2)]),axis=1) > 0.0, dtype=tf.float32)

  D = tf.reduce_mean(tf.square(feature_batch[:batch_size // 2] - feature_batch[batch_size // 2:2 * (batch_size // 2)]),axis=1)
  D = tf.clip_by_value(D, clip_value_min=1e-16, clip_value_max=10)
  D_sq = tf.sqrt(D)

  pos = D * hard_label
  neg = (1. - hard_label) * tf.square(tf.maximum(margin - D_sq, 0))
  semi_loss = tf.reduce_mean(pos + neg)

  BCE_loss_test =tf.compat.v1.losses.sigmoid_cross_entropy(multi_class_labels=labels_binary, logits=prediction_batch,weights=mask_batch)

  AE_loss_test = tf.reduce_mean(tf.math.pow(label_batch_mc - out_label, 2)) + tf.reduce_mean(tf.math.pow(in_feat - out_feat, 2))

  return cnn_accuracy, ae_accuracy,  BCE_loss_test, AE_loss_test, output_batch,out_label_ae, semi_loss

def fit_phase1(train_ds, valid_ds, test_ds,lr,lr_ae,epochs):
  m = 2
  m_ae = 2
  expon_moving_avg_old = np.inf
  expon_moving_avg_new = 0
  expon_moving_avg_old_ae = np.inf
  expon_moving_avg_new_ae = 0
  res_mAP = []

  for epoch in range(epochs):
    # Train
    acc_train = []
    acc_train_ae = []
    cnn_loss = []
    ae_loss = []
    semi_loss = []

    for n, element in train_ds.enumerate():
      image_batch = element['img']
      label_batch = element['attribute'][:,:,label_ratio]
      label_batch_true = element['attribute'][:,:,-1]


      accuracy_train, accuracy_train_ae, BCE_loss, AE_loss, output_batch, output_batch_ae, SEMI_loss = train_step(image_batch, label_batch, epoch)
      acc_train.append(accuracy_train)
      acc_train_ae.append(accuracy_train_ae)
      cnn_loss.append(BCE_loss)
      ae_loss.append(AE_loss)
      semi_loss.append(SEMI_loss)

      if n == 0:
        estimated_label = np.array(output_batch)
        train_label = np.array(label_batch_true)
        estimated_label_ae = np.array(output_batch_ae)
      else:
        estimated_label = np.append(estimated_label,np.array(output_batch),axis=0)
        train_label = np.append(train_label,np.array(label_batch_true),axis=0)
        estimated_label_ae = np.append(estimated_label_ae,np.array(output_batch_ae),axis=0)

    map = cal_map(train_label,estimated_label)
    map_ae = cal_map(train_label,estimated_label_ae)
    print ('Training at Epoch %d Loss_CNN = %4f Loss_AE = %4f Loss_SEMI = %4f MAP = %4f MAP-AE = %4f '%(epoch, np.mean(cnn_loss), np.mean(ae_loss), np.mean(semi_loss), map, map_ae))

    acc_valid = []
    acc_valid_ae = []
    cnn_loss_valid = []
    ae_loss_valid = []
    semi_loss_valid = []

    for n, element in valid_ds.enumerate():

      image_batch = element['img']
      label_batch = element['attribute'][:,:,-1]

      accuracy_valid, accuracy_valid_ae, BCE_loss_valid, AE_loss_valid, output_batch, output_batch_ae, SEMI_loss_valid= test_batch(image_batch, label_batch, epoch)
      acc_valid.append(accuracy_valid)
      acc_valid_ae.append(accuracy_valid_ae)
      cnn_loss_valid.append(BCE_loss_valid)
      ae_loss_valid.append(AE_loss_valid)
      semi_loss_valid.append(SEMI_loss_valid)

      if n == 0:
        estimated_label = np.array(output_batch)
        test_label = np.array(label_batch)
        estimated_label_ae = np.array(output_batch_ae)
      else:
        estimated_label = np.append(estimated_label, np.array(output_batch), axis=0)
        test_label = np.append(test_label, np.array(label_batch), axis=0)
        estimated_label_ae = np.append(estimated_label_ae, np.array(output_batch_ae), axis=0)

    map_valid = cal_map(test_label, estimated_label)
    map_ae_valid = cal_map(test_label, estimated_label_ae)
    res_mAP.append(map_ae_valid)
    best_mAP = np.max(res_mAP)

    print('Valid at Epoch %d Loss_CNN = %4f Loss_AE = %4f Loss_SEMI = %4f MAP = %4f MAP-AE = %4f MAP-BEST = %4f' % (epoch, np.mean(cnn_loss_valid), np.mean(ae_loss_valid),np.mean(semi_loss_valid), map_valid, map_ae_valid, best_mAP))
    expon_moving_avg_old = expon_moving_avg_new
    expon_moving_avg_new = expon_moving_avg_new * (1 - strength) + map_valid * strength

    expon_moving_avg_old_ae = expon_moving_avg_new_ae
    expon_moving_avg_new_ae = expon_moving_avg_new_ae * (1 - strength) + map_ae_valid * strength

    if expon_moving_avg_new_ae<expon_moving_avg_old_ae and lr_ae > lr_limit and m_ae <= 0:
        lr_ae = lr_ae * 0.8
        AE_optimizer.learning_rate.assign(lr_ae)
        print('learning_rate of BAE = %4f', lr_ae)
        m_ae = 2

    if expon_moving_avg_new<expon_moving_avg_old and lr > lr_limit and m <= 0:
        lr = lr * 0.8
        CNN_optimizer.learning_rate.assign(lr)
        print('learning_rate of CNN = %4f', lr)
        m = 2

    m -= 1
    m_ae -= 1

    if map_ae_valid >= best_mAP:
        acc_test = []
        acc_test_ae = []
        cnn_loss_test = []
        ae_loss_test = []
        semi_loss_test = []

        for n, element in test_ds.enumerate():

          image_batch = element['img']
          label_batch = element['attribute'][:,:,-1]

          accuracy_test, accuracy_test_ae, BCE_loss_test, AE_loss_test, output_batch, output_batch_ae, SEMI_loss_test = test_batch(image_batch, label_batch, epoch)
          acc_test.append(accuracy_test)
          acc_test_ae.append(accuracy_test_ae)
          cnn_loss_test.append(BCE_loss_test)
          ae_loss_test.append(AE_loss_test)
          semi_loss_test.append(SEMI_loss_test)

          if n == 0:
            estimated_label = np.array(output_batch)
            test_label = np.array(label_batch)
            estimated_label_ae = np.array(output_batch_ae)
          else:
            estimated_label = np.append(estimated_label, np.array(output_batch), axis=0)
            test_label = np.append(test_label, np.array(label_batch), axis=0)
            estimated_label_ae = np.append(estimated_label_ae, np.array(output_batch_ae), axis=0)

        map = cal_map(test_label, estimated_label)
        map_ae = cal_map(test_label, estimated_label_ae)

        Classifier.save_weights('./Cub_init/CNN_partial_init_%d'%index[label_ratio])
        BAE.save_weights('./Cub_init/BAE_partial_init_%d'%index[label_ratio])

        print('Test at Epoch %d Loss_CNN = %4f Loss_AE = %4f Loss_SEMI = %4f MAP = %4f MAP-AE = %4f' % (epoch, np.mean(cnn_loss_test), np.mean(ae_loss_test),np.mean(semi_loss_test), map, map_ae))
        print()

def fit_phase2(train_ds, test_ds, epochs):

  Classifier.load_weights('./Cub_init/CNN_partial_init_%d' %index[label_ratio])
  # BAE.load_weights('./Cub_init/BAE_%d_conv' %index[label_ratio])
  res_mAP = []
  res_mAP_test_cnn = []
  res_mAP_test_ae = []

  for epoch in range(20):
    # Train
    acc_train = []
    acc_train_ae = []
    cnn_loss = []
    ae_loss = []
    semi_loss = []
    for n, element in train_ds.enumerate():
      image_batch = element['img']
      label_batch = element['attribute'][:,:,label_ratio]
      label_batch_true = element['attribute'][:,:,-1]


      accuracy_train, accuracy_train_ae, BCE_loss, AE_loss, output_batch, output_batch_ae, SEMI_loss = train_step_ft(image_batch, label_batch, epoch)

      acc_train.append(accuracy_train)
      acc_train_ae.append(accuracy_train_ae)
      cnn_loss.append(BCE_loss)
      ae_loss.append(AE_loss)
      semi_loss.append(SEMI_loss)

      if n == 0:
        estimated_label = np.array(output_batch)
        train_label = np.array(label_batch_true)
        estimated_label_ae = np.array(output_batch_ae)
      else:
        estimated_label = np.append(estimated_label,np.array(output_batch),axis=0)
        train_label = np.append(train_label,np.array(label_batch_true),axis=0)
        estimated_label_ae = np.append(estimated_label_ae,np.array(output_batch_ae),axis=0)

    map = cal_map(train_label,estimated_label)
    map_ae = cal_map(train_label,estimated_label_ae)
    print ('Training at Epoch %d Loss_CNN = %4f Loss_AE = %4f Loss_SEMI = %4f MAP = %4f MAP-AE = %4f '%(epoch, np.mean(cnn_loss), np.mean(ae_loss), np.mean(semi_loss), map, map_ae))

    acc_test = []
    acc_test_ae = []
    cnn_loss_test = []
    ae_loss_test = []
    semi_loss_test = []

    for n, element in test_ds.enumerate():

      image_batch = element['img']
      label_batch = element['attribute'][:,:,-1]

      accuracy_test, accuracy_test_ae, BCE_loss_test, AE_loss_test, output_batch, output_batch_ae, SEMI_loss_test = test_batch(image_batch, label_batch, epoch)
      acc_test.append(accuracy_test)
      acc_test_ae.append(accuracy_test_ae)
      cnn_loss_test.append(BCE_loss_test)
      ae_loss_test.append(AE_loss_test)
      semi_loss_test.append(SEMI_loss_test)

      if n == 0:
        estimated_label = np.array(output_batch)
        test_label = np.array(label_batch)
        estimated_label_ae = np.array(output_batch_ae)
      else:
        estimated_label = np.append(estimated_label, np.array(output_batch), axis=0)
        test_label = np.append(test_label, np.array(label_batch), axis=0)
        estimated_label_ae = np.append(estimated_label_ae, np.array(output_batch_ae), axis=0)

    map = cal_map(test_label, estimated_label)
    map_ae = cal_map(test_label, estimated_label_ae)
    res_mAP_test_cnn.append(map)
    res_mAP_test_ae.append(map_ae)
    best_mAP_cnn = np.max(res_mAP_test_cnn)
    best_mAP_ae = np.max(res_mAP_test_ae)

    BAE.save_weights('./Cub_init/BAE_partial_init_%d' % index[label_ratio])

    print('Test at Epoch %d Loss_CNN = %4f Loss_AE = %4f Loss_SEMI = %4f MAP = %4f MAP-AE = %4f MAP_BEST = %4f MAP-AE_BEST = %4f' % (epoch,  np.mean(cnn_loss_test), np.mean(ae_loss_test),np.mean(semi_loss_test), map, map_ae, best_mAP_cnn, best_mAP_ae))
    print()


fit_phase1(train_batches,valid_batches, test_batches,lr,lr_ae,EPOCHS)
fit_phase2(train_batches, test_batches, EPOCHS)