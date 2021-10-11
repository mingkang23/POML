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
from data_ft import DATASETS

from os.path import join


parser = argparse.ArgumentParser()

parser.add_argument('--ratio', type=float, default=0.1, help='label ratio')
parser.add_argument('--lr', type=float, default=0.01, help='Learning Rate')
parser.add_argument('--lr_limit', type=float, default=0.0001, help='Learning Rate')
parser.add_argument('--lr_ae', type=float, default=0.01, help='Learning Rate of AE')
parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight Decay')
parser.add_argument('--hyper_semi',type=float, default=0.05,help='hyper of semi-loss')
parser.add_argument('--max_epoch', type=int, default=32,help='max num of epoch')
parser.add_argument('--num_class', type=int, default=1000,help='num of class')
parser.add_argument('--IMG_SIZE', type=int, default=224,help='image_size')
parser.add_argument('--batch_size', type=int, default=16,help='batch_size')
parser.add_argument('--dataset_name', type=str, default='voc',help='name of dataset')
parser.add_argument('--margin',type=float, default=1.0,help='margin')
parser.add_argument('--strength',type=float, default=0.35,help='exp strength')

FLAGS = parser.parse_args()

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

dataset = DATASETS['fundus']()

train_batches = dataset.train.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
valid_batches = dataset.valid.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

for element in train_batches.take(1):
    image_batch = element['img']
    label_batch = element['label']
    print(label_batch)
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

def ResNet_model():
    inputs = tf.keras.layers.Input(shape=[IMG_SIZE,IMG_SIZE,None])
    base_model = tf.keras.applications.VGG16(include_top=True, weights='imagenet', input_tensor=None,input_shape=(IMG_SIZE, IMG_SIZE, 3), pooling=None)
    inputs_preprocess = tf.keras.applications.vgg16.preprocess_input(inputs, data_format=None)
    regularizer_last = tf.keras.regularizers.l2(weight_decay)

    model=tf.keras.Model(base_model.input, outputs=base_model.get_layer('fc2').output)

    feature_batch = model(inputs_preprocess)
    prediction_batch = tf.keras.layers.Dense(num_class,kernel_regularizer=regularizer_last)(feature_batch)

    output_batch = tf.keras.activations.sigmoid(prediction_batch)
    return tf.keras.Model(inputs, outputs=[feature_batch, prediction_batch, output_batch])

Classifier = ResNet_model()
Classifier.summary()
Classifier.load_weights('./Pretrained_COCO/CNN_lr_0_001_tf')

[feature_batch, prediction_batch, output_batch] = Classifier(image_batch,training=False)
BCE_error = tf.keras.losses.BinaryCrossentropy(from_logits=False)

def BAE_model():

    lab = tf.keras.layers.Input(shape=[num_class], name='label_vector')
    feat = tf.keras.layers.Input(shape=[4096], name='feature_vector')
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
    x_feature = tf.keras.layers.Dense(4096,kernel_regularizer=regularizer)(x)

    return tf.keras.Model(inputs=[lab,feat], outputs=[lab, feat, x_label, x_feature])

BAE = BAE_model()
BAE.summary()
BAE.load_weights('./Pretrained_COCO/BAE_lr_0_01_tf_pretrain')

in_label, in_feat, out_label, out_feat = BAE([label_batch,feature_batch],training=False)


CNN_optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr, rho=0.9, momentum=0.9, epsilon=1.0)
AE_optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr_ae, rho=0.9, momentum=0.9, epsilon=1.0)

###############################  Training  #####################################
EPOCHS = max_epoch

import datetime
log_dir="logs/"

summary_writer = tf.summary.create_file_writer(log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

@tf.function
def train_step(image_batch,label_batch):
  with tf.GradientTape() as cl_tape, tf.GradientTape() as ae_tape:
    [feature_batch,prediction_batch,output_batch] = Classifier(image_batch, training=True)
    decoded_batch=tf.cast(output_batch > 0.5, dtype=tf.float32)
    cnn_accuracy = tf.reduce_mean(tf.cast(tf.equal(decoded_batch, label_batch), dtype=tf.float32))

    label_batch_mc = 2*label_batch-1
    weighted_mask = tf.abs(label_batch_mc + 2)

    mask_zero = tf.zeros_like(label_batch_mc, tf.float32)
    mask_one = tf.ones_like(label_batch_mc, tf.float32)

    in_label, in_feat, out_label, out_feat= BAE([mask_zero,feature_batch], training=True)
    # out_label = tf.clip_by_value(out_label, clip_value_min=-1, clip_value_max=1)
    out_label_ae = (out_label + 1)/2

    regularization_loss_ae = tf.math.add_n(BAE.losses)

    ae_decoded_batch = tf.cast(out_label > 0.0, dtype=tf.float32)
    ae_accuracy = tf.reduce_mean(tf.cast(tf.equal(ae_decoded_batch, label_batch), dtype=tf.float32))

    batch_size = tf.shape(feature_batch)[0]

    hard_label = tf.cast(-1.*tf.keras.losses.cosine_similarity(out_label[:batch_size // 2],  out_label[batch_size // 2:2*(batch_size // 2)], axis=1)> 0.98, dtype=tf.float32)

    similar = -1. * tf.keras.losses.cosine_similarity(out_label[:batch_size // 2],out_label[batch_size // 2:2 * (batch_size // 2)], axis=1)


    D = tf.reduce_mean(tf.square(feature_batch[:batch_size // 2] - feature_batch[batch_size // 2:2*(batch_size // 2)]), axis=1)
    D = tf.clip_by_value(D, clip_value_min=1e-16, clip_value_max=10)
    D_sq = tf.sqrt(D)

    pos = D * hard_label
    neg = (1. - hard_label) * tf.square(tf.maximum(margin- D_sq, 0))
    semi_loss = tf.reduce_mean(pos+neg)

    BCE_loss = tf.compat.v1.losses.sigmoid_cross_entropy(multi_class_labels=label_batch, logits=prediction_batch, weights=mask_one) + hyper_semi * semi_loss

    AE_loss = tf.reduce_sum(tf.math.pow(label_batch_mc - out_label, 2) * weighted_mask)/tf.reduce_sum(mask_one) + 0.1*tf.reduce_mean(tf.math.pow(in_feat - out_feat, 2))

  Classifier_gradients = cl_tape.gradient(BCE_loss, Classifier.trainable_variables)
  CNN_optimizer.apply_gradients(zip(Classifier_gradients, Classifier.trainable_variables))
  BAE_gradients = ae_tape.gradient(AE_loss, BAE.trainable_variables)
  AE_optimizer.apply_gradients(zip(BAE_gradients, BAE.trainable_variables))

  return cnn_accuracy, ae_accuracy, BCE_loss, AE_loss, output_batch,out_label_ae, semi_loss, hard_label, D_sq, similar

@tf.function
def test_batch(image_batch,label_batch):
  [feature_batch, prediction_batch, output_batch] = Classifier(image_batch, training=False)
  decoded_batch=tf.cast(output_batch > 0.5, dtype=tf.float32)
  cnn_accuracy = tf.reduce_mean(tf.cast(tf.equal(decoded_batch, label_batch), dtype=tf.float32))


  batch_size = tf.shape(feature_batch)[0]

  label_batch_mc = 2*label_batch-1


  mask_zero = tf.zeros_like(label_batch_mc, tf.float32)
  mask_one = tf.ones_like(label_batch_mc, tf.float32)

  in_label, in_feat, out_label, out_feat = BAE([mask_zero, feature_batch], training=False)
  out_label = tf.clip_by_value(out_label, clip_value_min=-1, clip_value_max=1)

  out_label_ae = (out_label + 1) / 2

  ae_decoded_batch = tf.cast(out_label > 0.0, dtype=tf.float32)
  ae_accuracy = tf.reduce_mean(tf.cast(tf.equal(ae_decoded_batch, label_batch), dtype=tf.float32))

  # hard_label = tf.cast(tf.reduce_sum(tf.math.multiply(label_batch[:batch_size // 2], label_batch[batch_size // 2:2 * (batch_size // 2)]),axis=1) > 0.0, dtype=tf.float32)
  hard_label = tf.cast(-1. * tf.keras.losses.cosine_similarity(label_batch[:batch_size // 2],
                                                               label_batch[batch_size // 2:2 * (batch_size // 2)],
                                                               axis=1) > 0.2, dtype=tf.float32)
  D = tf.reduce_mean(tf.square(feature_batch[:batch_size // 2] - feature_batch[batch_size // 2:2 * (batch_size // 2)]),axis=1)
  D = tf.clip_by_value(D, clip_value_min=1e-16, clip_value_max=10)
  D_sq = tf.sqrt(D)

  pos = D * hard_label
  neg = (1. - hard_label) * tf.square(tf.maximum(margin - D_sq, 0))
  semi_loss = tf.reduce_mean(pos + neg)

  BCE_loss_test =tf.compat.v1.losses.sigmoid_cross_entropy(multi_class_labels=label_batch, logits=prediction_batch,weights=mask_one)

  AE_loss_test = tf.reduce_mean(tf.math.pow(label_batch_mc - out_label, 2)) + tf.reduce_mean(tf.math.pow(in_feat - out_feat, 2))

  return cnn_accuracy, ae_accuracy,  BCE_loss_test, AE_loss_test, output_batch,out_label_ae, semi_loss

def fit_phase1(train_ds, valid_ds,lr,epochs):
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
      label_batch = element['label']

      accuracy_train, accuracy_train_ae, BCE_loss, AE_loss, output_batch, output_batch_ae, SEMI_loss, hard_label, D_sq, similar = train_step(image_batch, label_batch)
      acc_train.append(accuracy_train)
      acc_train_ae.append(accuracy_train_ae)
      cnn_loss.append(BCE_loss)
      ae_loss.append(AE_loss)
      semi_loss.append(SEMI_loss)

#      print(hard_label,D_sq, similar)


      if n == 0:
        estimated_label = np.array(output_batch)
        train_label = np.array(label_batch)
        estimated_label_ae = np.array(output_batch_ae)
      else:
        estimated_label = np.append(estimated_label,np.array(output_batch),axis=0)
        train_label = np.append(train_label,np.array(label_batch),axis=0)
        estimated_label_ae = np.append(estimated_label_ae,np.array(output_batch_ae),axis=0)

        if (n % int(6400/16))==0:
            acc_valid = []
            acc_valid_ae = []
            cnn_loss_valid = []
            ae_loss_valid = []
            semi_loss_valid = []

            for n_v, element in valid_ds.enumerate():

                image_batch = element['img']
                label_batch = element['label']

                accuracy_valid, accuracy_valid_ae, BCE_loss_valid, AE_loss_valid, output_batch, output_batch_ae, SEMI_loss_valid = test_batch(
                    image_batch, label_batch)
                acc_valid.append(accuracy_valid)
                acc_valid_ae.append(accuracy_valid_ae)
                cnn_loss_valid.append(BCE_loss_valid)
                ae_loss_valid.append(AE_loss_valid)
                semi_loss_valid.append(SEMI_loss_valid)

                if n_v == 0:
                    estimated_label_v = np.array(output_batch)
                    test_label = np.array(label_batch)
                    estimated_label_ae_v = np.array(output_batch_ae)
                else:
                    estimated_label_v = np.append(estimated_label_v, np.array(output_batch), axis=0)
                    test_label = np.append(test_label, np.array(label_batch), axis=0)
                    estimated_label_ae_v = np.append(estimated_label_ae_v, np.array(output_batch_ae), axis=0)

            map_valid = cal_map(test_label, estimated_label_v)
            map_ae_valid = cal_map(test_label, estimated_label_ae_v)
            res_mAP.append(map_valid)
            best_mAP = np.max(res_mAP)

            print(
            'Valid at Epoch %d at iter %d Accuracy = %4f AE-Accuracy = %4f Loss_CNN = %4f Loss_AE = %4f Loss_SEMI = %4f MAP = %4f MAP-AE = %4f MAP-BEST = %4f' % (
            epoch, n_v,  np.mean(acc_valid), np.mean(acc_valid_ae), np.mean(cnn_loss_valid), np.mean(ae_loss_valid),
            np.mean(semi_loss_valid), map_valid, map_ae_valid, best_mAP))

            # if map_valid >= best_mAP:
            #     Classifier.save_weights('./checkpoints_initial_coco_supple/CNN_lr_0_001_tf')
            #     BAE.save_weights('./checkpoints_initial_coco_supple/BAE_lr_0_001_tf')
            #     print('Save the Best Model')
            #     print()
    map = cal_map(train_label, estimated_label)
    map_ae = cal_map(train_label, estimated_label_ae)
    print ('Training at Epoch %d, %d Accuracy = %4f  AE-Accuracy = %4f Loss_CNN = %4f Loss_AE = %4f Loss_SEMI = %4f MAP = %4f MAP-AE = %4f ' % (epoch, n, np.mean(acc_train), np.mean(acc_train_ae), np.mean(cnn_loss), np.mean(ae_loss), np.mean(semi_loss),map, map_ae))

    lr = lr * 0.5
    CNN_optimizer.learning_rate.assign(lr)
    print('learning_rate of CNN = %4f', lr)


    acc_valid = []
    acc_valid_ae = []
    cnn_loss_valid = []
    ae_loss_valid = []
    semi_loss_valid = []

    for n_v, element in valid_ds.enumerate():
      image_batch = element['img']
      label_batch = element['label']

      accuracy_valid, accuracy_valid_ae, BCE_loss_valid, AE_loss_valid, output_batch, output_batch_ae, SEMI_loss_valid= test_batch(image_batch, label_batch)
      acc_valid.append(accuracy_valid)
      acc_valid_ae.append(accuracy_valid_ae)
      cnn_loss_valid.append(BCE_loss_valid)
      ae_loss_valid.append(AE_loss_valid)
      semi_loss_valid.append(SEMI_loss_valid)

      if n_v == 0:
        estimated_label_v = np.array(output_batch)
        test_label = np.array(label_batch)
        estimated_label_ae_v = np.array(output_batch_ae)
      else:
        estimated_label_v = np.append(estimated_label_v, np.array(output_batch), axis=0)
        test_label = np.append(test_label, np.array(label_batch), axis=0)
        estimated_label_ae_v = np.append(estimated_label_ae_v, np.array(output_batch_ae), axis=0)

    map_valid = cal_map(test_label, estimated_label_v)
    map_ae_valid = cal_map(test_label, estimated_label_ae_v)
    res_mAP.append(map_valid)
    best_mAP = np.max(res_mAP)

    print('Valid at Epoch %d Accuracy = %4f AE-Accuracy = %4f Loss_CNN = %4f Loss_AE = %4f Loss_SEMI = %4f MAP = %4f MAP-AE = %4f MAP-BEST = %4f' % (epoch, np.mean(acc_valid), np.mean(acc_valid_ae), np.mean(cnn_loss_valid), np.mean(ae_loss_valid),np.mean(semi_loss_valid), map_valid, map_ae_valid, best_mAP))

    # if map_valid >= best_mAP:
    #     Classifier.save_weights('./checkpoints_initial_coco_supple/CNN_lr_0_001_tf')
    #     BAE.save_weights('./checkpoints_initial_coco_supple/BAE_lr_0_001_tf')
    #     print('Save the Best Model')
    #     print()

fit_phase1(train_batches, valid_batches, lr, EPOCHS)
