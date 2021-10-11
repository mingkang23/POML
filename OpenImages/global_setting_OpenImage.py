# -*- coding: utf-8 -*-
"""
Created on Sat Jul 21 17:10:18 2018

@author: badat
"""
import D_utility
#%%
path = './'
labelmap_path = path+'data/OpenImages/2017_11/classes-trainable.txt'
dict_path =  path+'data/OpenImages/2017_11/class-descriptions.csv'
model_path =  path+'result/Pretrained_Openimages/model_n_feature 20000.ckpt'


record_path = path+'TFRecords/train_feature.tfrecords'
validation_path = path+'TFRecords/validation_feature.tfrecords'
test_path = path+'TFRecords/test_feature.tfrecords'


batch_size = 32#1000#
learning_rate_base = 0.0001
e2e_learning_rate_base = 1e-5
e2e_limit_learning_rate = 1.25e-9
thresold_coeff = 1e-3
limit_learning_rate = 1.25e-4
decay_rate_cond = 0.8
signal_strength = 0.1
report_interval=1000
early_stopping = True
n_cycles = 3448376//batch_size#D_utility.count_records(record_path)//batch_size

