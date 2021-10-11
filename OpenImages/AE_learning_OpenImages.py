from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import pandas as pd
import os.path
import os
import numpy as np
import time
from nets import resnet_v1
from measurement import apk,compute_number_misclassified
import D_utility
import global_setting_OpenImage
import pdb
from tensorflow.contrib import slim
from sklearn.metrics import average_precision_score
from preprocessing import preprocessing_factory
import random

#%% override
global_setting_OpenImage.batch_size=32
global_setting_OpenImage.report_interval = 100
global_setting_OpenImage.e2e_n_cycles = 3448376//global_setting_OpenImage.batch_size
global_setting_OpenImage.signal_strength *= global_setting_OpenImage.report_interval/100
global_setting_OpenImage.e2e_learning_rate_base = 1e-9
global_setting_OpenImage.saturated_Thetas_model  = './result/Pretrained_Openimages/model.npz'
change_interval = 20000
n_batch = 8

is_G = True
is_nonzero_G = True
is_sum_1=True
is_optimize_all_G = False
#
is_use_batch_norm = True
capacity = -1
val_capacity = 20
dictionary_evaluation_interval=250
partition_size = 200
strength_identity = 1
idx_GPU=0

#22,28
train_data_path= '/home/dclab/hard/cvpr20_IMCL/image_data/OpenImages/train/'
validation_data_path= '/home/dclab/hard/cvpr20_IMCL/image_data/OpenImages/validation/'


DATA_DIR = './TFRecords_div/'
TRAIN_DATA_sparse = []

num_idx=np.arange(216)
random.shuffle(num_idx)
for j in range(216):
    TRAIN_DATA_sparse.append(DATA_DIR + 'train_feature_{}.tfrecords'.format(j))


os.environ["CUDA_VISIBLE_DEVICES"]="{}".format(idx_GPU)
template_name='e2e_asym_OpenImage_c_{}_f_{}_{}_{}_{}_signal_str_{}_{}_GPU_{}_thresCoeff_{}_c_{}_stamp_{}'
list_alphas_colaborative = [1]#
list_alphas_feature = [0.0]#0,0.5,1,2
global_step = tf.Variable(0, trainable=False,dtype=tf.float32)
learning_rate = tf.Variable(global_setting_OpenImage.e2e_learning_rate_base,trainable = False,dtype=tf.float32)
learning_rate_ae = tf.Variable(global_setting_OpenImage.learning_rate_base,trainable = False,dtype=tf.float32)
n_epoches = 1
report_length = global_setting_OpenImage.e2e_n_cycles*n_epoches//global_setting_OpenImage.report_interval +1 #in case that my lousy computation is wrong
patient=report_length//100
c = 2.0
is_save = True
parallel_iterations = 1
#%%
print('number of cycles {}'.format(global_setting_OpenImage.n_cycles))
print('number partition_size ',partition_size)
#%%
def compute_AP(Prediction,Label):
    num_class = Prediction.shape[1]
    ap=np.zeros(num_class)
    for idx_cls in range(num_class):
        prediction = np.squeeze(Prediction[:,idx_cls])
        label = np.squeeze(Label[:,idx_cls])
        mask = np.abs(label)==1
        if np.sum(label>0)==0:
            continue
        binary_label=np.clip(label[mask],0,1)
        ap[idx_cls]=average_precision_score(binary_label,prediction[mask])#AP(prediction,label,names)
    return ap
#%%
def collapse_Theta(data):
    Thetas_1 = data['Thetas_1']
    Thetas_f = data['Thetas_f']
    Theta_1 = Thetas_1[:,:,-1]
#        pdb.set_trace()
    # reserving perspective transformation
    theta_1_n_row =Theta_1.shape[0]
    Theta_1=np.concatenate((Theta_1,np.zeros((theta_1_n_row,1))),axis=1)
    Theta_1[-1,-1]=1
    #
    
    Theta_f = Thetas_f[:,:,-1]
    Theta = np.matmul(Theta_1,Theta_f)
    return Theta
#%% label mapping function
def LoadLabelMap(labelmap_path, dict_path):
  """Load index->mid and mid->display name maps.

  Args:
    labelmap_path: path to the file with the list of mids, describing
        predictions.
    dict_path: path to the dict.csv that translates from mids to display names.
  Returns:
    labelmap: an index to mid list
    label_dict: mid to display name dictionary
  """
  labelmap = [line.rstrip() for line in tf.gfile.GFile(labelmap_path)]

  label_dict = {}
  for line in tf.gfile.GFile(dict_path):
    words = [word.strip(' "\n') for word in line.split(',', 1)]
    label_dict[words[0]] = words[1]

  return labelmap, label_dict
#%%
labelmap, label_dict = LoadLabelMap(global_setting_OpenImage.labelmap_path, global_setting_OpenImage.dict_path)
list_label = []
for id_name in labelmap:
    list_label.append(label_dict[id_name])
n_class = len(list_label)
#%% Dataset
image_size = resnet_v1.resnet_v1_101.default_image_size
height = image_size
width = image_size
def PreprocessImage(image, network='resnet_v1_101'):
      # If resolution is larger than 224 we need to adjust some internal resizing
      # parameters for vgg preprocessing.
      preprocessing_kwargs = {}
      preprocessing_fn = preprocessing_factory.get_preprocessing(name=network, is_training=False)
      height = image_size
      width = image_size
      image = preprocessing_fn(image, height, width, **preprocessing_kwargs)
      image.set_shape([height, width, 3])
      return image

def read_img(img_id,data_path):
    compressed_image = tf.read_file(data_path+img_id+'.jpg', 'rb')
    image = tf.image.decode_jpeg(compressed_image, channels=3)
    processed_image = PreprocessImage(image)
    return processed_image

def read_raw_img(img_id,data_path):
    return tf.read_file(data_path+img_id+'.jpg','rb')

def parser_train(record):
    feature = {'img_id': tf.FixedLenFeature([], tf.string),
               'label': tf.FixedLenFeature([], tf.string)}
    
    parsed = tf.parse_single_example(record, feature)
    img_id =  parsed['img_id']
    label = tf.decode_raw( parsed['label'],tf.int32)
    img = read_raw_img(img_id,train_data_path)
    return img_id,img,label

def parser_validation(record):
    feature = {'img_id': tf.FixedLenFeature([], tf.string),
               'label': tf.FixedLenFeature([], tf.string)}
    
    parsed = tf.parse_single_example(record, feature)
    img_id =  parsed['img_id']
    label = tf.decode_raw( parsed['label'],tf.int32)
    img = read_raw_img(img_id,validation_data_path)
    return img_id,img,label
#%%
# def compute_feature_prediction_large_batch(img,is_silent = False):
#     prediction_l = []
#     feature_l = []
#     tic = time.clock()
#     for idx_partition in range(img.shape[0]//partition_size+1):
#         if not is_silent:
#             print('{}.'.format(idx_partition),end='')
#         prediction_partition,feature_partition = sess.run([Prediction,features_concat],{img_input_ph:img[idx_partition*partition_size:(idx_partition+1)*partition_size]})
#         prediction_l.append(prediction_partition)
#         feature_l.append(feature_partition)
#     if not is_silent:
#         print('time: ',time.clock()-tic)
#     prediction = np.concatenate(prediction_l)
#     feature = np.concatenate(feature_l)
#     #print()
#     return prediction,feature

def compute_feature_prediction_large_batch(img,labels):
    prediction_l = []
    prediction_l_AE = []
    feature_l = []
    for idx_partition in range(img.shape[0]//partition_size+1):
        print('{}.'.format(idx_partition),end='')
        prediction_partition,feature_partition,prediction_partition_ae= sess.run([Prediction,features_concat, outputs_labels],{img_input_ph:img[idx_partition*partition_size:(idx_partition+1)*partition_size],labels_ae:labels[idx_partition*partition_size:(idx_partition+1)*partition_size],index_point:partition_size})
        prediction_l.append(prediction_partition)
        prediction_l_AE.append(prediction_partition_ae)
        feature_l.append(feature_partition)
    print()
    prediction = np.concatenate(prediction_l)
    feature = np.concatenate(feature_l)
    prediction_ae = np.concatenate(prediction_l_AE)
    return prediction,feature,prediction_ae

def load_memory(iterator_next,size,capacity = -1):
    labels_l = []
    ids_l=[]
    imgs_l = []
    print('load memory')
    if capacity == -1:
        n_p = size//partition_size+1
    else:
        n_p = capacity
    for idx_partition in range(n_p):
        print('{}.'.format(idx_partition),end='')
        (img_ids_p,img_p,labels_p) = sess.run(iterator_next)
        labels_l.append(labels_p)
        ids_l.append(img_ids_p)
        imgs_l.append(img_p)
    print()
    labels = np.concatenate(labels_l)
    ids = np.concatenate(ids_l)
    imgs = np.concatenate(imgs_l)
    return ids,imgs,labels

def compute_feature_prediction_large_batch_iterator(iterator_next,size):
    prediction_l = []
    feature_l = []
    labels_l = []
    ids_l=[]
    print('compute large batch')
    for idx_partition in range(10):#range(size//partition_size+1):
        print('partition ',idx_partition)
        tic = time.clock()
        (img_ids_p,img_p,labels_p) = sess.run(iterator_next)
        print(time.clock()-tic)
        tic = time.clock()
        prediction_partition,feature_partition = sess.run([Prediction,features_concat],{img_input_ph:img_p})
        print(time.clock()-tic)
        prediction_l.append(prediction_partition)
        feature_l.append(feature_partition)
        labels_l.append(labels_p)
        ids_l.append(img_ids_p)
    prediction = np.concatenate(prediction_l)
    feature = np.concatenate(feature_l)
    labels = np.concatenate(labels_l)
    ids_l = np.concatenate(ids_l)
    return prediction,ids_l,feature,labels

def get_img_sparse_dict_support_v2(support_ids):
    imgs = []
    for s_id in support_ids:
        imgs.append(read_img(s_id.decode("utf-8"),train_data_path)[tf.newaxis,:,:,:])
    imgs = sess.run(imgs)
    return np.concatenate(imgs)
def get_img_sparse_dict_support(idx_support,iterator_next,size):
    imgs_l = []
    labels_l = []
    print('get img dict support')
    for idx_partition in range(size//partition_size+1):
        print('partition ',idx_partition)
        (img_ids_p,img_p,labels_p) = sess.run(iterator_next)
        min_idx = idx_partition*partition_size
        max_idx = min_idx+img_p.shape[0]
        selector = np.where((idx_support>=min_idx) & (idx_support<max_idx))
        imgs_l.append(img_p[selector])
        labels_l.append(labels_p[selector])
    imgs = np.concatenate(imgs_l)
    labels = np.concatenate(labels_l)
    return imgs,labels
#%% load in memory
sess = tf.InteractiveSession()#tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True))
g = tf.get_default_graph()
#%%
Theta = tf.get_variable('Theta',shape=[2049,n_class])
learning_rate_fh=tf.placeholder(dtype=tf.float32,shape=())
op_assign_learning_rate = learning_rate.assign(learning_rate_fh)

learning_rate_fh_ae=tf.placeholder(dtype=tf.float32,shape=())
op_assign_learning_rate_ae = learning_rate_ae.assign(learning_rate_fh_ae)

#%%
dataset = tf.data.TFRecordDataset(global_setting_OpenImage.record_path)
dataset = dataset.map(parser_train)
dataset = dataset.shuffle(2000)
dataset = dataset.batch(global_setting_OpenImage.batch_size)
dataset = dataset.repeat()
iterator = dataset.make_initializable_iterator()
(img_ids,img,labels) = iterator.get_next()

#in memory
dataset_in_1 = tf.data.TFRecordDataset(TRAIN_DATA_sparse)
dataset_in_1 = dataset_in_1.map(parser_train).batch(partition_size)
sparse_dict_iterator_next = dataset_in_1.make_one_shot_iterator().get_next()
#sparse_dict_img_id,sparse_dict_img,sparse_dict_labels = sess.run([sparse_dict_img_id,sparse_dict_img,sparse_dict_labels])

dataset_in_2 = tf.data.TFRecordDataset(global_setting_OpenImage.validation_path)
dataset_in_2 = dataset_in_2.map(parser_validation)#.take(val_capacity*partition_size)
dataset_in_2 = dataset_in_2.batch(partition_size)
val_iterator_next = dataset_in_2.make_one_shot_iterator().get_next()
#(img_val_ids,val_img_v,val_labels)=sess.run([img_val_ids,val_img,val_labels])
#%%
n_sparse_dict = 20000
n_val = D_utility.count_records(global_setting_OpenImage.validation_path)
sparse_dict_img_ids,sparse_dict_imgs,sparse_dict_labels = load_memory(sparse_dict_iterator_next,n_sparse_dict,capacity)
img_val_ids,val_imgs_v,val_labels = load_memory(val_iterator_next,n_val,val_capacity)
#%%
#with slim.arg_scope(resnet_v1.resnet_arg_scope()):
saver = tf.train.import_meta_graph('./model/resnet/oidv2-resnet_v1_101.ckpt.meta')
img_input_ph = g.get_tensor_by_name('input_values:0')
features_concat = g.get_tensor_by_name('resnet_v1_101/pool5:0')
features_concat = tf.squeeze(features_concat)
#%% normalize norm
#features_concat=D_utility.project_unit_norm(features_concat)
#%%
features_concat = tf.concat([features_concat,tf.ones([tf.shape(features_concat)[0],1])],axis = 1,name='feature_input_point')
index_point = tf.placeholder(dtype=tf.int32,shape=())
F = features_concat[:index_point,:]
sparse_dict = features_concat[index_point:,:]
F_concat_ph = g.get_tensor_by_name('feature_input_point:0')
#%%
alpha_colaborative_var = tf.get_variable('alphha_colaborative',dtype=tf.float32,trainable=False, shape=())
alpha_colaborative_var_fh = tf.placeholder(dtype=tf.float32, shape=())

alpha_feature_var = tf.get_variable('alpha_feature',dtype=tf.float32,trainable=False, shape=())
alpha_feature_var_fh = tf.placeholder(dtype=tf.float32, shape=())

alpha_regularizer_var = tf.get_variable('alpha_regularizer',dtype=tf.float32,trainable=False, shape=())
alpha_regularizer_var_fh = tf.placeholder(dtype=tf.float32, shape=())
#%%
op_alpha_colaborative_var = alpha_colaborative_var.assign(alpha_colaborative_var_fh)
op_alpha_feature_var = alpha_feature_var.assign(alpha_feature_var_fh)
op_alpha_regularizer = alpha_regularizer_var.assign(alpha_regularizer_var_fh)
#%%

G = np.load('./result/Pretrained_Openimages/model.npz')['Gs'][:,:,0]

if is_sum_1:
    G = D_utility.preprocessing_graph(G)
else:
    np.fill_diagonal(G,strength_identity)

G_empty_diag = G - np.diag(np.diag(G))
if is_optimize_all_G:
    G_init=G[G!=0]
else:
    G_init=G_empty_diag[G_empty_diag!=0]
    
G_var = tf.get_variable("G_var", G_init.shape)
op_G_var=G_var.assign(G_init)
op_G_nonnegative = G_var.assign(tf.clip_by_value(G_var,0,1))
op_G_constraint = G_var.assign(tf.clip_by_value(G_var,-1,0.5))
indices = []
counter = 0
diag_G = tf.diag(np.diag(G))
#pdb.set_trace()
for idx_row in range(G_empty_diag.shape[1]):
    if is_optimize_all_G:
        idx_cols = np.where(G[idx_row,:]!=0)[0]
    else:
        idx_cols = np.where(G_empty_diag[idx_row,:]!=0)[0]
    for idx_col in idx_cols:
        if G[idx_row,idx_col]-G_init[counter] != 0:
            raise Exception('error relation construction')
        indices.append([idx_row,idx_col])
        counter += 1
if is_G:
    if is_optimize_all_G:
        part_G_var = tf.scatter_nd(indices, G_var, G.shape)
    else:
        # part_G_var = diag_G+tf.scatter_nd(indices, G_var, G.shape)#tf.eye(5000) #
        part_G_var =tf.eye(5000)+ tf.scatter_nd(indices, G_var, G.shape)  # tf.eye(5000) #
else:
    part_G_var = tf.eye(5000)


labels_ph = tf.placeholder(dtype=tf.float32, shape=(None,n_class)) #Attributes[:,:,fraction_idx_var]


hyper = tf.placeholder(dtype=tf.float32,shape=())
coeff_semi = tf.placeholder(dtype=tf.float32,shape=())

with tf.variable_scope("logistic"):
    logits = tf.matmul(F,Theta)
    labels_binary = tf.clip_by_value(labels_ph,0,1)
    labels_weight = tf.abs(tf.clip_by_value(labels_ph,-1,1))
    true_label = tf.clip_by_value(labels_ph, -1, 1)
    loss_logistic = tf.losses.sigmoid_cross_entropy(multi_class_labels=labels_binary[:global_setting_OpenImage.batch_size], logits=logits[:global_setting_OpenImage.batch_size],weights=labels_weight[:global_setting_OpenImage.batch_size])

with tf.variable_scope("regularizer"):
    loss_regularizer = tf.square(tf.norm(Theta[:-1,:]))


labels_ae = tf.placeholder(dtype=tf.float32, shape=(None,n_class)) #Attributes[:,:,fraction_idx_var]

with tf.variable_scope("BAE"):
    w_init = tf.contrib.layers.xavier_initializer(uniform=False, seed=0)
    labels_given = tf.clip_by_value(labels_ae, -1, 1)
    labels_ae_weight = tf.abs(labels_given)
    zero_masked = tf.zeros_like(labels_ae_weight)
    ones_masked = tf.ones_like(labels_ae_weight)

    aug_inputs = tf.concat([zero_masked, F], axis=1)


    Enc_1 = tf.get_variable('encoder_1', shape=[5000+2049, 2048], initializer=w_init,regularizer=tf.contrib.layers.l2_regularizer(0.001))
    Enc_b1 = tf.get_variable('bias_1', shape=[2048], initializer=None,regularizer=tf.contrib.layers.l2_regularizer(0.001))
    enc_1 = tf.matmul(aug_inputs,Enc_1) + Enc_b1
    enc_1 = tf.nn.leaky_relu(enc_1, alpha=0.1)

    Enc_2 = tf.get_variable('encoder_2', shape=[2048, 2048], initializer=w_init,regularizer=tf.contrib.layers.l2_regularizer(0.001))
    Enc_b2 = tf.get_variable('bias_2', shape=[2048], initializer=None,regularizer=tf.contrib.layers.l2_regularizer(0.001))
    enc_2 = tf.matmul(enc_1,Enc_2) + Enc_b2
    enc_2 = tf.nn.leaky_relu(enc_2, alpha=0.1)

    Enc_3 = tf.get_variable('encoder_3', shape=[2048, 2048], initializer=w_init,regularizer=tf.contrib.layers.l2_regularizer(0.001))
    Enc_b3 = tf.get_variable('bias_3', shape=[2048], initializer=None,regularizer=tf.contrib.layers.l2_regularizer(0.001))
    enc_3 = tf.matmul(enc_2,Enc_3) + Enc_b3
    enc_3 = tf.nn.leaky_relu(enc_3, alpha=0.1)

    Enc_4 = tf.get_variable('encoder_4', shape=[2048, 2048], initializer=w_init,regularizer=tf.contrib.layers.l2_regularizer(0.001))
    Enc_b4 = tf.get_variable('bias_4', shape=[2048], initializer=None,regularizer=tf.contrib.layers.l2_regularizer(0.001))
    feat = tf.matmul(enc_3,Enc_4) + Enc_b4
    feat = tf.nn.tanh(feat)

    Dec_1 = tf.get_variable('decoder_1', shape=[2048, 5000], initializer=w_init,regularizer=tf.contrib.layers.l2_regularizer(0.001))
    outputs_labels = tf.matmul(feat,Dec_1)

    Dec_2 = tf.get_variable('decoder_2', shape=[2048, 2049], initializer=w_init,regularizer=tf.contrib.layers.l2_regularizer(0.001))
    outputs_data = tf.matmul(feat,Dec_2)

    outputs_labels = tf.matmul(outputs_labels,part_G_var)

    ae_decoded_binary=tf.cast(outputs_labels > 0.0, dtype=tf.float32)

    batch_size_t=tf.shape(outputs_labels)[0]

    hardlabel = tf.cast(tf.keras.losses.cosine_similarity(outputs_labels[:batch_size_t // 2],  outputs_labels[batch_size_t // 2:2*(batch_size_t // 2)], axis=1)> 0.95, dtype=tf.float32)

    similar = tf.keras.losses.cosine_similarity(outputs_labels[:batch_size_t // 2],outputs_labels[batch_size_t // 2:2*(batch_size_t // 2)], axis=1)

    D = tf.reduce_mean(tf.square(F[:batch_size_t // 2] - F[batch_size_t // 2:]), axis=1)
    D = tf.clip_by_value(D, clip_value_min=1e-16, clip_value_max=100)
    D_sq = tf.sqrt(D)

    pos = (D * hardlabel)
    neg = (1. - hardlabel) * tf.square(tf.nn.relu(2.5 - D_sq))
    semi_loss =  tf.reduce_mean(pos+0.2*neg)
    semi_loss_pos = tf.reduce_mean(pos)

    c_reg = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    blank_ratio = tf.reduce_sum(ones_masked)/tf.reduce_sum(labels_ae_weight)
    loss_BAE = 10.0* blank_ratio * tf.reduce_mean(tf.pow(labels_given - outputs_labels, 2)* labels_ae_weight) + 0.1 * tf.reduce_mean(tf.pow(F - outputs_data, 2))


tf.global_variables_initializer().run()
sess.run(iterator.initializer)
#%%



def append_info(m_AP, sum_num_miss_p, sum_num_miss_n,loss_tot, loss_cnn, loss_mc, loss_graph, lr_v, lr_ae_v):
    res_mAP[index] = m_AP
    res_loss_tot[index] = loss_tot
    res_loss_cnn[index] = loss_cnn
    res_loss_mc[index] = loss_mc
    res_loss_graph[index] = loss_graph
    res_lr[index] = lr_v
    res_lr_ae[index] = lr_ae_v
    res_sum_num_miss_p[index] = sum_num_miss_p
    res_sum_num_miss_n[index] = sum_num_miss_n

    df_result['mAP: ' + extension] = res_mAP
    df_result['sum_num_miss_p: ' + extension] = res_sum_num_miss_p
    df_result['sum_num_miss_n: ' + extension] = res_sum_num_miss_n
    df_result['loss_tot: ' + extension] = res_loss_tot
    df_result['loss_cnn: ' + extension] = res_loss_cnn
    df_result['loss_mc: ' + extension] = res_loss_mc
    df_result['loss_graph: ' + extension] = res_loss_graph
    df_result['lr: ' + extension] = res_lr
    df_result['lr_ae: ' + extension] = res_lr_ae


#%%
print('placeholder assignment')
#%%
Theta_fh = tf.placeholder(dtype=tf.float32, shape=[2049,n_class])
op_assign_Theta = Theta.assign(Theta_fh)

#%% compute normalizer
trainable_vars = Theta#tf.trainable_variables()[:-3]
grad_logistic = tf.gradients(loss_logistic,trainable_vars)

norm_grad_logistic = tf.norm(grad_logistic)
Prediction = tf.matmul(features_concat,Theta)


optimizer_ae = tf.train.RMSPropOptimizer(
      learning_rate_ae,
      0.9,  # decay
      0.9,  # momentum
      1.0   #rmsprop_epsilon
  )

optimizer = tf.train.RMSPropOptimizer(
      learning_rate,
      0.9,  # decay
      0.9,  # momentum
      1.0   #rmsprop_epsilon
  )
loss = loss_logistic + coeff_semi* hyper* semi_loss

train_vars_CNN = [var for var in tf.trainable_variables() if var.name.startswith("resnet_v1_101") or var.name.startswith("Theta")]
train_vars_ae = [var for var in tf.trainable_variables() if var.name.startswith("BAE") or var.name.startswith("G_var")]
saver_ae=tf.compat.v1.train.Saver(var_list=train_vars_ae, max_to_keep=1)

grad_var_all = optimizer.compute_gradients(loss,train_vars_CNN)
train = optimizer.apply_gradients(grad_var_all)

grad_var_ae = optimizer_ae.compute_gradients(loss_BAE,train_vars_ae)
train_ae = optimizer_ae.apply_gradients(grad_var_ae)

with tf.get_default_graph().control_dependencies([train]):
    train_ae = optimizer_ae.apply_gradients(grad_var_ae)
Tot_op = tf.group(train, train_ae)

print('done placeholder assignment')
def experiment_cond_success():
    return True#(alpha_colaborative_o >0) or (alpha_colaborative_o + alpha_feature_o==0)

n_experiment= 0

for idx_alpha_colaborative,alpha_colaborative_o in enumerate(list_alphas_colaborative):
    for idx_alpha_feature,alpha_feature_o in enumerate(list_alphas_feature):
        for idx_alpha_regularizer,alpha_regularizer_o in enumerate([0]):
            if not experiment_cond_success():#index_column <= 4:#(idx_alpha_colaborative == 0 and idx_alpha_feature != 1) or idx_alpha_regularizer != 0 or 
                print('skip')
                continue
            n_experiment += 1
print('Total number of experiment: {}'.format(n_experiment))
#%%

print('-'*30)
df_result = pd.DataFrame()
pos_idx = 0
# input('hardcode position of Thetas={}: '.format(pos_idx))
data=np.load(global_setting_OpenImage.saturated_Thetas_model)
print('done')
# init_Theta = collapse_Theta(data)#data['Thetas'][:,:,pos_idx]
init_Theta = data['Thetas'][:,:,pos_idx]
#pdb.set_trace()
tf.global_variables_initializer().run()
#%%
sess.run(op_G_var)
sess.run(op_assign_Theta,{Theta_fh:init_Theta})
sess.run(iterator.initializer)
saver.restore(sess, global_setting_OpenImage.model_path)
sess.run(op_alpha_colaborative_var,{alpha_colaborative_var_fh:1e5})
sess.run(op_alpha_feature_var,{alpha_feature_var_fh:1})
#%%
#pdb.set_trace()
img_ids_v,img_v,labels_v=sess.run([img_ids,img,labels])
print('compute subset dict')

_,sparse_dict_feature,_=compute_feature_prediction_large_batch(sparse_dict_imgs, sparse_dict_labels)


print('done')

raitio_regularizer_grad_v=1


name = template_name.format(list_alphas_colaborative[0],list_alphas_feature[0],global_setting_OpenImage.e2e_learning_rate_base,global_setting_OpenImage.batch_size,global_setting_OpenImage.decay_rate_cond
                                               ,global_setting_OpenImage.signal_strength,global_setting_OpenImage.n_cycles,
                                               idx_GPU,global_setting_OpenImage.thresold_coeff,c,time.time())
#%% create dir
if not os.path.exists('./result/'+name) and is_save:
    os.makedirs('./result/'+name)
#%%
Thetas = np.zeros((2049,n_class,n_experiment))
Gs = np.zeros((n_class,n_class,n_experiment))
idx_experiment = 0
for idx_alpha_colaborative,alpha_colaborative_o in enumerate(list_alphas_colaborative):
    for idx_alpha_feature,alpha_feature_o in enumerate(list_alphas_feature):
        for idx_alpha_regularizer,alpha_regularizer_o in enumerate([0]):


            if not experiment_cond_success():#index_column <= 4:#(idx_alpha_colaborative == 0 and idx_alpha_feature != 1) or idx_alpha_regularizer != 0 or
                print('skip')
                continue

            print('report length {}'.format(report_length))
            res_mAP = np.zeros(report_length)
            res_loss_cnn = np.zeros(report_length)
            res_loss_mc = np.zeros(report_length)
            res_loss_graph = np.zeros(report_length)
            res_loss_tot=np.zeros(report_length)
            res_sum_num_miss_p=np.zeros(report_length)
            res_sum_num_miss_n=np.zeros(report_length)
            res_grad_logistic=np.zeros(report_length)
            res_lr=np.zeros(report_length)
            res_lr_ae=np.zeros(report_length)
            res_norm_f=np.zeros(report_length)

            tf.global_variables_initializer().run()
            print('reset Theta')
            sess.run(iterator.initializer)
            sess.run(op_G_var)
            saver.restore(sess, global_setting_OpenImage.model_path)
            sess.run(op_assign_Theta,{Theta_fh:init_Theta})
            saver_ae.restore(sess,'./result/Pretrained_Openimages/model_n_feature 20000_AE.ckpt')

            extension = 'n_feature {}'.format(n_sparse_dict)

            #exponential moving average
            expon_moving_avg_old = np.inf
            expon_moving_avg_new = 0
            expon_moving_avg_old_ae = np.inf
            expon_moving_avg_new_ae = 0
            #
            m = 0
            m_ae = 0
            df_ap = pd.DataFrame()
            df_ap['label']=list_label
            # print('lambda colaborative: {} lambda_feature: {} regularizer: {}'.format(alpha_colaborative,alpha_feature,alpha_regularizer))
            n_nan = 0
            n_error = 0
            #%%
            tic = time.clock()
            for idx_cycle in range(global_setting_OpenImage.e2e_n_cycles):
                try:
                    index = (idx_cycle*n_epoches)//global_setting_OpenImage.report_interval
                    img_ids_v,img_v,labels_v=sess.run([img_ids,img,labels])
                    is_first_start = idx_alpha_colaborative+idx_alpha_feature+idx_alpha_regularizer+idx_cycle==0
                    if idx_cycle%dictionary_evaluation_interval==0 and (not is_first_start):
                        print('evalutation of Dictionary')

                    idx_select = np.random.choice(n_sparse_dict,n_batch,replace=False)

                    img_v_concat = np.concatenate([img_v,sparse_dict_imgs[idx_select]],axis=0)

                    labels_concat = np.concatenate([labels_v, sparse_dict_labels[idx_select]],axis=0)

                    _,loss_tot, loss_cnn, loss_mc, loss_graph, lr_v, lr_ae_v, D_sq_v, similar_v = sess.run([Tot_op, loss, loss_logistic, loss_BAE, semi_loss,learning_rate, learning_rate_ae, D_sq, similar], {img_input_ph:img_v_concat, index_point:img_v_concat.shape[0], labels_ae: labels_concat, labels_ph:labels_concat, hyper: 1.0, coeff_semi:0.5})

                    if (idx_cycle * n_epoches) % change_interval == 0 and (not is_first_start):
                        print('change the features')
                        _, sparse_dict_imgs, sparse_dict_labels = load_memory(sparse_dict_iterator_next,n_sparse_dict, capacity)


                    if (idx_cycle*n_epoches) % global_setting_OpenImage.report_interval == 0 :#or idx_iter == n_epoches-1:

                        print('Elapsed time udapte: {}'.format(time.clock()-tic))
                        tic = time.clock()
                        time_o = time.clock()
                        print('n_error {} n_nan {}'.format(n_error,n_nan))
                        print('index {} -- compute mAP'.format(index))
                        validate_Prediction_val,_,validate_Prediction_val_ae=compute_feature_prediction_large_batch(val_imgs_v, val_labels)
                        ap = compute_AP(validate_Prediction_val,val_labels)
                        ap_ae = compute_AP(validate_Prediction_val_ae,val_labels)
                        num_mis_p,num_mis_n=compute_number_misclassified(validate_Prediction_val_ae,val_labels)
                        df_ap['index {}: ap'.format(index)]=ap
                        df_ap['index {}: num_mis_p'.format(index)]=num_mis_p
                        df_ap['index {}: num_mis_n'.format(index)]=num_mis_n
                        m_AP=np.mean(ap)
                        m_AP_ae=np.mean(ap_ae)

                        sum_num_miss_p = np.sum(num_mis_p)
                        sum_num_miss_n = np.sum(num_mis_n)

                        print('mAP {} mAP AE {} sum_num_miss_p {} sum_num_miss_n {}'.format(m_AP, m_AP_ae, sum_num_miss_p,sum_num_miss_n))
                        print(loss_tot, loss_cnn, loss_mc,loss_graph, lr_v, lr_ae_v)
                        #exponential_moving_avg
                        expon_moving_avg_old = expon_moving_avg_new
                        expon_moving_avg_new = expon_moving_avg_new * (1 - global_setting_OpenImage.signal_strength) + m_AP * global_setting_OpenImage.signal_strength

                        expon_moving_avg_old_ae = expon_moving_avg_new_ae
                        expon_moving_avg_new_ae = expon_moving_avg_new_ae * (1 - global_setting_OpenImage.signal_strength) + m_AP_ae * global_setting_OpenImage.signal_strength

                        if expon_moving_avg_new < expon_moving_avg_old and learning_rate.eval() >= global_setting_OpenImage.e2e_limit_learning_rate and m <= 0:
                            print('Adjust learning rate')
                            sess.run(op_assign_learning_rate, {learning_rate_fh: learning_rate.eval() * global_setting_OpenImage.decay_rate_cond})
                            m = patient
                        m -= 1

                        if expon_moving_avg_new_ae < expon_moving_avg_old_ae and learning_rate_ae.eval() >= 1e-6 and m_ae <= 0:
                            print('Adjust learning rate')
                            sess.run(op_assign_learning_rate_ae, {learning_rate_fh_ae: learning_rate_ae.eval() * global_setting_OpenImage.decay_rate_cond})
                            m_ae = patient
                        m_ae -= 1

                        print('Loss {} Loss_CNN {} Loss_MC {} Loss_Graph {} lr_cnn {} lr_ae {} dist {} similar_mean {} similar_max {}'.format(loss_tot, loss_cnn,loss_mc,loss_graph, lr_v, lr_ae_v,np.mean(D_sq_v), np.mean(similar_v), np.max(similar_v)))
                        append_info(m_AP,sum_num_miss_p,sum_num_miss_n,loss_tot,loss_cnn,loss_mc,loss_graph,lr_v,lr_ae_v)

                        if is_save:
                            Thetas[:,:,idx_experiment]=Theta.eval()
                            Gs[:,:,idx_experiment]=part_G_var.eval()
                            df_result.to_csv('./result/'+name+'/mAP.csv')
                            ap_save_name = './result/'+name+'/n_feature {}.csv'
                            df_ap.to_csv(ap_save_name.format(n_sparse_dict))
                            np.savez('./result/'+name+"/model", Thetas=Thetas, Gs=Gs)
                            model_name = 'model_'+extension+'.ckpt'
                            model_name_ae = 'model_'+extension+'_AE.ckpt'
                            saver.save(sess,'./result/'+name+"/"+model_name)
                            saver_ae.save(sess, './result/' + name + "/" + model_name_ae)
                            print(model_name_ae)


                except Exception as e:
                    n_error+=1
                    if np.isnan(loss_mc):
                        print('nan encounter')
                        n_nan += 1
                        sess.run(op_assign_Theta,{Theta_fh:Thetas[:,:,idx_experiment]})
                        sess.run(op_G_var)
                        sess.run(op_assign_learning_rate,{learning_rate_fh:lr_v})
                        m = patient
# #%%
sess.close()
tf.reset_default_graph()
