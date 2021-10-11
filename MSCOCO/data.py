import tensorflow as tf
import numpy as np
import glob
import pickle

from tensorflow.python.client import device_lib
# import global_setting_CUB

IMAGE_SIZE = 224
_GPUS = None

path = './'

def load_1k_name():
    path = './data/MSCOCO_1k/vocab_coco.pkl'
    with open(path,'rb') as f:
        vocab = pickle.load(f)
    return vocab['words']
classes = load_1k_name()
n_classes = len(classes)


def record_parse(serialized_example):
    features = tf.io.parse_single_example(
        serialized_example,
    features = {'img': tf.io.FixedLenFeature([], tf.string),
               'label': tf.io.FixedLenFeature([], tf.string)})

    img = tf.reshape(tf.io.decode_raw(features['img'], tf.float32), [IMAGE_SIZE, IMAGE_SIZE, 3])
    label =  tf.clip_by_value(tf.io.decode_raw(features['label'],tf.int32),-1,1)

    return dict(img=img, label=label)

def record_parse_test(serialized_example):
    features = tf.io.parse_single_example(
        serialized_example,
    features = {'img': tf.io.FixedLenFeature([], tf.string),
               'label_1k': tf.io.FixedLenFeature([], tf.string)})

    img = tf.reshape(tf.io.decode_raw(features['img'], tf.float32), [IMAGE_SIZE, IMAGE_SIZE, 3])
    label =  tf.clip_by_value(tf.io.decode_raw(features['label_1k'],tf.int32),-1,1)

    return dict(img=img, label=label)

def default_parse(dataset: tf.data.Dataset, parse_fn=record_parse) -> tf.data.Dataset:
    # para = 4 * max(1, len(get_available_gpus())) * 4
    para = 4 * max(1, len(get_available_gpus()))
    return dataset.map(parse_fn, num_parallel_calls=para)

def default_parse_test(dataset: tf.data.Dataset, parse_fn_test=record_parse_test) -> tf.data.Dataset:
    # para = 4 * max(1, len(get_available_gpus())) * 4
    para = 4 * max(1, len(get_available_gpus()))
    return dataset.map(parse_fn_test, num_parallel_calls=para)

def dataset(filenames: list) -> tf.data.Dataset:
    filenames = sorted(sum([glob.glob(x) for x in filenames], []))
    if not filenames:
        raise ValueError('Empty dataset, did you mount gcsfuse bucket?')
    #dataset = filenames.interleave(
    #    tf.data.TFRecordDataset, cycle_length=FLAGS.para_parse,
    #    num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return tf.data.TFRecordDataset(filenames,compression_type='ZLIB')


def augment_shift(x, w):
    y = tf.pad(x, [[0] * 2, [w] * 2, [w] * 2, [0] * 2], mode='REFLECT')
    return tf.image.random_crop(y, tf.shape(x))

def augment_flip(x):
    return tf.image.random_flip_left_right(x, seed=None)

def augment_noise(x):
    return x + tf.random.normal(tf.shape(x), mean=0.0, stddev=0.15)

def get_available_gpus():
    global _GPUS
    if _GPUS is None:
        local_device_protos = device_lib.list_local_devices()
        _GPUS = tuple([x.name for x in local_device_protos if x.device_type == 'GPU'])
    return _GPUS

class DataSet:
    def __init__(self, name, train, valid,
                 height=IMAGE_SIZE, width=IMAGE_SIZE, colors=3, nclass=1000):
        self.name   = name
        self.train  = train
        self.valid  = valid
        self.height = height
        self.width  = width
        self.colors = colors
        self.nclass = nclass


    @classmethod
    def creator(cls, name, augment, augment_valid, parse_fn=default_parse, parse_fn_test=default_parse_test, colors=3, nclass=1000, height=IMAGE_SIZE, width=IMAGE_SIZE):
        # fn = lambda x: x.repeat()
        fn = lambda x: x
        def create():
            DATA_DIR = './mscoco_data/'
            TRAIN_DATA = []
            VALID_DATA = []

            TRAIN_DATA.append(DATA_DIR + 'train_MSCOCO_img_ZLIB.tfrecords')
            VALID_DATA.append(DATA_DIR + 'test_MSCOCO_img_ZLIB.tfrecords')

            para = max(1, len(get_available_gpus())) * 4

                    # TRAIN_DATA.append(EXT_DIR + 'extdev{}.chunk*.tfrecord'.format(i))
            print(TRAIN_DATA)



            #train_data = parse_fn(dataset(TRAIN_DATA).shuffle(10000, reshuffle_each_iteration=True))
            train_data = parse_fn(dataset(TRAIN_DATA))
            valid_data = parse_fn_test(dataset(VALID_DATA))



            return cls(name,
                       train=fn(train_data).batch(1).map(augment, para),
                       valid=valid_data.batch(100).map(augment_valid, para),
                       nclass=nclass, colors=colors,
                       height=height, width=width)

        return name, create

augment_train = lambda x: ({'img'           : x['img'],
                            'label'           : tf.cast(x['label'],tf.float32)})

augment_valid = lambda x: ({'img'           : x['img'],
                            'label'           : tf.cast(x['label'],tf.float32)})


DATASETS = {}
DATASETS.update([DataSet.creator('fundus', augment_train, augment_valid)])
