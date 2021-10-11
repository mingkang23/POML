import tensorflow as tf
import numpy as np
import glob

from tensorflow.python.client import device_lib
# import global_setting_CUB

# parser = argparse.ArgumentParser()
# parser.add_argument('--seed', type=int, default=0, help='seed')
# FLAGS = parser.parse_args()
# seed = FLAGS.seed
# tf.random.set_seed(seed)
# os.environ['TF_DETERMINISTIC_OPS'] = '1'


IMAGE_SIZE = 224
_GPUS = None


def LoadLabelMap(attr_name_file, class_name_file):
    attr_name = []
    class_name = []
    with open(attr_name_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            idx, name = line.rstrip('\n').split(' ')
            attr_name.append(name)

    with open(class_name_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            idx, name = line.rstrip('\n').split(' ')
            class_name.append(name)
    return attr_name, class_name

path = './'
attr_name_file = path+'data/CUB/CUB_200_2011/attributes/attributes.txt'
class_name_file =  path+'data/CUB/CUB_200_2011/classes.txt'
class_signature_file = path+'data/CUB/CUB_200_2011/attributes/class_attribute_labels_continuous.txt'

attr_name, class_name = LoadLabelMap(attr_name_file, class_name_file)
n_attr = len(attr_name)

def record_parse(serialized_example):
    features = tf.io.parse_single_example(
        serialized_example,
    features = {'img_id': tf.io.FixedLenFeature([], tf.string),
               'img': tf.io.FixedLenFeature([], tf.string),
               'label': tf.io.FixedLenFeature([], tf.string),
               'attribute': tf.io.FixedLenFeature([], tf.string)})

    img_id = tf.io.decode_raw(features['img_id'], tf.int32)
    img = tf.reshape(tf.io.decode_raw(features['img'], tf.float32), [IMAGE_SIZE, IMAGE_SIZE, 3])
    attribute = tf.reshape(tf.io.decode_raw(features['attribute'], tf.int32), [len(attr_name), -1])
    label = tf.io.decode_raw(features['label'], tf.int32)

    return dict(img_id=img_id, img=img, attribute=attribute,label=label)

def default_parse(dataset: tf.data.Dataset, parse_fn=record_parse) -> tf.data.Dataset:
    # para = 4 * max(1, len(get_available_gpus())) * 4
    para = 1 * max(1, len(get_available_gpus()))
    return dataset.map(parse_fn, num_parallel_calls=para)

def dataset(filenames: list) -> tf.data.Dataset:
    filenames = sorted(sum([glob.glob(x) for x in filenames], []))
    if not filenames:
        raise ValueError('Empty dataset, did you mount gcsfuse bucket?')
    #dataset = filenames.interleave(
    #    tf.data.TFRecordDataset, cycle_length=FLAGS.para_parse,
    #    num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return tf.data.TFRecordDataset(filenames)


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
    def __init__(self, name, train, valid, test,
                 height=IMAGE_SIZE, width=IMAGE_SIZE, colors=3, nclass=312):
        self.name   = name
        self.train  = train
        self.valid  = valid
        self.test  = test
        self.height = height
        self.width  = width
        self.colors = colors
        self.nclass = nclass


    @classmethod
    def creator(cls, name, augment, augment_valid, augment_test, parse_fn=default_parse, colors=3, nclass=312, height=IMAGE_SIZE, width=IMAGE_SIZE):
        # fn = lambda x: x.repeat()
        fn = lambda x: x
        def create():
            DATA_DIR = './cub_data/'
            TRAIN_DATA = []
            VALID_DATA = []
            TEST_DATA = []

            TRAIN_DATA.append(DATA_DIR + 'zs_mask_train_CUB_img.tfrecords')
            VALID_DATA.append(DATA_DIR + 'zs_validation_CUB_img.tfrecords')
            TEST_DATA.append(DATA_DIR + 'zs_test_CUB_img.tfrecords')

            para = max(1, len(get_available_gpus())) * 4

                    # TRAIN_DATA.append(EXT_DIR + 'extdev{}.chunk*.tfrecord'.format(i))
            print(TRAIN_DATA)


            #train_data = parse_fn(dataset(TRAIN_DATA).shuffle(5011, reshuffle_each_iteration=True))
            train_data = parse_fn(dataset(TRAIN_DATA).shuffle(5011,seed=0, reshuffle_each_iteration=True))
            valid_data = parse_fn(dataset(VALID_DATA))
            test_data = parse_fn(dataset(TEST_DATA))

            return cls(name,
                       train=fn(train_data).batch(32).map(augment, para),
                       valid=valid_data.batch(32).map(augment_valid, para),
                       test=test_data.batch(32).map(augment_test, para),
                       nclass=nclass, colors=colors,
                       height=height, width=width)

        return name, create

augment_train = lambda x: ({'img_id'         :x['img_id'],
                            'img'           : x['img'],
                            'attribute'           : tf.cast(x['attribute'],tf.float32),
                            'label'           : x['label']})

augment_valid = lambda x: ({'img_id'         :x['img_id'],
                            'img'           : x['img'],
                            'attribute'           : tf.cast(x['attribute'],tf.float32),
                            'label'           : x['label']})

augment_test = lambda x: ({'img_id'         :x['img_id'],
                            'img'           : x['img'],
                            'attribute'           : tf.cast(x['attribute'],tf.float32),
                            'label'           : x['label']})

DATASETS = {}
DATASETS.update([DataSet.creator('fundus', augment_train, augment_valid, augment_test)])
