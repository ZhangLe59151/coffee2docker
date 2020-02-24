# -- Created by ZhangLe --#
# -- Created on 2020-02-23 --#

import tensorflow as tf 
from tensorflow import keras

# Just disables the warning, doesn't enable AVX/FMA
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

base_root = '/Users/zhangle/Documents/IS/coffee2docker/'
checkpoint_path = base_root + "/model/cp.ckpt"

print(checkpoint_path)


sess = tf.Session()
with gfile.FastGFile(pb_file_path+'model.pb', 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='') # 导入计算图