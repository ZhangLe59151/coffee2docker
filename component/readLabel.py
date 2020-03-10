# -- Created on 2020-03-09 --#

import tensorflow as tf

def get_label_dict(label_file_path):
  with tf.gfile.GFile(label_file_path, 'r') as fid:
    label_dict_string = fid.read()
    # parsor string to dict

