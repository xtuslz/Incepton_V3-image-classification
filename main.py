# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

uid_to_human = {}  # 字符串到英文的映射
for line in tf.gfile.GFile('imagenet_synset_to_human_label_map.txt').readlines():
    items = line.strip().split('\t')
    uid_to_human[items[0]] = items[1]

node_id_to_uid = {}  # 整数到字符串的映射
for line in tf.gfile.GFile('imagenet_2012_challenge_label_map_proto.pbtxt').readlines():
    if line.startswith('  target_class:'):
        target_class = int(line.split(': ')[1])
    if line.startswith('  target_class_string:'):
        target_class_string = line.split(': ')[1].strip('\n').strip('\"')
        node_id_to_uid[target_class] = target_class_string

node_id_to_name = {}
for key, value in node_id_to_uid.items():
    node_id_to_name[key] = uid_to_human[value]


def create_graph():  # 加载模型
    with tf.gfile.FastGFile('classify_image_graph_def.pb', 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')


def classify_image(image, top_k=1):  # 定义一个分类图片的函数
    image_data = tf.gfile.FastGFile(image, 'rb').read()

    create_graph()

    with tf.Session() as sess:
        # 'softmax:0': A tensor containing the normalized prediction across 1000 labels
        # 'pool_3:0': A tensor containing the next-to-last layer containing 2048 float description of the image
        # 'DecodeJpeg/contents:0': A tensor containing a string providing JPEG encoding of the image
        softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')
        predictions = sess.run(softmax_tensor, feed_dict={'DecodeJpeg/contents:0': image_data})
        predictions = np.squeeze(predictions)

        top_k = predictions.argsort()[-top_k:]
        for node_id in top_k:
            human_string = node_id_to_name[node_id]
            score = predictions[node_id]
            print('%s (score = %.5f)' % (human_string, score))


classify_image('test2.png')