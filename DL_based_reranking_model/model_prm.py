# coding: utf-8
from __future__ import print_function
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.python.feature_column.feature_column import _LazyBuilder
from tensorflow.python.ops import partitioned_variables
from tensorflow.contrib.layers import l1_l2_regularizer

import importlib
import numpy as np
import os


class PRM(object):

    @property
    def hash_features(self):
        return set()

    def label_parse(self, label, mode):
        return tf.minimum(label, 1)

    def __init__(self, input_format):
        self.input_format = input_format
        self.optimizers = {}
        self.uid = tf.placeholder(tf.int32, shape=[None, None])
        self.rank_skuid = tf.placeholder(tf.int32, shape=[None, None])
        self.rank_cateid = tf.placeholder(tf.int32, shape=[None, None])

        self.posid = tf.placeholder(tf.int32, shape=[None, None])

        self.label = tf.placeholder(tf.float32, shape=[None, None])

        self.top_k = 25

        self.params = input_format
        self.emb_user = tf.get_variable('user_embedding', shape=[self.params['user_num'], self.params['dims']],
                                       trainable=True,
                                       initializer=tf.contrib.layers.variance_scaling_initializer(factor=2.0,
                                                                                                  mode='FAN_OUT',
                                                                                                  uniform=True))
        self.emb_sku = tf.get_variable('sku_embedding', shape=[self.params['item_num'], self.params['dims']], trainable=True, initializer=tf.contrib.layers.variance_scaling_initializer(factor=2.0,
                                                                                                       mode='FAN_OUT',
                                                                                                       uniform=True))
        self.emb_cate = tf.get_variable('cate3_embedding', shape=[self.params['cate_num'], self.params['dims']], trainable=True, initializer=tf.contrib.layers.variance_scaling_initializer(factor=2.0,
                                                                                                       mode='FAN_OUT',
                                                                                                      uniform=True))
        self.emb_pos = tf.get_variable('pos_embedding', shape=[300, self.params['dims']],
                                        trainable=True,
                                        initializer=tf.contrib.layers.variance_scaling_initializer(factor=2.0,
                                                                                                   mode='FAN_OUT',
                                                                                                   uniform=True))


        # user interaction encoding
        self.rank_item_in = tf.nn.embedding_lookup(self.emb_sku, self.rank_skuid)
        self.rank_cate_in = tf.nn.embedding_lookup(self.emb_cate, self.rank_cateid)
        self.rank_pos_in = tf.nn.embedding_lookup(self.emb_pos, self.posid)
        self.rank_join_in = tf.concat([self.rank_item_in, self.rank_cate_in, self.rank_pos_in], -1)

        self.rank_join_attn = self.get_sequence_embedding_with_transformer('rank_seq', self.rank_skuid,
                                                                              T=self.params['matching_seq_max_len'],
                                                                              mode='attention',
                                                                              seq_embedding=self.rank_join_in,
                                                                              user_embedding=None)

        # predicting layer
        self.deep_out = self.rank_join_attn

        self.common_deep_out1 = tf.layers.dense(self.deep_out, units=64, activation=tf.nn.elu, name='dense1')

        self.common_deep_out2 = tf.layers.dense(self.common_deep_out1, units=32, activation=tf.nn.elu, name='dense2')

        self.final_layer_out = tf.layers.dense(self.common_deep_out2, units=1, name='dense3')

        self.final_layer_out = tf.reshape(self.final_layer_out, [-1, self.params['matching_seq_max_len']])

        # loss.
        # self.label = tf.reshape(self.label, [-1, 1])
        # self.loss = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(
        #     targets=self.label, logits=self.final_layer_out, pos_weight=self.weight))

        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
           labels=self.label, logits=self.final_layer_out))

        self.optimizer = self.params['optimizer']
        self.train_op = self.optimizer.minimize(self.loss)


    def layer_normalize(self, inputs, epsilon=1e-8):
        '''
        Applies layer normalization
        Args:
            inputs: A tensor with 2 or more dimensions
            epsilon: A floating number to prevent Zero Division
        Returns:
            A tensor with the same shape and data dtype
        '''
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]

        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta = tf.Variable(tf.zeros(params_shape))
        gamma = tf.Variable(tf.ones(params_shape))
        normalized = (inputs - mean) / ((variance + epsilon) ** (.5))
        outputs = gamma * normalized + beta

        return outputs

    def self_multi_head_attn(self, queries,
                             keys,
                             values,
                             num_units=None,
                             num_heads=1,
                             is_drop=False,
                             dropout_keep_prob=1,
                             is_training=True,
                             has_residual=True):

        _, T, input_num_units = queries.get_shape().as_list()
        if num_units is None:
            num_units = input_num_units

        # Linear projections
        Q = tf.layers.dense(queries, num_units, activation=tf.nn.relu)
        K = tf.layers.dense(keys, num_units, activation=tf.nn.relu)
        V = tf.layers.dense(values, num_units, activation=tf.nn.relu)
        if has_residual:
            if num_units != input_num_units:
                V_res = tf.layers.dense(values, num_units, activation=tf.nn.relu)
            else:
                V_res = queries

        # Split and concat
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)

        # Multiplication
        weights = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))

        # Scale
        weights = weights / (K_.get_shape().as_list()[-1] ** 0.5)

        # Activation
        weights = tf.nn.softmax(weights)

        # Dropouts
        if is_drop:
            weights = tf.layers.dropout(weights, rate=1 - dropout_keep_prob,
                                        training=tf.convert_to_tensor(is_training))

        # Weighted sum
        outputs = tf.matmul(weights, V_)

        # Restore shape
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)

        # Residual connection
        if has_residual:
            outputs += V_res

        outputs = tf.nn.relu(outputs)
        # Normalize
        outputs = self.layer_normalize(outputs)

        return outputs

    def get_sequence_embedding_with_transformer(self, scope_name, seq_ids, T=50, target_embedding=None,mode='mean',k=25,flag='attention', seq_embedding=None, user_embedding=None):

        mask = tf.cast(tf.not_equal(seq_ids, 0), tf.float32)
        mask = tf.expand_dims(mask, axis=-1)
        mask = (1 - mask) * (-1e9)
        seq_embedding = seq_embedding + mask
        paddings = tf.ones([tf.shape(seq_embedding)[0], T, tf.shape(seq_embedding)[2]]) * (-1e9)
        embedding_concat = tf.concat([seq_embedding, paddings], 1)
        seq_emb = tf.slice(embedding_concat, [0, 0, 0], [-1, T, -1])

        # self-attention encoding
        seq_emb = self.self_multi_head_attn(queries=seq_emb,
                                            keys=seq_emb,
                                            values=seq_emb,
                                            num_units=self.params['block_shape'][0],
                                            num_heads=self.params['heads'],
                                            is_drop=self.params['is_drop'],
                                            dropout_keep_prob=self.params['dropout_keep_prob']
                                            )

        return seq_emb


    def attention_sum_pooling(self, query, key, dims, flag):
        """
        :param query: [batch_size, query_size] -> [batch_size, time, query_size]
        :param key:   [batch_size, time, key_size]
        :return:      [batch_size, 1, time]
            query_size should keep the same dim with key_size
        """
        query = tf.expand_dims(query, 1)
        key_transpose = tf.transpose(key, [0, 2, 1])
        align = tf.matmul(query, key_transpose)
        align = tf.nn.softmax(align)

        output = tf.matmul(align, key)  # [batch_size, 1, time] * [batch_size, time, key_size] -> [batch_size, 1, key_size]
        # output = tf.squeeze(output)
        output = tf.reshape(output, [-1, dims])
        return output

    def multilabel_parse(self, label, features):
        msLabel = features['isMeishiIntent']
        labels = tf.where(tf.equal(msLabel, 1),
                          label,
                          tf.minimum(label, 3))
        labels = tf.cast(labels, tf.float32)
        return tf.minimum(tf.divide(labels, 5.0), 1.0)

    def save(self, sess, path):
        if not os.path.exists(path):
            os.makedirs(path)
        saver = tf.train.Saver()
        saver.save(sess, path + 'model.ckpt')

    def restore(self, sess, path):
        saver = tf.train.Saver()
        saver.restore(sess, path + 'model.ckpt')
        print('model restored from %s' % path)

    def get_sku_emb(self, sess):
        return sess.run(self.emb_sku)


    def train(self, features, sess, data_type):


        loss, train_op = sess.run([self.loss, self.train_op],
                                  feed_dict={self.rank_skuid: features[0], self.rank_cateid: features[1],
                                             self.label: features[2], self.posid: features[3]}
                                             )

        return loss, train_op

    def test(self, features, sess, data_type):

        return sess.run([tf.nn.sigmoid(self.final_layer_out)], feed_dict={self.rank_skuid: features[0], self.rank_cateid: features[1],
                                             self.label: features[2], self.posid: features[3]}
                                             )


