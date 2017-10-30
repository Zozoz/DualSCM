#!/usr/bin/env python
# encoding: utf-8
# @author: newbie
# email: zhengshiliang0@gmail.com


import os, sys
sys.path.append(os.getcwd())

import numpy as np
import tensorflow as tf

from utils.config import *
from newbie_nn.nn_layer import bi_dynamic_rnn, softmax_layer
from newbie_nn.att_layer import mlp_attention_layer
from utils.data_helper import load_w2v, batch_index, load_inputs_document_nohn


def lstm_sen(inputs, sen_len, keep_prob1, keep_prob2):
    print 'I am dual-sen!'
    cell = tf.contrib.rnn.LSTMCell
    inputs = tf.nn.dropout(inputs, keep_prob=keep_prob1)
    hiddens = bi_dynamic_rnn(cell, inputs, FLAGS.n_hidden, sen_len, FLAGS.max_sentence_len, 'sentence_o', 'last')
    logits = softmax_layer(hiddens, 2 * FLAGS.n_hidden, FLAGS.random_base, keep_prob2, FLAGS.l2_reg, FLAGS.n_class)
    return logits


def lstm_att_sen(inputs, sen_len, keep_prob1, keep_prob2):
    print 'I am dual-att-sen!'
    cell = tf.contrib.rnn.LSTMCell
    inputs = tf.nn.dropout(inputs, keep_prob=keep_prob1)
    hiddens = bi_dynamic_rnn(cell, inputs, FLAGS.n_hidden, sen_len, FLAGS.max_sentence_len, 'sentence_o', 'all')

    alpha = mlp_attention_layer(hiddens, sen_len, 2 * FLAGS.n_hidden, FLAGS.l2_reg, FLAGS.random_base, 'sen')
    outputs = tf.reshape(tf.matmul(alpha, hiddens), [-1, 2 * FLAGS.n_hidden])
    logits = softmax_layer(outputs, 2 * FLAGS.n_hidden, FLAGS.random_base, keep_prob2, FLAGS.l2_reg, FLAGS.n_class)
    return logits


def main(_):
    word_id_mapping, w2v = load_w2v(FLAGS.embedding_file, FLAGS.embedding_dim, True)
    word_embedding = tf.constant(w2v, dtype=tf.float32)

    with tf.name_scope('inputs'):
        keep_prob1 = tf.placeholder(tf.float32)
        keep_prob2 = tf.placeholder(tf.float32)
        x = tf.placeholder(tf.int32, [None, FLAGS.max_sentence_len])
        sen_len = tf.placeholder(tf.int32, [None])
        y = tf.placeholder(tf.float32, [None, FLAGS.n_class])

        inputs = tf.nn.embedding_lookup(word_embedding, x)
        inputs = tf.reshape(inputs, [-1, FLAGS.max_sentence_len, FLAGS.embedding_dim])

        if FLAGS.method.lower() == 'att':
            prob = lstm_att_sen(inputs, sen_len, keep_prob1, keep_prob2)
        else:
            prob = lstm_sen(inputs, sen_len, keep_prob1, keep_prob2)

    with tf.name_scope('loss'):
        reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prob, labels=y)) + tf.add_n(reg_loss)
        all_vars = [var for var in tf.global_variables()]

    with tf.name_scope('train'):
        global_step = tf.Variable(0, name='global_step', trainable=False)
        optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
        grads, global_norm = tf.clip_by_global_norm(tf.gradients(loss, all_vars), 5.0)
        train_op = optimizer.apply_gradients(zip(grads, all_vars), name='train_op', global_step=global_step)

    with tf.name_scope('predict'):
        cor_pred = tf.equal(tf.argmax(prob, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(cor_pred, tf.float32))
        accuracy_num = tf.reduce_sum(tf.cast(cor_pred, tf.int32))

    title = '-d1-{}d2-{}b-{}r-{}l2-{}sen-{}dim-{}h-{}c-{}'.format(
        FLAGS.keep_prob1,
        FLAGS.keep_prob2,
        FLAGS.batch_size,
        FLAGS.learning_rate,
        FLAGS.l2_reg,
        FLAGS.max_sentence_len,
        FLAGS.embedding_dim,
        FLAGS.n_hidden,
        FLAGS.n_class
    )

    conf = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=conf) as sess:
        import time
        timestamp = str(int(time.time()))
        _dir = 'summary/' + str(timestamp) + '_' + title
        test_loss = tf.placeholder(tf.float32)
        test_acc = tf.placeholder(tf.float32)
        train_summary_op, test_summary_op, validate_summary_op, train_summary_writer, test_summary_writer, \
        validate_summary_writer = summary_func(loss, accuracy, test_loss, test_acc, _dir, title, sess)

        save_dir = 'temp_model/' + str(timestamp) + '_' + title + '/'
        saver = saver_func(save_dir)

        init = tf.global_variables_initializer()
        sess.run(init)

        # saver.restore(sess, '/-')

        tr_x, tr_y, tr_sen_len = load_inputs_document_nohn(
            FLAGS.train_file,
            word_id_mapping,
            FLAGS.max_sentence_len
        )
        te_x, te_y, te_sen_len = load_inputs_document_nohn(
            FLAGS.test_file,
            word_id_mapping,
            FLAGS.max_sentence_len
        )

        def get_batch_data(xo, slo, yy, batch_size, kp1, kp2, is_shuffle=True):
            for index in batch_index(len(yy), batch_size, 1, is_shuffle):
                feed_dict = {
                    x: xo[index],
                    y: yy[index],
                    sen_len: slo[index],
                    keep_prob1: kp1,
                    keep_prob2: kp2,
                }
                yield feed_dict, len(index)

        max_acc = 0.
        max_prob = None
        step = 0
        for i in xrange(FLAGS.n_iter):
            for train, _ in get_batch_data(tr_x, tr_sen_len, tr_y, FLAGS.batch_size, FLAGS.keep_prob1, FLAGS.keep_prob2):
                _, step = sess.run([train_op, global_step], feed_dict=train)
                # train_summary_writer.add_summary(summary, step)

            acc, cost, cnt = 0., 0., 0
            p = []
            for test, num in get_batch_data(te_x, te_sen_len, te_y, 2000, 1.0, 1.0, False):
                _loss, _acc, _p = sess.run([loss, accuracy_num, prob], feed_dict=test)
                p += list(_p)
                acc += _acc
                cost += _loss * num
                cnt += num
            print 'all samples={}, correct prediction={}'.format(cnt, acc)
            acc = acc / cnt
            cost = cost / cnt
            print 'Iter {}: mini-batch loss={:.6f}, test acc={:.6f}'.format(i, cost, acc)
            summary = sess.run(test_summary_op, feed_dict={test_loss: cost, test_acc: acc})
            test_summary_writer.add_summary(summary, step)
            if acc > max_acc:
                max_acc = acc
                max_prob = p

        fp = open(FLAGS.prob_file, 'w')
        for item in max_prob:
            fp.write(' '.join([str(it) for it in item]) + '\n')
        print 'Optimization Finished! Max acc={}'.format(max_acc)

        print 'Learning_rate={}, iter_num={}, batch_size={}, hidden_num={}, l2={}'.format(
            FLAGS.learning_rate,
            FLAGS.n_iter,
            FLAGS.batch_size,
            FLAGS.n_hidden,
            FLAGS.l2_reg
        )


if __name__ == '__main__':
    tf.app.run()
