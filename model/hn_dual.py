#!/usr/bin/env python
# encoding: utf-8
# @author: newbie
# email: zhengshiliang0@gmail.com


import os, sys
sys.path.append(os.getcwd())

import numpy as np
import tensorflow as tf

from sklearn.metrics import precision_score, recall_score, f1_score
from utils.config import *
from newbie_nn.nn_layer import dynamic_rnn, bi_dynamic_rnn, softmax_layer
from newbie_nn.att_layer import mlp_attention_layer, softmax_with_len
from utils.data_helper import load_w2v, batch_index, load_word_embedding, load_inputs_document


def hn_inter_att(inputs_o, sen_len_o, doc_len_o, inputs_r, sen_len_r, doc_len_r, keep_prob1, keep_prob2, _id='0'):
    print 'I am hn-inter-att!'
    cell = tf.contrib.rnn.LSTMCell
    # Original Part
    inputs_o = tf.nn.dropout(inputs_o, keep_prob=keep_prob1)
    sen_len_o = tf.reshape(sen_len_o, [-1])
    hiddens_sen_o = bi_dynamic_rnn(cell, inputs_o, FLAGS.n_hidden, sen_len_o, FLAGS.max_sentence_len, 'sentence_o', 'all')
    alpha_sen_o = mlp_attention_layer(hiddens_sen_o, sen_len_o, 2 * FLAGS.n_hidden, FLAGS.l2_reg, FLAGS.random_base, 'sentence_o')
    outputs_sen_o = tf.reshape(tf.matmul(alpha_sen_o, hiddens_sen_o), [-1, FLAGS.max_doc_len, 2 * FLAGS.n_hidden])
    hiddens_doc_o = bi_dynamic_rnn(cell, outputs_sen_o, FLAGS.n_hidden, doc_len_o, FLAGS.max_doc_len, 'doc_o', 'all')

    # Reverse Part
    inputs_r = tf.nn.dropout(inputs_r, keep_prob=keep_prob1)
    sen_len_r = tf.reshape(sen_len_r, [-1])
    hiddens_sen_r = bi_dynamic_rnn(cell, inputs_r, FLAGS.n_hidden, sen_len_r, FLAGS.max_sentence_len, 'sentence_r', 'all')
    alpha_sen_r = mlp_attention_layer(hiddens_sen_r, sen_len_r, 2 * FLAGS.n_hidden, FLAGS.l2_reg, FLAGS.random_base, 'sentence_r')
    outputs_sen_r = tf.reshape(tf.matmul(alpha_sen_r, hiddens_sen_r), [-1, FLAGS.max_doc_len, 2 * FLAGS.n_hidden])
    hiddens_doc_r = bi_dynamic_rnn(cell, outputs_sen_r, FLAGS.n_hidden, doc_len_r, FLAGS.max_doc_len, 'doc_r', 'all')

    # Combined Part
    hiddens_doc = tf.concat([hiddens_doc_o, hiddens_doc_r], 2)  # batch_size * max_doc_len * 4n_hidden
    alpha_doc = mlp_attention_layer(hiddens_doc, doc_len_o, 4 * FLAGS.n_hidden, FLAGS.l2_reg, FLAGS.random_base, 'doc')
    outputs_doc = tf.reshape(tf.matmul(alpha_doc, hiddens_doc), [-1, 4 * FLAGS.n_hidden])
    logits = softmax_layer(outputs_doc, 4 * FLAGS.n_hidden, FLAGS.random_base, keep_prob2, FLAGS.l2_reg, FLAGS.n_class)
    return logits


def main(_):
    # word_id_mapping_o, w2v_o = load_w2v(FLAGS.embedding_file, FLAGS.embedding_dim, True)
    word_id_mapping_o, w2v_o = load_word_embedding(FLAGS.word_id_file, FLAGS.embedding_file, FLAGS.embedding_dim, True)
    word_embedding_o = tf.constant(w2v_o, dtype=tf.float32)
    # word_id_mapping_r, w2v_r = load_w2v(FLAGS.embedding_file_r, FLAGS.embedding_dim, True)
    # word_id_mapping_r, w2v_r = load_word_embedding(FLAGS.word_id_file, FLAGS.embedding_file_r, FLAGS.embedding_dim, True)
    word_id_mapping_r = word_id_mapping_o
    word_embedding_r = tf.constant(w2v_o, dtype=tf.float32)

    with tf.name_scope('inputs'):
        keep_prob1 = tf.placeholder(tf.float32)
        keep_prob2 = tf.placeholder(tf.float32)
        x_o = tf.placeholder(tf.int32, [None, FLAGS.max_doc_len, FLAGS.max_sentence_len])
        x_r = tf.placeholder(tf.int32, [None, FLAGS.max_doc_len, FLAGS.max_sentence_len])
        sen_len_o = tf.placeholder(tf.int32, [None, FLAGS.max_doc_len])
        sen_len_r = tf.placeholder(tf.int32, [None, FLAGS.max_doc_len])
        doc_len_o = tf.placeholder(tf.int32, None)
        doc_len_r = tf.placeholder(tf.int32, None)
        y = tf.placeholder(tf.float32, [None, FLAGS.n_class])

        inputs_o = tf.nn.embedding_lookup(word_embedding_o, x_o)
        inputs_o = tf.reshape(inputs_o, [-1, FLAGS.max_sentence_len, FLAGS.embedding_dim])
        inputs_r = tf.nn.embedding_lookup(word_embedding_r, x_r)
        inputs_r = tf.reshape(inputs_r, [-1, FLAGS.max_sentence_len, FLAGS.embedding_dim])

    prob = hn_inter_att(inputs_o, sen_len_o, doc_len_o, inputs_r, sen_len_r, doc_len_r, keep_prob1, keep_prob2)

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

    true_y = tf.argmax(y, 1)
    pred_y = tf.argmax(prob, 1)

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

    def get_batch_data(xo, slo, dlo, xr, slr, dlr, yy, batch_size, kp1, kp2, is_shuffle=True):
        for index in batch_index(len(yy), batch_size, 1, is_shuffle):
            feed_dict = {
                x_o: xo[index],
                x_r: xr[index],
                y: yy[index],
                sen_len_o: slo[index],
                sen_len_r: slr[index],
                doc_len_o: dlo[index],
                doc_len_r: dlr[index],
                keep_prob1: kp1,
                keep_prob2: kp2,
            }
            yield feed_dict, len(index)

    conf = tf.ConfigProto(allow_soft_placement=True)
    conf.gpu_options.allow_growth = True
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

        tr_x, tr_y, tr_sen_len, tr_doc_len = load_inputs_document(
            FLAGS.train_file,
            word_id_mapping_o,
            FLAGS.max_sentence_len,
            FLAGS.max_doc_len
        )
        te_x, te_y, te_sen_len, te_doc_len = load_inputs_document(
            FLAGS.test_file,
            word_id_mapping_o,
            FLAGS.max_sentence_len,
            FLAGS.max_doc_len
        )
        tr_x_r, tr_y_r, tr_sen_len_r, tr_doc_len_r = load_inputs_document(
            FLAGS.train_file_r,
            word_id_mapping_r,
            FLAGS.max_sentence_len,
            FLAGS.max_doc_len
        )
        te_x_r, te_y_r, te_sen_len_r, te_doc_len_r = load_inputs_document(
            FLAGS.test_file_r,
            word_id_mapping_r,
            FLAGS.max_sentence_len,
            FLAGS.max_doc_len
        )
        # v_x, v_y, v_sen_len, v_doc_len = load_inputs_document(
        #     FLAGS.validate_file_path,
        #     word_id_mapping,
        #     FLAGS.max_sentence_len,
        #     FLAGS.max_doc_len
        # )

        # v_x, v_y, v_sen_len, v_doc_len = load_inputs_document(
        #     FLAGS.validate_file_path,
        #     word_id_mapping,
        #     FLAGS.max_sentence_len,
        #     FLAGS.max_doc_len
        # )

        max_acc, max_prob, step = 0., None, None
        max_ty, max_py = None, None
        for i in xrange(FLAGS.n_iter):
            for train, _ in get_batch_data(tr_x, tr_sen_len, tr_doc_len, tr_x_r, tr_sen_len_r, tr_doc_len_r, tr_y,
                                           FLAGS.batch_size, FLAGS.keep_prob1, FLAGS.keep_prob2):
                _, step, summary = sess.run([train_op, global_step, train_summary_op], feed_dict=train)
                train_summary_writer.add_summary(summary, step)
                # embed_update = tf.assign(word_embedding, tf.concat([tf.zeros([1, FLAGS.embedding_dim]), word_embedding[1:]]), 0)
                # sess.run(embed_update)

            acc, cost, cnt = 0., 0., 0
            p, ty, py = [], [], []
            for test, num in get_batch_data(te_x, te_sen_len, te_doc_len, te_x_r, te_sen_len_r, te_doc_len_r, te_y, FLAGS.batch_size, 1.0, 1.0, False):
                _loss, _acc, _p, _ty, _py = sess.run([loss, accuracy_num, prob, true_y, pred_y], feed_dict=test)
                p += list(_p)
                ty += list(_ty)
                py += list(_py)
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
                max_ty = ty
                max_py = py
                # saver.save(sess, save_dir, global_step=step)

        print 'P:', precision_score(max_ty, max_py, average=None)
        print 'R:', recall_score(max_ty, max_py, average=None)
        print 'F:', f1_score(max_ty, max_py, average=None)

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
