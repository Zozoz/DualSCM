#!/usr/bin/env python
# encoding: utf-8
# @author: newbie
# email: zhengshiliang0@gmail.com


import os, sys


data_dir = 'analysis_dir'
label_name = ['neg', 'pos']


def split_test_to_error_and_right(test_f, prob_f, error_f=None, right_f=None):
    if error_f is None:
        error_f = 'error.txt'
    if right_f is None:
        right_f = 'right.txt'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    ep = open(os.path.join(data_dir, error_f), 'w')
    rp = open(os.path.join(data_dir, right_f), 'w')
    cnt = 1
    for prob, line in zip(open(prob_f), open(test_f)):
        label = line.split('||')[0].strip()
        prob = [float(item) for item in prob.split()]
        max_index = prob.index(max(prob))
        if label_name.index(label) == max_index:
            rp.write(str(cnt) + '||' + line)
        else:
            ep.write(str(cnt) + '||' + label_name[max_index] + '||' + line)
        cnt += 1
    ep.close()
    rp.close()


if __name__ == '__main__':
    split_test_to_error_and_right(sys.argv[1], sys.argv[2])
