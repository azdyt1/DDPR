import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from model_prm import PRM
from model_dlcm import DLCM

#from model_dien import DIEN
import numpy as np
import time
from tqdm import tqdm
import random
from sklearn.metrics import roc_auc_score
import sys
import math
from copy import deepcopy
#from utils import *

def compute_emb_similarity(embedding1, embedding2):
    embedding1 = np.reshape(embedding1, [-1])
    embedding2 = np.reshape(embedding2, [-1])
    num = np.dot(embedding1, embedding2.T)
    denom = np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
    cos = num / denom
    # cos = cos * 0.5 + 0.5
    return cos

def compute_emb_distance(embedding1, embedding2):
    embedding1 = np.reshape(embedding1, [-1])
    embedding2 = np.reshape(embedding2, [-1])
    return np.linalg.norm(embedding1 - embedding2)

def dcg_score(y_true, y_score, k=20):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gains = 2 ** y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)


def ndcg_score(y_true, y_score, k=20):
    best = dcg_score(y_true, y_true, k)
    actual = dcg_score(y_true, y_score, k)
    return actual / best

def ild_compute(emb_list):
    val = 0.0
    for i in range(len(emb_list)):
        for j in range(len(emb_list)):
            if i == j:
                continue
            val += (1 - np.sqrt(np.sum(np.square(emb_list[i]-emb_list[j]), -1)))
    return val / (len(emb_list)*len(emb_list)-1)

def mr_compute(sku_list):
    tot = 0.0
    for i in range(len(sku_list)):
        tot = tot + (0.85**(sku_list[i]['rank']-1)) * sku_list[i]['prediction']

    return tot / len(sku_list)

def epc_compute(sku_list, sku_novelty_map):
    tot = 0.0
    for i in range(len(sku_list)):
        tot = tot + (0.85 ** (sku_list[i]['rank'] - 1)) * sku_novelty_map[sku_list[i]['skuid']]

    return tot / len(sku_list)

def diversity_compute(cate_list):
    val = 0.0
    for i in range(len(cate_list)):
        for j in range(i+1, len(cate_list)):
            val += cate_list[i] != cate_list[j]
    return val / (len(cate_list)*(len(cate_list)-1)/2)

def entropy_compute(cate_list):
    val = 0.0
    cate = {}
    for i in range(len(cate_list)):
        if cate_list[i] not in cate.keys():
            cate[cate_list[i]] = 0
        cate[cate_list[i]] += 1
    for k in cate.keys():
        pro = cate[k] / len(cate_list)
        val += pro*math.log(pro)
    return val


input_format = {
    'num_epoch': 1000,
    'batch_size': 256,
    'test_batch_size': 256,
    'user_num': 10000,
    'item_num': 70000,
    'cate_num': 2000,
    'nums_expert':1,
    'expert_units': 64,
    'final_layer_hidden_units': [[64],[32]],
    'final_layer_activation': tf.nn.relu,
    'nums_label':1,
    'dims': 16,
    'seq_dims': 16,
    'eval_topk': 20,
    'rerank_topk': 100,
    'matching_seq_max_len': 200,
    'his_long_seq_max_len': 100,
    'his_short_seq_max_len': 25,
    'lambdaConstantMMR': 0.8,
    'lambdaConstantDPP': 0.9,
    'lambdaConstantCATE': 0.01,
    'lambdaConstantPMFA': 0.5,
    'lambdaConstantPMFB': 0.5,
    'lambdaConstantSSD': 0.01,
    'loss': 'logits_loss',
    'max_train_steps': None,
    'optimizer': tf.train.AdamOptimizer(learning_rate=0.001),
    'blocks': 1,
    'block_shape': [32],
    'heads': 1,
    'is_drop': False,
    'dropout_keep_prob': 1,
    'label_name': ['label'],
    'dense_dropout_keep_prob': 1.0,
    'patience': 10
}


user_train_matching_seq_map = {}
user_valid_matching_seq_map = {}
user_test_matching_seq_map = {}

user_train_long_seq_map = {}
user_valid_long_seq_map = {}
user_test_long_seq_map = {}

item_list = []
cate_list = []
his_item_seq_list = []
his_cate_seq_list = []
rank_item_seq_list = []
rank_label_seq_list = []
rank_pos_seq_list = []
match_item_seq_list = []
match_cate_seq_list = []
label_list = []
weight_list = []
user_list = []
query_list = []

item_valid_list = []
cate_valid_list = []
his_item_seq_valid_list = []
his_cate_seq_valid_list = []
rank_item_seq_valid_list = []
rank_label_seq_valid_list = []
rank_pos_seq_valid_list = []
match_item_seq_valid_list = []
match_cate_seq_valid_list = []
label_valid_list = []
weight_valid_list = []
user_valid_list = []
query_valid_list = []

item_test_list = []
cate_test_list = []
his_item_seq_test_list = []
his_cate_seq_test_list = []
rank_item_seq_test_list = []
rank_label_seq_test_list = []
rank_pos_seq_test_list = []
match_item_seq_test_list = []
match_cate_seq_test_list = []
label_test_list = []
weight_test_list = []
user_test_list = []
query_test_list = []

def load_sku_info_map(sku_info_file):
    print('begin load sku info:')
    sku_info_map = {}
    sku_info_map[0] = {'sku_id':0, 'cate_id':0}
    with open(sku_info_file, 'r') as f:
        for line in f:
            line_split = line.strip('\n').split(',')
            item_id = int(line_split[0])
            cate_id = int(line_split[1])
            sku_info_map[item_id] = {'sku_id':item_id, 'cate_id':cate_id}
        return sku_info_map

def read_usercf_seq(file_path_user_matching_seq):
    print('begin load user_cf seq:')
    with open(file_path_user_matching_seq, 'r') as f:
        first = True
        for line in tqdm(f, total=10000):
            if first == True:
                first = False
                continue
            line_split = line.strip('\n').split('\t')
            uid = line_split[0]
            if line_split[1] == '' or line_split[1] == 'NULL':
                matching_seq = [0] * 200
            else:
                matching_seq = [int(i) for i in line_split[1].split('-')] + [0]*200
                matching_seq = matching_seq[:200]
            user_train_matching_seq_map[uid] = matching_seq

def read_user_comirec_matching_seq(file_path_train_matching_seq, file_path_valid_matching_seq, file_path_test_matching_seq, max_len):
    print('begin load user matching seq:')
    with open(file_path_train_matching_seq, 'r') as f:
        first = True
        for line in tqdm(f, total=1000000):
            if first == True:
                first = False
                continue
            line_split = line.strip('\n').split('\t')
            uid = line_split[0]
            if line_split[1] == '' or line_split[1] == 'NULL':
                matching_seq = [0] * max_len
            else:
                matching_seq = [int(i) for i in line_split[1].split('-')] + [0]*max_len
                matching_seq = matching_seq[:max_len]
            user_train_matching_seq_map[uid] = matching_seq

    with open(file_path_valid_matching_seq, 'r') as f:
        first = True
        for line in tqdm(f, total=100000):
            if first == True:
                first = False
                continue
            line_split = line.strip('\n').split('\t')
            uid = line_split[0]
            if line_split[1] == '' or line_split[1] == 'NULL':
                matching_seq = [0] * max_len
            else:
                matching_seq = [int(i) for i in line_split[1].split('-')] + [0] * max_len
                matching_seq = matching_seq[:max_len]
            user_valid_matching_seq_map[uid] = matching_seq

    with open(file_path_test_matching_seq, 'r') as f:
        first = True
        for line in tqdm(f, total=100000):
            if first == True:
                first = False
                continue
            line_split = line.strip('\n').split('\t')
            uid = line_split[0]
            if line_split[1] == '' or line_split[1] == 'NULL':
                matching_seq = [0] * max_len
            else:
                matching_seq = [int(i) for i in line_split[1].split('-')] + [0] * max_len
                matching_seq = matching_seq[:max_len]
            user_test_matching_seq_map[uid] = matching_seq

def read_user_his_long_seq(file_path_train_long_seq, file_path_valid_long_seq, file_path_test_long_seq, max_len):
    print('begin load user history long seq:')
    with open(file_path_train_long_seq, 'r') as f:
        first = True
        for line in tqdm(f, total=1000000):
            if first == True:
                first = False
                continue
            line_split = line.strip('\n').split('\t')
            uid = line_split[0]
            if line_split[1] == '' or line_split[1] == 'NULL':
                matching_seq = [0] * max_len
            else:
                matching_seq = [int(i) for i in line_split[1].split('-')] + [0]*max_len
                matching_seq = matching_seq[:max_len]
            user_train_long_seq_map[uid] = matching_seq

    with open(file_path_valid_long_seq, 'r') as f:
        first = True
        for line in tqdm(f, total=100000):
            if first == True:
                first = False
                continue
            line_split = line.strip('\n').split('\t')
            uid = line_split[0]
            if line_split[1] == '' or line_split[1] == 'NULL':
                matching_seq = [0] * max_len
            else:
                matching_seq = [int(i) for i in line_split[1].split('-')] + [0] * max_len
                matching_seq = matching_seq[:max_len]
            user_valid_long_seq_map[uid] = matching_seq

    with open(file_path_test_long_seq, 'r') as f:
        first = True
        for line in tqdm(f, total=100000):
            if first == True:
                first = False
                continue
            line_split = line.strip('\n').split('\t')
            uid = line_split[0]
            if line_split[1] == '' or line_split[1] == 'NULL':
                matching_seq = [0] * max_len
            else:
                matching_seq = [int(i) for i in line_split[1].split('-')] + [0] * max_len
                matching_seq = matching_seq[:max_len]
            user_test_long_seq_map[uid] = matching_seq

def read_train_file(file_path_train):
    print('read train file:')
    cnt = 0
    with open(file_path_train, 'r') as f:
        first = True
        for line in tqdm(f, total=3000000):
            if first == True:
                first = False
                continue
            # cnt += 1
            # if cnt >= 10000:
            #     break
            line_split = line.strip('\n').split('\t')
            queryid = line_split[0]
            rank_item_seq = line_split[1]
            rank_label_seq = line_split[2]
            rank_pos_seq = line_split[3]

            if rank_item_seq == '' or rank_item_seq == 'NULL':
                rank_item_seq_padding = [0] * 200
            else:
                rank_item_seq_padding = [int(x) for x in rank_item_seq.split('-')] + [0] * 200
                rank_item_seq_padding = rank_item_seq_padding[:200]

            if rank_label_seq == '' or rank_label_seq == 'NULL':
                rank_label_seq_padding = [0] * 200
            else:
                rank_label_seq_padding = [int(x) for x in rank_label_seq.split('-')] + [0] * 200
                rank_label_seq_padding = rank_label_seq_padding[:200]

            if rank_pos_seq == '' or rank_pos_seq == 'NULL':
                rank_pos_seq_padding = [0] * 200
            else:
                rank_pos_seq_padding = [int(x) for x in rank_pos_seq.split('-')] + [0] * 200
                rank_pos_seq_padding = rank_pos_seq_padding[:200]

            rank_item_seq_list.append(rank_item_seq_padding)
            rank_label_seq_list.append(rank_label_seq_padding)
            rank_pos_seq_list.append(rank_pos_seq_padding)
            query_list.append(queryid)

def read_valid_file(file_path_train):
    print('read valid file:')
    cnt = 0
    with open(file_path_train, 'r') as f:
        first = True
        for line in tqdm(f, total=200000):
            if first == True:
                first = False
                continue
            # cnt += 1
            # if cnt >= 1000:
            #     break
            line_split = line.strip('\n').split('\t')
            queryid = line_split[0]
            rank_item_seq = line_split[1]
            rank_label_seq = line_split[2]
            rank_pos_seq = line_split[3]

            if rank_item_seq == '' or rank_item_seq == 'NULL':
                rank_item_seq_padding = [0] * 200
            else:
                rank_item_seq_padding = [int(x) for x in rank_item_seq.split('-')] + [0] * 200
                rank_item_seq_padding = rank_item_seq_padding[:200]

            if rank_label_seq == '' or rank_label_seq == 'NULL':
                rank_label_seq_padding = [0] * 200
            else:
                rank_label_seq_padding = [int(x) for x in rank_label_seq.split('-')] + [0] * 200
                rank_label_seq_padding = rank_label_seq_padding[:200]

            if rank_pos_seq == '' or rank_pos_seq == 'NULL':
                rank_pos_seq_padding = [0] * 200
            else:
                rank_pos_seq_padding = [int(x) for x in rank_pos_seq.split('-')] + [0] * 200
                rank_pos_seq_padding = rank_pos_seq_padding[:200]

            rank_item_seq_valid_list.append(rank_item_seq_padding)
            rank_label_seq_valid_list.append(rank_label_seq_padding)
            rank_pos_seq_valid_list.append(rank_pos_seq_padding)
            query_valid_list.append(queryid)

def read_test_file(file_path_test):
    print('read test file:')
    cnt = 0
    with open(file_path_test, 'r') as f:
        first = True
        for line in tqdm(f, total=200000):
            if first == True:
                first = False
                continue
            #cnt += 1
            #if cnt >= 10000:
            #    break
            line_split = line.strip('\n').split('\t')
            queryid = line_split[0]
            rank_item_seq = line_split[1]
            rank_label_seq = line_split[2]
            rank_pos_seq = line_split[3]

            if rank_item_seq == '' or rank_item_seq == 'NULL':
                rank_item_seq_padding = [0] * 200
            else:
                rank_item_seq_padding = [int(x) for x in rank_item_seq.split('-')] + [0] * 200
                rank_item_seq_padding = rank_item_seq_padding[:200]

            if rank_label_seq == '' or rank_label_seq == 'NULL':
                rank_label_seq_padding = [0] * 200
            else:
                rank_label_seq_padding = [int(x) for x in rank_label_seq.split('-')] + [0] * 200
                rank_label_seq_padding = rank_label_seq_padding[:200]

            if rank_pos_seq == '' or rank_pos_seq == 'NULL':
                rank_pos_seq_padding = [0] * 200
            else:
                rank_pos_seq_padding = [int(x) for x in rank_pos_seq.split('-')] + [0] * 200
                rank_pos_seq_padding = rank_pos_seq_padding[:200]

            rank_item_seq_test_list.append(rank_item_seq_padding)
            rank_label_seq_test_list.append(rank_label_seq_padding)
            rank_pos_seq_test_list.append(rank_pos_seq_padding)
            query_test_list.append(queryid)


def generate_train_batch(batch_data_index):

    rank_item_seq = [rank_item_seq_list[i] for i in batch_data_index]
    rank_cate_seq = []
    for l in rank_item_seq:
        rank_cate_seq.append([sku_info_map[i]['cate_id'] for i in l])
    rank_label_seq = [rank_label_seq_list[i] for i in batch_data_index]
    rank_pos_seq = [rank_pos_seq_list[i] for i in batch_data_index]

    query = np.reshape([query_list[i] for i in batch_data_index], [-1,1])

    return [rank_item_seq, rank_cate_seq, rank_label_seq, rank_pos_seq, query]


def generate_valid_batch(batch_size, global_index):

    rank_item_seq = rank_item_seq_valid_list[global_index-batch_size:global_index]
    rank_cate_seq = []
    for l in rank_item_seq:
        rank_cate_seq.append([sku_info_map[i]['cate_id'] for i in l])
    rank_label_seq = rank_label_seq_valid_list[global_index-batch_size:global_index]
    rank_pos_seq = rank_pos_seq_valid_list[global_index-batch_size:global_index]

    query = np.reshape(query_valid_list[global_index-batch_size:global_index], [-1, 1])

    return [rank_item_seq, rank_cate_seq, rank_label_seq, rank_pos_seq, query]

def generate_test_batch(batch_size, global_index):

    rank_item_seq = rank_item_seq_test_list[global_index - batch_size:global_index]
    rank_cate_seq = []
    for l in rank_item_seq:
        rank_cate_seq.append([sku_info_map[i]['cate_id'] for i in l])
    rank_label_seq = rank_label_seq_test_list[global_index - batch_size:global_index]
    rank_pos_seq = rank_pos_seq_test_list[global_index - batch_size:global_index]

    query = np.reshape(query_test_list[global_index - batch_size:global_index], [-1, 1])

    return [rank_item_seq, rank_cate_seq, rank_label_seq, rank_pos_seq, query]

def eval_full_data(trace_list, sku_info_map, emb_sku, flag='rank'):

    coverage = []
    ild = []
    aucs_ctr = []
    ndcgs_ctr = []
    diversity = []
    entropy = []
    cnt_ctr_label = 0
    cnt_trace_20 = 0
    topN = 50

    print('evaluate all trace:')
    for trace_id, sku_list in tqdm(trace_list.items()):
        cnt_trace_20 += 1
        # sku_sorted_list = sorted(sku_list, key=lambda item: -item['prediction'], reverse=False)
        if flag == 'rank':
            sku_sorted_list = sorted(sku_list, key=lambda x: x['prediction'], reverse=True)
        else:
            sku_sorted_list = sku_list

        sku_id_list = [x['skuid'] for x in sku_sorted_list]
        label_ctr = [x['ctr_label'] for x in sku_sorted_list]
        new_score = [-i for i in range(len(sku_sorted_list))]

        if sum(label_ctr) != 0.0 and sum(label_ctr) != len(label_ctr):
            aucs_ctr.append(roc_auc_score(label_ctr, new_score))
            ndcgs_ctr.append(ndcg_score(label_ctr, new_score, len(label_ctr)))
            cnt_ctr_label += 1

            sku_id_list = sku_id_list[:topN]
            coverage.append(len(set([sku_info_map[x]['cate_id'] for x in sku_id_list])))
            ild.append(ild_compute([emb_sku[i] for i in sku_id_list]))
            diversity.append(diversity_compute([sku_info_map[x]['cate_id'] for x in sku_id_list]))
            entropy.append(entropy_compute([sku_info_map[x]['cate_id'] for x in sku_id_list]))

    return np.mean(aucs_ctr), np.mean(ndcgs_ctr), np.mean(coverage), np.mean(ild), np.mean(diversity), np.mean(entropy)


def CATEMAX(sku_list, sku_info_map, lambdaConstant=0.2):
    s, r = [], sku_list.copy()
    r = sorted(r,key=lambda item:-item['prediction'])
    new_score = []
    len_r = len(r)
    while len(s) < len_r:
        score = 0.0
        selectOne = None
        second = 0
        for i in range(len(r)):
            item_dict = r[i]
            firstPart = item_dict['prediction']
            secondPart = 0.0
            for j in s:
                # sim2 = cosine_compute(sku_embedding_map[i], sku_embedding_map[j])
                if sku_info_map[item_dict['skuid']]['cate_id'] != sku_info_map[j['skuid']]['cate_id']:
                    secondPart += 1.0

            equationScore = firstPart + lambdaConstant * secondPart
            if equationScore > score:
                score = equationScore
                selectOne = i
                second = secondPart
        if selectOne == None:
            selectOne = i
        s.append(r[selectOne])
        # new_score.append(str(round(score,4)) + '-' + str(round(r[selectOne]['prediction'],4)) + '-' + str(second))
        r.pop(selectOne)
    return s, new_score

def MMR(sku_list, sku_emb, lambdaConstant=0.9):
    s, r = [], sku_list.copy()
    r = sorted(r,key=lambda item:-item['prediction'])
    new_score = []
    len_r = len(r)
    while len(s) < len_r:
        score = -10000.0
        selectOne = None
        for i in range(len(r)):
            item_dict = r[i]
            firstPart = item_dict['prediction']
            item_id1 = int(item_dict['skuid'])
            secondPart = -1.0
            for j in s:
                item_id2 = int(j['skuid'])
                sim2 = compute_emb_similarity(sku_emb[item_id1], sku_emb[item_id2])
                if sim2 > secondPart:
                    secondPart = sim2
            equationScore = lambdaConstant * firstPart - (1-lambdaConstant) * secondPart
            if equationScore > score:
                score = equationScore
                selectOne = i
                second = secondPart
        if selectOne == None:
            selectOne = i
        s.append(r[selectOne])
        # new_score.append(str(score) + '-' + str(r[selectOne]['modelscore']) + '-' + str(second))
        r.pop(selectOne)
    # if len(s) < len_r:
    #     s.extend(r)
    return s, new_score

def PMF(sku_list, sku_emb, lambdaConstantA=0.5, lambdaConstantB=0.5):
    s, r = [], sku_list.copy()
    r = sorted(r,key=lambda item:-item['prediction'])
    new_score = []
    len_r = len(r)
    while len(s) < len_r:
        score = -10000.0
        selectOne = None
        second = 0.0
        for i in range(len(r)):
            item_dict = r[i]
            firstPart = item_dict['prediction']
            item_id1 = int(item_dict['skuid'])
            secondPart = -10000.0
            thirdPart = -10000.0
            for j in s:
                item_id2 = int(j['skuid'])
                ic = np.dot(sku_emb[item_id1], sku_emb[item_id2])
                if ic > secondPart:
                    secondPart = ic
                ed = compute_emb_distance(sku_emb[item_id1], sku_emb[item_id2])
                if ed > thirdPart:
                    thirdPart = ed

            equationScore = firstPart + lambdaConstantA * secondPart + lambdaConstantB * thirdPart
            if equationScore > score:
                score = equationScore
                selectOne = i
                second = secondPart
        if selectOne == None:
            selectOne = i
        s.append(r[selectOne])
        # new_score.append(str(score) + '-' + str(r[selectOne]['modelscore']) + '-' + str(second))
        r.pop(selectOne)
    # if len(s) < len_r:
    #     s.extend(r)
    return s, new_score

def similar_matrix_gen(sku_list, sku_emb):
    similar_matrix = np.ones([len(sku_list) + 1, len(sku_list) + 1])
    idx = 0
    new_index = {}
    for sku in sku_list:
        new_index[sku['skuid']] = idx
        idx += 1
    skuid_list = [int(sku['skuid']) for sku in sku_list]
    for sku_root in skuid_list:
        for sku_one in skuid_list:
            sim = compute_emb_similarity(sku_emb[sku_root], sku_emb[sku_one])
            similar_matrix[new_index[sku_root], new_index[sku_one]] = sim

    return new_index, similar_matrix

def DPP(sku_list, sku_emb, lambdaConstant=0.7):

    new_index, similar_matrix = similar_matrix_gen(sku_list, sku_emb)

    item_index = []
    idx = 0
    idx_map = {}
    max_iter = len(sku_list)
    for i in range(len(sku_list)):
        item_index.append(new_index[sku_list[i]['skuid']])
        idx_map[idx] = sku_list[i]
        idx += 1
    sim_matrix = similar_matrix[item_index] # item之间的相似度矩阵
    sim_matrix = sim_matrix[:,item_index]
    rank_score = np.array([x['prediction'] for x in sku_list])
    kernel_matrix = rank_score.reshape((len(sku_list), 1)) \
                         * sim_matrix * rank_score.reshape((1, len(sku_list)))

    c = np.zeros((max_iter, len(sku_list)))
    d = np.copy(np.diag(kernel_matrix))
    j = np.argmax(d)
    Yg = [j]
    iter = 0
    new_score = [d[j]]
    Z = list(range(len(sku_list)))

    while len(Yg) < len(sku_list):
        Z_Y = set(Z).difference(set(Yg))
        for i in Z_Y:
            if iter == 0:
                ei = kernel_matrix[j, i] / np.sqrt(d[j])
            else:
                ei = (kernel_matrix[j, i] - np.dot(c[:iter, j], c[:iter, i])) / np.sqrt(d[j])
            c[iter, i] = ei
            d[i] = abs(d[i] - ei * ei)
        d[j] = 0.0
        j = np.argmax(d)
        if d[j] == 0.0:
            break
        # if d[j] < epsilon:
        #     break
        Yg.append(j)
        new_score.append(d[j])
        iter += 1
    # print(d)
    # print('len_origin:',len(itemScoreDict))
    # print('len_Yg:',len(Yg))
    Yg = [idx_map[i] for i in Yg]
    return Yg, new_score

def RLprop(sku_list, sku_emb, sku_novelty_map):
    s, r = [], sku_list.copy()
    r = sorted(r,key=lambda item:-item['prediction'])
    new_score = []
    len_r = len(r)

    TOT = 0.0
    g_mr = 0.0
    g_ild = 0.0
    g_epc = 0.0

    while len(s) < len_r:
        score = -10000.0
        selectOne = None

        g_mr_select = 0.0
        g_ild_select = 0.0
        g_epc_select = 0.0

        for i in range(len(r)):

            #g_mr_i = (0.85 ** (r[i]['rank'] - 1)) * r[i]['prediction']
            g_mr_i = r[i]['prediction']
            g_ild_i = 0.0
            for j in s:
                temp = np.sqrt(np.sum(np.square(sku_emb[r[i]['skuid']] - sku_emb[j['skuid']]), -1))
                if temp > g_ild_i:
                    g_ild_i = temp
            #g_epc_i = (0.85 ** (r[i]['rank'] - 1)) * sku_novelty_map[r[i]['skuid']]
            g_epc_i = sku_novelty_map[r[i]['skuid']]

            TOT_temp = max(TOT, TOT+g_mr_i+g_ild_i+g_epc_i)

            r_mr = TOT_temp * 0.4 - g_mr
            r_ild = TOT_temp * 0.3 - g_ild
            r_epc = TOT_temp * 0.3 - g_epc

            g_i = 0.0

            if g_mr_i >= 0:
                g_i = g_i + max(0.0, min(g_mr_i, r_mr))
            else:
                g_i = g_i + min(0.0, g_mr_i-r_mr)

            if g_ild_i >= 0:
                g_i = g_i + max(0.0, min(g_ild_i, r_ild))
            else:
                g_i = g_i + min(0.0, g_ild_i-r_ild)

            if g_epc_i >= 0:
                g_i = g_i + max(0.0, min(g_epc_i, r_epc))
            else:
                g_i = g_i + min(0.0, g_epc_i-r_epc)

            equationScore = g_i

            if equationScore > score:
                score = equationScore
                selectOne = i
                g_mr_select = g_mr_i
                g_ild_select = g_ild_i
                g_epc_select = g_epc_i

        if selectOne == None:
            selectOne = i
        s.append(r[selectOne])
        r.pop(selectOne)

        g_mr = g_mr + g_mr_select
        g_ild = g_ild + g_ild_select
        g_epc = g_epc + g_epc_select
        TOT = max(0.0, g_mr) + max(0.0, g_ild) + max(0.0, g_epc)

    return s, new_score

def SSD(sku_list, sku_emb, lambdaConstantA=0.5):
    s, r = [], sku_list.copy()
    sku_emb_copy = deepcopy(sku_emb)
    r = sorted(r,key=lambda item:-item['prediction'])
    new_score = []
    len_r = len(r)
    max_val = -10000.0
    max_idx = 0
    for i in range(len_r):
        if max_val < r[i]['prediction']:
            max_val = r[i]['prediction']
            max_idx = i
    s.append(r[max_idx])
    r.pop(max_idx)

    v = lambdaConstantA * np.linalg.norm(np.reshape(sku_emb_copy[r[max_idx]['skuid']], [-1]))
    item_id_pre = max_idx

    while len(s) < len_r:
        score = -10000.0
        selectOne = None
        for i in range(len(r)):
            item_dict = r[i]
            rel = item_dict['prediction']
            item_id = int(item_dict['skuid'])

            sku_emb_copy[item_id] = sku_emb_copy[item_id] - (np.dot(sku_emb_copy[item_id], sku_emb_copy[item_id_pre]) / np.dot(sku_emb_copy[item_id_pre], sku_emb_copy[item_id_pre])) * sku_emb_copy[item_id_pre]
            vv = np.linalg.norm(sku_emb_copy[item_id])*v

            equationScore = rel + vv
            if equationScore > score:
                score = equationScore
                selectOne = i

        if selectOne == None:
            selectOne = i
        s.append(r[selectOne])
        r.pop(selectOne)
        item_id_pre = selectOne
        v = v * np.linalg.norm(sku_emb_copy[selectOne])

    return s, new_score

def train(sku_info_map, exp_name, model_name, sku_novelty_map):
    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        input_format['data_type'] = data_type
        if model_name == 'PRM':
            model = PRM(input_format)
        elif model_name == 'DLCM':
            model = DLCM(input_format)
        else:
            model = PRM(input_format)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        global_auc = 0.
        global_patience = 0

        file_log = open(exp_name+'_log.txt', 'w')

        best_model_path = './'+exp_name+'/best_model/'

        print('start training:')
        start_time = time.time()
        # train
        try:
            for i in range(input_format['num_epoch']):
                new_index = np.arange(len(user_list)).tolist()
                np.random.shuffle(new_index)

                loss_sum = 0
                print('train epoch:', i)
                file_log.write('train epoch: ' + str(i) + '\n')
                pbar = tqdm(total=len(user_list)//input_format['batch_size'])
                global_index = input_format['batch_size']
                while global_index <= len(user_list):
                    batch_data = generate_train_batch(new_index[global_index-input_format['batch_size']:global_index])
                    loss, op_ = model.train(batch_data, sess, data_type)
                    loss_sum += loss
                    global_index += input_format['batch_size']
                    pbar.update(1)
                pbar.close()
                print('training loss:{}, global_step:{}'.format(loss_sum / (len(user_list)//input_format['batch_size']), i))
                file_log.write('training loss: '+str(loss_sum / (len(user_list)//input_format['batch_size']))+', global_step: '+ str(i)+'\n')

                time_cur = time.time()
                time_cost = (time_cur-start_time)/60.0

                print('time for training {} epoch is {} min'.format(i, time_cost))
                file_log.write('time for training: '+str(i)+' epoch is '+str(time_cost)+' min'+'\n')

                # valid
                if i % 1 == 0:
                    print('begin valid after epoch:', i)
                    file_log.write('begin valid after epoch: ' + str(i) + '\n')
                    trace = {}
                    sku_emb = model.get_sku_emb(sess)
                    global_index = input_format['batch_size']
                    while global_index <= len(user_valid_list):
                        batch_data = generate_valid_batch(input_format['batch_size'], global_index)
                        predict = model.test(batch_data, sess, data_type)
                        predict_l = np.reshape(predict, [-1]).tolist()
                        skuid_l = np.reshape(batch_data[0], [-1]).tolist()
                        uid_l = np.reshape(batch_data[8], [-1]).tolist()
                        label_l = np.reshape(batch_data[6], [-1]).tolist()
                        for k in range(len(uid_l)):
                            uid = uid_l[k]
                            skuid = skuid_l[k]
                            predict_k = predict_l[k]
                            label_k = label_l[k]
                            if uid not in trace.keys():
                                trace[uid] = []
                            trace[uid].append({'prediction':predict_k, 'ctr_label':label_k, 'skuid':skuid})
                        global_index += input_format['batch_size']

                    auc, ndcg, coverage, ild, div, entropy = eval_full_data(trace, sku_info_map, sku_emb, 'rank')
                    if auc > global_auc:
                        global_auc = auc
                        global_patience = 0
                        model.save(sess, best_model_path)
                    else:
                        global_patience += 1
                        if global_patience >= input_format['patience']:
                            break
                    print('auc:{}, ndcg:{}, coverage:{}, ild:{}, div:{}, entropy:{}, global_step:{}'.format(auc,ndcg,coverage,ild,div,entropy, i))
                    file_log.write('auc: '+str(auc)+', ndcg: '+str(ndcg)+', coverage: '+str(coverage)+', ild: '+str(ild)+', div: '+str(div)+', entropy: '+str(entropy)+', global_step: '+str(i)+'\n\n')
        except KeyboardInterrupt:
            model.save(sess, best_model_path)

        model.restore(sess, best_model_path)
        file_log.write('begin test after epoch: ' + str(i) + '\n')
        print('begin test after epoch: ', i)
        trace = {}
        sku_emb = model.get_sku_emb(sess)
        global_index = input_format['batch_size']
        while global_index <= len(user_test_list):
            batch_data = generate_test_batch(input_format['batch_size'], global_index)
            predict = model.test(batch_data, sess, data_type)
            predict_l = np.reshape(predict, [-1]).tolist()
            skuid_l = np.reshape(batch_data[0], [-1]).tolist()
            uid_l = np.reshape(batch_data[8], [-1]).tolist()
            label_l = np.reshape(batch_data[6], [-1]).tolist()
            for k in range(len(uid_l)):
                uid = uid_l[k]
                skuid = skuid_l[k]
                predict_k = predict_l[k]
                label_k = label_l[k]
                if uid not in trace.keys():
                    trace[uid] = []
                trace[uid].append({'prediction': predict_k, 'ctr_label': label_k, 'skuid': skuid})
            global_index += input_format['batch_size']

        file_log.write(model_name +' model results:\n')
        print(model_name + ' model results:')

        auc, ndcg, coverage, ild, div, entropy = eval_full_data(trace, sku_info_map, sku_emb, 'rank')
        file_log.write('auc: ' + str(auc) + ', ndcg: ' + str(ndcg) + ', coverage: ' + str(coverage) + ', ild: ' + str(
            ild) + ', div: ' + str(div) + ', entropy: ' + str(entropy) + ', global_step: ' + str(i) + '\n')
        print('auc:{}, ndcg:{}, coverage:{}, ild:{}, div:{}, entropy{}'.format(auc, ndcg, coverage, ild, div, entropy))

        file_log.write(model_name + '+Re-ranking model results:\n')
        print(model_name + '+Re-ranking model results:\n')
        trace_mmr, trace_dpp, trace_cate, trace_pmf, trace_ssd, trace_rl = rerank_trace(trace,
                                                                                        input_format['rerank_topk'],
                                                                                        input_format[
                                                                                            'lambdaConstantMMR'],
                                                                                        input_format[
                                                                                            'lambdaConstantDPP'],
                                                                                        input_format[
                                                                                            'lambdaConstantCATE'],
                                                                                        input_format[
                                                                                            'lambdaConstantPMFA'],
                                                                                        input_format[
                                                                                            'lambdaConstantPMFB'],
                                                                                        input_format[
                                                                                            'lambdaConstantSSD'],
                                                                                        sku_emb, sku_info_map,
                                                                                        sku_novelty_map)
        auc_mmr, ndcg_mmr, coverage_mmr, ild_mmr, div_mmr, entropy_mmr = eval_full_data(trace_mmr, sku_info_map,
                                                                                        sku_emb, 're-rank')
        auc_dpp, ndcg_dpp, coverage_dpp, ild_dpp, div_dpp, entropy_dpp = eval_full_data(trace_dpp, sku_info_map,
                                                                                        sku_emb, 're-rank')
        auc_cate, ndcg_cate, coverage_cate, ild_cate, div_cate, entropy_cate = eval_full_data(trace_cate, sku_info_map,
                                                                                              sku_emb,
                                                                                              're-rank')
        auc_pmf, ndcg_pmf, coverage_pmf, ild_pmf, div_pmf, entropy_pmf = eval_full_data(trace_pmf, sku_info_map,
                                                                                        sku_emb, 're-rank')
        auc_ssd, ndcg_ssd, coverage_ssd, ild_ssd, div_ssd, entropy_ssd = eval_full_data(trace_ssd, sku_info_map,
                                                                                        sku_emb, 're-rank')
        auc_rl, ndcg_rl, coverage_rl, ild_rl, div_rl, entropy_rl = eval_full_data(trace_rl, sku_info_map, sku_emb,
                                                                                  're-rank')

        file_log.write(
            'auc_mmr: ' + str(auc_mmr) + ', ndcg_mmr: ' + str(ndcg_mmr) + ', coverage_mmr: ' + str(
                coverage_mmr) + ', ild_mmr: ' + str(
                ild_mmr) + ', div_mmr: ' + str(div_mmr) + ', entropy_mmr: ' + str(entropy_mmr) + '\n')
        file_log.write(
            'auc_dpp: ' + str(auc_dpp) + ', ndcg_dpp: ' + str(ndcg_dpp) + ', coverage_dpp: ' + str(
                coverage_dpp) + ', ild_dpp: ' + str(
                ild_dpp) + ', div_dpp: ' + str(div_dpp) + ', entropy_dpp: ' + str(entropy_dpp) + '\n')
        file_log.write(
            'auc_cate: ' + str(auc_cate) + ', ndcg_cate: ' + str(ndcg_cate) + ', coverage_cate: ' + str(
                coverage_cate) + ', ild_cate: ' + str(
                ild_cate) + ', div_cate: ' + str(div_cate) + ', entropy_cate: ' + str(entropy_cate) + '\n')
        file_log.write(
            'auc_pmf: ' + str(auc_pmf) + ', ndcg_pmf: ' + str(ndcg_pmf) + ', coverage_pmf: ' + str(
                coverage_pmf) + ', ild_pmf: ' + str(
                ild_pmf) + ', div_pmf: ' + str(div_pmf) + ', entropy_pmf: ' + str(entropy_pmf) + '\n')
        file_log.write(
            'auc_ssd: ' + str(auc_ssd) + ', ndcg_ssd: ' + str(ndcg_ssd) + ', coverage_ssd: ' + str(
                coverage_ssd) + ', ild_ssd: ' + str(
                ild_ssd) + ', div_ssd: ' + str(div_ssd) + ', entropy_ssd: ' + str(entropy_ssd) + '\n')
        file_log.write(
            'auc_rl: ' + str(auc_rl) + ', ndcg_rl: ' + str(ndcg_rl) + ', coverage_rl: ' + str(
                coverage_rl) + ', ild_rl: ' + str(
                ild_rl) + ', div_rl: ' + str(div_rl) + ', entropy_rl: ' + str(entropy_rl) + '\n')

        print(
            'auc_mmr:{}, ndcg_mmr:{}, coverage_mmr:{}, ild_mmr:{}, div_mmr:{}, entropy_mmr{}'.format(auc_mmr, ndcg_mmr,
                                                                                                     coverage_mmr,
                                                                                                     ild_mmr, div_mmr,
                                                                                                     entropy_mmr))
        print(
            'auc_dpp:{}, ndcg_dpp:{}, coverage_dpp:{}, ild_dpp:{}, div_dpp:{}, entropy_dpp{}'.format(auc_dpp, ndcg_dpp,
                                                                                                     coverage_dpp,
                                                                                                     ild_dpp, div_dpp,
                                                                                                     entropy_dpp))
        print('auc_cate:{}, ndcg_cate:{}, coverage_cate:{}, ild_cate:{}, div_cate:{}, entropy_cate{}'.format(auc_cate,
                                                                                                             ndcg_cate,
                                                                                                             coverage_cate,
                                                                                                             ild_cate,
                                                                                                             div_cate,
                                                                                                             entropy_cate))
        print(
            'auc_pmf:{}, ndcg_pmf:{}, coverage_pmf:{}, ild_pmf:{}, div_pmf:{}, entropy_pmf{}'.format(auc_pmf, ndcg_pmf,
                                                                                                     coverage_pmf,
                                                                                                     ild_pmf, div_pmf,
                                                                                                     entropy_pmf))
        print(
            'auc_ssd:{}, ndcg_ssd:{}, coverage_ssd:{}, ild_ssd:{}, div_ssd:{}, entropy_ssd{}'.format(auc_ssd, ndcg_ssd,
                                                                                                     coverage_ssd,
                                                                                                     ild_ssd, div_ssd,
                                                                                                     entropy_ssd))
        print('auc_rl:{}, ndcg_rl:{}, coverage_rl:{}, ild_rl:{}, div_rl:{}, entropy_rl{}'.format(auc_rl, ndcg_rl,
                                                                                                 coverage_rl, ild_rl,
                                                                                                 div_rl, entropy_rl))

def test(best_model_path, sku_info_map, model_name, sku_novelty_map):
    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        input_format['data_type'] = data_type
        if model_name == 'PRM':
            model = PRM(input_format)
        elif model_name == 'DLCM':
            model = DLCM(input_format)
        else:
            model = PRM(input_format)
        model.restore(sess, best_model_path)

        file_log = open(model_name+'_test_result.txt', 'w')
        file_out = open(model_name+'_test_output.txt', 'w')

        # rerank_file_gen = open(model_name+'taobao_rerank_filefor_traindata.txt','w')
        # rerank_file_gen.write('queryid\titem_id\tprediction\tlabel\n')

        print('begin test:')
        trace = {}
        sku_emb = model.get_sku_emb(sess)
        global_index = input_format['batch_size']
        while global_index <= len(user_test_list):
            batch_data = generate_test_batch(input_format['batch_size'], global_index)
            predict = model.test(batch_data, sess, data_type)
            predict_l = np.reshape(predict, [-1]).tolist()
            skuid_l = np.reshape(batch_data[0], [-1]).tolist()
            uid_l = np.reshape(batch_data[8], [-1]).tolist()
            label_l = np.reshape(batch_data[6], [-1]).tolist()
            for k in range(len(uid_l)):
                uid = uid_l[k]
                skuid = skuid_l[k]
                predict_k = predict_l[k]
                label_k = label_l[k]
                if uid not in trace.keys():
                    trace[uid] = []
                trace[uid].append({'prediction': predict_k, 'ctr_label': label_k, 'skuid': skuid})
                file_out.write(str(uid)+'\t'+str(skuid)+'\t'+str(predict_k)+'\t'+str(label_k)+'\n')
            global_index += input_format['batch_size']

        file_log.write(model_name + ' model results:\n')

        auc, ndcg, coverage, ild, div, entropy = eval_full_data(trace, sku_info_map, sku_emb, 'rank')
        file_log.write('auc: ' + str(auc) + ', ndcg: ' + str(ndcg) + ', coverage: ' + str(coverage) + ', ild: ' + str(
            ild) + ', div: ' + str(div) + ', entropy: ' + str(entropy) + '\n')
        print('auc:{}, ndcg:{}, coverage:{}, ild:{}, div:{}, entropy{}'.format(auc,ndcg,coverage,ild,div,entropy))

        file_log.write(model_name + '+Re-ranking model results:\n')
        print(model_name + '+Re-ranking model results:\n')
        trace_mmr, trace_dpp, trace_cate, trace_pmf, trace_ssd, trace_rl = rerank_trace(trace,
                                                                                        input_format['rerank_topk'],
                                                                                        input_format[
                                                                                            'lambdaConstantMMR'],
                                                                                        input_format[
                                                                                            'lambdaConstantDPP'],
                                                                                        input_format[
                                                                                            'lambdaConstantCATE'],
                                                                                        input_format[
                                                                                            'lambdaConstantPMFA'],
                                                                                        input_format[
                                                                                            'lambdaConstantPMFB'],
                                                                                        input_format[
                                                                                            'lambdaConstantSSD'],
                                                                                        sku_emb, sku_info_map,
                                                                                        sku_novelty_map)
        auc_mmr, ndcg_mmr, coverage_mmr, ild_mmr, div_mmr, entropy_mmr = eval_full_data(trace_mmr, sku_info_map,
                                                                                        sku_emb, 're-rank')
        auc_dpp, ndcg_dpp, coverage_dpp, ild_dpp, div_dpp, entropy_dpp = eval_full_data(trace_dpp, sku_info_map,
                                                                                        sku_emb, 're-rank')
        auc_cate, ndcg_cate, coverage_cate, ild_cate, div_cate, entropy_cate = eval_full_data(trace_cate, sku_info_map,
                                                                                              sku_emb,
                                                                                              're-rank')
        auc_pmf, ndcg_pmf, coverage_pmf, ild_pmf, div_pmf, entropy_pmf = eval_full_data(trace_pmf, sku_info_map,
                                                                                        sku_emb, 're-rank')
        auc_ssd, ndcg_ssd, coverage_ssd, ild_ssd, div_ssd, entropy_ssd = eval_full_data(trace_ssd, sku_info_map,
                                                                                        sku_emb, 're-rank')
        auc_rl, ndcg_rl, coverage_rl, ild_rl, div_rl, entropy_rl = eval_full_data(trace_rl, sku_info_map, sku_emb,
                                                                                  're-rank')

        file_log.write(
            'auc_mmr: ' + str(auc_mmr) + ', ndcg_mmr: ' + str(ndcg_mmr) + ', coverage_mmr: ' + str(
                coverage_mmr) + ', ild_mmr: ' + str(
                ild_mmr) + ', div_mmr: ' + str(div_mmr) + ', entropy_mmr: ' + str(entropy_mmr) + '\n')
        file_log.write(
            'auc_dpp: ' + str(auc_dpp) + ', ndcg_dpp: ' + str(ndcg_dpp) + ', coverage_dpp: ' + str(
                coverage_dpp) + ', ild_dpp: ' + str(
                ild_dpp) + ', div_dpp: ' + str(div_dpp) + ', entropy_dpp: ' + str(entropy_dpp) + '\n')
        file_log.write(
            'auc_cate: ' + str(auc_cate) + ', ndcg_cate: ' + str(ndcg_cate) + ', coverage_cate: ' + str(
                coverage_cate) + ', ild_cate: ' + str(
                ild_cate) + ', div_cate: ' + str(div_cate) + ', entropy_cate: ' + str(entropy_cate) + '\n')
        file_log.write(
            'auc_pmf: ' + str(auc_pmf) + ', ndcg_pmf: ' + str(ndcg_pmf) + ', coverage_pmf: ' + str(
                coverage_pmf) + ', ild_pmf: ' + str(
                ild_pmf) + ', div_pmf: ' + str(div_pmf) + ', entropy_pmf: ' + str(entropy_pmf) + '\n')
        file_log.write(
            'auc_ssd: ' + str(auc_ssd) + ', ndcg_ssd: ' + str(ndcg_ssd) + ', coverage_ssd: ' + str(
                coverage_ssd) + ', ild_ssd: ' + str(
                ild_ssd) + ', div_ssd: ' + str(div_ssd) + ', entropy_ssd: ' + str(entropy_ssd) + '\n')
        file_log.write(
            'auc_rl: ' + str(auc_rl) + ', ndcg_rl: ' + str(ndcg_rl) + ', coverage_rl: ' + str(
                coverage_rl) + ', ild_rl: ' + str(
                ild_rl) + ', div_rl: ' + str(div_rl) + ', entropy_rl: ' + str(entropy_rl) + '\n')

        print(
            'auc_mmr:{}, ndcg_mmr:{}, coverage_mmr:{}, ild_mmr:{}, div_mmr:{}, entropy_mmr{}'.format(auc_mmr, ndcg_mmr,
                                                                                                     coverage_mmr,
                                                                                                     ild_mmr, div_mmr,
                                                                                                     entropy_mmr))
        print(
            'auc_dpp:{}, ndcg_dpp:{}, coverage_dpp:{}, ild_dpp:{}, div_dpp:{}, entropy_dpp{}'.format(auc_dpp, ndcg_dpp,
                                                                                                     coverage_dpp,
                                                                                                     ild_dpp, div_dpp,
                                                                                                     entropy_dpp))
        print('auc_cate:{}, ndcg_cate:{}, coverage_cate:{}, ild_cate:{}, div_cate:{}, entropy_cate{}'.format(auc_cate,
                                                                                                             ndcg_cate,
                                                                                                             coverage_cate,
                                                                                                             ild_cate,
                                                                                                             div_cate,
                                                                                                             entropy_cate))
        print(
            'auc_pmf:{}, ndcg_pmf:{}, coverage_pmf:{}, ild_pmf:{}, div_pmf:{}, entropy_pmf{}'.format(auc_pmf, ndcg_pmf,
                                                                                                     coverage_pmf,
                                                                                                     ild_pmf, div_pmf,
                                                                                                     entropy_pmf))
        print(
            'auc_ssd:{}, ndcg_ssd:{}, coverage_ssd:{}, ild_ssd:{}, div_ssd:{}, entropy_ssd{}'.format(auc_ssd, ndcg_ssd,
                                                                                                     coverage_ssd,
                                                                                                     ild_ssd, div_ssd,
                                                                                                     entropy_ssd))
        print('auc_rl:{}, ndcg_rl:{}, coverage_rl:{}, ild_rl:{}, div_rl:{}, entropy_rl{}'.format(auc_rl, ndcg_rl,
                                                                                                 coverage_rl, ild_rl,
                                                                                                 div_rl, entropy_rl))

def rerank_trace(trace, topk, lambdaConstantMMR=0.9, lambdaConstantDPP=0.9, lambdaConstantCATE=0.01, lambdaConstantPMFA=0.5, lambdaConstantPMFB=0.5, lambdaConstantSSD=0.5, sku_emb=None, sku_info_map=None, sku_novelty_map=None):

    trace_mmr = {}
    trace_dpp = {}
    trace_cate = {}
    trace_pmf = {}
    trace_ssd = {}
    trace_rl = {}

    print('begin re-rank all trace:')
    for trace_, sku_list in tqdm(trace.items()):

        sku_list_s = sorted(sku_list, key=lambda item: item['prediction'], reverse=True)
        for i in range(len(sku_list_s)):
            sku_list_s[i]['rank'] = i+1

        sku_list_s_k = sku_list_s[:topk]
        sku_list_s_k_ = sku_list_s[topk:]

        sku_sorted_list_cate, new_score = CATEMAX(sku_list_s_k.copy(), sku_info_map, lambdaConstantCATE)

        sku_sorted_list_mmr, new_score = MMR(sku_list_s_k.copy(), sku_emb, lambdaConstantMMR)

        sku_sorted_list_dpp, new_score = DPP(sku_list_s_k.copy(), sku_emb, lambdaConstantDPP)

        sku_sorted_list_pmf, new_score = PMF(sku_list_s_k.copy(), sku_emb, lambdaConstantPMFA, lambdaConstantPMFB)

        sku_sorted_list_ssd, new_score = SSD(sku_list_s_k.copy(), sku_emb, lambdaConstantSSD)

        sku_sorted_list_rl, new_score = RLprop(sku_list_s_k.copy(), sku_emb, sku_novelty_map)

        sku_sorted_list_cate.extend(sku_list_s_k_)
        sku_sorted_list_mmr.extend(sku_list_s_k_)
        sku_sorted_list_dpp.extend(sku_list_s_k_)
        sku_sorted_list_pmf.extend(sku_list_s_k_)
        sku_sorted_list_ssd.extend(sku_list_s_k_)
        sku_sorted_list_rl.extend(sku_list_s_k_)

        trace_mmr[trace_] = sku_sorted_list_mmr
        trace_dpp[trace_] = sku_sorted_list_dpp
        trace_cate[trace_] = sku_sorted_list_cate
        trace_pmf[trace_] = sku_sorted_list_pmf
        trace_ssd[trace_] = sku_sorted_list_ssd
        trace_rl[trace_] = sku_sorted_list_rl

    return trace_mmr, trace_dpp, trace_cate, trace_pmf, trace_ssd, trace_rl

if len(sys.argv) > 1:
    data_type = sys.argv[1]
    file_path_train = sys.argv[2]
    file_path_valid = sys.argv[3]
    file_path_test = sys.argv[4]
    sku_info_file = sys.argv[5]
    file_path_user_cf_seq = sys.argv[6]
    file_path_train_matching_seq = sys.argv[7]
    file_path_valid_matching_seq = sys.argv[8]
    file_path_test_matching_seq = sys.argv[9]
    file_path_train_long_seq = sys.argv[10]
    file_path_valid_long_seq = sys.argv[11]
    file_path_test_long_seq = sys.argv[12]
    mode = sys.argv[13]
    model_name = sys.argv[14]
    exp_name = sys.argv[15]
else:
    data_type = 'taobao'
    file_path_train = 'taobao_traindata_6000user02_offline_200_matchingneg2_newweight2_all.txt'
    file_path_valid = 'taobao_testdata_6000user02_offline_200.txt'
    file_path_test = 'taobao_testdata_6000user02_offline_200.txt'
    sku_info_file = 'taobao_item_cate_6000user02.txt'
    file_path_user_cf_seq = 'taobao_usercf_seq_200_10000.txt'
    file_path_train_matching_seq = 'taobao_train_data6000user02_matching_list_200.txt'
    file_path_valid_matching_seq = 'taobao_test_data6000user02_matching_list_200.txt'
    file_path_test_matching_seq = 'taobao_test_data6000user02_matching_list_200.txt'
    file_path_train_long_seq = 'taobao_traindata_6000user02_longseq.txt'
    file_path_valid_long_seq = 'taobao_testdata_6000user02_longseq.txt'
    file_path_test_long_seq = 'taobao_testdata_6000user02_longseq.txt'
    mode = 'test'
    model_name = 'DRN'
    exp_name = 'drn_taobao_batch256_dim16_2'


# data = np.load('20220810_sku_map_spu.npz', allow_pickle=True)
# sku_info_map = data['sku_info_map'].tolist()

# data_matrix = np.load('similar_matrix_0808_0809_cate3.npz', allow_pickle=True)
similar_matrix = []
new_index = []

SEED = 2
tf.set_random_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

if mode == 'train':
    sku_info_map = load_sku_info_map(sku_info_file)
    read_train_file(file_path_train)
    read_valid_file(file_path_valid)
    read_test_file(file_path_test)
    read_user_comirec_matching_seq(file_path_train_matching_seq, file_path_valid_matching_seq, file_path_test_matching_seq, input_format['matching_seq_max_len'])
    read_user_his_long_seq(file_path_train_long_seq, file_path_valid_long_seq, file_path_test_long_seq, input_format['his_long_seq_max_len'])

    data = np.load('taobao_6000user02_item_novelty.npz', allow_pickle=True)
    sku_novelty_map = data['sku_novelty_map'].tolist()
    train(sku_info_map, exp_name, model_name, sku_novelty_map)

elif mode == 'test':
    best_model_path = 'drn_taobao_batch256_dim16/best_model/'
    read_test_file(file_path_test)
    read_user_comirec_matching_seq(file_path_train_matching_seq, file_path_test_matching_seq, file_path_test_matching_seq, input_format['matching_seq_max_len'])
    read_user_his_long_seq(file_path_train_long_seq, file_path_valid_long_seq, file_path_test_long_seq, input_format['his_long_seq_max_len'])

    sku_info_map = load_sku_info_map(sku_info_file)
    data = np.load('taobao_item_novelty.npz', allow_pickle=True)
    sku_novelty_map = data['sku_novelty_map'].tolist()
    test(best_model_path, sku_info_map, model_name, sku_novelty_map)

elif mode == 'test-rerank':
    best_model_path = ''
    # test_rerank(test_batch_data, best_model_path, sku_info_map, topk, lambdaConstantMMR, lambdaConstantDPP, lambdaConstantCATE, similar_matrix, new_index)
