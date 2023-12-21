import os
import random
from tqdm import tqdm
import time
import numpy as np
import math

user_interaction = {}
item_cnt = {}
cnt = 0
dt_map = {}


item_map = {'0':'0'}
with open('data/movie_data/movie_item_map.txt', 'r') as f:
    for line in f:
        line_split = line.strip('\n').split(',')
        item_id = line_split[0]
        map_id = line_split[1]
        item_map[item_id] = map_id

item_cate_map = {'0':'0'}
with open('data/movie_data/movie_item_cate.txt', 'r') as f:
    for line in f:
        line_split = line.strip('\n').split(',')
        item_id = line_split[0]
        cate_id = line_split[1]
        item_cate_map[item_id] = cate_id

user_all = []
with open('ratings.dat','r') as f:
    for line in tqdm(f,total=1100000):
        line_split = line.strip('\n').split('::')
        user_id = line_split[0]
        item_id = line_split[1]
        rates = float(line_split[2])
        timestamp = line_split[3]
        if rates < 4.0:
            continue
        if user_id not in user_interaction.keys():
            user_interaction[user_id] = []
            user_all.append(user_id)
        user_interaction[user_id].append({'item_id':item_id,'dt':timestamp})
        if item_id not in item_cnt.keys():
            item_cnt[item_id] = 0
        item_cnt[item_id] += 1

# item_map = {'0':0}
# sorted_item_list = sorted(item_cnt.items(), key=lambda x:x[1], reverse=True)
# for (item, cnt) in sorted_item_list:
#     if cnt >= 10:
#         item_map[item] = len(item_map)

train_user_matching_seq = {}
with open('movie_train_data_matching_list_200.txt', 'r') as f:
    first = True
    for line in tqdm(f, total=400000):
        line_split = line.strip('\n').split('\t')
        queryid = line_split[0]
        matching_seq = line_split[1]
        train_user_matching_seq[queryid] = matching_seq.split('-')

valid_user_matching_seq = {}
with open('movie_valid_data_matching_list_200.txt', 'r') as f:
    first = True
    for line in tqdm(f, total=1000):
        line_split = line.strip('\n').split('\t')
        queryid = line_split[0]
        matching_seq = line_split[1]
        valid_user_matching_seq[queryid] = matching_seq.split('-')

test_user_matching_seq = {}
with open('movie_test_data_matching_list_200.txt', 'r') as f:
    first = True
    for line in tqdm(f, total=1000):
        line_split = line.strip('\n').split('\t')
        queryid = line_split[0]
        matching_seq = line_split[1]
        test_user_matching_seq[queryid] = matching_seq.split('-')

print('filter_item_cnt:', len(item_map))

total_interaction = 0

user_cnt = 0

filter_user = []
for user in tqdm(user_all):
    # total_interaction += len(user_interaction[user])
    new_list = []
    cate_list = []
    user_his_train = []
    for item in user_interaction[user]:
        if item_cnt[item['item_id']] >= 10:
            new_list.append(item)

    user_interaction[user] = new_list
    if len(user_interaction[user]) >= 10:
        user_cnt += 1
        filter_user.append(user)
        total_interaction += len(user_interaction[user])

split_train_valid = math.ceil(len(filter_user)*0.8)
split_valid_test = math.ceil(len(filter_user)*0.9)

train_user = filter_user[:split_train_valid]
valid_user = filter_user[split_train_valid:split_valid_test]
test_user = filter_user[split_valid_test:]

print('user_filter_cnt:',user_cnt)
print('total_part_interaction:',total_interaction)

file_user = open('movie_user_map.txt', 'w')
file_user.write('uid\tmap_id\n')
user_map = {'0':0}
for user in filter_user:
    user_map[user] = len(user_map)
    file_user.write(user+'\t'+str(user_map[user])+'\n')

# data2 = np.load('taobao_hot_item_200_80000user.npz',allow_pickle=True)
# hot_item_list_200 = [item_map[x] for x in data2['hot_item'].tolist()]

file_train = open('movie1m_traindata_matchingneg2_newweight3.txt','w')
file_valid = open('movie1m_validdata_08_matching.txt','w')
file_test = open('movie1m_testdata_08_matching.txt','w')
file_train.write('queryid'+'\t'+'uuid'+'\t'+'item_id'+'\t'+'cate_id'+'\t'+'his_item_seq'+'\t'+'label'+'\t'+'weight'+'\n')
file_valid.write('queryid'+'\t'+'uuid'+'\t'+'item_id'+'\t'+'cate_id'+'\t'+'his_item_seq'+'\t'+'label'+'\t'+'weight'+'\t'+'match_score'+'\n')
file_test.write('queryid'+'\t'+'uuid'+'\t'+'item_id'+'\t'+'cate_id'+'\t'+'his_item_seq'+'\t'+'label'+'\t'+'weight'+'\t'+'match_score'+'\n')

query_id = 0
weight_cnt = {}
item_all = []
for i in range(1, 2812):
    item_all.append(str(i))

cnt_train = 0
print('begin generating train data:')
for user in tqdm(train_user):
    interaction_list = user_interaction[user]
    interaction_list = sorted(interaction_list, key=lambda x:x['dt'])
    his_item_list = [item_map[i['item_id']] for i in interaction_list]
    his_cate_list = [item_cate_map[item_map[i['item_id']]] for i in interaction_list]

    his_item_map = {}
    for item in his_item_list:
        his_item_map[item] = 1

    user_his_all_one = his_cate_list
    cate_cnt = {}
    for x in user_his_all_one:
        if x not in cate_cnt.keys():
            cate_cnt[x] = 0
        cate_cnt[x] += 1
    max_val_ori = 0
    min_val_ori = 100000
    for key in cate_cnt.keys():
        if cate_cnt[key] > max_val_ori:
            max_val_ori = cate_cnt[key]
        if cate_cnt[key] < min_val_ori:
            min_val_ori = cate_cnt[key]
    for key in cate_cnt.keys():
        cate_cnt[key] = (max_val_ori // cate_cnt[key])

    max_val = 0
    min_val = 100000
    for key in cate_cnt.keys():
        if cate_cnt[key] > max_val:
            max_val = cate_cnt[key]
        if cate_cnt[key] < min_val:
            min_val = cate_cnt[key]

    # his_cate_set = user_his_cate[user]
    for index in range(len(his_item_list)):
        query_id += 1
        item_id = his_item_list[index]
        cate = his_cate_list[index]

        matching_seq = train_user_matching_seq[str(query_id)]
        if item_id in matching_seq:

            cnt_train += 1

            if index <= 25:
                file_train.write(str(query_id)+'\t'+user+'\t'+item_id+'\t'+cate+'\t'+'-'.join(his_item_list[:index])+'\t1\t')
            else:
                file_train.write(str(query_id) + '\t' + user + '\t' + item_id + '\t' + cate + '\t' + '-'.join(
                    his_item_list[index-25:index]) + '\t1\t')

            if max_val == min_val:
                weight = 1
            else:
                std = (cate_cnt[cate] - min_val) / (max_val - min_val)
                weight = math.ceil(std * 3 + 1)

            if weight not in weight_cnt.keys():
                weight_cnt[weight] = 0
            weight_cnt[weight] += 1
            file_train.write(str(weight) + '\n')

            for j in range(2):
                cnt_train += 1
                neg = random.choice(matching_seq)
                while neg in his_item_map.keys():
                    neg = random.choice(matching_seq)

                if index <= 25:
                    file_train.write(str(query_id) + '\t' + user + '\t' + neg + '\t' + item_cate_map[neg] + '\t' + '-'.join(
                        his_item_list[:index]) + '\t0\t1\n')
                else:
                    file_train.write(str(query_id) + '\t' + user + '\t' + neg + '\t' + item_cate_map[neg] + '\t' + '-'.join(
                        his_item_list[index - 25:index]) + '\t0\t1\n')

                if item_cate_map[neg] in his_cate_list:
                    if max_val_ori == min_val_ori:
                        weight = 1
                    else:
                        std = (cate_cnt[cate] - min_val_ori) / (max_val_ori - min_val_ori)
                        weight = math.ceil(std * 3 + 1)
                else:
                    weight = 1
                file_train.write(str(weight)+'\n')



print(weight_cnt)
print('cnt_train:', cnt_train)

cnt_valid = 0
print('begin generating valid data:')
for user in tqdm(valid_user):
    interaction_list = user_interaction[user]
    interaction_list = sorted(interaction_list, key=lambda x: x['dt'])
    his_item_list = [item_map[i['item_id']] for i in interaction_list]
    his_cate_list = [item_cate_map[item_map[i['item_id']]] for i in interaction_list]

    his_item_map = {}
    for item in his_item_list:
        his_item_map[item] = 1

    query_id += 1
    split_index = math.ceil(0.8 * len(his_item_list))
    pos_list = his_item_list[split_index:len(his_item_list)]

    matching_seq = valid_user_matching_seq[user]
    flag = False

    for index in range(split_index, len(interaction_list)):
        item_id = his_item_list[index]
        cate = his_cate_list[index]

        if item_id in matching_seq:
            flag = True
            break

    if flag == True:

        score = 0
        for ii in matching_seq:
            if ii in pos_list:
                label = '1'
            else:
                label = '0'

            score -= 1
            cnt_valid += 1
            if split_index <= 25:
                file_valid.write(str(query_id) + '\t' + user + '\t' + ii + '\t' + item_cate_map[ii] + '\t' + '-'.join(
                    his_item_list[:split_index]) + '\t'+label+'\t1\t'+str(score)+'\n')
            else:
                file_valid.write(str(query_id) + '\t' + user + '\t' + ii + '\t' + item_cate_map[ii] + '\t' + '-'.join(
                    his_item_list[split_index - 25:split_index]) + '\t'+label+'\t1\t'+str(score)+'\n')


print('cnt_valid:', cnt_valid)

cnt_test = 0
print('begin generating test data:')
for user in tqdm(test_user):
    interaction_list = user_interaction[user]
    interaction_list = sorted(interaction_list, key=lambda x: x['dt'])
    his_item_list = [item_map[i['item_id']] for i in interaction_list]
    his_cate_list = [item_cate_map[item_map[i['item_id']]] for i in interaction_list]

    his_item_map = {}
    for item in his_item_list:
        his_item_map[item] = 1

    query_id += 1
    split_index = math.ceil(0.8 * len(his_item_list))
    pos_list = his_item_list[split_index:len(his_item_list)]

    matching_seq = test_user_matching_seq[user]
    flag = False

    for index in range(split_index, len(interaction_list)):
        item_id = his_item_list[index]
        cate = his_cate_list[index]

        if item_id in matching_seq:
            flag = True
            break

    if flag == True:

        score = 0
        for ii in matching_seq:
            if ii in pos_list:
                label = '1'
            else:
                label = '0'
            score -= 1
            cnt_test += 1
            if split_index <= 25:
                file_test.write(str(query_id) + '\t' + user + '\t' + ii + '\t' + item_cate_map[ii] + '\t' + '-'.join(
                    his_item_list[:split_index]) + '\t' + label + '\t1\t'+str(score)+'\n')
            else:
                file_test.write(str(query_id) + '\t' + user + '\t' + ii + '\t' + item_cate_map[ii] + '\t' + '-'.join(
                    his_item_list[split_index - 25:split_index]) + '\t' + label + '\t1\t'+str(score)+'\n')

print('cnt_test:', cnt_test)