import os
import random
from tqdm import tqdm
import time
import numpy as np
import math

item_map = {}
with open('data/taobao_data/taobao_item_map.txt', 'r') as f:
    for line in f:
        line_split = line.strip('\n').split(',')
        item_id = line_split[0]
        map_id = line_split[1]
        item_map[item_id] = map_id

item_cate_map = {}
with open('data/taobao_data/taobao_item_cate.txt', 'r') as f:
    for line in f:
        line_split = line.strip('\n').split(',')
        item_id = line_split[0]
        cate_id = line_split[1]
        item_cate_map[item_id] = cate_id

user_map = {}
with open('data/taobao_data/taobao_user_map.txt', 'r') as f:
    for line in f:
        line_split = line.strip('\n').split(',')
        user_id = line_split[0]
        index_id = line_split[1]
        user_map[user_id] = index_id

item_cate_map['0'] = '0'

user_interaction = {}
cate_part = {}
item_cnt = {}
cnt = 0
dt_map = {}

data = np.load('taobao_user_6000_02.npz', allow_pickle=True)
user_part = data['user_part'].tolist()
user_part_map = {}
user_part = [str(user) for user in user_part]
for user in user_part:
    user_part_map[user] = 1

with open('UserBehavior.csv', 'r') as f:
    for line in tqdm(f,total=100200000):
        line_split = line.strip('\n').split(',')
        user_id = line_split[0]
        if user_id not in user_part_map.keys():
            continue
        item_id = line_split[1]
        cate_id = line_split[2]
        event_type = line_split[3]
        if event_type != 'pv':
            continue
        ts = int(line_split[4])
        # dt = line_split[5]
        # dt_map[dt] = 1

        if user_id not in user_interaction.keys():
            user_interaction[user_id] = []
        user_interaction[user_id].append({'item_id':item_id,'dt':ts})
        if item_id not in item_cnt.keys():
            item_cnt[item_id] = 0
        item_cnt[item_id] += 1



# print('total_user_cnt:',len(user_interaction))
# print('total_cate_cnt:',len(cate_map))

total_interaction = 0

user_cnt = 0
filter_user = []
for user in tqdm(user_part):
    new_list = []
    cate_list = []
    user_his_train = []
    if user not in user_interaction.keys():
        continue
    for item in user_interaction[user]:
        item_id = item['item_id']
        if item_cnt[item_id] >= 10:
            new_list.append(item)
            cate_id = item_cate_map[item_map[item_id]]
            if cate_id not in cate_part.keys():
                cate_part[cate_id] = {}
            cate_part[cate_id][item_map[item_id]] = 1

    user_interaction[user] = new_list
    if len(user_interaction[user]) >= 10:
        user_cnt += 1
        filter_user.append(user)
        total_interaction += len(user_interaction[user])

print('user_filter_cnt:',user_cnt)
print('total_part_interaction:',total_interaction)
print('cate_part_len:',len(cate_part))

split_train_valid = math.ceil(len(filter_user)*0.8)
split_valid_test = math.ceil(len(filter_user)*0.9)
train_user = filter_user[:split_train_valid]
valid_user = filter_user[split_train_valid:split_valid_test]
test_user = filter_user[split_valid_test:]
# print('begin cate item map:')
# for cate in tqdm(cate_part.keys()):
#     cate_part[cate] = [item_map[x] for x in list(cate_part[cate].keys())]

# data1 = np.load('taobao_hot_item_1000.npz',allow_pickle=True)
# hot_item_list = set(data1['item_list'].tolist())
#
# data2 = np.load('taobao_hot_item_10000.npz',allow_pickle=True)
# hot_item_list_10000 = [item_map[x] for x in data2['taobao_hot_item_10000'].tolist()]

train_user_matching_seq = {}
with open('taobao_train_data6000user02_matching_list_200.txt', 'r') as f:
    first = True
    for line in tqdm(f, total=400000):
        line_split = line.strip('\n').split('\t')
        queryid = line_split[0]
        matching_seq = line_split[1]
        train_user_matching_seq[queryid] = matching_seq.split('-')

valid_user_matching_seq = {}
with open('taobao_valid_data6000user02_matching_list_200.txt', 'r') as f:
    first = True
    for line in tqdm(f, total=1000):
        line_split = line.strip('\n').split('\t')
        queryid = line_split[0]
        matching_seq = line_split[1]
        valid_user_matching_seq[queryid] = matching_seq.split('-')

test_user_matching_seq = {}
with open('taobao_test_data6000user02_matching_list_200.txt', 'r') as f:
    first = True
    for line in tqdm(f, total=1000):
        line_split = line.strip('\n').split('\t')
        queryid = line_split[0]
        matching_seq = line_split[1]
        test_user_matching_seq[queryid] = matching_seq.split('-')

# test_user_matching_seq = {}
# with open('taobao_test_data_1000_matching_list_top200_addpos.txt', 'r') as f:
#     first = True
#     for line in tqdm(f, total=1000):
#         line_split = line.strip('\n').split('\t')
#         queryid = line_split[0]
#         matching_seq = line_split[1]
#         test_user_matching_seq[queryid] = matching_seq.split('-')

file_train = open('taobao_traindata_6000user02_offline_200_matchingneg5_newweight3.txt','w')
file_valid = open('taobao_validdata_6000user02_offline_200.txt','w')
file_test = open('taobao_testdata_6000user02_offline_200.txt','w')

file_train.write('queryid'+'\t'+'uuid'+'\t'+'item_id'+'\t'+'cate_id'+'\t'+'his_item_seq'+'\t'+'label'+'\t'+'weight'+'\n')
file_valid.write('queryid'+'\t'+'uuid'+'\t'+'item_id'+'\t'+'cate_id'+'\t'+'his_item_seq'+'\t'+'label'+'\t'+'weight'+'match_score'+'\n')
file_test.write('queryid'+'\t'+'uuid'+'\t'+'item_id'+'\t'+'cate_id'+'\t'+'his_item_seq'+'\t'+'label'+'\t'+'weight'+'match_score'+'\n')

# hot_item_200 = np.load('taobao_hot200_item_10000user.npz', allow_pickle=True)
# hot_item_200 = hot_item_200['hot_item_200'].tolist()

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
                file_train.write(str(query_id)+'\t'+user_map[user]+'\t'+item_id+'\t'+cate+'\t'+'-'.join(his_item_list[:index])+'\t1\t')
            else:
                file_train.write(str(query_id) + '\t' + user_map[user] + '\t' + item_id + '\t' + cate + '\t' + '-'.join(
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

            for j in range(5):
                cnt_train += 1
                neg = random.choice(matching_seq)
                while neg in his_item_map.keys():
                    neg = random.choice(matching_seq)

                if index <= 25:
                    file_train.write(str(query_id) + '\t' + user_map[user] + '\t' + neg + '\t' + item_cate_map[neg] + '\t' + '-'.join(
                        his_item_list[:index]) + '\t0\t')
                else:
                    file_train.write(str(query_id) + '\t' + user_map[user] + '\t' + neg + '\t' + item_cate_map[neg] + '\t' + '-'.join(
                        his_item_list[index - 25:index]) + '\t0\t')

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

    matching_seq = valid_user_matching_seq[user_map[user]]
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
                file_valid.write(str(query_id) + '\t' + user_map[user] + '\t' + ii + '\t' + item_cate_map[ii] + '\t' + '-'.join(
                    his_item_list[:split_index]) + '\t'+label+'\t1\t'+str(score)+'\n')
            else:
                file_valid.write(str(query_id) + '\t' + user_map[user] + '\t' + ii + '\t' + item_cate_map[ii] + '\t' + '-'.join(
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

    matching_seq = test_user_matching_seq[user_map[user]]
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
                file_test.write(str(query_id) + '\t' + user_map[user] + '\t' + ii + '\t' + item_cate_map[ii] + '\t' + '-'.join(
                    his_item_list[:split_index]) + '\t' + label + '\t1\t'+str(score)+'\n')
            else:
                file_test.write(str(query_id) + '\t' + user_map[user] + '\t' + ii + '\t' + item_cate_map[ii] + '\t' + '-'.join(
                    his_item_list[split_index - 25:split_index]) + '\t' + label + '\t1\t'+str(score)+'\n')

print('cnt_test:', cnt_test)