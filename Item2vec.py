#—*-coding:utf8-*-
'''
item2vec Algo
author:andjsmile
date:2018.12

'''
import pandas as pd
import operator
import warnings
import numpy as np
import os
warnings.filterwarnings('ignore')

def produce_train_data(input_file, output_file):
    '''
    Args:
        input_file:user behavior file
        output_file:output file

    '''
    if not os.path.exists(input_file):
        return {}
    df = pd.read_csv(input_file)
    df = df[df['rating'] >= 4].reset_index(drop=True)[['userId', 'movieId']].astype(str)
    user_item_dict = df.groupby('userId')['movieId'].apply(list).to_dict()

    fw = open(output_file, 'w')
    for k, v in user_item_dict.items():
        fw.write(' '.join(v) + '\n')
    fw.close()


def load_item_vec(input_file):
    '''
    Args:
        input_file:item vec file
        注意：这里的 item vec file 是由word2vec库生成的。
    Return:
        dict key:itemid value:np.array([num1,num2,...])

    '''
    if not os.path.exists(input_file):
        return {}
    linenum = 0
    item_vec = {}
    fp = open(input_file)
    for line in fp:
        if linenum == 0:
            linenum += 1
            continue
        item = line.strip().split()
        if len(item) < 129:
            continue
        itemid = item[0]
        if itemid == '</s>':
            continue
        item_vec[itemid] = np.array([float(ele) for ele in item[1:]])
    fp.close()
    return item_vec

def cal_item_sim(item_vec,itemid,output_file):
    '''
    Args:
        item_vec:item embedding vector
        itemid:fixed itemid to clac item sim
        output_file:the file to store result
    '''
    if itemid not in item_vec:
        return
    score={}
    topk=10
    fix_item_vec=item_vec[itemid]
    for tmp_itemid in item_vec:
        if tmp_itemid==itemid:
            continue
        tmp_itemvec=item_vec[tmp_itemid]
        fenmu=np.linalg.norm(fix_item_vec)*np.linalg.norm(tmp_itemvec)
        if fenmu==0:
            score[tmp_itemid]==0
        else:
            score[tmp_itemid]=round(np.dot(fix_item_vec,tmp_itemvec)/fenmu,3)
    fw=open(output_file,'w+')
    out_str=itemid+'\t'
    tmp_list=[]
    for zuhe in sorted(score.items(),key=operator.itemgetter(1),reverse=True)[:topk]:
        tmp_list.append(zuhe[0] +'_' +str(zuhe[1]))
    out_str+=';'.join(tmp_list)
    print(out_str)
    fw.write(out_str)
    fw.close()

def main_flow():
    item_vec = load_item_vec('ml-20m/item_vec.txt')
    cal_item_sim(item_vec, '27', 'ml-20m/sim_result.txt')

if __name__=='__main__':
    main_flow()