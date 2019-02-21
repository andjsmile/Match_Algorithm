#—*-coding:utf8-*-
'''
Content Based Algo
author:andjsmile
date:2018.12

'''
import pandas as pd
import operator
import warnings
import os
warnings.filterwarnings('ignore')

def get_ave_score(input_file):
    '''
    得到每个item的平均得分
    Args:
        input_file:user rating file
    Return:
        a dict,key:itemid value:ave_score
    '''
    if not os.path.exists(input_file):
        return {}
    df=pd.read_csv(input_file)
    df=df[df['rating']>=4][['movieId','rating']]
    ave_score=df.groupby('movieId').rating.mean().to_dict()
    return ave_score


def get_item_cate(ave_score, input_file):
    '''
    Args:
        ave_score:a dict,key itemid ,value rating score
        input_file:item info file
    Return:
        a dict:key itemid value a dict,key:cate value:ratio
        a dict:key cate value[itemid1,itemid2,itemid3...]
    '''
    item_info = pd.read_csv(input_file)
    item_cate = {}
    record = {}
    topk = 100
    cate_item_sort = {}
    for i in range(item_info.shape[0]):
        itemid = item_info.loc[i, 'movieId']
        cate_str = item_info.loc[i, 'genres']
        cate_list = cate_str.strip().split('|')
        ratio = 1 / len(cate_list)
        if itemid not in item_cate:
            item_cate[itemid] = {}
        for fix_cate in cate_list:
            item_cate[itemid][fix_cate] = ratio

    for itemid in item_cate:
        for cate in item_cate[itemid]:
            if cate not in record:
                record[cate] = {}
            itemid_rating_score = ave_score.get(itemid, 0)
            record[cate][itemid] = itemid_rating_score

    for cate in record:
        if cate not in cate_item_sort:
            cate_item_sort[cate] = []
        for zuhe in sorted(record[cate].items(), key=operator.itemgetter(1), reverse=True)[:topk]:
            cate_item_sort[cate].append(zuhe[0])
    return item_cate, cate_item_sort

def get_up(item_cate,input_file):
    '''
    Args:
        item_cate:a dict,key itemid,value a dict,key cate,value ratio
        input_file:user rating file
    Return:
        a dict:key userid,value[(cate1,ratio1),(cate2,ratio2)]
    '''
    record={}
    up={}
    topk=2#给每个用户取两个最受欢迎的类别
    if not os.path.exists(input_file):
        return {}
    df=pd.read_csv(input_file)
    df=df[df['rating']>=4].reset_index(drop=True)
    for i in range(df.shape[0]):
        userid=df.loc[i,'userId']
        itemid=df.loc[i,'movieId']
        rating=df.loc[i,'rating'].astype(float)
        timestamp=df.loc[i,'timestamp'].astype(int)
        if itemid not in item_cate:
            continue
        time_score=get_time_score(timestamp)
        if userid not in record:
            record[userid]={}
        for fix_cate in item_cate[itemid]:
            if fix_cate not in record[userid]:
                record[userid][fix_cate]=0
            record[userid][fix_cate]+=rating*time_score*item_cate[itemid][fix_cate]
    for userid in record:
        if userid not in up:
            up[userid]=[]
        total_score=0 #对某种类别的喜好程度的归一化值得分母
        for zuhe in sorted(record[userid].items(),key=operator.itemgetter(1),reverse=True)[:topk]:
            up[userid].append((zuhe[0],zuhe[1]))
            total_score+=float(zuhe[1])
        for index in range(len(up[userid])):
            up[userid][index]=(up[userid][index][0],round(up[userid][index][1]/total_score,3))
    return up

def get_time_score(timestamp):
    fix_time_stamp=1427784002 #最大的时间戳
    total_sec=24*60*60
    delta=(fix_time_stamp-timestamp)/total_sec/10000#将‘秒’转为‘天’
    return round(1/(1+delta),3)

def recom(cate_item_sort,up,userid,topk=10):
    '''
    Args:
        cate_item_sort:reverse sort
        up:user profile
        userid:fix userid to recom
        topk:recom num
    Return:
        a dict,key userid,value[itemid1,itemid2,...]
    '''
    if userid not in up:
        return {}
    recom_result={}
    if userid not in recom_result:
        recom_result[userid]=[]
    for zuhe in up[userid]:
        cate=zuhe[0]
        ratio=zuhe[1]
        num=int(topk*ratio)+1
        if cate not in cate_item_sort:
            continue
        recom_list=cate_item_sort[cate][:num]
        recom_result[userid]+=recom_list
    return recom_result

def main_flow():
    ave_score = get_ave_score('Movielens_1M/ml-latest-small/ratings.csv')
    item_cate, cate_item_sort = get_item_cate(ave_score, 'Movielens_1M/ml-latest-small/movies.csv')
    up = get_up(item_cate, 'Movielens_1M/ml-latest-small/ratings.csv')
    print(up)

if __name__=='__main__':
    main_flow()