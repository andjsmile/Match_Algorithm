#—*-coding:utf8-*-
'''
item cf Algo
author:andjsmile
date:2018.10

'''
import pandas as pd
import math
import operator
import warnings
warnings.filterwarnings('ignore')

def get_user_click(path):
    '''
    获得每个userid行为过的movieid，构造成字典的形式。
    get user click list
    Args:
        rating_file:input file
    Return:
        dict,key:userid,value:[itemid1,itemid2,..]
    '''
#     if not os.path.exists(rating_file):
#         return {}
#     else:
    df=pd.read_csv(path)
    data=df[df['rating']>=3][['userId','movieId']]
    user_click=data.groupby('userId').movieId.apply(list).to_dict()
    return user_click

def get_item_info(path):
    '''
    获得每个movieid的title以及genres.
    get item info[title,genres]
    Args:
        item_file:input iteminfo file
    Return:
        a dict,key itemid,value:[title,genres]
    '''
#     if not os.path.exists(rating_file):
#         return {}
#     else:
    df=pd.read_csv(path)
    item_info = {}
    for i in range(len(df.index)):
        item_info[df.loc[i,'movieId']]=[df.loc[i,'title'],df.loc[i,'genres']]
    return item_info

def base_contribute_score():
    '''
    item cf bse sim contribution score
    '''
    return 1
def cal_item_sim(user_click):
    '''
    计算item之间的相似度得分。
    Args:
        user_item:a dict,key userid,value [itemid1,itemid2,...]
    Return:
        a dict:key itemid,value a dict:key itemid ,value:sim_score
    '''
    co_appear = {}
    item_user_click_time = {}  # 记录这个item被多少用户点击过，用于分母
    for user, itemlist in user_click.items():
        for index_i in range(0, len(itemlist)):
            itemid_i = itemlist[index_i]
            item_user_click_time.setdefault(itemid_i, 0)  # setdefault对与不存在的key给予一个初值的设定
            item_user_click_time[itemid_i] += 1
            for index_j in range(index_i + 1, len(itemlist)):
                itemid_j = itemlist[index_j]
                co_appear.setdefault(itemid_i, {})
                co_appear[itemid_i].setdefault(itemid_j, 0)
                co_appear[itemid_i][itemid_j] += base_contribute_score()

                co_appear.setdefault(itemid_j, {})
                co_appear[itemid_j].setdefault(itemid_i, 0)
                co_appear[itemid_j][itemid_i] += base_contribute_score()

    item_sim_score = {}
    item_sim_score_sorted = {}
    for itemid_i, relate_item in co_appear.items():
        for itemid_j, co_time in relate_item.items():
            sim_score = co_time / (
                        math.sqrt(item_user_click_time[itemid_i]) * math.sqrt(item_user_click_time[itemid_j]))
            item_sim_score.setdefault(itemid_i, {})
            item_sim_score[itemid_i].setdefault(itemid_j, 0)
            item_sim_score[itemid_i][itemid_j] = sim_score

    for itemid in item_sim_score:
        item_sim_score_sorted[itemid] = sorted(item_sim_score[itemid].items(), key= \
            operator.itemgetter(1), reverse=True)
    return item_sim_score_sorted

def cal_recom_result(sim_info,user_click):
    recent_click_num=3
    topk=5
    recom_info={}
    for user in user_click:
        click_list=user_click[user]
        recom_info.setdefault(user,{})
        for itemid in click_list[:recent_click_num]:
            if itemid not in sim_info:
                continue
            for itemsimzuhe in sim_info[itemid][:topk]:
                itemsimid=itemsimzuhe[0]
                itemsimscore=itemsimzuhe[1]
                recom_info[user][itemsimid]=itemsimscore
    return recom_info

def main_flow():
    '''
    main flow of item cf

    '''
    user_click = get_user_click('Movielens_1M/ml-latest-small/ratings.csv')
    sim_info = cal_item_sim(user_click)
    recom_result = cal_recom_result(sim_info, user_click)
    print(recom_result[1])
if __name__=='__main__':
    main_flow()
