#—*-coding:utf8-*-
'''
LFM Algo
author:andjsmile
date:2018.10

'''
import pandas as pd
import operator
import warnings
import numpy as np
warnings.filterwarnings('ignore')

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

def get_ave_score(path):
    '''
    get item ave rating score
    Args:
        input file:user rating file
    Return:
        a dict,key:itemid ,value:ave_score
    '''
    user_rating=pd.read_csv(path)
    item_dict=user_rating[['movieId','rating']].groupby(['movieId']).rating.mean().to_dict()
    return item_dict


def get_train_data(path):
    '''
    get train data for LFM model train
    Args:
        path:user item rating file
    Return:
        a list[(userid,itemid,label),(userid1,itemid1,label1)]
    '''
    score_dict = get_ave_score(path)
    df = pd.read_csv(path)
    train_data = []
    df_pos = df[df['rating'] >= 4][['userId', 'movieId', 'rating']].reset_index(drop=True)
    pos_dict = {}  # 收集正样本的字典
    for i in range(0, df_pos.shape[0]):
        if df_pos.loc[i, 'userId'] not in pos_dict:
            pos_dict[df_pos.loc[i, 'userId']] = []
        pos_dict[df_pos.loc[i, 'userId']].append((df_pos.loc[i, 'movieId'], 1))

    df_neg = df[df['rating'] < 4][['userId', 'movieId', 'rating']].reset_index(drop=True)
    neg_dict = {}  # 收集负样本的字典
    for i in range(0, df_neg.shape[0]):
        if df_neg.loc[i, 'userId'] not in neg_dict:
            neg_dict[df_neg.loc[i, 'userId']] = []
        score = score_dict.get(df_neg.loc[i, 'movieId'], 0)
        neg_dict[df_neg.loc[i, 'userId']].append((df_neg.loc[i, 'movieId'], score))  # 这里负样本的label先用ave_score代替，因为后面要负采样

    for userid in pos_dict:
        if userid not in neg_dict:
            continue
        num_data = min(len(pos_dict[userid]), len(neg_dict[userid]))
        if num_data > 0:
            train_data += [(userid, zuhe[0], zuhe[1]) for zuhe in pos_dict[userid]][:num_data]
        else:
            continue
        sorted_neg_list = sorted(neg_dict[userid], key=lambda ele: ele[1], reverse=True)[:num_data]
        train_data += [(userid, zuhe[0], 0) for zuhe in sorted_neg_list]
    return train_data

def lfm_train(train_data,F,alpha,beta,step):
    '''
    Args:
        train:train data for lfm
        F:user vector len,item vector len
        alpha:regularization factor
        beta:learning rate
        step:iteration num
    Return:
        dict:key itemid,value:list
        dict:keyu userid,value:list
    '''
    user_vec={}
    item_vec={}
    for i in range(step):
        for ele in train_data:
            userid,itemid,label=ele
            if userid not in user_vec:
                user_vec[userid]=init_model(F)
            if itemid not in item_vec:
                item_vec[itemid]=init_model(F)
        loss=label-model_predict(user_vec[userid],item_vec[itemid])
        for index in range(F):
            user_vec[userid][index]-=beta*(user_vec[userid][index]-loss*item_vec[itemid][index])
            item_vec[itemid][index]-=beta*(item_vec[itemid][index]-loss*user_vec[userid][index])
        beta*=0.9
    return user_vec,item_vec

def init_model(vector_len):
    return np.random.randn(vector_len)

def model_predict(user_vector,item_vector):
    res=np.dot(user_vector,item_vector)/(np.linalg.norm(user_vector)*np.linalg.norm(item_vector))
    return res

def model_trian_process():
    train_data=get_train_data('Movielens_1M/ml-latest-small/ratings.csv')
    user_vec,item_vec=lfm_train(train_data,50,0.01,0.1,50)
    recom_list=give_recom_result(user_vec,item_vec,24)
    ana_recom_result(train_data,24,recom_list)

def give_recom_result(user_vec,item_vec,userid):
    fix_num=10
    if userid not in user_vec:
        return []
    record={}#存储每一个item与user_vec之间的距离
    recom_list=[]
    user_vector=user_vec[userid]
    for itemid in item_vec:
        item_vector=item_vec[itemid]
        res=np.dot(user_vector,item_vector)/(np.linalg.norm(user_vector)*np.linalg.norm(item_vector))
        record[itemid]=res
    for zuhe in sorted(record.items(),key=operator.itemgetter(1),reverse=True)[:fix_num]:
        itemid=zuhe[0]
        score=round(zuhe[1],3)
        recom_list.append((itemid,score))
    return recom_list

def ana_recom_result(train_data,userid,recom_list):
    item_info=get_item_info('Movielens_1M/ml-latest-small/movies.csv')
    for data_instance in train_data:
        tempid,itemid,label=data_instance
        if tempid==userid and label==1:
            print(item_info[itemid])
    print('recom result')
    for zuhe in recom_list:
        print(item_info[zuhe[0]])

if __name__=='__main__':
    model_trian_process()