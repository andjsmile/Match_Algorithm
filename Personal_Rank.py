#—*-coding:utf8-*-
'''
Person Rank Algo
author:andjsmile
date:2018.11

'''
import pandas as pd
import os
import operator
import warnings
import numpy as np
warnings.filterwarnings('ignore')

def get_graph_from_data(input_file):
    '''
    Args:
        input_file:user item rating file
    Return:
        a dict:{userA:{itemb:1,itemc:1},itemb:{userA:1}}
    '''
    if not os.path.exists(input_file):
        return {}
    graph={}
    df=pd.read_csv(input_file)
    df=df[df['rating']>=4].reset_index(drop=True)
    for i in range(df.shape[0]):
        if df.loc[i,'userId'] not in graph:
            graph[df.loc[i,'userId']]={}
        graph[df.loc[i,'userId']]['item_'+ df.loc[i,'movieId'].astype(str)]=1
        if 'item_'+df.loc[i,'movieId'].astype(str) not in graph:
            graph['item_'+df.loc[i,'movieId'].astype(str)]={}
        graph['item_'+ df.loc[i,'movieId'].astype(str)][df.loc[i,'userId']]=1
    return graph


def personal_rank(graph, root, alpha, iter_num, recom_num=10):
    '''
    Args:
        graph:user item graph
        root:the fixed user for which to recom
        alpha:the prob to go to random walk
        iter_num: iteration num
        recom_num:recom item num
    Return:
        a dick,key itemid,value pr
    '''
    recom_result = {}
    right_num = 0
    rank = {}
    rank = {point: 0 for point in graph}  # personal rank算法的的初始条件时：除root顶点初始化为1外，其余顶点的pr值都初始化为0
    rank[root] = 1
    for iter_index in range(iter_num):
        tmp_rank = {}
        tmp_rank = {point: 0 for point in graph}  # 存储该迭代轮次下，其余顶点对root顶点的pr值
        for out_point, out_dict in graph.items():
            for inner_point, value in graph[out_point].items():
                tmp_rank[inner_point] += round(alpha * rank[out_point] / len(out_dict), 4)
                if inner_point == root:
                    tmp_rank[inner_point] += round(1 - alpha, 4)
        if tmp_rank == rank:  # 认为这种情况下迭代充分了！
            break
        rank = tmp_rank

    for zuhe in sorted(rank.items(), key=operator.itemgetter(1), reverse=True):
        point, pr_score = zuhe[0], zuhe[1]
        if len(point.split('_')) < 2:  # 如果这个顶点并不是item顶点则过滤
            continue
        if point in graph[root]:  # 如果是item顶点但是之前被root顶点行为过，也同样过滤
            continue
        recom_result[point] = pr_score
        right_num += 1  # 记录满足要求的推荐item的个数
        if right_num > recom_num:
            break
    return recom_result

from scipy.sparse import coo_matrix
def graph_to_m(graph):
    '''
    Args:
        graph:user item graph
    Return:
        a coo_matrix sparse matrix M
        a list,total_user item point
        a dict,
    '''
    vertex=graph.keys()
    vertex=list(vertex)
    total_len=len(vertex)
    address_dict={}#存储每一个顶点位置
    for index in range(len(vertex)):
        address_dict[vertex[index]]=index
    row=[]
    col=[]
    data=[]
    for element_i in graph:#Mij表示顶点i到顶点j有没有看路径连通，如果有，数值就是顶点i出度的倒数
        weight=round(1/len(graph[element_i]),3)
        row_index=address_dict[element_i]
        for element_j in graph[element_i]:
            col_index=address_dict[element_j]
            row.append(row_index)
            col.append(col_index)
            data.append(weight)
    row=np.array(row)
    col=np.array(col)
    data=np.array(data)
    m=coo_matrix((data,(row,col)),shape=(total_len,total_len))
    return m,vertex,address_dict

def mat_all_point(m_mat,vertex,alpha):
    '''
    get E-alpha*m_mat.T
    Args:
        m_mat
        vertex:total item  and user point
        alpha:the prob for random walking
    Return:
        a sparse
    '''
    total_len=len(vertex)
    row=[]
    col=[]
    data=[]
    for index in range(total_len):
        row.append(index)
        col.append(index)
        data.append(1)
    row=np.array(row)
    col=np.array(col)
    data=np.array(data)
    eye_t=coo_matrix((data,(col,row)),shape=(total_len,total_len))
    return eye_t.tocsr()-alpha*m_mat.tocsr().transpose()

from scipy.sparse.linalg import gmres #用来求稀疏矩阵的逆
def personal_rank_mat(graph,root,alpha,recom_num=10):
    m,vertex,address_dict=graph_to_m(graph)
    if root not in address_dict:
        return {}
    score_dict={}
    recom_dict={}
    mat_all=mat_all_point(m,vertex,alpha)
    index=address_dict[root]
    initial_list=[[0] for row in range(len(vertex))]
    initial_list[index]=[1]
    r_zero=np.array(initial_list)
    res=gmres(mat_all,r_zero,tol=1e-8)[0]#tol是误差,gmres()输出一个元组，第一维度是一个数组，里面是其余所有顶点对该root顶点的pr值的得分
    for index in range(len(res)):
        point=vertex[index]
        if isinstance(point, np.int64):
            continue
        if point in graph[root]:#如果该顶点是root顶点已经行为过的，则过滤掉
            continue
        score_dict[point]=round(res[index],3)
    for zuhe in sorted(score_dict.items(),key=operator.itemgetter(1),reverse=True)[:recom_num]:
        point,score=zuhe[0],zuhe[1]
        recom_dict[point]=score
    return recom_dict


def get_one_user_by_mat():
    '''
    give a fixed user by mat

    '''
    user = 1
    alpha = .8
    graph = get_graph_from_data('Movielens_1M/ml-latest-small/ratings.csv')
    recom_result = personal_rank_mat(graph, user, alpha, 100)
    return recom_result

def main_flow():
    recom_result = get_one_user_by_mat()
    print(recom_result)

if __name__=='__main__':
    main_flow()