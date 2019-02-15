# -*- coding: utf-8 -*-

import pickle
import matplotlib.pyplot as plt
import numpy as np
import datetime

plk_list = ['local','batch','switch']
title_list = ['foreground','background']

fg_range = np.arange(0,5.1,.1)
bg_range = np.arange(0,2.1,.1)

def read_plk(plk_name):
    file_name = './'+plk_name+'_mnist.pkl'
    res_dic = pickle.load(open( file_name, "rb" ))
    return res_dic
"""
noisy_result['org_test'] = res
noisy_result['foreground'] = test_noisy(gaussian_range,x_test,0,0)
"""
def read_plks(plk_list):
    fg_matrix = []
    bg_matrix = []
    for p in plk_list:
        tmp = read_plk(p)
        fg_matrix.append(tmp['foreground'])
        bg_matrix.append(tmp['background'])
    fg_matrix =  np.array(fg_matrix)[:,:,1]
    bg_matrix =  np.array(bg_matrix)[:,:,1]
    return fg_matrix,bg_matrix

def plot_mat(x_range, x_mat,plk_list,title_str):
    for i in range(len(plk_list)):
        plt.plot(x_range,x_mat[i],label=plk_list[i])
    plt.title(title_str)
    plt.legend()
    plt.show()

def plot_all(m_dic,
            r=[fg_range,bg_range],
            plk_list=plk_list,
            title_list = title_list):
    fig, axs = plt.subplots(1, 2,figsize=(15,15))
    for q in range(len(title_list)):
        tit = title_list[q]
        for i in range(len(plk_list)):
            v = m_dic[tit][i,:]
            axs[q].plot(r[q],v,label=plk_list[i])
        axs[q].set_title(tit, fontsize=10)
        axs[q].set_ylim([0, 1])

    fig.subplots_adjust(bottom=0.2, wspace=0.5)
    axs[0].legend(loc='upper center', 
                bbox_to_anchor=(0.5, -0.2),
                fancybox=False, shadow=False, ncol=3)
    now = datetime.datetime.now()
    now_str = now.strftime("%Y-%m-%d %H:%M:%S")
    plt.savefig(now_str+'_mnist.png')



fg_mat,bg_mat = read_plks(plk_list)
print(bg_mat.shape)
# plot_mat(fg_range,bg_mat,plk_list,'foreground')
#plot_mat(gaussian_range,g_neg_mat,plk_list,'Gaussian-NEG')

keys = title_list
values = [fg_mat,bg_mat]
mat_dic = {key:value for key, value in zip(keys, values)}
plot_all(mat_dic)
