import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import copy
from collections import deque
from pandas.core.frame import DataFrame
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st
    
def sns_muldf_distplot_(df_list,sample_space_,colors_,des_label_,
                        xlim_left_,xlim_right_,
                        filter_cond=None,bin_=20,xlabel_='Group',
                        ylabel_='Amount',size=[3,3],
                        fontsize_=9,labelsize_=9,font_scale_=1):
    '''
    在 DataFrame-i 中，透過篩選不同條件k(filter_cond)，呈現第j個變數(sample_space_)
    的機率分布，並畫在子圖j上。
    ->plt subfig-j :P_Sj_Di(Y|cond=k), Di = i-th in DataFrame list
    
    df_list: 
        Dataframe-i in Dataframe list.
    sample_space: 
        Feature name. 
        Plt feature data of Dataframe-i at subfigure.
    xlim_left_:
        int,float
        subfigure xlimit-max.
    xlim_right_:
        int,float
        subfigure xlimit-min.
    colors_:
        Color of line .
        [D1-Cond1-Color, D1-Cond2-Color, ..., Di-Condk-1-Color,Di-Condk-Color, ...]
    des_label_:
        line description.
        [D1-Cond1-Description, D1-Cond2-Description, ..., Di-Condk-1-Description,Di-Condk-Description, ...]
    filter_cond:
        default is None.
        dictionary. {"feature name":value}
        Plt line of diff feature value at subfigure.
    bin_:
        default is 20.
        number of interval.
    xlabel_:
        xlabel_ description.
    ylabel_:
        yLabel description.
    size:
        nXm subfigure in one figure.
    '''
    
    # 畫布設定
    sns.set(style='ticks',palette='muted',color_codes=True,font_scale=font_scale_)
    
    #畫布初始化
    plt.figure()
    fig, ax = plt.subplots(size[0], size[1], figsize=(24, 30))

    #调整子图间距
    plt.subplots_adjust(wspace=0.3, hspace=0.5)
    
    # 篩選設定
    fv_flag = filter_cond
    
    # 不同畫布，不同樣本空間
    for j,col_ in enumerate(sample_space_):
        # 第j+1畫布
        plt.subplot(size[0], size[1], j+1)
        plt.title(col_,fontsize=fontsize_)
        
        #樣本空間，變數x範圍
        plt.xlim([xlim_left_[j],xlim_right_[j]])
        
        t_ = 0
        for i,df_ in enumerate(df_list):
            if fv_flag == None:
                filter_cond = {'fv':[True]}
                df_['fv'] = True
            # 資料範圍要符合樣本空間
            cond_left = np.array(xlim_left_[j]<=df_[col_])
            cond_right = np.array(df_[col_]<=xlim_right_[j])
            cond = cond_left & cond_right
            plot_data = df_.loc[cond,:]
            
            # plt P_Di(S_j|cond=k), Di = i-th in DataFrame list
            for k1,f_ in enumerate(filter_cond):   
                for k2,fv_ in enumerate(filter_cond[f_]):  
                    color_index = colors_[t_]
                    label_index = des_label_[t_]
                    t_ += 1

                    # 拉出分佈資料
                    plot_data_tmp_=plot_data.loc[plot_data[f_]==fv_,col_]
                    
                    if plot_data_tmp_.shape[0] == 0:
                        continue;
                        
                    xlim_arr = np.array([xlim_left_[j],xlim_right_[j]])
                    plot_data_tmp_ = plot_data_tmp_.append(pd.Series(xlim_arr),ignore_index = True) 
    
                    # 劃出資料分佈
                    sns.distplot(plot_data_tmp_,    # 資料
                                 bins=bin_[j],      # 間距個數
                                 hist=True,         # 是否畫出長條圖
                                 norm_hist=True,    # normal distribution
                                 kde=True,          # 是否畫出kde
                                 kde_kws={'bw':xlim_right_[j]/bin_[j]/2},   #kde 範圍
                                 color=color_index,          # 分佈顏色 
                                 label=label_index)          # 分佈文字說明
                    
            plt.ylabel(ylabel_, fontsize=fontsize_)
            plt.xlabel(xlabel_, fontsize=fontsize_)
            locs, labels = plt.xticks()
            plt.tick_params(axis="x", which="both", labelsize=labelsize_, pad=4)
            plt.tick_params(axis="y", which="both", labelsize=labelsize_)
            plt.legend()
        
    plt.show()

#%%
if __name__ == "__main__":
    # 常態分佈數據``
    mu, sigma = 10, 5
    df_1 = pd.DataFrame(st.norm(mu, sigma).rvs(100000),
                        columns=["feat"]  )
    df_2 = pd.DataFrame(st.norm(mu-4, sigma+4).rvs(100000),
                        columns=["feat"]  )
    df = pd.concat([df_1,df_2],axis=0)
    
    # 多數據畫圖 1
    sns_muldf_distplot_(df_list = [df_1,df_2],
                        sample_space_ = ['feat'],
                        colors_ = ['#003366','#FB3523'],
                        des_label_ = ['Y-0','Y-1'],
                        xlabel_ = 'X',
                        ylabel_ = 'PDF',
                        size=[1,1],
                        bin_ = [20],
                        xlim_left_ = [df['feat'].min()],
                        xlim_right_ =[df['feat'].max()])


