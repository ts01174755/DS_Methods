#%%
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import copy
from collections import deque
from pandas.core.frame import DataFrame
from sklearn import metrics
import scipy.stats as st
from collections import defaultdict
from scipy.interpolate import interp1d
import math

#%%
def df_get_pdf(df,X,Y,X_len,delta=None):
    def feature_cdf(df,X,Y,ascending=True):
        def zero():
            return 0    
        # df sort 
        arr_len = df.shape[0]
        feat_N = df[Y].unique()
        
        # Y_arrary
        Y_arr = np.array(df[Y])
        PY_arr = np.zeros((arr_len,feat_N.shape[0]),dtype=float)
        columns = ["%s_cdf(%s=%s)" %("As" if ascending==True else "Ds",Y,v) for i,v in enumerate(feat_N)]
    
        tmp_N = defaultdict(zero)
        for ind_,val_ in enumerate(Y_arr):
            for fi_,fv_ in enumerate(feat_N):
                if val_ == fv_:
                    tmp_N[fv_] += 1
                    PY_arr[ind_,fi_] = tmp_N[fv_]
                else:
                    PY_arr[ind_,fi_] = tmp_N[fv_]
                    
        for fi_,fv_ in enumerate(feat_N):
            PY_arr[:,fi_] = PY_arr[:,fi_] / tmp_N[fv_]
    
        df_cdf = pd.concat([df[X],
                            pd.DataFrame(PY_arr,columns=columns)],axis=1)\
                    .groupby(by=[X]).max()
        return df_cdf.sort_index()
    
    def get_interfunc(df,Y):
        X_Y = np.zeros((df.shape[0] + 2,2))
        X_Y[1:-1,0] = df.index
        X_Y[1:-1,1] = df[Y]
        X_Y[0] = 2 * X_Y[1] - X_Y[2]
        X_Y[-1] = 2 * X_Y[-2] - X_Y[-3]
        X_Y[0,1] = 0
        X_Y[-1,1] = 1

        func_ = interp1d(X_Y[:,0], X_Y[:,1], kind = 'quadratic')
        return func_, X_Y[-2,0] - X_Y[-3,0]
    
    df_sort = df.sort_values(by=[X]).reset_index(drop=True)
    feat_X = df_sort[X].to_numpy()

    interval_n = math.ceil((feat_X.max()-feat_X.min())/X_len)
    interval_v = [feat_X.min() + (i-0.5)*X_len for i in range(interval_n+1)]
    for i in range(interval_n):
        feat_X[(interval_v[i] <= feat_X) & (feat_X < interval_v[i+1]) ] = \
            (interval_v[i] + interval_v[i+1])/2
    df_sort[X] = feat_X

    feat_cdf = feature_cdf(df_sort, X=X, Y=Y)
    cols = feat_cdf.columns    
    
    if delta == None:
        delta = X_len/2
    
    for col in cols:
        XY_func,delta_tmp = get_interfunc(feat_cdf,col)
        delta = delta_tmp if delta_tmp < delta else delta
        X_delta1 = feat_X + delta
        X_delta2 = feat_X - delta

        Y_delta1 = XY_func(X_delta1)
        Y_delta2 = XY_func(X_delta2)
    
        df_sort['pdf' + col[6:]] = (Y_delta2 - Y_delta1 ) /(X_delta2 - X_delta1)
    
    return df_sort


#%%
if __name__ == "__main__":
    #data prepare
    mu, sigma = 10, 5
    s1 = st.norm(mu, sigma).rvs(100000)
    s2 = st.norm(mu-4, sigma+4).rvs(10000)

    df_1 = pd.DataFrame(s1,columns=["feat"]  )
    df_1['Y'] = 0
    
    df_2 = pd.DataFrame(s2,columns=["feat"]  )
    df_2['Y'] = 1
    df = pd.concat([df_1,df_2],axis=0).reset_index(drop=True)
    
    # df_get_pdf
    df_pdf = df_get_pdf(df, X='feat', Y='Y', X_len=1, delta=.1)
    df_pdf['pdf_diff'] = df_pdf['pdf(Y=1)'] - df_pdf['pdf(Y=0)']

    #plt
    plt.figure(figsize=(15,10),dpi=100,linewidth = 2)
    plt.plot(df_pdf['feat'],df_pdf['pdf(Y=1)'],'-',color = 'r', label="pdf(Y=1)")
    plt.plot(df_pdf['feat'],df_pdf['pdf(Y=0)'],'-',color = 'g', label="pdf(Y=0)")
    plt.plot(df_pdf['feat'],df_pdf['pdf_diff'],'-',color = 'b', label="pdf_diff")
    
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    
    # 標示x,y軸(labelpad代表與圖片的距離)
    plt.xlabel("feat", fontsize=30, labelpad = 15)
    plt.ylabel("pdf", fontsize=30, labelpad = 20)
    
    # 顯示出線條標記位置
    plt.legend(loc = "best", fontsize=20)
    plt.show()
    
