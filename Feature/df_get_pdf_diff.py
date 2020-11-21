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
#%%
def df_get_pdf_diff(df, feature, cla_col, cla_val,Interval):
    '''
    獲得目標變數的 正例pdf 以及 反例pdf，並計算兩者之間的pdf差，並返回。
    '''
    def PDF_func(dataframe,feature,start,end,Interval):
        #initial
        N = len(dataframe[feature])
        seq = dataframe[feature].sort_values().reset_index(drop=True)
        pdf_range = range(start,end,Interval)
       
        tmp_pdframe = pd.Series(
                [sum((pdf_range[i-1]<seq ) & (seq<pdf_range[i]))/N for i in range(1,len(pdf_range))]
                ,name='pdf')
        #N個數有N-1個區間
        #區間長度:Index~Index+Interval
        tmp_pdframe.index = pdf_range[:-1]
        return tmp_pdframe

    df_pdf1 = PDF_func(df[df[cla_col]==cla_val],
                       feature,
                       df[feature].astype(int).min(),
                       df[feature].astype(int).max(),
                       Interval)
    
    df_pdf2 = PDF_func(df[df[cla_col]!=cla_val],
                       feature,
                       df[feature].astype(int).min(),
                       df[feature].astype(int).max(),
                       Interval)

    pdf_diff = df_pdf1 - df_pdf2
    df['%s_pdf_diff'%feature] = df[feature].astype(int).map(pdf_diff.to_dict())

    return df

#%%
if __name__ == "__main__":
    mu, sigma = 10, 5
    s1 = st.norm(mu, sigma).rvs(1000)
    s2 = st.norm(mu-4, sigma+4).rvs(1000)

    df_1 = pd.DataFrame(s1,columns=["feat"]  )
    df_1['Y'] = 0
    df_2 = pd.DataFrame(s2,columns=["feat"]  )
    df_2['Y'] = 1
    df = pd.concat([df_1,df_2],axis=0).reset_index(drop=True)
    
    df = df_get_pdf_diff(df, feature='feat', cla_col='Y', cla_val=1, Interval=1)

    sns.pairplot(df.sample(n=1000),hue="Y")
