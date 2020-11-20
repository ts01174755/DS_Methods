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
from collections import defaultdict
        
def df_get_cdf(df,X,Y,ascending=True):
    def zero():
        return 0
    # df sort 
    df_sort = df.sort_values(by=[X],ascending=ascending).reset_index(drop=True)
    arr_len = df_sort.shape[0]
    
    # Y_arrary
    Y_arr = np.array(df_sort[Y])
    PY_arr = np.zeros(arr_len,dtype=float)
    tmp_N = defaultdict(zero)

    for ind_,val_ in enumerate(Y_arr):
        PY_arr[ind_] = tmp_N[val_] + 1
        tmp_N[val_] += 1
    for ind_,val_ in enumerate(Y_arr):
        PY_arr[ind_] = PY_arr[ind_] / tmp_N[val_]

    df_cdf = pd.concat([df_sort,pd.DataFrame(PY_arr,columns=["%s_cdf"%Y])],axis=1)
    
    return df_cdf.sort_values(by=[X],ascending=True).reset_index(drop=True)

#%%
if __name__ == "__main__":
    mu, sigma = 10, 5
    s1 = st.norm(mu, sigma).rvs(100000)
    s2 = st.norm(mu-4, sigma+4).rvs(10000)

    df_1 = pd.DataFrame(s1,columns=["feat"]  )
    df_1['Y'] = 0
    df_2 = pd.DataFrame(s2,columns=["feat"]  )
    df_2['Y'] = 1
    df = pd.concat([df_1,df_2],axis=0).reset_index(drop=True)
    df_cdf1 = df_get_cdf(df, X="feat", Y="Y", ascending=True)
    df_cdf2 = df_get_cdf(df, X="feat", Y="Y", ascending=False)
    
    sns.lineplot(data=df_cdf1, x="feat", y="Y_cdf", hue="Y", style="Y")
    sns.lineplot(data=df_cdf2, x="feat", y="Y_cdf", hue="Y", style="Y")
    plt.legend(loc="best")
    plt.show()

#%% cdf
import pandas as pd
import numpy as np
import copy
from collections import deque
from pandas.core.frame import DataFrame
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st

#
mu, sigma = 10, 5
s1 = st.norm(mu, sigma).rvs(1000000)
s2 = st.norm(mu-4, sigma+4).rvs(100000)

df_1 = pd.DataFrame(s1,columns=["feat"]  )
df_1['Y'] = 0
df_2 = pd.DataFrame(s2,columns=["feat"]  )
df_2['Y'] = 1

df = pd.concat([df_1,df_2],axis=0).reset_index(drop=True)
#
feature_cols = [col for col in df.columns if col != 'Y']
target_cols = [col for col in df.columns if col not in feature_cols]

#
def plot_feature_distribution(df1, df2, label1, label2, features,plot_rows,plot_cols):
    i = 0
    sns.set_style('whitegrid')
    plt.figure()
    fig, ax = plt.subplots(1,1,figsize=(12,7))

    for feature in features:
        i += 1
        plt.subplot(plot_rows,plot_cols,i)
        sns.distplot(df1[feature], hist=False,label=label1)
        sns.distplot(df2[feature], hist=False,label=label2)
        plt.xlabel(feature, fontsize=9)
        locs, labels = plt.xticks()
        plt.tick_params(axis='x', which='major', labelsize=6, pad=-6)
        plt.tick_params(axis='y', which='major', labelsize=6)
    plt.show();

t0 = df.loc[df['Y'] == 0]
t1 = df.loc[df['Y'] == 1]
#features = df.columns.values[0]
plot_feature_distribution(t0, t1, '0', '1', feature_cols,1,1)

##%%
#Input一個Feat & Label, 得到各點P(Y=1) - P(Y=0)的機率
#若該值是正,代表該值域下1的樣本點比0多
def PDF_func(dataframe,feature,start,end,step,FileName):
    #initial
    N = len(dataframe[feature])
    seq = dataframe[feature].sort_values().reset_index(drop=True)
    pdf_range = range(start,end,step)
    
    tmp_pdframe = pd.DataFrame(
            [sum((pdf_range[i-1]<seq ) & (seq<pdf_range[i]))/N for i in range(1,len(pdf_range))]
            ,columns=['pdf'])
    #N個數有N-1個區間
    #區間長度:Index~Index+step
    tmp_pdframe.index = pdf_range[:-1]
    globals()['pdf_frame_'+str(FileName)] = tmp_pdframe
    print("Done!File Name: pdf_frame_"+ str(FileName))

PDF_func(df[df['Y']==1],'feat',int(min(df['feat'])),int(max(df['feat'])),1,'feat_1')
PDF_func(df[df['Y']==0],'feat',int(min(df['feat'])),int(max(df['feat'])),1,'feat_0')

pdf_diff = pdf_frame_feat_1-pdf_frame_feat_0
#找出差異最大的區間

#Create Feature
df['pdf_diff_feat']=pdf_diff.loc[round(df['feat'])].reset_index(drop=True)

#
def CDF_func(dataframe,feature,start,end,step,FileName):
    #initial
    N = len(dataframe[feature])
    seq = dataframe[feature].sort_values().reset_index(drop=True)
    cdf_range = range(start,end,step)

    tmp_cdframe = pd.DataFrame(
            [sum(seq<value)/N for value in cdf_range],columns=['cdf'])
    tmp_cdframe.index = cdf_range
    
    globals()['cdf_frame_'+str(FileName)] = tmp_cdframe
    print("Done!File Name: cdf_frame_"+ str(FileName))

CDF_func(df[df['Y']==1],'feat',int(min(df['feat']))-2,int(max(df['feat']))+2,1,'feat_1')
CDF_func(df[df['Y']==0],'feat',int(min(df['feat']))-2,int(max(df['feat']))+2,1,'feat_0')

cdf_div_10 = cdf_frame_feat_1/cdf_frame_feat_0
cdf_div_01 = cdf_frame_feat_0/cdf_frame_feat_1

cdf_feat = cdf_div_10+cdf_div_01

#Create Feature
df['cdf_div_feat']=cdf_feat.loc[round(df['feat'])].reset_index(drop=True)







###########################
#
import xgboost as xgb
param = {'max_depth': 8,
         'learning_rate ': 0.02,
         'silent': 1,
         'objective': 'binary:logistic',
         "eval_metric":"auc"
         #"scale_pos_weight":10,
         #"subsample":0.9,
         #"min_child_weight":5,
          }

x_col=['feat','pdf_diff_feat','cdf_div_feat']
train_X = df[x_col]

y_col=['Y']
train_Y = df[y_col]

dtrain = xgb.DMatrix(train_X,label=train_Y)

cv_res= xgb.cv(param,
               dtrain,
               num_boost_round=1000,#830
               early_stopping_rounds=10,
               nfold=3, metrics='logloss',show_stdv=True)

print(cv_res)

#cv_res.shape[0]為最佳迭代次?
bst = xgb.train(param,dtrain,num_boost_round=cv_res.shape[0])
bst.get_score(importance_type='gain')


