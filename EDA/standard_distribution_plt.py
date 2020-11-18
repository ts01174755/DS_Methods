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
        
def plot_feature_distribution(df1, df2, label1, label2, features,plot_rows,plot_cols):
    i = 0
    sns.set_style('whitegrid')
    plt.figure()
    fig, ax = plt.subplots(1,1,figsize=(20,15))

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

#%%
if __name__ == "__main__":
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
    
    t0 = df.loc[df['Y'] == 0]
    t1 = df.loc[df['Y'] == 1]
    t = t0.append(t1,ignore_index = True)
    
    plot_feature_distribution(t0, t1, '0', '1', feature_cols,1,1)
    












