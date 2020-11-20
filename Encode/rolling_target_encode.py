
#%% Rolling Target Encoding
import pandas as pd
import numpy as np
def rolling_target_encode(df, 
                          df_tar_col, 
                          df_seq_col,
                          df_cla_col,
                          seq_init = '20200101',
                          seq_format = '%Y%m%d',
                          seq_unit = 'Day',
                          cla_pos = '1',
                          col_isna = True,
                          Target_lambda = 0.5, 
                          col_newname = 'rolling_tar_encode'):
    def rolling_prob(df_tmp_, 
                     df_seq_col,
                     df_cla_col,
                     cla_pos,
                     col_newname):
        
        #print(df_tmp_)

        # Seq & col of df
        df = df_tmp_[[df_seq_col,df_cla_col]]

        # Sequential Statistics of df
        df_nparray = np.array(df)
        df_seq_unique = pd.unique(df[df_seq_col])
        df_seq_nparray = np.array(df[df_seq_col])
        df_seq_cond = np.zeros((df_seq_nparray.shape[0],3))
        
        # Rolling conditional probability
        df_nparray_tmp_posnum = df_nparray_tmp_num = 0
        for index_ in df_seq_unique:
            #print(index_)
            index_l = np.searchsorted(df_seq_nparray, index_, side='left')
            index_r = np.searchsorted(df_seq_nparray, index_, side='right')
            df_nparray_tmp_ = df_nparray[index_l:index_r]
            df_nparray_tmp_num = df_nparray_tmp_num + df_nparray_tmp_.shape[0]
            df_nparray_tmp_posnum = \
                df_nparray_tmp_posnum \
                + df_nparray_tmp_[df_nparray_tmp_[:,1]==cla_pos].shape[0]
            df_seq_cond[index_l:index_r,0] = df_nparray_tmp_posnum
            df_seq_cond[index_l:index_r,1] = df_nparray_tmp_num
            #print(index_,df_nparray_tmp_posnum,df_nparray_tmp_num)
            
        df_seq_cond[:,2] = df_seq_cond[:,0] / df_seq_cond[:,1] 

        df_prob = pd.DataFrame(data = df_seq_cond[:,2],
                               columns = [col_newname],
                               dtype='float')

        return pd.concat([df_tmp_.reset_index(drop=True),df_prob],axis=1)\
                 .reset_index(drop=True)
    
    if seq_unit=='Day':
        df['init'] = seq_init
        df['%s2Day' %df_seq_col] = (\
            pd.to_datetime(df[df_seq_col] ,format=seq_format)\
            - pd.to_datetime(df['init'] ,format=seq_format)\
            ).dt.days
        df_seq_col = '%s2Day' %df_seq_col
        
    
    index_cond = df[df_cla_col]!=cla_pos
    if np.all(index_cond) is True:
        print("Only one cla in cla_col!")
        return None
            
    # data processing
    if col_isna and np.any(df[df_tar_col].isna()):
       print('df_tar_col:%s isna!' %df_tar_col)
       return None   
    elif col_isna and np.any(df[df_seq_col].isna()):
       print('df_seq_col:%s isna!' %df_seq_col)
       return None
    elif col_isna and np.any(df[df_cla_col].isna()):
       print('df_cla_col:%s isna!' %df_cla_col)
       return None
       
    # rolling prob
    df = df.sort_values(df_seq_col, ascending=True, na_position='first')
    df_rolling = rolling_prob(df, df_seq_col, df_cla_col, cla_pos, 
                              col_newname = 'rolling_prob')

    #rolling cond prob
    
    df = df.sort_values([df_tar_col,df_seq_col], ascending=True, na_position='first')
    df_rolling = df_rolling.groupby(df_tar_col)\
                           .apply(rolling_prob,
                                  df_seq_col,
                                  df_cla_col,
                                  cla_pos,
                                  'rolling_tar_prob')
        
    #rolling target encode
    df_rolling[col_newname] = \
        Target_lambda*df_rolling['rolling_prob'] \
        + (1-Target_lambda)*df_rolling['rolling_tar_prob']

    return df_rolling

    
#%%    
if __name__ == "__main__":
    df = data.copy().reset_index(drop=True)
    df = df[['City_1','SubmissionDate','RE_FLAG']].head(10000)
    df['City_1'] = df['City_1'].fillna("?")
    df['RE_FLAG'] = df['RE_FLAG'].fillna("0")
    
    df_rolling_target = rolling_target_encode(\
                          df,
                          df_tar_col = 'City_1',
                          df_seq_col = 'SubmissionDate',
                          df_cla_col = 'RE_FLAG',
                          seq_init = '20200101',
                          seq_format = '%Y%m%d',
                          seq_unit = 'Day',
                          cla_pos = "1",
                          col_isna = True,
                          Target_lambda = 0.5,
                          col_newname = 'rolling_tar_encode'\
                      ).reset_index(drop=True)
