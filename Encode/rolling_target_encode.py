
#%% Rolling Target Encoding
import pandas as pd
import numpy as np
def rolling_target_encode(df, 
                          df_tar_col, 
                          df_seq_col,
                          df_cla_col,
                          cla_pos,
                          Target_lambda = 0.5, 
                          col_newname = 'rolling_tar_encode'):
    def rolling_prob(df_tmp_, 
                     df_seq_col,
                     df_cla_col,
                     cla_pos,
                     col_newname):
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

        return pd.concat([df_tmp_.reset_index(drop=True),df_prob],axis=1).reset_index(drop=True)

            
    # data processing
    index_cond = df[df_cla_col]!=cla_pos
    if np.all(index_cond) is True:
        print("Only one cla in cla_col!")
        return None
    if np.any(df[df_tar_col].isna()):
       print('df_tar_col:%s has isna!' %df_tar_col)
       return None   
    elif np.any(df[df_seq_col].isna()):
       print('df_seq_col:%s has isna!' %df_seq_col)
       return None
    elif np.any(df[df_cla_col].isna()):
       print('df_cla_col:%s has isna!' %df_cla_col)
       return None
       
    # rolling prob
    df = df.sort_values(df_seq_col, ascending=True, na_position='first')
    col_name1 = 'P(%s-roll)' %df_seq_col
    df_rolling = rolling_prob(df, df_seq_col, df_cla_col, cla_pos, col_name1)

    #rolling cond prob
    col_name2 = 'P(%s-roll | %s=%s)' %(df_seq_col,df_tar_col,cla_pos)
    df = df.sort_values([df_tar_col,df_seq_col], ascending=True, na_position='first')
    df_rolling = df_rolling.groupby(df_tar_col)\
                           .apply(rolling_prob,df_seq_col,df_cla_col, cla_pos, col_name2)
        
    #rolling target encode
    col_newname = '%s'%Target_lambda + col_name1 + ' + %s'%(1-Target_lambda) + col_name2
    df_rolling[col_newname] = \
        Target_lambda*df_rolling[col_name1] \
        + (1-Target_lambda)*df_rolling[col_name2]

    return df_rolling

#%%    
if __name__ == "__main__":
    tmp = [['A',1,1,False,True,True,True,True,-5],
           ['B',3,0,False,True,True,False,False,8],
           ['C',2,1,False,True,False,True,False,11],
           ['B',2,1,True,False,True,True,False,7],
           ['A',2,1,True,False,True,True,True,-4],
           ['C',4,0,False,False,True,True,False,13],
           ['B',4,0,True,True,False,True,True,9],
           ['C',1,1,True,True,True,True,False,10],
           ['A',6,0,False,True,True,True,False,0],
           ['D',1,0,True,False,False,True,False,14],
           ['B',1,0,True,True,True,True,False,6],
           ['A',3,1,True,True,False,True,True,-3],
           ['A',4,0,True,True,True,False,True,-2],
           ['C',3,1,True,True,True,True,True,12],
           ['A',5,1,True,True,True,True,False,-1],
           ['D',2,1,True,True,False,True,True,15]]
    df = pd.DataFrame(tmp,columns = ['X','seq','Y','var_4','var_5',
                                     'var_6','var_7','var_8','var_9'])
    df_rolling_target = rolling_target_encode(\
                          df,
                          df_tar_col = 'X',
                          df_seq_col = 'seq',
                          df_cla_col = 'Y',
                          cla_pos = 1,
                          Target_lambda = 0.5,
                          col_newname = 'rolling_tar_encode'\
                      ).reset_index(drop=True)



 
