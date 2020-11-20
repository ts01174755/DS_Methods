
#%% xgb
import xgboost as xgb
param = {'max_depth': 8,
         #'learning_rate ': 0.02,
         #'silent': 1,
         'objective': 'binary:logistic',
         "eval_metric":"auc"
         #"scale_pos_weight":10,
         #"subsample":0.9,
         #"min_child_weight":5,
          }

x_col = list(df_cdf.drop(['Y'],axis=1).columns)
y_col = ['Y']

train_X = df_cdf[x_col]
train_Y = df_cdf[y_col]

dtrain = xgb.DMatrix(train_X,label=train_Y)

cv_res= xgb.cv(param,
               dtrain,
               num_boost_round=1000,#830
               early_stopping_rounds=10,
               nfold=3, metrics='logloss',show_stdv=True)

print(cv_res)

#cv_res.shape[0]為最佳迭代次?
bst = xgb.train(param,dtrain,num_boost_round=cv_res.shape[0])
print(bst.get_score(importance_type='gain'))


