import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
Train=pd.DataFrame(pd.read_csv('train_xy.csv'))
Test=pd.DataFrame(pd.read_csv('test_all.csv'))

Test['y']=-1
Total_Data= pd.concat([Train, Test],ignore_index=True) #type: pd.DataFrame


#移除唯一值列
Total_Data.pop('x_110')
Total_Data.pop('x_112')
save_value_col=Total_Data.columns.values.tolist()
save_value_col.remove('cust_id')
save_value_col.remove('cust_group')


save_value_col=Total_Data.columns.values.tolist()
save_value_col.remove('cust_id')
save_value_col.remove('cust_group')
save_value_col.remove('y')
concat_ses_list=[]

#将缺失值改为-1，减少缺失值的处罚力度
concat_ses_list.append(Total_Data['y'])
for s in save_value_col:
    l=Total_Data[s].copy()
    for i in range(len(l)):
        if l[i]==-99:
            l[i]=-1
    concat_ses_list.append(l)


Concat_df = pd.concat(concat_ses_list,axis=1)


Train_df=Concat_df[Concat_df['y']!=-1]
Test_df=Concat_df[Concat_df['y']==-1]

Concat_df_list=Concat_df.columns.values.tolist()
Concat_df_list.remove('y')

Train_lg=Train_df[Concat_df_list].values
Train_label=Train_df['y'].values
Test_lg=Test_df[Concat_df_list].values


#xgboost，lr模型stacking融合
clf= XGBClassifier(max_depth=4, learning_rate=0.1,
                 n_estimators=80, silent=True,
                 objective="binary:logistic", booster='gbtree',min_child_weight=3,subsample=0.8,
 gamma=0)
 
 from sklearn.linear_model import LogisticRegression
clf2= LogisticRegression(C=0.1, penalty='l2', tol=1e-4)

from sklearn.ensemble import RandomForestClassifier
clf4= RandomForestClassifier(n_estimators=400,oob_score=True)



from sklearn.ensemble import VotingClassifier
from mlxtend.classifier import StackingClassifier


eclf= StackingClassifier(classifiers=[clf,clf2],
                          meta_classifier=LogisticRegression(C=0.1, penalty='l2', tol=1e-4), use_probas=True,  verbose=3)
eclf.fit(Train_lg, Train_label)
R=eclf.predict_proba(Test_lg)
instance_id_list=Test['cust_id'].values

with open('1.csv','w') as f:
    f.write('cust_id,pred_prob\n')
    for i in range(len(instance_id_list)):
        f.write('%d,%f\n'%(instance_id_list[i],float(R[i][1])))