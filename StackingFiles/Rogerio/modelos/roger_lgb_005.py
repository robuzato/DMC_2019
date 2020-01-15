import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import Normalizer

train = pd.read_csv("G:\\Meu Drive\\LP&D\\DMC_2019_task\\DMC\\Roger\\train.csv", delimiter='|')
test = pd.read_csv("G:\\Meu Drive\\LP&D\\DMC_2019_task\\DMC\\Roger\\test.csv", delimiter='|')
identificador = "001"
seed = 1981
limiar = 0.64
pesoZero = 2
pesoUm = 1
def scr(pred, v2, limiar):
    #limiar = 0.55
    if((pred >= limiar) and v2 == 1):
        return 5
    if((pred >= limiar) and v2 == 0):
        return -25
    if((pred < limiar) and v2 == 1):
        return -5
    else:
        return 0
    
def dmc_eval_score(preds, labels, limiar):
    err = [-25 if (a>=limiar) and (b == 0) else (5 if (a>=limiar) and (b == 1) else (-5 if (a < limiar) and (b == 1) else 0)) for (a,b) in zip(preds, labels)]
    return sum(err)    

def dmc_fobj(preds, dtrain):
    labels = dtrain.get_label()
    weight = dtrain.get_weight()
    preds = 1 / (1 + np.exp(-preds))
    grad = (preds - labels) * weight
    hess = (preds * (1 - preds))
    return grad, hess

def dmc_eval(preds, dtrain):
    labels = dtrain.get_label()
    err = [-25 if (a>=limiar) and (b == 0) else (5 if (a>=limiar) and (b == 1) else (-5 if (a < limiar) and (b == 1) else 0)) for (a,b) in zip(preds, labels)]
    return 'error', sum(err), True
 
def dmc_percentil(array):    
    pp = np.percentile(array , [5,95])
    return pp[0] , pp[1] 

def train_lgb(d_train, d_valid, val_sets, n_rounds):
    params = {   
		'boosting_type': 'dart',
		#'objective': 'xentropy',
		'learning_rate': 0.05,
		'max_depth': 100,
		'subsample': 0.4,
		'colsample_bytree': 0.7,       
		'colsample_bylevel': 0.7,
		'alpha': 0.6,
		'random_state': seed,
       'verbosity' :-1,
       'num_threads': -1,
       'num_leaves': 100,
       'xgboost_dart_mode': True,
       'uniform_drop': True,
       'lambda_l1': 0.4,
       'lambda_l2': 0.5
       }    
    
    model = lgb.train(params, 
            train_set=d_train, 
            valid_sets=[d_train, d_valid], 
            valid_names=['train','valid'],
            fobj = dmc_fobj,
            feval = dmc_eval, 
            num_boost_round=n_rounds,
            #early_stopping_rounds=25,
            verbose_eval=n_rounds)
        
    return model

#%%
    
train_trustLevel = train['trustLevel']
train = train.drop(['trustLevel'],axis=1)
test_trustLevel = test['trustLevel']
test = test.drop(['trustLevel'],axis=1)
target = train['fraud']
train = train.drop('fraud', axis=1)
geral_df = pd.concat([train,test], keys=['t1', 't2'])
scaler = Normalizer()
features =  geral_df.columns
for feat in features:
    inf ,sup = dmc_percentil(geral_df[feat])
    geral_df[feat] = np.where(geral_df[feat] > sup , sup , geral_df[feat])
    geral_df[feat] = np.where(geral_df[feat] < inf , inf , geral_df[feat])
    geral_df[feat] = (geral_df[feat] - inf)/ (sup-inf)
    
#%%

for i in range(0, len(features)) :
    for j in  range(i+1 , len(features)):
        if(i != j):
            geral_df[features[i]+features[j]] = geral_df[features[i]] * geral_df[features[j]]

geral_df.reset_index(inplace=True)
geral_df = geral_df.drop('level_1', axis = 1)
train = geral_df[geral_df['level_0'] == 't1']
test = geral_df[geral_df['level_0'] == 't2']
train = train.drop('level_0', axis = 1)
test = test.drop('level_0', axis = 1)
train['fraud'] = target
    
#%%
preds = np.zeros(1879)
score = 0
limiarMin = limiar
limiarMax = limiar
for i in range(0, 1879, 1):
    print("Iteração " + str(i+1))
    x_valid = train[i:i+1].copy(deep=True)
    y_valid = x_valid['fraud']
    x_valid = x_valid.drop('fraud',axis=1)
    x_train = train.drop(i, axis = 0)
    y_train = x_train['fraud']
    x_train = x_train.drop('fraud', axis=1)
    x_weights = np.where(y_train == 0, pesoZero, pesoUm )
    d_train = lgb.Dataset(x_train, y_train, weight = x_weights)
    d_valid = lgb.Dataset(x_valid, y_valid)
    model = train_lgb(d_train, d_valid, val_sets=[d_train, d_valid], n_rounds=1)
    model = train_lgb(d_train, d_valid, val_sets=[d_train, d_valid], n_rounds=300)
    #score = score + dmc_eval_score(model.predict(x_valid)[0], y_valid[i], limiar)
    preds[i] = model.predict(x_valid)
    score_act = -1000
    for j in np.arange(0, 1, 0.01):

        score = dmc_eval_score(preds[:i], train.fraud[:i], j)
        if (score > score_act):
            score_act = score
            limiarMin = j
        if (score >= score_act):
            #score_act = score
            limiarMax = j

    print("Limiar min: (" +str(limiarMin) + "), Limiar max: (" + str(limiarMax) + "), Score: "+ str(score_act))

preds = 1 / (1 + np.exp(-preds))
preds_df = pd.DataFrame()
preds_df['Score'] = preds
preds_df.reset_index(inplace=True)
#%%
preds_df['index'] = preds_df['index']+1
preds_df.rename(columns={'index': 'ID'},inplace=True)

best_score = -10000
best_limiar = -1
for i in range(0, 100, 1):
    timing_score = 0
    for j in range(0, 1879, 1):
        timing_score = timing_score + scr(preds[j], train['fraud'][j], (i/100))
    if best_score < timing_score:
        best_score = timing_score
        best_limiar = (i/100)
print("Melhor limiar: " + str(best_limiar) + " Pontuação: " + str(best_score))
#%%
preds_df.to_csv("G:\\Meu Drive\\LP&D\\DMC_2019_task\\DMC\\Roger\\Staking\\roger_lgb_"+identificador+"_train.csv",index=False)

features =  test.columns
#Train na base toda
y_train = train['fraud']
x_train = train.drop('fraud', axis = 1)
x_weights = np.where(y_train == 0, pesoZero, pesoUm )
d_train = lgb.Dataset(x_train, y_train, weight = x_weights)
d_valid = lgb.Dataset(y_train)
model = train_lgb(d_train,d_train, val_sets=[d_train], n_rounds=300)
preds = model.predict(test[features])
preds_df = pd.DataFrame()
preds_df['Score'] = preds
preds_df.reset_index(inplace=True)
preds_df['index'] = preds_df['index']+1
preds_df.rename(columns={'index': 'ID'},inplace=True)
preds_df.to_csv("G:\\Meu Drive\\LP&D\\DMC_2019_task\\DMC\\Roger\\Staking\\roger_lgb_"+identificador+"_test.csv",index=False)
    
#print("Best iteration for validation: ")
#max(model['error-mean'])