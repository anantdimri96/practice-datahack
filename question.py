
"""
ANANT DIMRI
Spyder Editor


This is a temporary script file.
"""



import numpy as np
import xgboost as xgb
import pandas as pd


from sklearn.preprocessing import LabelEncoder

# Reading datasets
train = pd.read_csv("/home/anant_dimri/Desktop/train.csv")
test = pd.read_csv("/home/anant_dimri/Desktop/test.csv")


# Saving id variables to create final submission
test_id = test['User_ID'].copy()
test_productid = test['Product_ID'].copy()

# Reducing boundaries to decrease RMSE
cutoff_purchase = np.percentile(train['Purchase'], 99.9) 
train.ix[train['Purchase'] > cutoff_purchase, 'Purchase'] = cutoff_purchase

# Label Encoding User_IDs
le = LabelEncoder()
train['User_ID'] = le.fit_transform(train['User_ID'])
test['User_ID'] = le.transform(test['User_ID'])

# Label Encoding Product_IDs
new_product_ids = list(set(pd.unique(test['Product_ID'])) - set(pd.unique(train['Product_ID'])))

le = LabelEncoder()
train['Product_ID'] = le.fit_transform(train['Product_ID'])
test.ix[test['Product_ID'].isin(new_product_ids), 'Product_ID'] = -1
new_product_ids.append(-1)

test.ix[~test['Product_ID'].isin(new_product_ids), 'Product_ID'] = le.transform(test.ix[~test['Product_ID'].isin(new_product_ids), 'Product_ID'])


y = train['Purchase']
train.drop(['Purchase', 'Product_Category_2', 'Product_Category_3'], inplace=True, axis=1)
test.drop(['Product_Category_2', 'Product_Category_3'], inplace=True, axis=1)

train = pd.get_dummies(train)
test = pd.get_dummies(test)

dtrain = xgb.DMatrix(train.values, label=y, missing=np.nan)

param = {'objective': 'reg:linear', 'booster': 'gbtree', 'silent': 1,
		 'max_depth': 10, 'eta': 0.1, 'nthread': 4,
		 'subsample': 0.8, 'colsample_bytree': 0.8, 'min_child_weight': 20,
		 }
num_round = 300

seeds = [3342,4556]
test_preds = np.zeros((len(test), len(seeds)))  


for run in range(len(seeds)):
	sys.stdout.write("\rXGB RUN:{}/{}".format(run+1, len(seeds)))
	sys.stdout.flush()
	param['seed'] = seeds[run]
	clf = xgb.train(param, dtrain, num_round)
	dtest = xgb.DMatrix(test.values, missing=np.nan)
test_preds[:, run] = clf.predict(dtest)

test_preds = np.mean(test_preds, axis=1)


   


submit = pd.DataFrame({'User_ID': test_id, 'Product_ID': test_productid, 'Purchase': test_preds})
submit = submit[['User_ID', 'Product_ID', 'Purchase']]

submit.ix[submit['Purchase'] < 0, 'Purchase'] = 12  # changing min prediction to min value in train
submit.to_csv("/home/anant_dimri/Desktop/anant_blackfriday.csv", index=False)
