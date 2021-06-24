import os
import pandas as pd
import numpy as np


def read_eval():
    eval_collection = None
    for folder_d in os.listdir(os.path.join(os.getcwd(), "data")):
        eval_result = os.path.join(
            os.path.join(os.path.join(os.getcwd(), "data"), folder_d, "images_cropped"), 'result.csv')
        if not os.path.exists(eval_result):
            continue
        data = pd.read_csv(eval_result)
        defect = pd.read_csv('defect.csv')
        data = data[~data['name'].isin(defect.name.to_list())]
        data = np.array(data.values.tolist())
        eval = data
        if eval_collection is not None:
            eval_collection = np.concatenate((eval_collection, eval), axis=0)
        else:
            eval_collection = eval
    mean = eval_collection[:, 1:].astype(float).mean()
    std = eval_collection[:, 1:].astype(float).std()
    z_norm = (eval_collection[:, 1:].astype(float) - mean) / std
    min = eval_collection[:, 1:].astype(float).min()
    max = eval_collection[:, 1:].astype(float).max()
    minmax_norm = (eval_collection[:, 1:].astype(float) - min) / (max - min)
    result = np.concatenate((eval_collection, z_norm, minmax_norm), axis=1)
    pd_eval = pd.DataFrame(result, columns=['name', 'eval','z_norm', 'minmax_norm' ])
    pd_eval.to_csv('eval.csv', index=False)

read_eval()
feature = pd.read_csv("merge_feature.csv")
eval_h= pd.read_csv("eval.csv")
join_eval = feature.merge(eval_h, left_on='name', right_on='name')
join_eval.to_csv('join_eval.csv', index=False)
corrmat = join_eval.corr(method ='pearson')
print(corrmat)
