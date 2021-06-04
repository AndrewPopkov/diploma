import pandas as pd
import numpy as np

data = pd.read_csv("measure_result.csv")

main_data = data[data.name.str.contains("main_")]
main_data.to_csv('main_data.csv', index=False)
mean = np.array(main_data.mean().tolist())
std = np.array(main_data.std().tolist())
d = np.array(data.values.tolist())
input = d[:, 1:].astype(float)

g = np.exp(-((input - mean) ** 2 / (2.0 * std ** 2)))
d[:, 1:] = g
normalase_feature = pd.DataFrame(d, columns=data.columns)
normalase_feature.to_csv('normalase_feature.csv', index=False)
mean_result = g.mean(axis=1).reshape(3311, 1)
merge_feature = pd.DataFrame(np.concatenate((d[:, :1], mean_result), axis=1), columns=['name', 'merge_result'])
merge_feature.to_csv('merge_feature.csv', index=False)
# print(merge_feature.head(100))
