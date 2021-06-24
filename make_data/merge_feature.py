import pandas as pd
import numpy as np
# name,rms_contrast,michelson_contrast,hbs_brightness,bezryadinetal_brightness,L1_norm,energy_Laplacian,
# kryszczuk_drygajlo_sharpness,gao_sharpness,tenengrad_sharpness,adaptive_tenengrad_sharpness,illumination
data = pd.read_csv("measure_result.csv")
main_data = data[data.name.str.contains("main_")]

main_data.to_csv('main_data.csv', index=False)
mean = np.array(main_data.mean().tolist())
std = np.array(main_data.std().tolist())
d = np.array(data.values.tolist())
input = d[:, 1:].astype(float)

g = np.exp(-((input - mean) ** 2 / (2.0 * std ** 2)))
d[:, 1:] = g
n_f = pd.DataFrame(d, columns=data.columns)
n_f.to_csv('normalase_feature.csv', index=False)
contrast =  (np.array(n_f.rms_contrast.tolist()).astype(np.float)+np.array(n_f.michelson_contrast.tolist()).astype(np.float))/2
brightness =  (np.array(n_f.hbs_brightness.tolist()).astype(np.float)+np.array(n_f.bezryadinetal_brightness.tolist()).astype(np.float))/2
focus= (np.array(n_f.L1_norm.tolist()).astype(np.float)+np.array(n_f.energy_Laplacian.tolist()).astype(np.float))/2
sharpness= (np.array(n_f.kryszczuk_drygajlo_sharpness.tolist()).astype(np.float)+np.array(n_f.gao_sharpness.tolist()).astype(np.float)+\
           np.array(n_f.tenengrad_sharpness.tolist()).astype(np.float)+np.array(n_f.adaptive_tenengrad_sharpness.tolist()).astype(np.float))/4
illumination=np.array(n_f.illumination.tolist()).astype(np.float)
norm_mean_feature=np.concatenate((d[:, :1],contrast.reshape(3311, 1), brightness.reshape(3311, 1),focus.reshape(3311, 1),
                                  sharpness.reshape(3311, 1),illumination.reshape(3311, 1)), axis=1)
n_m_f = pd.DataFrame(norm_mean_feature, columns=['name', 'contrast', 'brightness', 'focus', 'sharpness', 'illumination'])
n_m_f.to_csv('norm_mean_feature.csv', index=False)
# michelson_contrast, adaptive_tenengrad_sharpness, energy_Laplacian, illumination
m_c=np.array(n_f.michelson_contrast.tolist()).astype(np.float)
a_t_s=np.array(n_f.adaptive_tenengrad_sharpness.tolist()).astype(np.float)
e_l=np.array(n_f.energy_Laplacian.tolist()).astype(np.float)
il=np.array(n_f.illumination.tolist()).astype(np.float)
norm_selected_feature=np.concatenate((d[:, :1],m_c.reshape(3311, 1), a_t_s.reshape(3311, 1),e_l.reshape(3311, 1),
                                  il.reshape(3311, 1)), axis=1)

mean = norm_mean_feature[:, 1:].astype(np.float).mean(axis=1)
geo_mean= norm_mean_feature[:, 1:].astype(np.float).prod(axis=1)**(1.0/norm_mean_feature[:, 1:].shape[1])
merge_feature = pd.DataFrame(np.concatenate((d[:, :1], mean.reshape(3311, 1), geo_mean.reshape(3311, 1)), axis=1), columns=['name', 'mean', 'geo_mean'])
merge_feature.to_csv('merge_feature.csv', index=False)
# print(merge_feature.head(100))

selected_mean = norm_selected_feature[:, 1:].astype(np.float).mean(axis=1)
selected_geo_mean= norm_selected_feature[:, 1:].astype(np.float).prod(axis=1)**(1.0/norm_mean_feature[:, 1:].shape[1])
selected_merge_feature = pd.DataFrame(np.concatenate((d[:, :1], selected_mean.reshape(3311, 1), selected_geo_mean.reshape(3311, 1)), axis=1), columns=['name', 'mean', 'geo_mean'])
selected_merge_feature.to_csv('selected_merge_feature.csv', index=False)
