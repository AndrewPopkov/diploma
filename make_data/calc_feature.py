import os
import pandas as pd
from graphic_measure.brightness import *
from graphic_measure.contrast import *
from graphic_measure.focus_sharpness import *
from graphic_measure.luminance import illumination


def get_image_describe():
    img_result = []
    for folder_d in os.listdir(os.path.join(os.getcwd(), "data")):
        sample_Image = os.path.join(
            os.path.join(os.path.join(os.getcwd(), "data"), folder_d, "images_cropped"), folder_d + '.tif')
        for folder_i in os.listdir(os.path.join(os.path.join(os.getcwd(), "data"), folder_d, "images_cropped")):
            img_result.append(('main_' + folder_d + '.tif',  sample_Image))
            if not folder_i.endswith(".tif") and not folder_i.endswith(".csv"):
                for file_inner in os.listdir(
                        os.path.join(os.path.join(os.path.join(os.getcwd(), "data"), folder_d, "images_cropped"),
                                     folder_i)):
                    root_inner = os.path.join(
                        os.path.join(os.path.join(os.getcwd(), "data"), folder_d, "images_cropped"),
                        folder_i)
                    if file_inner.endswith(".tif"):
                        img_result.append(
                            (file_inner,  os.path.join(root_inner, file_inner)))
                        print(os.path.join(root_inner, file_inner))
    result = {}
    result['name'] = []
    result['rms_contrast'] = []
    result['michelson_contrast'] = []
    result['hbs_brightness'] = []
    result['bezryadinetal_brightness'] = []
    result['L1_norm'] = []
    result['energy_Laplacian'] = []
    result['kryszczuk_drygajlo_sharpness'] = []
    result['gao_sharpness'] = []
    result['tenengrad_sharpness'] = []
    result['adaptive_tenengrad_sharpness'] = []
    result['illumination'] = []
    i = 1
    tt=[]
    img_result=set(img_result)
    for pathFolder in img_result:
        image = Image.open(pathFolder[1])
        h,w,_ = np.array(image).shape
        tt.append([h,w])
        # if i % 500 == 0:
        #     measure = pd.DataFrame(result)
        #     measure.to_csv(str(i) + '_measure_result.csv', index=False)
        # result['name'].append(pathFolder[0])
        # result['rms_contrast'].append(calc_rms_contrast(pathFolder[1]))
        # result['michelson_contrast'].append(calc_michelson_contrast(pathFolder[1]))
        # result['hbs_brightness'].append(calc_hbs_brightness(pathFolder[1]))
        # result['bezryadinetal_brightness'].append(calc_bezryadinetal_brightness(pathFolder[1]))
        # result['L1_norm'].append(L1_norm(pathFolder[1]))
        # result['energy_Laplacian'].append(energy_Laplacian(pathFolder[1]))
        # result['kryszczuk_drygajlo_sharpness'].append(kryszczuk_drygajlo_sharpness(pathFolder[1]))
        # result['gao_sharpness'].append(gao_sharpness(pathFolder[1]))
        # result['tenengrad_sharpness'].append(tenengrad_sharpness(pathFolder[1]))
        # result['adaptive_tenengrad_sharpness'].append(adaptive_tenengrad_sharpness(pathFolder[1]))
        # result['illumination'].append(illumination(pathFolder[1]))
        print("calc measure for " + pathFolder[0])
        i = i + 1
    p=np.array(tt)
    h_a=p[:, 0:1]
    w_a = p[:, 1:]
    measure = pd.DataFrame(result)
    measure.to_csv('measure_result.csv', index=False)
    return result


get_image_describe()
# forPA05_06.tif
