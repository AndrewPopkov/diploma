import os
import cv2
import json
import numpy as np


def read_json(path_ann, main=True):
    with open(path_ann, 'r') as doc:
        if main:
            json_doc = json.load(doc)
            images = json_doc['photo']['quad']
            img_describes = {
                'id': path_ann,
                'bbox': images
            }
            return img_describes
        else:
            json_doc = json.load(doc)
            images = json_doc['quad']
            img_describes = {
                'id': path_ann,
                'bbox': images
            }
            return img_describes


def get_bar_sample(path):
    image = cv2.imread(path)
    h, w = image.shape[0:2]
    img_describes = {
        'id': path,
        'bbox': [[0, 0], [w, 0], [w, h], [0, h]]
    }
    return img_describes


# [[1.         1.        ], [0.87511478 0.80952381], [0.9513315  0.82795699], [3.46938776 0.91397849]]
# def calc_tranform(photo_on_doc, sample_bar_doc, bar_doc_real):
#     anc = np.array(bar_doc_real['bbox'][0])
#     photo_on_doc_dif = np.array(photo_on_doc['bbox']) + anc
#     box_doc = np.array(sample_bar_doc['bbox']) + anc
#     transformation_rates = np.array(bar_doc_real['bbox']) /box_doc
#     # tt=np.array(photo_json['bbox'])-np.array(sample_img_bar['bbox'])
#     dstTriangle = np.array([
#         box_doc[0],
#         box_doc[1],
#         box_doc[2],
#     ], dtype=np.float32)
#     srcTriangle = np.array([
#         bar_doc_real['bbox'][0],
#         bar_doc_real['bbox'][1],
#         bar_doc_real['bbox'][2],
#     ], dtype=np.float32)
#     rotMat = cv2.getAffineTransform(srcTriangle, dstTriangle )
#     return rotMat.tolist()


def get_image_describe():
    path_img_result = []
    for folder_d in os.listdir(os.path.join(os.getcwd(), "data")):
        sample_Image = os.path.join(
            os.path.join(os.path.join(os.getcwd(), "data"), folder_d, "images"), folder_d + '.tif')
        sample_Json = os.path.join(
            os.path.join(os.path.join(os.getcwd(), "data"), folder_d, "ground_truth"), folder_d + '.json')
        for folder_i in os.listdir(os.path.join(os.path.join(os.getcwd(), "data"), folder_d, "images")):
            path_img = {}
            path_img['Name'] = folder_d + "/" + folder_i
            path_img['Images'] = []
            path_img['Ground_Truth'] = []
            path_img['Sample_Image'] = sample_Image
            path_img['Sample_Json'] = sample_Json
            if not folder_i.endswith(".tif"):
                for file_inner in os.listdir(
                        os.path.join(os.path.join(os.path.join(os.getcwd(), "data"), folder_d, "images"), folder_i)):
                    root_inner = os.path.join(os.path.join(os.path.join(os.getcwd(), "data"), folder_d, "images"),
                                              folder_i)
                    if file_inner.endswith(".tif"):
                        path_img["Images"].append(
                            {'file_name': file_inner, 'file_path': os.path.join(root_inner, file_inner)})
                        print(os.path.join(root_inner, file_inner))
                for file_inner in os.listdir(
                        os.path.join(os.path.join(os.path.join(os.getcwd(), "data"), folder_d, "ground_truth"),
                                     folder_i)):
                    root_inner = os.path.join(os.path.join(os.path.join(os.getcwd(), "data"), folder_d, "ground_truth"),
                                              folder_i)
                    if file_inner.endswith(".json"):
                        path_img["Ground_Truth"].append(
                            {'file_name': file_inner, 'file_path': os.path.join(root_inner, file_inner)})
                        print(os.path.join(root_inner, file_inner))
            path_img_result.append(path_img)
    result = []
    for pathFolder in path_img_result:
        sample_json = read_json(pathFolder['Sample_Json'], True)
        sample_img_bar = get_bar_sample(pathFolder['Sample_Image'])
        for pair in zip(sorted(pathFolder['Ground_Truth'], key=lambda item: item['file_name']),
                        sorted(pathFolder['Images'], key=lambda item: item['file_name'])):
            img_describe = {}
            photo_json = read_json(pair[0]['file_path'], False)
            # img_describe['tranform_m'] = calc_tranform(sample_json, sample_img_bar, photo_json)
            img_describe['real_doc_json'] = photo_json['bbox']
            img_describe['photo_on_doc'] = sample_json['bbox']
            img_describe['Sample_Image'] = pathFolder['Sample_Image']
            img_describe['sample_bar_doc'] = sample_img_bar['bbox']
            img_describe['file_name'] = pair[1]['file_name']
            img_describe['file_path'] = pair[1]['file_path']
            result.append(img_describe)
    return result

# {'real_doc_json': [[65, 744], [973, 755], [954, 1382], [72, 1360]],
#  'photo_on_doc': [[47, 112], [298, 112], [298, 425], [47, 425]],
#  'sample_bar_doc': [[0, 0], [1040, 0], [1040, 732], [0, 732]],
#  'file_name': 'KA05_13.tif',
#  'file_path': '/home/andrey/repo/study/diploma/make_data/data/05_aze_passport/images/KA/KA05_13.tif',
#  'new_folder': '/home/andrey/repo/study/diploma/make_data/data/05_aze_passport/images_cropped/KA/'}

# {'real_doc_json': [[47, 758], [963, 740], [961, 1374], [76, 1376]],
#  'photo_on_doc': [[47, 112], [298, 112], [298, 425], [47, 425]],
#  'sample_bar_doc': [[0, 0], [1040, 0], [1040, 732], [0, 732]],
#  'file_name': 'KA05_14.tif',
#  'file_path': '/home/andrey/repo/study/diploma/make_data/data/05_aze_passport/images/KA/KA05_14.tif',
#  'new_folder': '/home/andrey/repo/study/diploma/make_data/data/05_aze_passport/images_cropped/KA/'}

def rotation(img):
    (h, w) = img.shape[:2]
    center = (int(w / 2), int(h / 2))
    rotation_matrix = cv2.getRotationMatrix2D(center, 180, 1)
    rotated = cv2.warpAffine(img, rotation_matrix, (w,h ))
    return rotated

def handled_image(img_describe):
    img = cv2.imread(img_describe['file_path'])
    doc = np.array(img_describe['real_doc_json'])
    rect = cv2.minAreaRect(doc)

    # the order of the box points: bottom left, top left, top right,
    # bottom right
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    cv2.drawContours(img, [box], 0, (0, 0, 255), 2)

    # get width and height of the detected rectangle
    width = int(rect[1][0])
    height = int(rect[1][1])

    sample = np.array(img_describe['sample_bar_doc'])

    w_s = sample[2][0] - sample[0][0]
    h_s = sample[2][1] - sample[0][1]

    k1, k2 = width / w_s, height / h_s
    src_pts = box.astype("float32")
    # coordinate of the points in box points after the rectangle has been
    # straightened
    dst_pts = np.array([[0, height - 1],
                        [0, 0],
                        [width - 1, 0],
                        [width - 1, height - 1]], dtype="float32")

    # the perspective transformation matrix
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    warped = cv2.warpPerspective(img, M, (width, height))
    if width<height:
        k1, k2 = height / w_s, width / h_s
        warped=np.rot90(warped)
    crop = img_describe['photo_on_doc']
    crop_img = warped[int(crop[0][1] * k2):int(crop[2][1] * k2), int(crop[0][0] * k1):int(crop[2][0] * k1)].copy()
    if not os.path.exists(img_describe['new_folder']):
        os.makedirs(img_describe['new_folder'])
    is_written = cv2.imwrite(os.path.join(img_describe['new_folder'], os.path.basename(img_describe['file_path'])),
                             crop_img)
    if is_written:
        print("save image to " + img_describe['new_folder'])


if __name__ == '__main__':
    for img_describe in get_image_describe():
        if not os.path.exists(img_describe['Sample_Image'].replace("images", "images_cropped")):
            # os.makedirs(img_describe['new_folder'])
            img = cv2.imread(img_describe['Sample_Image'])
            crop = img_describe['photo_on_doc']
            crop_img = img[int(crop[0][1]):int(crop[2][1]), int(crop[0][0]):int(crop[2][0])].copy()
            is_written = cv2.imwrite(img_describe['Sample_Image'].replace("images", "images_cropped"),
                                 crop_img)
        img_describe['new_folder'] = img_describe['file_path'] \
            .replace("images", "images_cropped") \
            .replace(os.path.basename(img_describe['file_path']), "")
        handled_image(img_describe)
