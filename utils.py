import cv2
import numpy as np


def make_student_config(config):
    student_config = config.copy()
    student_config['resolution'] = (16, 4)
    student_config['resnetfpn']['initial_dim'] = 8
    student_config['resnetfpn']['block_dims'] = [8, 16, 32, 32]  # s1, s2, s3

    student_config['coarse']['d_model'] = 32
    student_config['coarse']['d_ffn'] = 32
    student_config['coarse']['nhead'] = 1
    student_config['coarse']['layer_names'] = ['self', 'cross'] * 2
    return student_config


def get_coarse_match(conf_matrix, input_height, input_width, resolution):
    """
        Predicts coarse matches from conf_matrix
    Args:
        resolution: image
        input_width:
        input_height:
        conf_matrix: [N, L, S]

    Returns:
        mkpts0_c: [M, 2]
        mkpts1_c: [M, 2]
        mconf: [M]
    """

    hw0_i = (input_height, input_width)
    hw0_c = (input_height // resolution, input_width // resolution)
    hw1_c = hw0_c  # input images have the same resolution
    feature_num = hw0_c[0] * hw0_c[1]

    # 3. find all valid coarse matches
    # this only works when at most one `True` in each row
    b_ids, i_ids, j_ids  = np.nonzero(conf_matrix > 0.01)
    # all_j_ids = mask.argmax(axis=2)
    # j_ids = all_j_ids.squeeze(0)
    # b_ids = np.zeros_like(j_ids, dtype=np.long)
    # i_ids = np.arange(feature_num, dtype=np.long)

    mconf = conf_matrix[b_ids, i_ids, j_ids]

    # 4. Update with matches in original image resolution
    scale = hw0_i[0] / hw0_c[0]
    mkpts0_c = np.stack(
        [i_ids % hw0_c[1], np.trunc(i_ids / hw0_c[1])],
        axis=1) * scale
    mkpts1_c = np.stack(
        [j_ids % hw1_c[1], np.trunc(j_ids / hw1_c[1])],
        axis=1) * scale

    return mkpts0_c, mkpts1_c, mconf


def make_query_image(frame, ratio):
    query_img, (dw, dh) = letterbox(frame, ratio)
    query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2GRAY)
    # query_img = ratio_preserving_resize(query_img, img_size)
    return query_img, (dw, dh)

def letterbox(img, r, new_shape=(480, 640), color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    # r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    # dw /= 2  # divide padding into 2 sides
    # dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = 0, int(round(dh + 0.1))
    left, right = 0, int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, (dw, dh)


def ratio_preserving_resize(image, img_size):
    # ratio preserving resize
    img_h, img_w = image.shape
    scale_h = img_size[1] / img_h
    scale_w = img_size[0] / img_w
    scale_max = max(scale_h, scale_w)
    new_size = (int(img_w * scale_max), int(img_h * scale_max))
    image = cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR)
    # center crop
    x = new_size[0] // 2 - img_size[0] // 2
    y = new_size[1] // 2 - img_size[1] // 2
    image = image[y:y + img_size[1], x:x + img_size[0]]
    return image
