"""
这里假设channel last
"""
import copy
import numpy as np
import cv2
from microsnoop.datasets.datasets_utils import find_inst_bbox


def crop_inst(X, M, mode='bbox', meta=None,
              sta = 224, oversize_to_sta=False, crop_step=1,
              crop_one=False,
              crop_to_sta=False,
              rsc_crop=False, rsc_crop_ratio=2.5):
    """
    # 这里切好之后理论上就不动了，后面扩增只使用random_rotate_and_resize就行了
    # 每张切出来的全部混在一个列表里，不再给单张图片单独给一个列表
    # asset channel last
    # crop inst using bbox or mask mode

    crop_one：目前只在训练时使用，每次只随机分割一个细胞
    sta, oversize_to_sta, crop_step：分割过程参数
    rsc_crop, rsc_crop_ratio: 根据bbox找到中心点坐标,分割的长度根据bbox长度和rsc_crop_ratio求得
        -> crop_to_sta（用于控制将提供的分割的长度设置为sta;若打开,则crop_centroids_size无效）

    可以裁切的模式：
        1. X多通道，M单通道
        2. X单通道，M单通道
        3. X多通道，M多通道，默认使用第一维
        4. 增加了根据mask放缩裁切的模式：此模式自动转为bbox裁切模式，用于根据提供mask进行缩放范围，可以包含一定程度的细胞微环境
    """
    nimg = len(X)
    imgs = []
    masks = []
    metas = {'bbox': []}

    for i in range(nimg):  # 对于每一张图片
        maski = M[i]
        imgi = X[i]
        assert maski.ndim==2, "Please only choose the channel with biggest mask of one instance" \
                              "for example, when you have cyto or nuclei channels, you should choose the" \
                              "cyto channel to get the whole cell instance"

        inst_list = np.unique(maski)
        inst_list = np.setdiff1d(inst_list, np.array([0])) # 去除0值

        if crop_one:
            continue_choose_flag = True
            if len(inst_list) < 1: # 排除没有单细胞的情况，防止死循环
                continue_choose_flag = False
            while continue_choose_flag:
                j = np.random.choice(inst_list, 1)[0]
                maski_ = np.zeros_like(maski)
                maski_[maski == j] = 1
                bbox = find_inst_bbox(maski, j)  # 拿到bbox
                if bbox[0] == 0 or bbox[1] == 0 or bbox[2] == maski_.shape[0] or bbox[3] == maski_.shape[1]:
                    continue_choose_flag = True
                else:
                    continue_choose_flag = False
                    if rsc_crop:
                        mode = 'bbox'
                        bbox = list(bbox)
                        crop_size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
                        crop_center_r = bbox[0] + (bbox[2] - bbox[0]) // 2
                        crop_center_c = bbox[1] + (bbox[3] - bbox[1]) // 2
                        if crop_to_sta:
                            crop_size = sta // 2
                        else:
                            crop_size = crop_size * rsc_crop_ratio
                        crop_size_thred = sta // 2
                        if not oversize_to_sta:  # 如果不把大的图片放缩的话，就切一部分
                            if crop_size > crop_size_thred: crop_size = crop_size_thred
                        bbox[0] = max(int(crop_center_r - crop_size), 0)
                        bbox[1] = max(int(crop_center_c - crop_size), 0)
                        bbox[2] = min(int(crop_center_r + crop_size), maski_.shape[0])
                        bbox[3] = min(int(crop_center_c + crop_size), maski_.shape[1])
                    imgs, masks, metas = _crop_instij(i, imgs, masks, metas, mode, imgi, maski_, meta, bbox, sta, oversize_to_sta)
        else:
            if len(inst_list)>0:  # 有一个细胞都算进去
                use_crop_step = crop_step
                # if len(props) < 50: use_crop_step = 1  # 如果数量少于50个，就不挑选
                for j in inst_list[::use_crop_step]:  # 对于每一个实例
                    maski_ = np.zeros_like(maski)
                    maski_[maski == j] = 1
                    bbox = find_inst_bbox(maski, j)  # 拿到bbox
                    if bbox[0] == 0 or bbox[1] == 0 or bbox[2] == maski_.shape[0] or bbox[3] == maski_.shape[1]:
                        # 去掉边缘的
                        continue
                    else:
                        if rsc_crop:
                            mode = 'bbox'
                            bbox = list(bbox)
                            crop_size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
                            crop_center_r = bbox[0] + (bbox[2] - bbox[0]) // 2
                            crop_center_c = bbox[1] + (bbox[3] - bbox[1]) // 2
                            if crop_to_sta:
                                crop_size = sta // 2
                            else:
                                crop_size = crop_size * rsc_crop_ratio
                            crop_size_thred = sta // 2
                            if not oversize_to_sta:  # 如果不把大的图片放缩的话，就切一部分
                                if crop_size > crop_size_thred: crop_size = crop_size_thred
                            bbox[0] = max(int(crop_center_r - crop_size), 0)
                            bbox[1] = max(int(crop_center_c - crop_size), 0)
                            bbox[2] = min(int(crop_center_r + crop_size), maski_.shape[0])
                            bbox[3] = min(int(crop_center_c + crop_size), maski_.shape[1])
                        imgs, masks, metas = _crop_instij(i, imgs, masks, metas, mode, imgi, maski_, meta, bbox, sta, oversize_to_sta)

    return imgs, masks, metas

def _crop_instij(i, imgs, masks, metas, mode, imgi, maski_, meta, bbox, sta, oversize_to_sta):
    """
        从图像i中切出一个实例j并添加到imgs, masks, metas中
    """
    imgi_inst_0 = []
    if imgi.ndim == 2:
        imgi = np.expand_dims(imgi, axis=-1)
    for c in range(imgi.shape[-1]):
        if mode.lower() == 'mask':  # crop inst using mask
            imgi_inst_ci = imgi[..., c][bbox[0]:bbox[2], bbox[1]:bbox[3]] * maski_[bbox[0]:bbox[2],
                                                                            bbox[1]:bbox[3]]  # bbox
        elif mode.lower() == 'bbox':  # crop inst using bbox
            imgi_inst_ci = imgi[..., c][bbox[0]:bbox[2], bbox[1]:bbox[3]]
        else:
            print("Wrong crop mode provided, will use mask mode，You can choose mask or bbox mode when restarting")
            imgi_inst_ci = imgi[..., c][bbox[0]:bbox[2], bbox[1]:bbox[3]] * maski_[bbox[0]:bbox[2],
                                                                            bbox[1]:bbox[3]]  # bbox
        imgi_inst_0.append(imgi_inst_ci)  # 此时channel first
    if mode.lower() == 'mask':  # crop inst using mask
        maski_inst = maski_[bbox[0]:bbox[2], bbox[1]:bbox[3]]
    # 这里有个问题是只取这个实例的mask，还是把所有的一股脑都切出来
    _, shape_0, shape_1 = np.array(imgi_inst_0).shape  # (c, h, w)
    if oversize_to_sta:  # 如果比sta大，则把长边resize到sta, 否则后面就会被舍弃掉
        shape_max = max(shape_0, shape_1)
        if shape_max > sta:
            rescale = sta / shape_max
            imgi_inst_1 = []
            shape_0 = min(int(rescale * shape_0), sta)
            shape_1 = min(int(rescale * shape_1), sta)
            for imgi_inst_ci in imgi_inst_0:
                imgi_inst_ci_ = cv2.resize(imgi_inst_ci, (shape_1, shape_0), interpolation=cv2.INTER_LINEAR)
                imgi_inst_1.append(imgi_inst_ci_) # 这里resize时的shape是反的
            if mode.lower() == 'mask':  # crop inst using mask
                maski_inst = cv2.resize(maski_inst, (shape_1, shape_0), interpolation=cv2.INTER_NEAREST)
        else:
            imgi_inst_1 = imgi_inst_0
    else:
        imgi_inst_1 = imgi_inst_0

    add_0 = sta - shape_0
    add_0_l = add_0 // 2
    add_0_r = add_0 - add_0_l
    add_1 = sta - shape_1
    add_1_l = add_1 // 2
    add_1_r = add_1 - add_1_l
    if add_0 >= 0 and add_1 >= 0:  # 只能保存长宽比sta更小的
        if meta is not None:
            meta_keys = [key for key in meta.keys()]
            for key in meta_keys:
                if key not in metas.keys():
                    metas[key] = []
                metas[key].append(meta[key][i])
        metas['bbox'].append(bbox)

        if add_0 == 0 and add_1 == 0:
            imgi_inst = imgi_inst_1
        else:
            imgi_inst = []
            for imgi_inst_ci in imgi_inst_1:
                imgi_inst_ci_ = np.pad(imgi_inst_ci, ((add_0_l, add_0_r), (add_1_l, add_1_r)), 'constant',
                                       constant_values=(0, 0))
                imgi_inst.append(imgi_inst_ci_)
            if mode.lower() == 'mask':  # crop inst using mask
                maski_inst = np.pad(maski_inst, ((add_0_l, add_0_r), (add_1_l, add_1_r)), 'constant',
                                constant_values=(0, 0))

        imgs.append(np.float32(np.array(imgi_inst).transpose(1, 2, 0)))
        if mode.lower() == 'mask':  # crop inst using mask
            masks.append(np.uint16(maski_inst))
    return imgs, masks, metas
