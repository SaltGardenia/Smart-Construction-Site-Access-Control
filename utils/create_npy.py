import os
import cv2
import numpy as np
from xml.dom.minidom import parse
from random import shuffle

def cut_img(path, out_path):
    """
    截图安全帽
    :param path: 原始图片输入路径
    :param out_path: 截图后图片路径
    :return:
    """
    nonehat_index = 1
    hat_index = 1
    filenames = os.listdir(path)  # 读取文件夹里面的内  xml 和 jpg
    # print(filenames)

    for file in filenames:
        if file.split('.')[-1] == 'xml':  # 获取里面xml
            # 读取xml
            DomTree = parse(path + '/' + file)
            # 获取根节点
            annotation = DomTree.documentElement
            # 获取annotation下面的filename
            filename = annotation.getElementsByTagName('filename')[0].childNodes[
                0].data  # filename和childNodes可能有多个，所以要用s和[0]
            # print(filename)
            # 图片转灰度
            img = cv2.imread(path + '/' + filename, cv2.IMREAD_GRAYSCALE)

            # 获取图片被框的位置
            objects = annotation.getElementsByTagName('object')  # 所有是<object>都获取出来，放入数组中objects
            # 获取标记位置和颜色
            for object in objects:
                # print(object)
                # 颜色
                label = object.getElementsByTagName('name')[0].childNodes[0].data
                # print(label)
                bndbox = object.getElementsByTagName('bndbox')[0]
                xmin = int(bndbox.getElementsByTagName('xmin')[0].childNodes[0].data)
                ymin = int(bndbox.getElementsByTagName('ymin')[0].childNodes[0].data)
                xmax = int(bndbox.getElementsByTagName('xmax')[0].childNodes[0].data)
                ymax = int(bndbox.getElementsByTagName('ymax')[0].childNodes[0].data)

                # print(xmin, ymin, xmax, ymax)

                IMG = img[ymin:ymax, xmin:xmax]  # 行/列

                if label == 'none':
                    cv2.imwrite(out_path + '/' + 'none_' + str(nonehat_index) + '.jpg', IMG)
                    nonehat_index += 1;
                else:
                    cv2.imwrite(out_path + '/' + 'hat_' + str(hat_index) + '.jpg', IMG)
                    hat_index += 1;

cut_img('NA', 'hat_none')

def del_img(path):
    """
    删除比较小的图片 40*40 以下
    :param path: 图片路径
    :return:
    """
    filenames = os.listdir(path)
    for file in filenames:
        img = cv2.imread(path + '/' + file)
        if img.shape[0] < 40 or img.shape[1] < 40:
            os.remove(path + '/' + file)

del_img('hat_none')

def create_npy(path, out_path, img_size = 50):
    """
    创建npy
    :param paht: 原始图片路径
    :param out_path: npy输出路径
    :param img_size: 图片大小，用来定义图片训练时候的输入尺寸
    :return:
    """
    npy_data = []
    filenames = os.listdir(path)
    for file in filenames:
        img = cv2.imread(path + '/' + file, cv2.IMREAD_GRAYSCALE)
        new_img = cv2.resize(img, (img_size, img_size))
        label = file.split('_')[0]
        # 独热编码
        if label == 'hat':
            label_name = [1, 0]
        else:
            label_name = [0, 1]

        # 把这张图片和对应的标识存入集合
        npy_data.append([np.array(new_img), np.array(label_name)])

    shuffle(npy_data)
    np.save(out_path, npy_data)

create_npy('hat_none', 'hat_none.npy')

