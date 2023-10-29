# -*- coding: UTF-8 -*-
import numpy as np
import torch
from torch.autograd import Variable
from tqdm import *

from visdom import Visdom  # pip install Visdom
import settings
import datasets
from torch_util import validate_image_by_try_load_image
from models import CNN, RES18
import argparse
import one_hot_encoding
from models import *


def remove_module_from_keys(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace('module.', '')
        new_state_dict[new_key] = value
    return new_state_dict


def main():
    cnn = RES18()  # 创建一个CNN模型实例
    cnn.eval()  # 将模型设置为评估状态
    # cnn.load_state_dict(torch.load(args.model_path))  # 加载预训练模型参数，注释掉这一行

    parser = argparse.ArgumentParser(description="model path")  # 创建一个命令行参数解析器
    parser.add_argument("--model-path", type=str, default="weights/cnn_best.pt")  # 添加一个命令行参数，用于指定模型文件路径
    args = parser.parse_args()
    state_dict = torch.load(args.model_path)  # 加载预训练模型参数
    state_dict = remove_module_from_keys(state_dict)  # 去除state_dict中的module
    cnn.load_state_dict(state_dict)  # 加载预训练模型参数
    # cnn.load_state_dict(torch.load(args.model_path))  # 加载预训练模型参数

    predict_dataloader = datasets.get_predict_data_loader()  # 获取预测数据集的数据加载器
    correct = 0  # 正确预测的数量
    total = 0  # 总样本数量
    vis = Visdom()  # 创建一个Visdom实例，注释掉这一行
    pBar = tqdm(total=predict_dataloader.__len__())  # 进度条
    for i, (images, labels) in enumerate(predict_dataloader):  # 遍历预测数据集的数据加载器
        pBar.update(1)

        image = images  # 获取当前图像数据
        # if not validate_image_by_try_load_image(image):
        #     continue  # 如果图像数据不合法，则跳过当前循环
        vimage = Variable(image)  # 将图像数据转换为PyTorch变量
        predict_label = cnn(vimage)  # 使用CNN模型对图像进行预测

        c0 = settings.ALL_CHAR_SET[np.argmax(predict_label[0, 0:settings.ALL_CHAR_SET_LEN].data.numpy())]
        c1 = settings.ALL_CHAR_SET[
            np.argmax(predict_label[0, settings.ALL_CHAR_SET_LEN:2 * settings.ALL_CHAR_SET_LEN].data.numpy())]
        c2 = settings.ALL_CHAR_SET[
            np.argmax(predict_label[0, 2 * settings.ALL_CHAR_SET_LEN:3 * settings.ALL_CHAR_SET_LEN].data.numpy())]
        c3 = settings.ALL_CHAR_SET[
            np.argmax(predict_label[0, 3 * settings.ALL_CHAR_SET_LEN:4 * settings.ALL_CHAR_SET_LEN].data.numpy())]
        predict_label = '%s%s%s%s' % (c0, c1, c2, c3)  # 将预测结果的前4个字符拼接成一个字符串
        print(predict_label)  # 打印预测结果
        true_label = one_hot_encoding.decode(labels.numpy()[0])  # 获取真实标签
        total += labels.size(0)  # 更新总样本数量
        if (predict_label == true_label):  # 如果预测结果与真实标签相同，则正确预测数量加1
            correct += 1
        print('Test Accuracy of the model on the %d test images: %d' % (total, correct))
        vis.images(image, opts=dict(caption=predict_label))  # 将预测结果可视化，注释掉这一行


if __name__ == '__main__':
    main()  # 程序入口
