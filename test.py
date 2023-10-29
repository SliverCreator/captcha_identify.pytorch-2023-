# -*- coding: UTF-8 -*-
import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.checkpoint import checkpoint

import settings
import datasets
from models import *
import one_hot_encoding
import argparse
import torch_util
import os
from models import *
from tqdm import *
from torch_util import validate_image_by_try_load_image, plot_result, select_device

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"


# GPU / cpu
IS_USE_GPU = 1
# 将num_workers设置为等于计算机上的CPU数量
worker_num = 8

if IS_USE_GPU:
    import torch_util

    # 通过os.environ["CUDA_VISIBLE_DEVICES"]指定所要使用的显卡，如：
    # os.environ["CUDA_VISIBLE_DEVICES"] = "3,2,0,1"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch_util.select_device()
else:
    device = torch.device("cpu")


def main(model_path):
    cnn = CNN()  # 创建CNN模型实例
    cnn.eval()
    state_dict = torch.load(model_path, map_location=device)  # 加载预训练模型参数
    state_dict = remove_module_from_keys(state_dict)  # 去除state_dict中的module
    cnn.load_state_dict(state_dict)  # 加载预训练模型参数
    # cnn.load_state_dict(torch.load(model_path, map_location=device))  # 加载预训练模型参数
    # ck = torch.load(model_path, map_location=device)
    # cnn.load_state_dict({k.replace('module.', ''): v for k, v in ck['state_dict'].items()})
    print("load cnn net.")

    test_dataloader = datasets.get_test_data_loader()  # 获取测试数据集的迭代器

    correct = 0  # 正确预测的数量
    total = 0  # 总样本数量

    pBar = tqdm(total=test_dataloader.__len__())  # 进度条

    for i, (images, labels) in enumerate(test_dataloader):  # 遍历测试数据集
        pBar.update(1)

        image = images  # 获取当前图像
        vimage = Variable(image)  # 将图像转换为Variable类型，以便输入到神经网络中
        predict_label = cnn(vimage)  # 使用CNN模型进行预测

        c0 = settings.ALL_CHAR_SET[
            np.argmax(predict_label[0, 0:settings.ALL_CHAR_SET_LEN].data.numpy())]  # 获取预测结果的前4个字符
        c1 = settings.ALL_CHAR_SET[
            np.argmax(predict_label[0, settings.ALL_CHAR_SET_LEN:2 * settings.ALL_CHAR_SET_LEN].data.numpy())]
        c2 = settings.ALL_CHAR_SET[
            np.argmax(predict_label[0, 2 * settings.ALL_CHAR_SET_LEN:3 * settings.ALL_CHAR_SET_LEN].data.numpy())]
        c3 = settings.ALL_CHAR_SET[
            np.argmax(predict_label[0, 3 * settings.ALL_CHAR_SET_LEN:4 * settings.ALL_CHAR_SET_LEN].data.numpy())]
        predict_label = '%s%s%s%s' % (c0, c1, c2, c3)  # 将预测结果拼接为字符串
        true_label = one_hot_encoding.decode(labels.numpy()[0])  # 获取真实标签
        total += labels.size(0)  # 更新总样本数量
        if (predict_label == true_label):  # 如果预测结果与真实标签相同，则正确预测数量加1
            correct += 1
        # if(total%200==0):
        # print('Test Accuracy of the model on the %d test images: %f %%' % (total, 100 * correct / total))
        print('Test Accuracy of the model on the %d test images: %d' % (total, correct))


def remove_module_from_keys(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace('module.', '')
        new_state_dict[new_key] = value
    return new_state_dict
# def remove_module_from_keys(model_state_file):
#     from collections import OrderedDict
#     new_checkpoint = OrderedDict()
#     checkpoint = torch.load(model_state_file)['state_dict']
#     import pdb
#     pdb.set_trace()
#     for k, v in checkpoint.items():
#         name = k[7:]  # remove module.
#         new_checkpoint[name] = v
#     return new_checkpoint


def test_data(model_path):
    # plot_result()
    cnn = RES18()  # 创建CNN模型实例
    cnn.eval()
    state_dict = torch.load(model_path, map_location=device)  # 加载预训练模型参数
    state_dict = remove_module_from_keys(state_dict)  # 去除state_dict中的module
    cnn.load_state_dict(state_dict)  # 加载预训练模型参数
    # cnn.load_state_dict({k.replace('module.', ''): v for k, v in state_dict['state_dict'].items()})
    # cnn.load_state_dict(torch.load(model_path, map_location=device))  # 加载预训练模型参数
    test_dataloader = datasets.get_test_data_loader()  # 获取测试数据集的迭代器

    correct = 0  # 正确预测的数量
    total = 0  # 总样本数量

    for i, (images, labels) in enumerate(test_dataloader):  # 遍历测试数据集

        image = images  # 获取当前图像
        vimage = Variable(image)  # 将图像转换为Variable类型，以便输入到神经网络中
        predict_label = cnn(vimage)  # 使用CNN模型进行预测

        c0 = settings.ALL_CHAR_SET[
            np.argmax(predict_label[0, 0:settings.ALL_CHAR_SET_LEN].data.numpy())]  # 获取预测结果的前4个字符
        c1 = settings.ALL_CHAR_SET[
            np.argmax(predict_label[0, settings.ALL_CHAR_SET_LEN:2 * settings.ALL_CHAR_SET_LEN].data.numpy())]
        c2 = settings.ALL_CHAR_SET[
            np.argmax(predict_label[0, 2 * settings.ALL_CHAR_SET_LEN:3 * settings.ALL_CHAR_SET_LEN].data.numpy())]
        c3 = settings.ALL_CHAR_SET[
            np.argmax(predict_label[0, 3 * settings.ALL_CHAR_SET_LEN:4 * settings.ALL_CHAR_SET_LEN].data.numpy())]
        predict_label = '%s%s%s%s' % (c0, c1, c2, c3)  # 将预测结果拼接为字符串
        true_label = one_hot_encoding.decode(labels.numpy()[0])  # 获取真实标签
        total += labels.size(0)  # 更新总样本数量
        if (predict_label == true_label):  # 如果预测结果与真实标签相同，则正确预测数量加1
            correct += 1
        # if(total%200==0):
        # print('Test Accuracy of the model on the %d test images: %f %%' % (total, 100 * correct / total))
    # if not total or not correct:
    #     return 0
    return 100 * correct / total  # 返回测试准确率


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="test path")  # 创建命令行参数解析器
    parser.add_argument('--model-path', type=str, default="weights/cnn_best.pt")  # 添加模型路径参数
    args = parser.parse_args()  # 解析命令行参数
    main(args.model_path)  # 调用主函数进行测试
