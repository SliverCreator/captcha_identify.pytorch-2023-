# -*- coding: UTF-8 -*-
from captcha.image import ImageCaptcha  # 导入验证码生成库
from PIL import Image  # 导入图像处理库
import random  # 导入随机数库
import time  # 导入时间库
import settings  # 导入设置模块，包含MAX_CAPTCHA、ALL_CHAR_SET和TRAIN_DATASET_PATH等变量
import os  # 导入操作系统库，用于操作文件和目录
from multiprocessing import cpu_count  # 导入多进程库，用于并行生成验证码
from multiprocessing_on_dill.dummy import Process, Value, Lock  # 导入多进程库，用于并行生成验证码
import dill


# 定义一个函数，用于生成指定长度的随机字符串
def random_captcha():
    captcha_text = []  # 初始化一个空列表，用于存储生成的随机字符
    for i in range(settings.MAX_CAPTCHA):  # 循环MAX_CAPTCHA次，生成指定长度的随机字符串
        c = random.choice(settings.ALL_CHAR_SET)  # 从ALL_CHAR_SET中随机选择一个字符
        captcha_text.append(c)  # 将选中的字符添加到列表中
    return ''.join(captcha_text)  # 将列表中的字符连接成一个字符串并返回


# 定义一个函数，用于生成字符对应的验证码图片
def gen_captcha_text_and_image():
    image = ImageCaptcha()  # 创建一个ImageCaptcha对象，用于生成验证码图片
    captcha_text = random_captcha()  # 调用random_captcha函数生成随机字符串
    captcha_image = Image.open(image.generate(captcha_text))  # 使用ImageCaptcha对象的generate方法生成验证码图片，并将其转换为PIL.Image对象
    return captcha_text, captcha_image  # 返回生成的随机字符串和验证码图片


def generate_captchas(path, start, end, process_id, counter, lock):
    if not os.path.exists(path):  # 如果路径不存在，则创建该路径
        os.makedirs(path)  # 通过改变此处目录，以生成 训练、测试和预测用的验证码集

    for i in range(start, end):  # 循环生成指定数量的验证码
        now = str(int(time.time()))  # 获取当前时间戳，并将其转换为字符串
        text, image = gen_captcha_text_and_image()  # 调用gen_captcha_text_and_image函数生成随机字符串和验证码图片
        filename = text + '_' + now + '.jpg'  # 拼接文件名，格式为：随机字符串_时间戳.jpg
        image.save(path + os.path.sep + filename)  # 将验证码图片保存到指定路径下
        with lock:  # 使用锁机制，防止多个进程同时修改counter.value的值
            counter.value += 1  # 生成一个验证码，计数器counter的值加1
            if counter.value % 1000 == 0:  # 每生成1000个验证码，打印一次生成进度
                print(f'Total progress: {counter.value} captchas generated.')  # 打印生成进度


def gen_by_count_and_path(count, path):  # 定义一个函数，用于生成指定数量的验证码
    print(f"Will generate {count} png -> {path}")  # 打印生成信息
    num_processes = cpu_count()  # 获取计算机的CPU数量
    captchas_per_process = count // num_processes  # 计算每个进程需要生成的验证码数量

    processes = []  # 定义一个空列表，用于存储进程
    counter = Value('i', 0)  # 定义一个计数器，用于记录已生成的验证码数量
    lock = Lock()  # 定义一个锁，用于防止多个进程同时修改counter.value的值

    for i in range(num_processes):  # 循环生成进程
        start = i * captchas_per_process  # 计算每个进程需要生成的验证码的起始位置
        end = (i + 1) * captchas_per_process if i != num_processes - 1 else count  # 计算每个进程需要生成的验证码的结束位置
        process = Process(target=dill.loads(dill.dumps(generate_captchas)),
                          args=(path, start, end, i, counter, lock))  # 创建一个进程，用于生成验证码
        processes.append(process)  # 将进程添加到列表中
        process.start()  # 启动进程

    for process in processes:   # 循环所有进程
        process.join()  # 等待所有进程结束


def main():
    dill.loads(dill.dumps(gen_by_count_and_path(10000, settings.TRAIN_DATASET_PATH)))  # 生成训练集验证码
    dill.loads(dill.dumps(gen_by_count_and_path(1500, settings.PREDICT_DATASET_PATH)))  # 生成预测集验证码
    dill.loads(dill.dumps(gen_by_count_and_path(1500, settings.TEST_DATASET_PATH)))  # 生成测试集验证码


if __name__ == '__main__':
    main()
