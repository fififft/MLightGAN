import time
import os
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from pdb import set_trace as st
from util import html

opt = TestOptions().parse()
opt.nThreads = 1   # test code only supports nThreads = 1 只支持单线程运行
opt.batchSize = 1  # test code only supports batchSize = 1 批次大小为1
opt.serial_batches = True  # no shuffle 数据的处理顺序为串行
opt.no_flip = True  # no flip #不进行翻转

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
model = create_model(opt)   #创建用于测试的模型
visualizer = Visualizer(opt)
# create website 保存网络结果
web_dir = os.path.join("./ablation/", opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))
# test
print(len(dataset))
for i, data in enumerate(dataset): #遍历数据集
    model.set_input(data) #将当前数据 data 输入模型
    visuals = model.predict() #通过模型预测图像
    img_path = model.get_image_paths() #获取路径
    print('process image... %s' % img_path)
    visualizer.save_images(webpage, visuals, img_path)

webpage.save()
