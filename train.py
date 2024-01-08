import time
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer

def get_config(config):
    import yaml
    with open(config, 'r') as stream:
        return yaml.load(stream)
#获取训练配置
opt = TrainOptions().parse()
config = get_config(opt.config)

#创建数据类
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print('#training images = %d' % dataset_size)

#创建模型
model = create_model(opt)
visualizer = Visualizer(opt)

total_steps = 0

#模型训练
for epoch in range(1, opt.niter + opt.niter_decay + 1): #遍历训练的每个 epoch，从1到 opt.niter + opt.niter_decay + 1
    epoch_start_time = time.time() #记录当前 epoch 的开始时间，用于计算该 epoch 的训练耗时
    for i, data in enumerate(dataset): #遍历训练数据集中的每个样本
        iter_start_time = time.time() #计算每个 iteration 的训练耗时
        total_steps += opt.batchSize #更新总步数
        epoch_iter = total_steps - dataset_size * (epoch - 1)
        model.set_input(data) #加载数据
        model.optimize_parameters(epoch)#执行一次模型参数的优化（训练）

        if total_steps % opt.display_freq == 0:#可视化
            visualizer.display_current_results(model.get_current_visuals(), epoch)

        if total_steps % opt.print_freq == 0:
            errors = model.get_current_errors(epoch)
            t = (time.time() - iter_start_time) / opt.batchSize
            visualizer.print_current_errors(epoch, epoch_iter, errors, t)
            if opt.display_id > 0:
                visualizer.plot_current_errors(epoch, float(epoch_iter)/dataset_size, opt, errors)

        if total_steps % opt.save_latest_freq == 0:
            print('saving the latest model (epoch %d, total_steps %d)' %
                  (epoch, total_steps))
            model.save('latest')

    if epoch % opt.save_epoch_freq == 0:#
        print('saving the model at the end of epoch %d, iters %d' %
              (epoch, total_steps))
        model.save('latest')
        model.save(epoch)

    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

    if opt.new_lr: #迭代学习率
        if epoch == opt.niter:
            model.update_learning_rate()
        elif epoch == (opt.niter + 20):
            model.update_learning_rate()
        elif epoch == (opt.niter + 70):
            model.update_learning_rate()
        elif epoch == (opt.niter + 90):
            model.update_learning_rate()
            model.update_learning_rate()
            model.update_learning_rate()
            model.update_learning_rate()
    else:
        if epoch > opt.niter:
            model.update_learning_rate()
