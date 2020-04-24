import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from models.cycle_gan_model import CycleGANModel
from models.cycle_gan_with_distillation import CycleGANModelWithDistillation
from util.visualizer import Visualizer
from thop import profile
from tqdm import tqdm
from argparse import Namespace
import numpy as np
from PIL import Image
from util.visualizer import save_images
from util import html
import os

def map_image(img):
    new_range = 255
    old_range = 2
    scaled = np.array((img +1) / float(old_range), dtype=float)
    return scaled * new_range


if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)
    opt2 = Namespace(**vars(opt))
    opt2.name = 'monet2photo_pretrained'
    opt2.isTrain = False
    teacher = CycleGANModel(opt2)
    teacher.isTeacher = True
    teacher.setup(opt2)               # regular setup: load and print networks; create schedulers
    opt.netG = 'resnet_3blocks'
    opt.results_dir = 'results'
    model = CycleGANModelWithDistillation(opt, teacher)
    model.setup(opt)
    print(opt.dataset)
    web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations
    for epoch in tqdm(range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1)):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch

        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            teacher.set_input(data)
            teacher.test()
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                #save_result = total_iters % opt.update_html_freq == 0
                #model.compute_visuals()
                #visuals = model.get_current_visuals()  # get image results
                #img_path = model.get_image_paths()  # get image paths
                #print(img_path)
                #exit()
                #save_images(webpage, visuals, img_path)
                pass

            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()
        if epoch % 1 == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
        model.update_learning_rate()                     # update learning rates at the end of every epoch.

