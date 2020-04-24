import os
from options.test_options import TestOptions
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
from argparse import Namespace
from models.cycle_gan_with_distillation import  CycleGANModelWithDistillation
from models.cycle_gan_model import CycleGANModel
if __name__ == '__main__':
    opt = TrainOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    opt2 = Namespace(**vars(opt))
    opt2.name = 'monet2photo_pretrained'
    opt2.isTrain = False
    teacher = CycleGANModel(opt2)
    teacher.isTeacher = True
    opt.continue_train = True
    opt2.continue_train = True
    teacher.setup(opt2)  # regular setup: load and print networks; create schedulers
    opt.netG = 'resnet_3blocks'
    opt.results_dir = 'results'
    model = CycleGANModelWithDistillation(opt, teacher)      # create a model given opt.model and other options
    model.setup(opt)              # regular setup: load and print networks; create schedulers
    # create a website
    web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
    if opt.load_iter > 0:  # load_iter is 0 by default
        web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
    print('creating web directory', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    #if opt.eval:
       # model.eval()
    for i, data in enumerate(dataset):
        if i > 10:
            exit()
        #if i >= opt.num_test:  # only apply our model to opt.num_test images.
           # break
        #model.set_input(data)  # unpack data from data loader
        #model.test()         # run inference

        teacher.set_input(data)
        teacher.test()
        model.set_input(data)  # unpack data from dataset and apply preprocessing
        model.optimize_parameters()


        model.compute_visuals()
        visuals = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()  # get image paths
        print(img_path)
        #save_images(webpage, visuals, img_path)


        #visuals = model.get_current_visuals()  # get image results
        #img_path = model.get_image_paths()     # get image paths
        #if i % 5 == 0:  # save images to an HTML file
          #  print('processing (%04d)-th image... %s' % (i, img_path))
       # save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)
    webpage.save()  # save the HTML
