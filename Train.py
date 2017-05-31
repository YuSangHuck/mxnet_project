import mxnet as mx
from mxnet.io import DataBatch, DataIter
from importlib import import_module
import numpy as np
import logging
logger = logging.getLogger('Train')
import os
import time
import argparse
import Dataset
import network
import re

####################					in = parser				#################
#													 				 			#
####################				return = 					#################
#																	 			#
####################				explain						#################
# 																	 			#
# set default data args for Train												#
#																				#
def add_data_args(parser):
    data = parser.add_argument_group('Data', 'the input images')
    data.add_argument('--data-train', type=str, help='the training data')
    data.add_argument('--data-val', type=str, help='the validation data')
    data.add_argument('--rgb-mean', type=str, default='123.68,116.779,103.939',
                      help='a tuple of size 3 for the mean rgb')
    data.add_argument('--pad-size', type=int, default=0,
                      help='padding the input image')
    data.add_argument('--image-shape', type=str,
                      help='the image shape feed into the network, e.g. (3,224,224)')
    data.add_argument('--num-classes', type=int, help='the number of classes')
    data.add_argument('--num-examples', type=int, help='the number of training examples')
    data.add_argument('--data-nthreads', type=int, default=4,
                      help='number of threads for data decoding')
    return 

####################					in = parser				#################
#													 				 			#
####################				return =					#################
#																				#
####################				explain						#################
#																				#
# set default agumentation args													#
#																				# 
def add_data_aug_args(parser):
    data_aug = parser.add_argument_group(
        'Image augmentations', 'implemented in src/io/image_aug_default.cc')
    data_aug.add_argument('--random-crop', type=int, default=1,
                     help='if or not randomly crop the image')
    data_aug.add_argument('--random-mirror', type=int, default=1,
                     help='if or not randomly flip horizontally')
    data_aug.add_argument('--max-random-h', type=int, default=0,
                     help='max change of hue, whose range is [0, 180]')
    data_aug.add_argument('--max-random-s', type=int, default=0,
                     help='max change of saturation, whose range is [0, 255]')
    data_aug.add_argument('--max-random-l', type=int, default=0,
                     help='max change of intensity, whose range is [0, 255]')
    data_aug.add_argument('--max-random-aspect-ratio', type=float, default=0,
                     help='max change of aspect ratio, whose range is [0, 1]')
    data_aug.add_argument('--max-random-rotate-angle', type=int, default=0,
                     help='max angle to rotate, whose range is [0, 360]')
    data_aug.add_argument('--max-random-shear-ratio', type=float, default=0,
                     help='max ratio to shear, whose range is [0, 1]')
    data_aug.add_argument('--max-random-scale', type=float, default=1,
                     help='max ratio to scale')
    data_aug.add_argument('--min-random-scale', type=float, default=1,
                     help='min ratio to scale, should >= img_size/input_shape. otherwise use --pad-size')
    return 

####################					in = aug				#################
#													 				 			#
# parser.aug ~~																	#
#																				#
####################				return =					#################
#																	 			#
####################				explain						#################
#																	 			#
# set augmentation level														#
#																				#
def set_data_aug_level(aug, level):
    if level >= 1:
        aug.set_defaults(random_crop=1, random_mirror=1)
    if level >= 2:
        aug.set_defaults(max_random_h=36, max_random_s=50, max_random_l=50)
    if level >= 3:
        aug.set_defaults(max_random_rotate_angle=10, max_random_shear_ratio=0.1, max_random_aspect_ratio=0.25)

####################					in = args,kv			#################
#													 				 			#
####################				return = (train,val)		#################
#																	 			#
####################				explain						#################
#																	 			#
# args -> mx.io.ImageRecordIter() -> return(train, val)							#
# function that returns the train and val data iterators						#
def get_rec_iter(args, kv=None):
    image_shape = tuple([int(l) for l in args.image_shape.split(',')])
    dtype = np.float32;
    if kv:
        (rank, nworker) = (kv.rank, kv.num_workers)
    else:
        (rank, nworker) = (0, 1)
    rgb_mean = [float(i) for i in args.rgb_mean.split(',')]
    train = mx.io.ImageRecordIter(
        path_imgrec         = args.data_train, # path : /root/user/data.rec
        mean_r              = rgb_mean[0], # Red_mean : 120
        mean_g              = rgb_mean[1], # Green_mean :140
        mean_b              = rgb_mean[2], # Blue_mean : 110
        data_shape          = image_shape, # (3,228,228)
        batch_size          = args.batch_size, # 128
        pad                 = args.pad_size, # padding size
        fill_value          = 127, # paddig pixels value
        preprocess_threads  = args.data_nthreads) # number of decoding threads

    if args.data_val is None:
        return (train, None)
    val = mx.io.ImageRecordIter(
        path_imgrec         = args.data_val,
        mean_r              = rgb_mean[0],
        mean_g              = rgb_mean[1],
        mean_b              = rgb_mean[2],
        batch_size          = args.batch_size,
        data_shape          = image_shape,
        preprocess_threads  = args.data_nthreads)
    return (train, val)

####################					in =args,kv				#################
#													 				 			#
####################	return = mx.lr_scheduler.MultiFactorScheduler	#################
#																	 			#
####################				explain						#################
#																	 			#
# args.lr : initial learning rate												#
# args.lr_factor : ratio to reduce lr on each step								#
# args.lr_step_epochs : epochs to reduce the lr									#
#																				#
def _get_lr_scheduler(args, kv):
    if 'lr_factor' not in args or args.lr_factor >= 1:
        return (args.lr, None)
    epoch_size = args.num_examples / args.batch_size
    if 'dist' in args.kv_store: # multiple machine
        epoch_size /= kv.num_workers
    begin_epoch = args.load_epoch if args.load_epoch else 0
    step_epochs = [int(l) for l in args.lr_step_epochs.split(',')]
    lr = args.lr
    for s in step_epochs:
        if begin_epoch >= s:
            lr *= args.lr_factor
    if lr != args.lr:
        logging.info('Adjust learning rate to %e for epoch %d' %(lr, begin_epoch))

    steps = [epoch_size * (x-begin_epoch) for x in step_epochs if x-begin_epoch > 0]
    return (lr, mx.lr_scheduler.MultiFactorScheduler(step=steps, factor=args.lr_factor))

####################					in = args				#################
#													 				 			#
####################				return = save '~.params'	#################
#																	 			#
####################				explain						#################
#																	 			#
# save model -> 'args.model_prefic/~-epoch.params'								#
#																				#
def _save_model(args, rank=0):
    if args.model_prefix is None:
        return None
    dst_dir = os.path.join(args.model_prefix,args.network)
    if not os.path.isdir(args.model_prefix):
        os.makedirs(args.model_prefix)
    return mx.callback.do_checkpoint(dst_dir if rank == 0 else "%s-%d" % (
        args.model_prefix, rank))

####################					in = parser				#################
#													 				 			#
####################				return =					#################
#																	 			#
####################				explain						#################
#																	 			#
# set default args for Train													#
#																				#
def add_fit_args(parser):
    train = parser.add_argument_group('Training', 'model training')
    train.add_argument('--network', type=str,
                       help='the neural network to use')
    train.add_argument('--num-layers', type=int,
                       help='number of layers in the neural network, required by some networks such as resnet')
    train.add_argument('--gpus', type=str,
                       help='list of gpus to run, e.g. 0 or 0,2,5. empty means using cpu')
    train.add_argument('--kv-store', type=str, default='device',
                       help='key-value store type')
    train.add_argument('--num-epochs', type=int, default=100,
                       help='max num of epochs')
    train.add_argument('--lr', type=float, default=0.1,
                       help='initial learning rate')
    train.add_argument('--lr-factor', type=float, default=0.1,
                       help='the ratio to reduce lr on each step')
    train.add_argument('--lr-step-epochs', type=str,
                       help='the epochs to reduce the lr, e.g. 30,60')
    train.add_argument('--optimizer', type=str, default='sgd',
                       help='the optimizer type')
    train.add_argument('--mom', type=float, default=0.9,
                       help='momentum for sgd')
    train.add_argument('--wd', type=float, default=0.0001,
                       help='weight decay for sgd')
    train.add_argument('--batch-size', type=int, default=128,
                       help='the batch size')
    train.add_argument('--disp-batches', type=int, default=20,
                       help='show progress for every n batches')
    train.add_argument('--model-prefix', type=str,
                       help='model prefix')
    train.add_argument('--load-epoch', type=int,
                       help='load the model on an epoch using the model-load-prefix')
    return 

####################					in = 					#################
#													 				 			#
####################				return =					#################
#																				#
####################				explain						#################
#																	 			#
# train by setted args -> call model.fit										#
# set dataset by data_loader													#
# save result by _save_model													#
#																				#
def fit(args, network, data_loader, **kwargs):
    # kvstore
    kv = mx.kvstore.create(args.kv_store)

    # data iterators
    (train, val) = data_loader(args, kv)

    # save model(train result), epoch_end_callback : callbacks that run after each epoch
    checkpoint = _save_model(args, kv.rank)
    
    # logging form
    format = '[%(levelname)s:%(asctime)s],%(message)s'
    logging.basicConfig(filename = os.path.join(args.model_prefix,'create_train_db.log'),
                        filemode = 'w',
                        format = format,
                        level=logging.DEBUG)
    
    # devices for training(CPU or GPU)
    devs = mx.cpu() if args.gpus is None or args.gpus is '' else [
        mx.gpu(int(i)) for i in args.gpus.split(',')]

    # set learning rate
    lr, lr_scheduler = _get_lr_scheduler(args, kv)

    # create model
    model = mx.mod.Module(
        context       = devs,
        symbol        = network
    )

    lr_scheduler  = lr_scheduler
    optimizer_params = {
            'learning_rate': lr,
            'momentum' : args.mom,
            'wd' : args.wd,
            'lr_scheduler': lr_scheduler}

    if args.network == 'alexnet':
        # AlexNet will not converge using Xavier
        initializer = mx.init.Normal()
    else:
        initializer = mx.init.Xavier(
            rnd_type='gaussian', factor_type="in", magnitude=2)

    # evaluation metrices
    eval_metrics = ['accuracy']

    # batch_end_callback : callbacks that run after each batch
    batch_end_callbacks = [mx.callback.Speedometer(args.batch_size, args.disp_batches)]
    if 'batch_end_callback' in kwargs:
        cbs = kwargs['batch_end_callback']
        batch_end_callbacks += cbs if isinstance(cbs, list) else [cbs]

    # run
    model.fit(train_data   = data,
        begin_epoch        = args.load_epoch if args.load_epoch else 0,
        num_epoch          = args.num_epochs,
        eval_data          = val,
        eval_metric        = eval_metrics,
        kvstore            = kv,
        optimizer          = args.optimizer,
        optimizer_params   = optimizer_params,
        initializer        = initializer,
        batch_end_callback = batch_end_callbacks,
        epoch_end_callback = checkpoint,
        allow_missing      = True)

####################					in = file				#################
#													 				 			#
# text format file																#
#																				#
####################				return = number				#################
#																	 			#
# how many line number															#
#																				#
####################				explain						#################
#																	 			#
# read text_file -> return how many lines in the file							#
#																				#
def read_num(file):
    num = 0
    with open(file) as file:
        while True:
            line = file.readline()
            if not line:
                break
            num += 1
    return num

####################					in = file				#################
#													 				 			#
# text format file																#
#																				#
####################				return = kwargs				#################
#																	 			#
# return dictonary																#
#																				#
####################				explain						#################
#																	 			#
# read text format file															#
# key(word)=value(integer)														#
#																				#
# -> kwargs={key1:value1, key2:value2, key3:value3, ...}						#
#																				#
def read_info(file):
    kwargs = {}
    with open(file) as info:
        while True:
            line = info.readline()
            if not line:
                break
            key = re.search('\w+',line)
            value = re.search('\d+',line)
            kwargs[key.group()] = value.group()
         
    return kwargs


####################					in = 					#################
#													 				 			#
####################				return =					#################
#																	 			#
####################				explain						#################
#																	 			#
def Train_create(dataset_dir, framework, out_model_dir, max_epochs, mb_size, network_name, devs):
    if framework == 4: # check. is it mxnet?
        if not os.path.exists(dataset_dir):
            return print("Dataset directory is wrong")

        if not os.path.exists(out_model_dir):
            os.makedirs(out_model_dir)

        parser = argparse.ArgumentParser(description='Train',
                                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        add_fit_args(parser) # set default args for fit 
        add_data_args(parser) # set default args for data 
        add_data_aug_args(parser) # set default args for data_augmentation 
        set_data_aug_level(parser, 2) # set default args for augmentation_level

        num_examples = 0
        # set label_file(labels.txt), record_file(.rec), list_file(.lst), num_example, num_classes
        label_file = [labels for labels in Dataset.Dataset_result(dataset_dir) if 'labels.txt' in labels][0]
        if any('dataset.' in split for split in Dataset.Dataset_result(dataset_dir)):
            data_train = [data for data in Dataset.Dataset_result(dataset_dir) if '.rec' in data][0]
            data_val = None
        else:
            data = [data for data in Dataset.Dataset_result(dataset_dir) if '.rec' in data]
            data_train = [train for train in data if 'train.rec' in train][0]
            data_val = [val for val in data if 'test.rec' in val][0]
        lst = [lst for lst in Dataset.Dataset_result(dataset_dir) if '.lst' in lst]
        for lst_file in lst:
            num_examples += read_num(lst_file)
        num_classes = read_num(label_file)
        
        image_info = read_info(os.path.join(dataset_dir,'image_info.txt')) # read img_info
        image_shape    = '{},{},{}'.format(int(image_info['channel']),int(image_info['size']),int(image_info['size'])) # set image_shape
       
        parser.set_defaults(
            network        = network_name,
            # data
            data_train     = data_train,
            data_val       = data_val,
            num_classes    = num_classes,
            num_examples   = num_examples,
            image_shape    = image_shape, 
            pad_size       = 0,
            # train
            gpus          = devs,
            batch_size     = mb_size,
            num_epochs     = max_epochs,
            lr             = .005,
            lr_step_epochs = '200,250',
            disp_batches   = int(num_examples/mb_size/1),
            )
        args = parser.parse_args()
        args.model_prefix=out_model_dir 

        net = import_module('network.'+args.network)
        sym = net.get_symbol(**vars(args))
        fit(args, sym, get_rec_iter)
        print('Train_create finish')
    return True

####################					in = 					#################
#													 				 			#
####################				return =					#################
#																	 			#
####################				explain						#################
#																	 			#
def Train_result(model_dir):
    if not os.path.exists(model_dir):
        return print('model_dir is not found.')
    
    result = os.listdir(model_dir)
    result.sort()
    
    return result

in_dataset_dir = Dataset.in_dataset_dir
dataset_dir = Dataset.out_dataset_dir
out_model_dir = Dataset.out_dataset_dir + '/model'

if __name__ == '__main__':
    Train_create(dataset_dir = dataset_dir, 
                 framework = 4, 
                 out_model_dir = out_model_dir, 
                 max_epochs = 5, 
                 mb_size = 128, 
                 network_name = 'lenet', 
                 devs = '0,1,2')
    print(Train_result(model_dir = out_model_dir))

