import mxnet as mx
from mxnet.io import DataBatch, DataIter
import numpy as np
import logging
logger = logging.getLogger('Train')
import os
import time
import argparse
import Dataset
import network
import re

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
    data.add_argument('--benchmark', type=int, default=0,
                      help='if 1, then feed the network with synthetic data')
    data.add_argument('--dtype', type=str, default='float32',
                      help='data type: float32 or float16')
    return data

def add_data_aug_args(parser):
    aug = parser.add_argument_group(
        'Image augmentations', 'implemented in src/io/image_aug_default.cc')
    aug.add_argument('--random-crop', type=int, default=1,
                     help='if or not randomly crop the image')
    aug.add_argument('--random-mirror', type=int, default=1,
                     help='if or not randomly flip horizontally')
    aug.add_argument('--max-random-h', type=int, default=0,
                     help='max change of hue, whose range is [0, 180]')
    aug.add_argument('--max-random-s', type=int, default=0,
                     help='max change of saturation, whose range is [0, 255]')
    aug.add_argument('--max-random-l', type=int, default=0,
                     help='max change of intensity, whose range is [0, 255]')
    aug.add_argument('--max-random-aspect-ratio', type=float, default=0,
                     help='max change of aspect ratio, whose range is [0, 1]')
    aug.add_argument('--max-random-rotate-angle', type=int, default=0,
                     help='max angle to rotate, whose range is [0, 360]')
    aug.add_argument('--max-random-shear-ratio', type=float, default=0,
                     help='max ratio to shear, whose range is [0, 1]')
    aug.add_argument('--max-random-scale', type=float, default=1,
                     help='max ratio to scale')
    aug.add_argument('--min-random-scale', type=float, default=1,
                     help='min ratio to scale, should >= img_size/input_shape. otherwise use --pad-size')
    return aug

def set_data_aug_level(aug, level):
    if level >= 1:
        aug.set_defaults(random_crop=1, random_mirror=1)
    if level >= 2:
        aug.set_defaults(max_random_h=36, max_random_s=50, max_random_l=50)
    if level >= 3:
        aug.set_defaults(max_random_rotate_angle=10, max_random_shear_ratio=0.1, max_random_aspect_ratio=0.25)

class SyntheticDataIter(DataIter):
    def __init__(self, num_classes, data_shape, max_iter, dtype):
        self.batch_size = data_shape[0]
        self.cur_iter = 0
        self.max_iter = max_iter
        self.dtype = dtype
        label = np.random.randint(0, num_classes, [self.batch_size,])
        data = np.random.uniform(-1, 1, data_shape)
        self.data = mx.nd.array(data, dtype=self.dtype)
        self.label = mx.nd.array(label, dtype=self.dtype)
    def __iter__(self):
        return self
    @property
    def provide_data(self):
        return [mx.io.DataDesc('data', self.data.shape, self.dtype)]
    @property
    def provide_label(self):
        return [mx.io.DataDesc('softmax_label', (self.batch_size,), self.dtype)]
    def next(self):
        self.cur_iter += 1
        if self.cur_iter <= self.max_iter:
            return DataBatch(data=(self.data,),
                             label=(self.label,),
                             pad=0,
                             index=None,
                             provide_data=self.provide_data,
                             provide_label=self.provide_label)
        else:
            raise StopIteration
    def __next__(self):
        return self.next()
    def reset(self):
        self.cur_iter = 0

def get_rec_iter(args, kv=None):
    image_shape = tuple([int(l) for l in args.image_shape.split(',')])
    dtype = np.float32;
    if 'dtype' in args:
        if args.dtype == 'float16':
            dtype = np.float16
    if 'benchmark' in args and args.benchmark:
        data_shape = (args.batch_size,) + image_shape
        train = SyntheticDataIter(args.num_classes, data_shape, 50, dtype)
        return (train, None)
    if kv:
        (rank, nworker) = (kv.rank, kv.num_workers)
    else:
        (rank, nworker) = (0, 1)
    rgb_mean = [float(i) for i in args.rgb_mean.split(',')]
    train = mx.io.ImageRecordIter(
        path_imgrec         = args.data_train,
        label_width         = 1,
        mean_r              = rgb_mean[0],
        mean_g              = rgb_mean[1],
        mean_b              = rgb_mean[2],
        data_name           = 'data',
        label_name          = 'softmax_label',
        data_shape          = image_shape,
        batch_size          = args.batch_size,
        rand_crop           = args.random_crop,
        max_random_scale    = args.max_random_scale,
        pad                 = args.pad_size,
        fill_value          = 127,
        min_random_scale    = args.min_random_scale,
        max_aspect_ratio    = args.max_random_aspect_ratio,
        random_h            = args.max_random_h,
        random_s            = args.max_random_s,
        random_l            = args.max_random_l,
        max_rotate_angle    = args.max_random_rotate_angle,
        max_shear_ratio     = args.max_random_shear_ratio,
        rand_mirror         = args.random_mirror,
        preprocess_threads  = args.data_nthreads,
        shuffle             = True,
        num_parts           = nworker,
        part_index          = rank)
    if args.data_val is None:
        return (train, None)
    val = mx.io.ImageRecordIter(
        path_imgrec         = args.data_val,
        label_width         = 1,
        mean_r              = rgb_mean[0],
        mean_g              = rgb_mean[1],
        mean_b              = rgb_mean[2],
        data_name           = 'data',
        label_name          = 'softmax_label',
        batch_size          = args.batch_size,
        data_shape          = image_shape,
        preprocess_threads  = args.data_nthreads,
        rand_crop           = False,
        rand_mirror         = False,
        num_parts           = nworker,
        part_index          = rank)
    return (train, val)

def _get_lr_scheduler(args, kv):
    if 'lr_factor' not in args or args.lr_factor >= 1:
        return (args.lr, None)
    epoch_size = args.num_examples / args.batch_size
    if 'dist' in args.kv_store:
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

def _load_model(args, rank=0):
    if 'load_epoch' not in args or args.load_epoch is None:
        return (None, None, None)
    assert args.model_prefix is not None
    model_prefix = args.model_prefix
    if rank > 0 and os.path.exists("%s-%d-symbol.json" % (model_prefix, rank)):
        model_prefix += "-%d" % (rank)
    sym, arg_params, aux_params = mx.model.load_checkpoint(
        model_prefix, args.load_epoch)
    logging.info('Loaded model %s_%04d.params', model_prefix, args.load_epoch)
    return (sym, arg_params, aux_params)

def _save_model(args, rank=0):
    if args.model_prefix is None:
        return None
    dst_dir = os.path.join(args.model_prefix,args.network)
    if not os.path.isdir(args.model_prefix):
        os.makedirs(args.model_prefix)
    return mx.callback.do_checkpoint(dst_dir if rank == 0 else "%s-%d" % (
        args.model_prefix, rank))

def add_fit_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
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
    parser.add_argument('--monitor', dest='monitor', type=int, default=0,
                        help='log network parameters every N iters if larger than 0')
    train.add_argument('--load-epoch', type=int,
                       help='load the model on an epoch using the model-load-prefix')
    train.add_argument('--top-k', type=int, default=0,
                       help='report the top-k accuracy. 0 means no report.')
    train.add_argument('--test-io', type=int, default=0,
                       help='1 means test reading speed without training')
    return train

def fit(args, network, data_loader, **kwargs):
    """
    train a model
    args : argparse returns
    network : the symbol definition of the nerual network
    data_loader : function that returns the train and val data iterators
    """
    # kvstore
    kv = mx.kvstore.create(args.kv_store)

    # data iterators
    (train, val) = data_loader(args, kv)
    if args.test_io:
        tic = time.time()
        for i, batch in enumerate(train):
            for j in batch.data:
                j.wait_to_read()
            if (i+1) % args.disp_batches == 0:
                logging.info('Batch [%d]\tSpeed: %.2f samples/sec' % (
                    i, args.disp_batches*args.batch_size/(time.time()-tic)))
                tic = time.time()
        return

    # load model
    if 'arg_params' in kwargs and 'aux_params' in kwargs:
        arg_params = kwargs['arg_params']
        aux_params = kwargs['aux_params']
    else:
        sym, arg_params, aux_params = _load_model(args, kv.rank)
        if sym is not None:
            assert sym.tojson() == network.tojson()

    # save model
    checkpoint = _save_model(args, kv.rank)
    
    # logging
    format = '[%(levelname)s:%(asctime)s],%(message)s'
    logging.basicConfig(filename = os.path.join(args.model_prefix,'create_train_db.log'),
                        filemode = 'w',
                        format = format,
                        level=logging.DEBUG)
    
    # devices for training
    devs = mx.cpu() if args.gpus is None or args.gpus is '' else [
        mx.gpu(int(i)) for i in args.gpus.split(',')]

    # learning rate
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

    monitor = mx.mon.Monitor(args.monitor, pattern=".*") if args.monitor > 0 else None

    if args.network == 'alexnet':
        # AlexNet will not converge using Xavier
        initializer = mx.init.Normal()
    else:
        initializer = mx.init.Xavier(
            rnd_type='gaussian', factor_type="in", magnitude=2)

    # evaluation metrices
    eval_metrics = ['accuracy']
    if args.top_k > 0:
        eval_metrics.append(mx.metric.create('top_k_accuracy', top_k=args.top_k))

    # callbacks that run after each batch
    batch_end_callbacks = [mx.callback.Speedometer(args.batch_size, args.disp_batches)]
    if 'batch_end_callback' in kwargs:
        cbs = kwargs['batch_end_callback']
        batch_end_callbacks += cbs if isinstance(cbs, list) else [cbs]

    # run
    model.fit(train,
        begin_epoch        = args.load_epoch if args.load_epoch else 0,
        num_epoch          = args.num_epochs,
        eval_data          = val,
        eval_metric        = eval_metrics,
        kvstore            = kv,
        optimizer          = args.optimizer,
        optimizer_params   = optimizer_params,
        initializer        = initializer,
        arg_params         = arg_params,
        aux_params         = aux_params,
        batch_end_callback = batch_end_callbacks,
        epoch_end_callback = checkpoint,
        allow_missing      = True,
        monitor            = monitor)

def read_num(file):
    num = 0
    with open(file) as file:
        while True:
            line = file.readline()
            if not line:
                break
            num += 1
    return num

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


def Train_create(dataset_dir, framework, out_model_dir, max_epochs, mb_size, network_name, devs):
    if framework == 4: # check mxnet?
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
            lst = [lst for lst in Dataset.Dataset_result(dataset_dir) if '.lst' in lst]
        else:
            data = [data for data in Dataset.Dataset_result(dataset_dir) if '.rec' in data]
            data_train = [train for train in data if 'train.rec' in train][0]
            data_val = [val for val in data if 'test.rec' in val][0]
            lst = [lst for lst in Dataset.Dataset_result(dataset_dir) if '.lst' in lst]
        for lstfile in lst:
            num_examples += read_num(lstfile)
        num_classes = read_num(label_file)
        
        args = read_info(os.path.join(dataset_dir,'image_info.txt')) # read img_info
        image_shape    = '{},{},{}'.format(int(args['channel']),int(args['size']),int(args['size'])) # set image_shape
       
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
#            gpus          = devs,
            batch_size     = mb_size,
            num_epochs     = max_epochs,
            lr             = .005,
            lr_step_epochs = '200,250',
            disp_batches   = int(num_examples/mb_size/1),
            )
        args = parser.parse_args()
        args.model_prefix=out_model_dir 

        from importlib import import_module
        net = import_module('network.'+args.network)
        sym = net.get_symbol(**vars(args))
        print(args.image_shape)
        fit(args, sym, get_rec_iter)
        print('Training is finished')
        return True

def Train_result(model_dir):
    if not os.path.exists(model_dir):
        return print('model_dir is not found.')

    return os.listdir(model_dir)


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

