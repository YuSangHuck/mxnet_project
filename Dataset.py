from __future__ import print_function
import os
import sys
import logging

import mxnet as mx
import random
import argparse
import cv2
import time
import traceback

####################					in = 					#################
#													 				 			#
#																	 			#
####################				return =					#################
#																	 			#
#																				#
####################				explain						#################
#																	 			#
#																				#
####################											#################
def list_image(root, exts, out):
    i = 0
    cat = {}
    for path, dirs, files in os.walk(root, followlinks=True):
        dirs.sort()
        files.sort()
        for fname in files:
            fpath = os.path.join(path, fname)
            suffix = os.path.splitext(fname)[1].lower()
            if os.path.isfile(fpath) and (suffix in exts):
                if path not in cat:
                    cat[path] = len(cat)
                yield (i, os.path.relpath(fpath, root), cat[path])
                i += 1
    with open(out+'/labels.txt','w') as labels:
        for k, v in sorted(cat.items(), key=lambda x: x[1]):
            labels.write(os.path.basename(os.path.relpath(k, root))+'\n')

####################					in = 					#################
#													 				 			#
#																	 			#
####################				return =					#################
#																	 			#
#																				#
####################				explain						#################
#																	 			#
#																				#
####################											#################
def write_list(path_out, image_list):
    with open(path_out, 'w') as fout:
        for i, item in enumerate(image_list):
            line = '%d\t' % item[0]
            for j in item[2:]:
                line += '%f\t' % j
            line += '%s\n' % item[1]
            fout.write(line)

####################					in = args				#################
#													 				 			#
#																	 			#
####################				return = 					#################
#																	 			#
#																				#
####################				explain						#################
#																	 			#
#																				#
####################											#################
def make_list(args):
    image_list = list_image(args.root, args.exts, args.out)
    image_list = list(image_list)
    random.seed(100)
    random.shuffle(image_list)
    size = len(image_list)

#    chunk_size = (size + args.chunks - 1) / args.chunks
#    for i in range(args.chunks):
#        chunk = image_list[i * int(chunk_size):(i + 1) * int(chunk_size)]
#        sep = int(chunk_size * args.train_ratio)
#        sep_test = int(chunk_size * args.test_ratio)
#         
#        if args.train_ratio == 1.0:
#            write_list(args.out+'/dataset' + '.lst', chunk)
#        else:
#            if args.test_ratio:
#                write_list(args.out+'/dataset' + '_test.lst', chunk[:sep_test])
#            if args.train_ratio + args.test_ratio < 1.0:
#                write_list(args.out+'/dataset' + '_val.lst', chunk[sep_test + sep:])
#            write_list(args.out+'/dataset' + '_train.lst', chunk[sep_test:sep_test + sep])

    sep = int(size * args.train_ratio)
         
    if args.train_ratio == 1.0:
        write_list(args.out+'/dataset' + '.lst', image_list)
    else:
        write_list(args.out+'/dataset' + '_val.lst', image_list[sep:])
        write_list(args.out+'/dataset' + '_train.lst', image_list[:sep])

####################					in = 					#################
#													 				 			#
#																	 			#
####################				return =					#################
#																	 			#
#																				#
####################				explain						#################
#																	 			#
#																				#
####################											#################
def read_list(path_in):
    with open(path_in) as fin:
        while True:
            line = fin.readline()
            if not line:
                break
            line = [i.strip() for i in line.strip().split('\t')]
            item = [int(line[0])] + [line[-1]] + [float(i) for i in line[1:-1]]
            yield item

####################					in = 					#################
#													 				 			#
#																	 			#
####################				return =					#################
#																	 			#
#																				#
####################				explain						#################
#																	 			#
#																				#
####################											#################
def image_encode(args, i, item, q_out):
    fullpath = os.path.join(args.root, item[1])

    if len(item) > 3 and args.pack_label:
        header = mx.recordio.IRHeader(0, item[2:], item[0], 0)
    else:
        header = mx.recordio.IRHeader(0, item[2], item[0], 0)

    if args.pass_through:
        try:
            with open(fullpath) as fin:
                img = fin.read()
            s = mx.recordio.pack(header, img)
            q_out.put((i, s, item))
        except Exception as e:
            traceback.print_exc()
            print('pack_img error:', item[1], e)
            q_out.put((i, None, item))
        return
    try:
        img = cv2.imread(fullpath, args.color)
    except:
        traceback.print_exc()
        print('imread error trying to load file: %s ' % fullpath)
        q_out.put((i, None, item))
        return
    if img is None:
        print('imread read blank (None) image for file: %s' % fullpath)
        q_out.put((i, None, item))
        return
    if args.center_crop:
        if img.shape[0] > img.shape[1]:
            margin = (img.shape[0] - img.shape[1]) / 2;
            img = img[margin:margin + img.shape[1], :]
        else:
            margin = (img.shape[1] - img.shape[0]) / 2;
            img = img[:, margin:margin + img.shape[0]]
    if args.resize:
        img = cv2.resize(img, (args.resize, args.resize))

    try:
        s = mx.recordio.pack_img(header, img, quality=args.quality, img_fmt=args.encoding)
        q_out.put((i, s, item))
    except Exception as e:
        traceback.print_exc()
        print('pack_img error on file: %s' % fullpath, e)
        q_out.put((i, None, item))
        return

####################					in = 					#################
#													 				 			#
#																	 			#
####################				return =					#################
#																	 			#
#																				#
####################				explain						#################
#																	 			#
#																				#
####################											#################
def read_worker(args, q_in, q_out):
    while True:
        deq = q_in.get()
        if deq is None:
            break
        i, item = deq
        image_encode(args, i, item, q_out)

####################					in = 					#################
#													 				 			#
#																	 			#
####################				return =					#################
#																	 			#
#																				#
####################				explain						#################
#																	 			#
#																				#
####################											#################
def write_worker(q_out, fname, working_dir):
    pre_time = time.time()
    count = 0
    fname = os.path.basename(fname)
    fname_rec = os.path.splitext(fname)[0] + '.rec'
    fname_idx = os.path.splitext(fname)[0] + '.idx'
    record = mx.recordio.MXIndexedRecordIO(os.path.join(working_dir, fname_idx),
                                           os.path.join(working_dir, fname_rec), 'w')
    buf = {}
    more = True
    while more:
        deq = q_out.get()
        if deq is not None:
            i, s, item = deq
            buf[i] = (s, item)
        else:
            more = False
        while count in buf:
            s, item = buf[count]
            del buf[count]
            if s is not None:
                record.write_idx(item[0], s)

            if count % 1000 == 0:
                cur_time = time.time()
                print('time:', cur_time - pre_time, ' count:', count)
                pre_time = cur_time
            count += 1

####################					in = 					#################
#													 				 			#
#																	 			#
####################				return =					#################
#																	 			#
#																				#
####################				explain						#################
#																	 			#
#																				#
####################											#################
def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Create an image list or \
        make a record database by reading from an image list')
    parser.add_argument('--root', default='/', help='path to folder containing images.')
    parser.add_argument('--out', default='/', help='path to folder output dataset.')

    cgroup = parser.add_argument_group('Options for creating image lists')
    cgroup.add_argument('--list', type=bool, default=True,
                        help='If this is set im2rec will create image list(s) by traversing root folder\
        and output to <prefix>.lst.\
        Otherwise im2rec will read <prefix>.lst and create a database at <prefix>.rec')
    cgroup.add_argument('--exts', type=list, default=['.jpeg', '.jpg', '.png', '.bmp'],
                        help='list of acceptable image extensions.')
    cgroup.add_argument('--chunks', type=int, default=1, help='number of chunks.')
    cgroup.add_argument('--train-ratio', type=float, default=1.,
                        help='Ratio of images to use for training.')
    cgroup.add_argument('--test-ratio', type=float, default=0.,
                        help='Ratio of images to use for testing.')

    rgroup = parser.add_argument_group('Options for creating database')
    rgroup.add_argument('--pass-through', type=bool, default=False,
                        help='whether to skip transformation and save image as is')
    rgroup.add_argument('--resize', type=int, default=50,
                        help='resize the shorter edge of image to the newsize, original images will\
        be packed by default.')
    rgroup.add_argument('--center-crop', type=bool, default=False,
                        help='specify whether to crop the center image to make it rectangular.')
    rgroup.add_argument('--quality', type=int, default=95,
                        help='JPEG quality for encoding, 1-100; or PNG compression for encoding, 1-9')
    rgroup.add_argument('--num-thread', type=int, default=1,
                        help='number of thread to use for encoding. order of images will be different\
        from the input list if >1. the input list will be modified to match the\
        resulting order.')
    rgroup.add_argument('--color', type=int, default=1, choices=[-1, 0, 1],
                        help='specify the color mode of the loaded image.\
        1: Loads a color image. Any transparency of image will be neglected. It is the default flag.\
        0: Loads image in grayscale mode.\
        -1:Loads image as such including alpha channel.')
    rgroup.add_argument('--encoding', type=str, default='.png', choices=['.jpg', '.png'],
                        help='specify the encoding of the images.')
    rgroup.add_argument('--pack-label', type=bool, default=False,
        help='Whether to also pack multi dimensional label in the record file')
    args = parser.parse_args()

    return args

####################					in = 					#################
#													 				 			#
#																	 			#
####################				return =					#################
#																	 			#
#																				#
####################				explain						#################
#																	 			#
#																				#
####################											#################
def make_info(out_dataset_dir, **kwargs):
    abs_path_file = os.path.join(out_dataset_dir,'image_info.txt')
    with open(abs_path_file,'w') as image_info:
        for key in kwargs.keys():
            image_info.write('{}={}\n'.format(key,kwargs[key]))
    return


####################					in = 					#################
#													 				 			#
#																	 			#
####################				return =					#################
#																	 			#
#																				#
####################				explain						#################
#																	 			#
#																				#
####################											#################
def Dataset_create(in_dataset_dir, out_dataset_dir, resize, framework):
    if framework == 4: # check mxnet?
        if not (os.path.exists(in_dataset_dir) and os.listdir(in_dataset_dir)): # check input_dataset
            return print('in dataset is Wrong.')
    
        if not os.path.exists(out_dataset_dir): # check output-dataset directory
            os.makedirs(out_dataset_dir)

        make_info(out_dataset_dir, channel = 3, size = resize) # make information. key=keyvalue. used to Train.py

        args = parse_args() # set default args for making Dataset 
        args.root = in_dataset_dir
        args.out = out_dataset_dir
        args.resize = resize
        make_list(args) # make list_file(.lst) -> used to make record_file(.rec)

        working_dir = args.out # set working_dir. make record_file(.rec) at working_dir
        files = [os.path.join(working_dir, fname) for fname in os.listdir(working_dir)
                    if os.path.isfile(os.path.join(working_dir, fname))] # files = [abs_fnames]
        count = 0 # total list_file(.lst) number
        cnt = 0 # total image number
        logger = logging.getLogger('single_process')
        filehandler = logging.FileHandler(out_dataset_dir+'/create_val_db.log','w')
        streamhandler = logging.StreamHandler()
        formatter = logging.Formatter('[%(filename)s|%(asctime)s] %(message)s')
        filehandler.setFormatter(formatter)
        streamhandler.setFormatter(formatter)
        logger.addHandler(filehandler)
        logger.addHandler(streamhandler)
        logger.setLevel(logging.DEBUG)
        
        for fname in files:
            if fname.startswith(args.out) and fname.endswith('.lst'): # find list_file(.lst) of [fnames]
                count += 1
                image_list = read_list(fname) # read list_file(.lst) 
                # -- write_record -- #
                import queue
                q_out = queue.Queue()
                fname = os.path.basename(fname)
                fname_rec = os.path.splitext(fname)[0] + '.rec' # fname.rec
                fname_idx = os.path.splitext(fname)[0] + '.idx' # fname.idx
                record = mx.recordio.MXIndexedRecordIO(os.path.join(working_dir, fname_idx),
                                                        os.path.join(working_dir, fname_rec), 'w') # Read/write RecordIO format supporting random access
                pre_time = time.time()
                for i, item in enumerate(image_list):
                    image_encode(args, i, item, q_out) # i:number, item:[0number,1path,2label], q_out:Queue / img -> encoding,mx.recordio.pack_img -> Queue
                    if q_out.empty():
                        continue
                    _, s, _ = q_out.get() # s=mx.recordio.pack_img
                    record.write_idx(item[0], s) # write record_file(.rec) with index(.idx)
                    if cnt % 1000 == 0:
                        cur_time = time.time()
                        logger.info('time:{0:0.10f}, count:{1}'.format(cur_time - pre_time, cnt)) 
                    pre_time = cur_time
                    cnt += 1
        if not count:
            print('Did not find and list file with prefix %s'%args.out)

        print('Dataset creating finished')
        return True

####################					in = 					#################
#													 				 			#
#																	 			#
####################				return =					#################
#																	 			#
#																				#
####################				explain						#################
#																	 			#
#																				#
####################											#################
def Dataset_result(out_dataset_dir):
    found_dir = os.path.exists(out_dataset_dir)
    found_files = os.listdir(out_dataset_dir)
    found_dataset = []

    if not found_dir:
        return print('Directory is not found')    

    if not found_files:
        return print('Dataset is not found')
    
    for found_file in found_files:
        if os.path.splitext(found_file)[1] in ('.idx','.lst','.rec','.txt','.log'):
            found_file = os.path.join(out_dataset_dir,found_file)
            found_dataset.append(found_file)
    found_dataset.sort()
    return found_dataset


in_dataset_dir = '/root/git/dataset/img'
out_dataset_dir = in_dataset_dir + '/../out_dataset'

if __name__ == '__main__':
    Dataset_create(in_dataset_dir = in_dataset_dir,
                   out_dataset_dir = out_dataset_dir,
                   resize = 16,
                   framework = 4)
#    print(Dataset_result(out_dataset_dir))

