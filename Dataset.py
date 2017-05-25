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
import queue

####################					in = args				#################
#																				#
# args.root = in_dataset_dir											 		#
# args.out = out_dataset_dir													#
# args.exts = '.jpeg','.png','.bmp','.jpg'										#
# args.train_ratio = between 0,1 -> 0~1											#
#																				#
####################				return = X					#################
#																				#
####################				explain						#################
#																				#
# using in_dataset_dir images -> make list_file(.lst)				 			#
# calling list_image,write_list function										#
#																				#
# if train_ratio = 1.															#
#     output is 'dataset.lst'													#
# if train_ratio = 0.8 -> val_ratio = 0.2										#
#     output is 'dataset_train.lst, dataset_val.lst'							#
#																				#
def make_list(args):
    image_list = list_image(args.root, args.out, args.exts)
    image_list = list(image_list)
    random.seed(100)
    random.shuffle(image_list)
    size = len(image_list)
    sep = int(size * args.train_ratio)
         
    if args.train_ratio == 1.0:
        write_list(args.out+'/dataset' + '.lst', image_list)
    else:
        write_list(args.out+'/dataset' + '_val.lst', image_list[sep:])
        write_list(args.out+'/dataset' + '_train.lst', image_list[:sep])

####################				in = root,out,exts			#################
# 													 				 			#
# root = in_dataset_dir												 			#
# out = out_dataset_dir															#
# exts = extensoin ('.jpeg','.bmp', ...)										#
#																				#
####################				return = generator			#################
#																				#
# generator -> list [(example_number, rel_fname, label)					 		#
#																				#
####################				explain						#################
#																	 			#
# return generator(example_num, rel_fname, label)								#
# and write 'out_dataset_dir/labels.txt' 										#
#																				#
def list_image(root, out, exts):
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

####################		in = path_out, image_list			#################
#																				#
# path_out = out_dataset_dir													#
# image_list = list_file							 				 			#
#																	 			#
####################				return = X					#################
#																	 			#
# No return but write list_file(.lst)											#
#																				#
####################				explain						#################
#																	 			#
# write list_file(.lst)															#
# line = 'example_num, label, rel_fname)										#
#																				#
def write_list(path_out, image_list):
    with open(path_out, 'w') as fout:
        for i, item in enumerate(image_list):
            line = '%d\t' % item[0]
            for j in item[2:]:
                line += '%f\t' % j
            line += '%s\n' % item[1]
            fout.write(line)

####################					in = path_in			#################
#																				#
# path_in = in_list_file							  				 			#
#																	 			#
####################				return = item				#################
#																				#
# item = (example_num, rel_fname, label)							 			#
#																				#
####################				explain						#################
#																	 			#
# read list_file(.lst)															#
# line -> item																	#
#																				#
def read_list(path_in):
    with open(path_in) as fin:
        while True:
            line = fin.readline()
            if not line:
                break
            line = [i.strip() for i in line.strip().split('\t')]
            item = [int(line[0])] + [line[-1]] + [float(i) for i in line[1:-1]]
            yield item

####################			in = args,i,item,q_out			#################
#																				#
# args = color,centor_crop,resize,quality										#
# item = (example_num, rel_fname, label)										#
# q_out = Queue														 			#
#																	 			#
####################				return = X					#################
# 																				#
# No return but put encoding image information to Queue							#
#																				#
####################				explain						#################
#																	 			#
# raw_image -> packing(encoding_images) -> Queue								#
#																				#
def image_encode(args, i, item, q_out):
    fullpath = os.path.join(args.root, item[1])

    if len(item) > 3 and args.pack_label:
        header = mx.recordio.IRHeader(0, item[2:], item[0], 0)
    else:
        header = mx.recordio.IRHeader(0, item[2], item[0], 0)

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

####################					in = X 					#################
#													 				 			#
####################				return = args				#################
# 																	 			#
#      = root,out																#
# args = cgroup(exts,train-ratio) for list_file(.lst)							#
#      = rgroup(resize,center-crop,quality,color,encoding,pack-label)			#
#																				#
####################				explain						#################
#																	 			#
# set defaul args for making list_file(.lst) and record_file(.rec)				#
#																				#
def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Create an image list or \
        make a record database by reading from an image list')
    parser.add_argument('--root', default='/', help='path to folder containing images.')
    parser.add_argument('--out', default='/', help='path to folder output dataset.')

    cgroup = parser.add_argument_group('Options for creating image lists file')
    cgroup.add_argument('--exts', type=list, default=['.jpeg', '.jpg', '.png', '.bmp'],
                        help='list of acceptable image extensions.')
    cgroup.add_argument('--train-ratio', type=float, default=1.,
                        help='Ratio of images to use for training.')

    rgroup = parser.add_argument_group('Options for creating record file')
    rgroup.add_argument('--resize', type=int, default=50,
                        help='resize the shorter edge of image to the newsize, original images will\
        be packed by default.')
    rgroup.add_argument('--center-crop', type=bool, default=False,
                        help='specify whether to crop the center image to make it rectangular.')
    rgroup.add_argument('--quality', type=int, default=9,
                        help='JPEG quality for encoding, 1-100; or PNG compression for encoding, 1-9')
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

####################		in = out_dataset_dir,kwargs			#################
#													 				 			#
# out_dataset_dir = out_dataset_dir												#
# kwargs = resize, channel														#
#																				#
####################				return = X					#################
#																				#
####################				explain						#################
#																				#
# write information for Train													#
#																				#
def make_info(out_dataset_dir, **kwargs):
    abs_path_file = os.path.join(out_dataset_dir,'image_info.txt')
    with open(abs_path_file,'w') as image_info:
        for key in kwargs.keys():
            image_info.write('{}={}\n'.format(key,kwargs[key]))
    return


####################					in						 #################
#													 				 			#
# in_dataset_dir = raw dataset directory										#
# out_dataset_dir = out dataset directory										#
# resize = resizing scale														#
# framework = MXnet,CNTK, ...													#
#																				#
####################				return = True/False			#################
#																				#
####################				explain						#################
#																	 			#
# in & out directory check														#
# set default dataset_args for making Dataset											#
# make list_file(.lst) -> make record_file(.rec)								#
# logging by 1000																#
#																				#
def Dataset_create(in_dataset_dir, out_dataset_dir, resize, framework):
    if framework == 4: # check. is it mxnet?
        if not (os.path.exists(in_dataset_dir) and os.listdir(in_dataset_dir)): # check input_dataset
            return print('in dataset is Wrong.')
    
        if not os.path.exists(out_dataset_dir): # check output-dataset directory
            os.makedirs(out_dataset_dir)
        
        # set default dataset_args for making Dataset 
        dataset_args = parse_args()
        dataset_args.root = in_dataset_dir
        dataset_args.out = out_dataset_dir
        dataset_args.resize = resize

        # make information. key=keyvalue. used to Train.py
        if dataset_args.color == 1:
            make_info(out_dataset_dir, channel = 3, size = resize)
        # make list_file(.lst) -> used to make record_file(.rec)
        make_list(dataset_args) 

        # set working_dir. make record_file(.rec) at working_dir
        working_dir = dataset_args.out
        files = [os.path.join(working_dir, fname) for fname in os.listdir(working_dir)
                    if os.path.isfile(os.path.join(working_dir, fname))] # files = [abs_fnames]
        count = 0 # total list_file(.lst) number
        cnt = 0 # total image number
        logger = logging.getLogger('single_process')
        filehandler = logging.FileHandler(out_dataset_dir+'/create_val_db.log','w')
        streamhandler = logging.StreamHandler()
        formatter = logging.Formatter('[%(levelname)s|%(asctime)s] %(message)s')
        filehandler.setFormatter(formatter)
        streamhandler.setFormatter(formatter)
        logger.addHandler(filehandler)
        logger.addHandler(streamhandler)
        logger.setLevel(logging.DEBUG)
        
        for fname in files:
            if fname.startswith(dataset_args.out) and fname.endswith('.lst'): # find list_file(.lst) of [fnames]
                count += 1
                image_list = read_list(fname) # read list_file(.lst) count,path,label
                # -- write_record -- #
                q_out = queue.Queue()
                fname = os.path.basename(fname)
                fname_rec = os.path.splitext(fname)[0] + '.rec' # fname.rec
                fname_idx = os.path.splitext(fname)[0] + '.idx' # fname.idx
                record = mx.recordio.MXIndexedRecordIO(os.path.join(working_dir, fname_idx),
                                                        os.path.join(working_dir, fname_rec), 'w') # Read/write RecordIO format supporting random access
                pre_time = time.time()
                for i, item in enumerate(image_list):
                    image_encode(dataset_args, i, item, q_out) # i:number, item:[0number,1path,2label], q_out:Queue / img -> encoding,mx.recordio.pack_img -> Queue
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
            print('Did not find and list file with prefix %s'%dataset_args.out)

        print('Dataset_create finish')
    return True

####################					in = out_dataset_dir	#################
#													 				 			#
# out_dataset_dir = out_dataset directory										#
#																				#
####################				return = found_dataset		#################
#																	 			#
# found_dataset created by Dataset_create										#
#																				#
####################				explain						#################
#																	 			#
# out_dir & out_dataset check													#
# return found_dataset															#
#																				#
def Dataset_result(out_dataset_dir):
    found_dir = os.path.exists(out_dataset_dir)
    found_files = os.listdir(out_dataset_dir)
    found_dataset = []

    if not found_dir:
        return print('Directory is not found')    

    if not found_files:
        return print('Dataset is not found')
    
    for found_file in found_files:
        if os.path.splitext(found_file)[1] in ('.lst','.rec','.txt'):
            found_file = os.path.join(out_dataset_dir,found_file)
            found_dataset.append(found_file)
    found_dataset.sort()

    return found_dataset


in_dataset_dir = '/root/git/dataset/img'
out_dataset_dir = in_dataset_dir + '/../out_dataset'

if __name__ == '__main__':
    Dataset_create(in_dataset_dir = in_dataset_dir,
                   out_dataset_dir = out_dataset_dir,
                   resize = 32,
                   framework = 4)
    print(Dataset_result(out_dataset_dir))

