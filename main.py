import Dataset
import Train

in_dataset_dir = Dataset.in_dataset_dir
out_dataset_dir = Dataset.out_dataset_dir
out_model_dir = Train.out_model_dir

if __name__ == '__main__':
    Dataset.Dataset_create(in_dataset_dir = in_dataset_dir,
                           out_dataset_dir = out_dataset_dir,
                           resize = 32,
                           framework = 4)
    print(Dataset.Dataset_result(out_dataset_dir))
    Train.Train_create(dataset_dir = out_dataset_dir,
                       framework = 4,
                       out_model_dir = out_model_dir,
                       max_epochs = 4,
                       mb_size = 128,
                       network_name = 'lenet',
                       devs = '1,2')
    print(Train.Train_result(model_dir=out_model_dir))

