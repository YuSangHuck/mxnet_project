import Dataset
import Train

# for windows settings
#in_dataset_dir = 'D:\Github\dataset\img\mydataset'
#out_dataset_dir = 'D:\Github\dataset\img\mydataset\out'

# for linux settings
in_dataset_dir = Dataset.in_dataset_dir
out_dataset_dir = Dataset.out_dataset_dir

if __name__ == '__main__':
#    Dataset.Dataset_create(Dataset.in_dataset_dir, Dataset.out_dataset_dir, 32, 4)
    print(Dataset.Dataset_result(out_dataset_dir))
