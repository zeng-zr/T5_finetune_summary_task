#首先编写继承自 Dataset 类的自定义数据集用于组织样本和标签。考虑到使用 LCSTS 两百多万条样本进行训练耗时过长，这里我们只抽取训练集中的前 20 万条数据

from torch.utils.data import Dataset

max_dataset_size = 200000

class LCSTS(Dataset):
    def __init__(self, data_file):
        self.data = self.load_data(data_file)
    
    def load_data(self, data_file):
        Data = {}
        with open(data_file, 'rt', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                if idx >= max_dataset_size:
                    break
                items = line.strip().split('!=!')
                assert len(items) == 2
                Data[idx] = {
                    'title': items[0],
                    'content': items[1]
                }
        return Data
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

train_data = LCSTS('dataset/data1.tsv')
valid_data = LCSTS('dataset/data2.tsv')
test_data = LCSTS('dataset/data3.tsv')

if __name__=='__main__':
    print(f'train set size: {len(train_data)}')
    print(f'valid set size: {len(valid_data)}')
    print(f'test set size: {len(test_data)}')
    print(next(iter(train_data)))