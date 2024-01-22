import argparse
import json
import os
import random
import sys

import torch
import numpy

root_path = r'C:\Users\16230\Desktop\ABSA-PyTorch-master'
sys.path.append(root_path)
from dependency_graph import process
from models import ATAE_LSTM
from train import Instructor


def set_seed(seed):
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


# dataset
# lr
# dropout
# num_epoch
# batch_size
# patience
# seeds
# valset_ratio
# train_file
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='atae_lstm', type=str)
    parser.add_argument('--dataset', type=str, help='twitter, restaurant, laptop')
    parser.add_argument('--lr', type=float, help='try 5e-5, 2e-5 for BERT, 1e-3 for others')
    parser.add_argument('--dropout', type=float)
    parser.add_argument('--num_epoch', type=int, help='try larger number for non-BERT models')  # 训练次数
    parser.add_argument('--batch_size', type=int, help='try 16, 32, 64 for BERT models')
    parser.add_argument('--train_file', type=str,
                        help='set ratio between 0 and 1 for validation support')
    parser.add_argument('--seeds', type=list, nargs='+', help='set seed for reproducibility')
    parser.add_argument('--valset_ratio', type=float,
                        help='set ratio between 0 and 1 for validation support')  # 验证集比例
    parser.add_argument('--optimizer', default='adagrad', type=str)
    parser.add_argument('--initializer', default='xavier_uniform_', type=str)
    parser.add_argument('--l2reg', default=0.01, type=float)
    parser.add_argument('--log_step', default=10, type=int)  # 记录损失
    parser.add_argument('--embed_dim', default=300, type=int)  # 查询向量（嵌入层）的维度（超参数），将稀疏高维转向稠密低维
    parser.add_argument('--hidden_dim', default=300, type=int)  # 隐藏层维数，即隐藏层节点个数
    parser.add_argument('--bert_dim', default=768, type=int)
    parser.add_argument('--pretrained_bert_name', default='bert-base-uncased', type=str)
    parser.add_argument('--max_seq_len', default=85, type=int)
    parser.add_argument('--polarities_dim', default=3, type=int)
    parser.add_argument('--hops', default=3, type=int)
    parser.add_argument('--patience', default=5, type=int)  # 连续多少次损失不下降，即停止训练
    parser.add_argument('--device', default=None, type=str, help='e.g. cuda:0')

    opt = parser.parse_args()

    # set_seed(opt.seed)

    optimizers = {
        'adadelta': torch.optim.Adadelta,  # default lr=1.0
        'adagrad': torch.optim.Adagrad,  # default lr=0.01
        'adam': torch.optim.Adam,  # default lr=0.001
        'adamax': torch.optim.Adamax,  # default lr=0.002
        'asgd': torch.optim.ASGD,  # default lr=0.01
        'rmsprop': torch.optim.RMSprop,  # default lr=0.01
        'sgd': torch.optim.SGD,
    }

    opt.model_class = ATAE_LSTM
    dataset_files = {
        'twitter': {
            'train': opt.train_file,
            'test': root_path + '/datasets/acl-14-short-data/test.raw'
        },
        'restaurant': {
            'train': opt.train_file,
            'test': root_path + '/datasets/semeval14/Restaurants_Test_Gold.xml.seg'
        },
        'laptop': {
            'train': opt.train_file,
            'test': root_path + '/datasets/semeval14/Laptops_Test_Gold.xml.seg'
        }
    }
    opt.inputs_cols = ['text_indices', 'aspect_indices']
    opt.initializer = torch.nn.init.xavier_uniform_
    opt.optimizer = optimizers[opt.optimizer]
    opt.dataset_file = dataset_files[opt.dataset]
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
        if opt.device is None else torch.device(opt.device)

    # 训练邻接矩阵
    process(opt.train_file)
    json_f = {}
    for seed in iter(opt.seeds):
        seed = ''.join(seed)
        seed = int(seed)
        set_seed(seed)
        ins = Instructor(opt)
        test_acc, test_f1, epoch = ins.run()
        json_f[seed] = {'test_acc': test_acc, 'test_f1': test_f1, 'epoch': epoch}
    # 结果保存
    fout = open(opt.train_file + '.outcome.json', 'w')
    fout.write(json.dumps(json_f))
