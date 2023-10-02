import argparse
import os
import json
import torch
from src.exp_main import Exp_Main
import random
import numpy as np

fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

def get_defaults(args):
	if  args.data =='exchange':
		args.data_path = 'exchange_rate.csv'
		args.seq_len = 96
		args.label_len = 48
		args.pred_len = 192
		args.c_out = 8
	elif  args.data =='etth1':
		args.data_path = 'ETTh1.csv'
		args.seq_len = 96
		args.label_len = 48
		args.pred_len = 192
		args.c_out = 1
	elif args.data == 'ill':
		args.data_path = 'national_illness.csv'
		args.seq_len = 36
		args.label_len = 18
		args.pred_len = 36
		args.c_out = 7
	return args

def get_args():
	parser = argparse.ArgumentParser(description='Training on Time Series Prediction')

	# data loader
	parser.add_argument('--data', type=str, default='etth1', help='dataset type') # 'etth1,'ill,'exchange'
	parser.add_argument('--root_path', type=str, default='./dataset/', help='root path of the data file')
	parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
	parser.add_argument('--save_dir', type=str, default='./results', help='save_dir')

	# forecasting task
	parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
	parser.add_argument('--label_len', type=int, default=48, help='start token length')
	parser.add_argument('--pred_len', type=int, default=192, help='prediction sequence length')
	# model
	parser.add_argument('--d_model', type=int, default=128, help='hidden dimension')
	# optimization
	parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
	parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
	parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
	parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
	parser.add_argument('--learning_rate', type=float, default=0.001, help='optimizer learning rate')
	parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')

	# GPU
	parser.add_argument('--use_gpu', type=int, default=1, help='use gpu')  # True
	parser.add_argument('--gpu', type=int, default=0, help='gpu')

	args = parser.parse_args()
	return args

def training(args):
	args = get_defaults(args)
	print('Args in experiment:')
	print(args)
	Exp = Exp_Main
	# setting record of experiments
	setting = '{}/MLP_sl{}_ll{}_pl{}'.format(
		args.data,
		args.seq_len,
		args.label_len,
		args.pred_len)
	os.makedirs(os.path.join(args.save_dir, setting), exist_ok=True)
	with open(os.path.join(args.save_dir, setting, 'args.json'), 'w') as f:
		json.dump(args.__dict__, f)
	exp = Exp(args)  # set experiments
	print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
	exp.train(setting)

if __name__=='__main__':
	args = get_args()
	training(args)