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
	parser = argparse.ArgumentParser(description='Adaptation on Time Series Prediction')

	# data loader
	parser.add_argument('--data', type=str, default='etth1', help='dataset type') # 'etth1,'ill,'exchange'
	parser.add_argument('--root_path', type=str, default='./dataset/', help='root path of the data file')
	parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
	parser.add_argument('--save_dir', type=str, default='./results', help='save_dir')

	# forecasting task
	parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
	parser.add_argument('--label_len', type=int, default=48, help='start token length')
	parser.add_argument('--pred_len', type=int, default=192, help='prediction sequence length')  # 96
	# adaptation
	parser.add_argument("--adapt_layer", default='norm', type=str,choices=['all', 'norm']) # norm
	parser.add_argument("--adapt", default='sgd', type=str, choices=['none', 'sgd', 'mekf'])
	parser.add_argument("--update_step", default=10, type=int) # 5
	parser.add_argument("--lr", default=0.1, type=float)  # SGD lr
	parser.add_argument("--p0", default=0.1, type=float) #  mekf (modified EKF) parameters
	parser.add_argument("--eps", default=1e-5, type=float) #  mekf (modified EKF) parameters
	parser.add_argument("--lbd", default=1, type=float) #  mekf (modified EKF) parameters
	parser.add_argument("--buffer_size", default=1000, type=int) #1000
	parser.add_argument("--sim_update_thresh", default=0.2, type=float)  # using most similar data to update the model
	parser.add_argument("--prior_thresh", default=None, type=float)
	parser.add_argument("--max_fit_iter", default=1, type=float)

	# model
	parser.add_argument('--d_model', type=int, default=128, help='hidden dimension')  # 10
	# optimization
	parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')  # 10
	# GPU
	parser.add_argument('--use_gpu', type=int, default=1, help='use gpu')  # True
	parser.add_argument('--gpu', type=int, default=0, help='gpu')

	args = parser.parse_args()
	return args

def adaptation(args):
	args =  get_defaults(args)
	print('Args in experiment:')
	print(args)
	Exp = Exp_Main
	setting = '{}/MLP_sl{}_ll{}_pl{}'.format(
		args.data,
		args.seq_len,
		args.label_len,
		args.pred_len)

	adapt_setting = os.path.join(setting, 'adaptation/adapt{}/adapt{}_st{}_sgd{}'.format(
		args.adapt,args.adapt_layer, args.update_step,args.lr))
	os.makedirs(os.path.join(args.save_dir, adapt_setting), exist_ok=True)
	with open(os.path.join(args.save_dir, adapt_setting, 'args.json'), 'w') as f:
		json.dump(args.__dict__, f)

	exp = Exp(args)  # set experiments
	adapt_layers = []
	all_weight_cnt, adapt_weight_cnt = 0, 0
	for name, p in exp.model.named_parameters():
		if args.adapt_layer == 'norm':
			if 'norm' in name : # or 'decoder.' in name
				adapt_layers.append(name)
				adapt_weight_cnt += p.data.nelement()
		elif args.adapt_layer == 'all':
			adapt_layers.append(name)
			adapt_weight_cnt += p.data.nelement()
		else:
			p.required_grad=False
		all_weight_cnt += p.data.nelement()
	param_num = all_weight_cnt / 1024. / 1024.
	adapt_num = adapt_weight_cnt / 1024. / 1024.
	print("Number of model parameters: {} M, adaptable parameters: {} M".format(param_num, adapt_num))

	load_path = os.path.join(args.save_dir, setting, 'checkpoint.pth')
	print('>>>>>>>adaptable prediction : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
	exp.adaptable_predict(adapt_setting, adapt_layers=adapt_layers, load_path=load_path)

if __name__=='__main__':
	args = get_args()
	adaptation(args)