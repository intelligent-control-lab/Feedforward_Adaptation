import os
import numpy as np
import pandas as pd
import os
import joblib
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')


class Dataset_ETT_hour(Dataset):
	def __init__(self, root_path, flag='train', size=None,
	             features='S', data_path='ETTh1.csv',
	             target='OT', scale=True,):
		# size [seq_len, label_len, pred_len]
		# info
		if size == None:
			self.seq_len = 24 * 4 * 4
			self.label_len = 24 * 4
			self.pred_len = 24 * 4
		else:
			self.seq_len = size[0]
			self.label_len = size[1]
			self.pred_len = size[2]
		# init
		assert flag in ['train', 'test', 'val']
		type_map = {'train': 0, 'val': 1, 'test': 2}
		self.set_type = type_map[flag]

		self.features = features
		self.target = target
		self.scale = scale

		self.root_path = root_path
		self.data_path = data_path
		self.__read_data__()

	def __read_data__(self):
		self.scaler = StandardScaler()
		df_raw = pd.read_csv(os.path.join(self.root_path,
		                                  self.data_path))

		border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
		border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
		border1 = border1s[self.set_type]
		border2 = border2s[self.set_type]

		if self.features == 'M' or self.features == 'MS':
			cols_data = df_raw.columns[1:]
			df_data = df_raw[cols_data]
		elif self.features == 'S':
			df_data = df_raw[[self.target]]

		if self.scale:
			train_data = df_data[border1s[0]:border2s[0]]
			self.scaler.fit(train_data.values)
			data = self.scaler.transform(df_data.values)
		else:
			data = df_data.values

		df_stamp = df_raw[['date']][border1:border2]
		df_stamp['date'] = pd.to_datetime(df_stamp.date)

		self.data_x = data[border1:border2]
		self.data_y = data[border1:border2]

	def __getitem__(self, index):
		s_begin = index
		s_end = s_begin + self.seq_len
		r_begin = s_end - self.label_len
		r_end = r_begin + self.label_len + self.pred_len

		seq_x = self.data_x[s_begin:s_end]
		seq_y = self.data_y[r_begin:r_end]

		return seq_x, seq_y

	def __len__(self):
		return len(self.data_x) - self.seq_len - self.pred_len + 1

	def inverse_transform(self, data):
		return self.scaler.inverse_transform(data)

	def normalize(self,data_x,data_y):
		return data_x,data_y

	def denormalize(self,data_x,data_y, output, u= None):
		if u is not None:
			return data_x,data_y, output, u
		return data_x,data_y, output


class Dataset_Custom(Dataset):
	def __init__(self, root_path, flag='train', size=None,
	             features='M', data_path='national_illness.csv',
	             target='OT', scale=True):
		# size [seq_len, label_len, pred_len]
		# info
		if size == None:
			self.seq_len = 24 * 4 * 4
			self.label_len = 24 * 4
			self.pred_len = 24 * 4
		else:
			self.seq_len = size[0]
			self.label_len = size[1]
			self.pred_len = size[2]
		# init
		assert flag in ['train', 'test', 'val']
		type_map = {'train': 0, 'val': 1, 'test': 2}
		self.set_type = type_map[flag]

		self.features = features
		self.target = target
		self.scale = scale

		self.root_path = root_path
		self.data_path = data_path
		self.__read_data__()

	def __read_data__(self):
		self.scaler = StandardScaler()
		df_raw = pd.read_csv(os.path.join(self.root_path,
		                                  self.data_path))

		'''
		df_raw.columns: ['date', ...(other features), target feature]
		'''
		cols = list(df_raw.columns)
		cols.remove(self.target)
		cols.remove('date')
		df_raw = df_raw[['date'] + cols + [self.target]]
		# print(cols)
		num_train = int(len(df_raw) * 0.7)
		num_test = int(len(df_raw) * 0.2)
		num_vali = len(df_raw) - num_train - num_test
		border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
		border2s = [num_train, num_train + num_vali, len(df_raw)]
		border1 = border1s[self.set_type]
		border2 = border2s[self.set_type]

		if self.features == 'M' or self.features == 'MS':
			cols_data = df_raw.columns[1:]
			df_data = df_raw[cols_data]
		elif self.features == 'S':
			df_data = df_raw[[self.target]]

		if self.scale:
			train_data = df_data[border1s[0]:border2s[0]]
			self.scaler.fit(train_data.values)
			# print(self.scaler.mean_)
			# exit()
			data = self.scaler.transform(df_data.values)
		else:
			data = df_data.values

		self.data_x = data[border1:border2]
		self.data_y = data[border1:border2]
		vel_x = np.zeros_like(self.data_x)
		vel_x[1:] = self.data_x[1:] - self.data_x[:-1]
		vel_y = np.zeros_like(self.data_y)
		vel_y[1:] = self.data_y[1:] - self.data_y[:-1]
		self.vel_x = vel_x
		self.vel_y = vel_y
		diff = data[border1+1:border2,] - data[border1:border2-1,]
		self.vel_mean = np.mean(diff, axis=0)
		self.vel_std = np.std(diff, axis=0)

	def __getitem__(self, index):
		s_begin = index
		s_end = s_begin + self.seq_len
		r_begin = s_end - self.label_len
		r_end = r_begin + self.label_len + self.pred_len

		seq_x = self.data_x[s_begin:s_end]
		seq_y = self.data_y[r_begin:r_end]

		return seq_x, seq_y

	def __len__(self):
		return len(self.data_x) - self.seq_len - self.pred_len + 1

	def inverse_transform(self, data):
		return self.scaler.inverse_transform(data)

	def normalize(self,data_x,data_y):
		return data_x,data_y

	def denormalize(self,data_x,data_y, output, u= None):
		if u is not None:
			return data_x,data_y, output, u
		return data_x,data_y, output
