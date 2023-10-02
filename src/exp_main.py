from src.data_factory import data_provider
from src.replay import ReplayBuffer
from src.mekf import MEKF
from src.MLP import Model
from src.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop
from src.metrics import metric, MSE, MAE, CORR

import torch
import torch.nn as nn
from torch import optim
import joblib

import os
import time
import random
import warnings
import matplotlib.pyplot as plt
import numpy as np
from  scipy import stats

warnings.filterwarnings('ignore')



def resample(samples, weights):
    '''systematic resampling'''
    # weight normalization
    w = np.array(weights)
    assert w.sum() > 0, 'all weights are zero'
    w /= w.sum()
    w = w.cumsum()
    M = len(samples)
    ptrs = (random.random() + np.arange(M)) / M
    new_samples = []
    i = 0
    j = 0
    while i < M:
        if ptrs[i] < w[j]:
            new_samples.append(samples[j])
            i += 1
        else:
            j += 1
    return np.asarray(new_samples)

class Exp_Main(object):
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)
    def _acquire_device(self):
        if self.args.use_gpu:
            #os.environ["CUDA_VISIBLE_DEVICES"] = str(
            #    self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _build_model(self):
        model = Model(self.args).float()
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x, batch_y = vali_data.normalize(batch_x, batch_y)

                # encoder - decoder
                outputs = self.model(batch_x)
                f_dim =  0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:]

                batch_x, batch_y,outputs = vali_data.denormalize(batch_x, batch_y,outputs)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.save_dir, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)

                batch_x,batch_y = train_data.normalize(batch_x,batch_y)

                # encoder - decoder
                outputs = self.model(batch_x)
                # print(outputs.shape,batch_y.shape)
                f_dim = 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            # if early_stopping.early_stop:
            # 	print("Early stopping")
            # 	break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = os.path.join(path,'checkpoint.pth')
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def infer_sample(self, batch_x, batch_y, return_feature=False):

        # encoder - decoder
        outputs = self.model(batch_x)
        feat = batch_x
        f_dim = 0
        # print(outputs.shape,batch_y.shape)
        outputs = outputs[:, -self.args.pred_len:, f_dim:]
        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
        if return_feature:
            return outputs, batch_y, batch_x, feat
        else:
            return outputs, batch_y, batch_x

    def adaptable_predict(self, setting, adapt_layers, load_path=None):
        self.args.batch_size = 1
        self.args.update_step = min(self.args.update_step, self.args.buffer_size)
        args = self.args

        if load_path is not None:
            print('loading model', load_path)
            self.model.load_state_dict(torch.load(load_path,map_location=self.device))

        replay_buffer = ReplayBuffer(max_size=self.args.buffer_size,  delay=self.args.update_step,device=self.device)
        adapt_weights = []
        for name, p in self.model.named_parameters():
            if name in adapt_layers:
                adapt_weights.append(p)
                print(name, p.size())

        if self.args.adapt == 'mekf':
            optimizer = MEKF(adapt_weights, dim_out=self.args.update_step * self.args.c_out,
                             p0=self.args.p0, eps=self.args.eps, lbd=self.args.lbd,)
        elif self.args.adapt == 'sgd':
            optimizer = torch.optim.SGD(adapt_weights, lr=self.args.lr)
        elif self.args.adapt == 'adam':
            optimizer = torch.optim.Adam(adapt_weights, lr=self.args.lr)
        else:
            optimizer = None

        preds = []
        trues = []
        inputx = []
        traj_err_list,traj_preds, traj_inputs, traj_trues =[],[],[],[]
        real_errs = []
        folder_path = os.path.join(self.args.save_dir, setting)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()

        i_iter = 0
        non_sims=0
        test_data, test_loader = self._get_data(flag='test')
        for i, (data_x, data_y, ) in enumerate(test_loader):
            data_x = data_x.float().to(self.device)
            data_y = data_y.float().to(self.device)
            data_x, data_y = test_data.normalize(data_x, data_y)


            data_rep = self.args.adapt != 'none' and len(replay_buffer.data) >= 1
            with torch.no_grad():
                outputs, batch_y, batch_x, batch_feat = self.infer_sample(data_x, data_y,
                                                                          return_feature=True)
            # filter out abnormal points
            prior_e = ((outputs - batch_y) ** 2).mean().data.item()
            if args.prior_thresh is not None:
                if prior_e > self.args.prior_thresh:
                    data_rep=False
                    print('anomly point', i)

            # feedforward compensation
            if data_rep:
                current_feature = batch_x
                features_rep, labels_rep, inp_trajs_rep, rewards = replay_buffer.get_sample(
                    size=1, current_feature=current_feature)
                cur_sample_sims = replay_buffer.cur_sample_sims

                fit_err = 0
                for i_fit in range( args.max_fit_iter):
                    outputs_rep, y_rep, x_rep, feat_rep = self.infer_sample(inp_trajs_rep, labels_rep, return_feature=True)

                    #print(cur_sample_sims[0].item())
                    if cur_sample_sims[0] > self.args.sim_update_thresh:
                        y_adapt_rep = y_rep[:, :self.args.update_step].contiguous().view((len(y_rep), -1))
                        y_hat_adapt_rep = outputs_rep[:, :self.args.update_step].contiguous().view((len(y_rep), -1))
                        err_adapt_rep = (y_adapt_rep - y_hat_adapt_rep).detach()
                        fit_err = (y_adapt_rep - y_hat_adapt_rep).pow(2).mean().item()

                        if self.args.adapt in ['sgd','adam']:
                            optimizer.zero_grad()
                            loss_rep = torch.nn.functional.mse_loss(y_hat_adapt_rep, y_adapt_rep)
                            loss_rep.backward()
                            optimizer.step()
                        elif self.args.adapt in ['mekf']:
                            y_adapt_i = y_adapt_rep.view((-1, 1))
                            y_hat_adapt_i = y_hat_adapt_rep.view((-1, 1))
                            err_adapt_i = err_adapt_rep.view((-1, 1))
                            def nrls_closure_replay(index=0):
                                optimizer.zero_grad()
                                dim_out = optimizer.state['dim_out']
                                retain = index < dim_out - 1
                                y_hat_adapt_i[index].backward(retain_graph=retain)
                                return err_adapt_i
                            optimizer.step(nrls_closure_replay)

                        else:
                            fit_err = 0


            #  prediction
            self.model.eval()
            with torch.no_grad():
                outputs, batch_y, batch_x, batch_feat = self.infer_sample(data_x, data_y,  return_feature=True)
                prior_e = ((outputs - batch_y) ** 2).mean().data
            preds.append(outputs.detach().cpu().numpy())
            trues.append(batch_y.detach().cpu().numpy())
            inputx.append(batch_x.detach().cpu().numpy())

            real_errs.append(torch.abs(outputs - batch_y).detach().cpu().numpy())

            # save buffer
            if args.adapt !='none':
                feat = data_x
                replay_buffer.push(feat, data_y, data_x, reward=[i_iter] * len(feat),task=0)

            traj_x, traj_y, traj_outputs = test_data.denormalize(batch_x, batch_y, outputs)
            traj_preds.append(traj_outputs.detach().cpu().numpy())
            traj_trues.append(traj_y.detach().cpu().numpy())
            traj_inputs.append(traj_x.detach().cpu().numpy())
            traj_err_list.append(((traj_outputs - traj_y) ** 2).mean().item())

            #self.model.train()
            if i_iter % 10 == 0:
                print(f'step:{i}, pred_error:{prior_e},')

        print('non_sims:', non_sims)
        preds =np.concatenate(preds,axis=0) #np.array(preds)
        trues = np.concatenate(trues,axis=0)
        inputx = np.concatenate(inputx,axis=0)
        real_errs = np.concatenate(real_errs,axis=0)

        traj_preds = np.concatenate(traj_preds, axis=0)
        traj_trues = np.concatenate(traj_trues, axis=0)
        traj_inputs = np.concatenate(traj_inputs, axis=0)


        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
        print('mse:{}, mae:{}, rmse:{}'.format(mse, mae,rmse))
        traj_mae, traj_mse, traj_rmse, traj_mape, traj_mspe, traj_rse, traj_corr = metric(traj_preds, traj_trues)
        print('traj_mse:{}, traj_mae:{}, traj_rmse:{}'.format(traj_mse, traj_mae, traj_rmse))

        f = open(os.path.join(folder_path, "result.txt"), 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, rmse:{}, rse:{}, corr:{}'.format(mse, mae,rmse, rse, corr))
        f.write('\n')
        f.write('traj_mse:{}, traj_mae:{}, traj_rmse:{}, traj_rse:{}, traj_corr:{}'.format(traj_mse, traj_mae,traj_rmse, traj_rse, traj_corr))
        f.write('\n')
        f.write('\n')
        f.close()

        save_dict = {'pred': preds, 'label': trues, 'input': inputx,
                     'metrics': np.array([mae, mse, rmse, mape, mspe, rse]),
                     'real_errs': real_errs,
                     'traj_pred':traj_preds, 'traj_true':traj_trues, 'traj_input':traj_inputs,'traj_err_list':traj_err_list,}
        #joblib.dump(save_dict, os.path.join(folder_path, "result.pkl"))

        return save_dict
