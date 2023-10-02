import random
import torch
import math
import numpy as np

def mse_sim(t1, t2):
    dist = torch.sqrt(((t1 - t2)**2).mean(axis=1))
    #sim =  1 - dist
    sim = 1/(1+dist) # torch.exp(-dist)
    return sim




class ReplayBuffer(): # push_mode = time_simple time_decay
    def __init__(self, max_size=1000, push_mode='time_decay',pull_mode='feature_sim', sim_func='mse', sim_thresh=None, delay=1, device='cpu'): # random, task_random, task_reward
        self.max_size = max_size
        self.push_mode=push_mode
        self.pull_mode = pull_mode
        self.delay = delay
        self.data = []
        self.label = []
        self.inp_traj = []
        self.rewards = []
        self.tasks = []
        self.ids = []
        self.cur_sample_sims=[-1]
        self.sim_thresh = sim_thresh
        self.device = device

        self.delay_buffer = []
        if sim_func=='cos':
            self.sim_func = torch.nn.CosineSimilarity(dim=1, eps=1e-08)
        elif sim_func=='mse':
            self.sim_func = mse_sim
        else:
            raise NotImplementedError


    def process_delay_buffer(self,feature,label,inp_traj, reward, task):
        feature = feature.cpu().data
        label = label.cpu().data
        inp_traj=inp_traj.cpu().data
        current_step = reward[0]
        self.delay_buffer.append([feature,label, inp_traj, reward, task, current_step])
        if len(self.delay_buffer) >= self.delay:
            [feature, label,inp_traj, reward, task, current_step] = self.delay_buffer[0]
            self.delay_buffer = self.delay_buffer[1:]
            ret_data = [feature,label,inp_traj, reward, task]
        else:
            ret_data = None
        return ret_data

    def del_buffer_items(self,del_indices):
        L = len(self.data)
        self.data = [self.data[ii] for ii in range(L) if ii not in del_indices]
        self.label = [self.label[ii] for ii in range(L) if ii not in del_indices]
        self.inp_traj = [self.inp_traj[ii] for ii in range(L) if ii not in del_indices]
        self.rewards = [self.rewards[ii] for ii in range(L) if ii not in del_indices]
        self.tasks = [self.tasks[ii] for ii in range(L) if ii not in del_indices]
        self.ids = [self.ids[ii] for ii in range(L) if ii not in del_indices]


    def push(self, feature, label, inp_traj, reward, task):
        ''' Update data into cached dynamic memory
            data[Tensor]: [N, 22, 64, 64]
            data[Tensor]: [N, 22, 1]
        '''
        extra_size=100

        ret_data = self.process_delay_buffer(feature,label, inp_traj, reward, task)
        if ret_data is None:
            return

        feature, label,inp_traj, reward, task = ret_data
        if self.push_mode in ['random']:
            buf_size = min(len(feature), self.max_size)
            if buf_size <= self.max_size - len(self.data)+extra_size:
                old_indices = []
            else:
                old_indices = np.random.choice(len(self.data), buf_size - (self.max_size - len(self.data)),
                                               replace=False)
                self.del_buffer_items(old_indices)
                old_indices = []
            new_indices = np.random.choice(len(feature), buf_size, replace=False)
        elif self.push_mode in ['time_decay']: # remove very old features
            buf_size = min(len(feature), self.max_size)
            if buf_size <= self.max_size - len(self.data)+extra_size:
                old_indices = []
            else:
                old_len = buf_size - (self.max_size - len(self.data))
                time_rewards = torch.tensor(self.rewards)
                val, indices = torch.topk(time_rewards, old_len, dim=0, largest=False)
                old_indices = indices.cpu().numpy()
                self.del_buffer_items(old_indices)
                old_indices = []
            new_indices = np.random.choice(len(feature), buf_size, replace=False)
        elif self.push_mode in ['time_simple']: # remove very old features
            buf_size = min(len(feature), self.max_size)
            if buf_size <= self.max_size - len(self.data)+extra_size:
                old_indices = []
            else:
                old_len = buf_size - (self.max_size - len(self.data))
                old_indices = np.arange(len(self.data)-old_len, len(self.data))
                #time_rewards = torch.tensor(self.rewards)
                self.del_buffer_items(old_indices)
                old_indices = []
            new_indices = np.random.choice(len(feature), buf_size, replace=False)
        else:
            raise NotImplementedError
        for i in range(len(new_indices)):
            new_i = new_indices[i]
            if i < len(old_indices):
                old_i = old_indices[i]
                self.data[old_i] = torch.unsqueeze(feature[new_i], 0)
                self.label[old_i] = torch.unsqueeze(label[new_i], 0)
                self.inp_traj[old_i] = torch.unsqueeze(inp_traj[new_i], 0)
                self.rewards[old_i] = reward[new_i]
                self.tasks[old_i] = task
                self.ids[old_i] = new_i
            else:
                self.data.append(torch.unsqueeze(feature[new_i], 0))
                self.label.append(torch.unsqueeze(label[new_i], 0))
                self.inp_traj.append(torch.unsqueeze(inp_traj[new_i], 0))
                self.rewards.append(reward[new_i])
                self.tasks.append(task)
                self.ids.append(new_i)

    def get_sample_indices(self, size,current_feature=None):
        if self.pull_mode in ['feature_sim']:
            assert current_feature is not None
            current_feature=current_feature.cpu()
            features = torch.cat(self.data)
            if len(features.size())>2:
                features = features.view(len(features),-1)
                current_feature = current_feature.view(len(current_feature), -1)
            sims = self.sim_func(features,current_feature)
            val,indices = torch.topk(sims, size, dim=0, largest=True)
            if self.sim_thresh is not None:
                new_indices = []
                for ind in indices:
                    if sims[ind] >= self.sim_thresh:
                        new_indices.append(ind)
                indices = new_indices
            self.cur_sample_sims = sims[indices].data.numpy()
        elif self.pull_mode in ['time_decay']:
            time_rewards = torch.tensor(self.rewards)
            val, indices = torch.topk(time_rewards, size, dim=0, largest=True)
            indices = indices.cpu().numpy()
        elif self.pull_mode in ['random']:
            if size<=len(self.data):
                indices = np.random.choice(len(self.data), size=size, replace=False)
            else:
                indices = np.random.choice(len(self.data), size=size, replace=True)
        else:
            raise NotImplementedError
        return indices

    def get_sample(self, size=1, current_feature=None):
        indices =self.get_sample_indices(size,current_feature)

        self.indices = indices
        features = []
        labels = []
        inp_trajs=[]
        rewards = []
        for cur_index in indices:
            features.append(self.data[cur_index])
            labels.append(self.label[cur_index])
            inp_trajs.append(self.inp_traj[cur_index])
            rewards.append(self.rewards[cur_index])
        features = torch.squeeze(torch.stack(features, 0), 1).to(self.device)
        labels = torch.squeeze(torch.stack(labels, 0), 1).to(self.device)
        inp_trajs = torch.squeeze(torch.stack(inp_trajs, 0), 1).to(self.device)

        return features, labels,inp_trajs, rewards

