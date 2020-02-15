from torch.distributions.categorical import Categorical
import torch
import torch.nn as nn
import torch.nn.functional as F
from env_FindGoals import EnvFindGoals
import matplotlib.pyplot as plt
from utils import *

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class Actor(nn.Module):
    def __init__(self, N_action):
        super(Actor, self).__init__()
        self.N_action = N_action
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1)
        self.flat1 = Flatten()
        self.fc1 = nn.Linear(16, 32)
        self.fc2 = nn.Linear(32, self.N_action)

    def get_action(self, h):
        h1 = F.relu(self.conv1(h))
        h1 = self.flat1(h1)
        h1 = F.relu(self.fc1(h1))
        h = F.softmax(self.fc2(h1), dim=1)
        m = Categorical(h.squeeze(0))
        return m.sample().item(), h

class Critic(nn.Module):
    def __init__(self, N_action):
        super(Critic, self).__init__()
        self.N_action = N_action
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1)
        self.flat1 = Flatten()
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1)
        self.flat2 = Flatten()
        self.fc1 = nn.Linear(32, 64)
        self.fc2 = nn.Linear(64, N_action*N_action)
        self.fc1_1 = nn.Linear(16, 64)
        self.fc1_2 = nn.Linear(64, 1)
        self.fc2_1 = nn.Linear(16, 64)
        self.fc2_2 = nn.Linear(64, 1)

    def get_value(self, s1, s2):
        h1 = F.relu(self.conv1(s1))
        h1 = self.flat1(h1)
        h2 = F.relu(self.conv2(s2))
        h2 = self.flat2(h2)
        h = torch.cat([h1, h2], 1)
        x = F.relu(self.fc1(h))
        x = self.fc2(x)
        return x
        
    def get_value_1(self,s1):
        h1 = F.relu(self.conv1(s1))
        h1 = self.flat1(h1)
        x = F.relu(self.fc1_1(h1))
        x = self.fc1_2(x)
        return x
    
    def get_value_2(self,s2):
        h2 = F.relu(self.conv2(s2))
        h2 = self.flat2(h2)
        x = F.relu(self.fc2_1(h2))
        x = self.fc2_2(x)
        return x
        
    
class RNDModel(nn.Module):
    def __init__(self, output_size=64):
        super(RNDModel, self).__init__()
        self.output_size = output_size
        self.conv1_tar = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1)
        self.flat1_tar = Flatten()
        self.fc1_tar = nn.Linear(16, 64)
        self.fc2_tar = nn.Linear(64, output_size)
        self.conv1_pred = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1)
        self.flat1_pred = Flatten()
        self.fc1_pred = nn.Linear(16, 64)
        self.fc2_pred = nn.Linear(64, output_size)
        
    def target_forward(self, next_obs):
        #target_feature = self.target(next_obs)
        #predict_feature = self.predictor(next_obs)
        h1 = F.relu(self.conv1_tar(next_obs))
        h1 = self.flat1_tar(h1)
        x = F.relu(self.fc1_tar(h1))
        x = self.fc2_tar(x)
        return x
    
    def predictor_forward(self, next_obs):
        #target_feature = self.target(next_obs)
        #predict_feature = self.predictor(next_obs)
        h1 = F.relu(self.conv1_pred(next_obs))
        h1 = self.flat1_pred(h1)
        x = F.relu(self.fc1_pred(h1))
        x = self.fc2_pred(x)
        return x
        
class COMA(object):
    def __init__(self, N_action):
        self.N_action = N_action
        self.actor1 = Actor(self.N_action)
        self.actor2 = Actor(self.N_action)
        self.critic = Critic(self.N_action)
        #1st version shared RND network, 2nd version serperate for each network
        self.rnd1 = RNDModel(64)
        self.rnd2 = RNDModel(64)
        self.gamma = 0.95
        self.c_loss_fn = torch.nn.MSELoss()
        self.use_cuda = False
        self.device = torch.device('cuda' if self.use_cuda else 'cpu') #added later

    def get_action(self, obs1, obs2):
        action1, pi_a1 = self.actor1.get_action(self.img_to_tensor(obs1).unsqueeze(0))
        action2, pi_a2 = self.actor2.get_action(self.img_to_tensor(obs2).unsqueeze(0))
        v1 = self.critic.get_value_1(self.img_to_tensor(obs1).unsqueeze(0))
        v2 = self.critic.get_value_2(self.img_to_tensor(obs2).unsqueeze(0))
        return action1, pi_a1, action2, pi_a2, v1, v2

    def img_to_tensor(self, img):
        img_tensor = torch.FloatTensor(img)
        img_tensor = img_tensor.permute(2, 0, 1)
        return img_tensor
    
    def compute_intrinsic_reward1(self, obs):
        obs = torch.FloatTensor(obs).to(self.device)

        target_next_feature = self.rnd1.target_forward(self.img_to_tensor(obs).unsqueeze(0))
        predict_next_feature = self.rnd1.predictor_forward(self.img_to_tensor(obs).unsqueeze(0))
        intrinsic_reward = (target_next_feature - predict_next_feature).pow(2).sum(1) / 2

        return intrinsic_reward.data.cpu().numpy()
    
    def compute_intrinsic_reward2(self, obs):
        obs = torch.FloatTensor(obs).to(self.device)

        target_next_feature = self.rnd2.target_forward(self.img_to_tensor(obs).unsqueeze(0))
        predict_next_feature = self.rnd2.predictor_forward(self.img_to_tensor(obs).unsqueeze(0))
        intrinsic_reward = (target_next_feature - predict_next_feature).pow(2).sum(1) / 2

        return intrinsic_reward.data.cpu().numpy()
        

    def cross_prod(self, pi_a1, pi_a2):
        new_pi = torch.zeros(1, self.N_action*self.N_action)
        for i in range(self.N_action):
            for j in range(self.N_action):
                new_pi[0, i*self.N_action+j] = pi_a1[0, i]*pi_a2[0, j]
        return new_pi

    def train(self, o1_list, a1_list, pi_a1_list, o2_list, a2_list, pi_a2_list, r_list,adv1_int,adv2_int):
        a1_optimizer = torch.optim.Adam(self.actor1.parameters(), lr=3e-4)
        a2_optimizer = torch.optim.Adam(self.actor2.parameters(), lr=3e-4)
        c_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-3)
        rnd1_optimizer = torch.optim.Adam(self.rnd1.parameters(), lr=1e-3)
        rnd2_optimizer = torch.optim.Adam(self.rnd2.parameters(), lr=1e-3)
        
        #obs_for_rnd1 = torch.from_numpy(o1_list[0])
        #obs_for_rnd2 = torch.from_numpy(o2_list[0])

        T = len(r_list)
        obs1 = self.img_to_tensor(o1_list[0]).unsqueeze(0)
        obs2 = self.img_to_tensor(o2_list[0]).unsqueeze(0)
        
        obs_for_rnd1 = self.img_to_tensor(o1_list[0]).unsqueeze(0)
        obs_for_rnd2 = self.img_to_tensor(o2_list[0]).unsqueeze(0)
        
        for t in range(1, T):
            temp_obs1 = self.img_to_tensor(o1_list[t]).unsqueeze(0)
            obs1 = torch.cat([obs1, temp_obs1], dim=0)
            temp_obs2 = self.img_to_tensor(o2_list[t]).unsqueeze(0)
            obs2 = torch.cat([obs2, temp_obs2], dim=0)
        Q = self.critic.get_value(obs1, obs2)
        Q_est = Q.clone()
        for t in range(T - 1):
            a_index = a1_list[t]*self.N_action + a2_list[t]
            Q_est[t][a_index] = r_list[t] + self.gamma * torch.sum(self.cross_prod(pi_a1_list[t+1], pi_a2_list[t+1])*Q_est[t+1, :])
        a_index = a1_list[T - 1] * self.N_action + a2_list[T - 1]
        Q_est[T - 1][a_index] = r_list[T - 1]
        c_loss = self.c_loss_fn(Q, Q_est.detach())
        c_optimizer.zero_grad()
        c_loss.backward()
        c_optimizer.step()

        A1_list = []
        for t in range(T):
            temp_Q1 = torch.zeros(1, self.N_action)
            for a1 in range(self.N_action):
                temp_Q1[0, a1] = Q[t][a1*self.N_action + a2_list[t]]
            a_index = a1_list[t] * self.N_action + a2_list[t]
            temp_A1 = Q[t, a_index] - torch.sum(pi_a1_list[t]*temp_Q1)
            A1_list.append(temp_A1)

        A2_list = []
        for t in range(T):
            temp_Q2 = torch.zeros(1, self.N_action)
            for a2 in range(self.N_action):
                temp_Q2[0, a2] = Q[t][a1_list[t] * self.N_action + a2]
            a_index = a1_list[t] * self.N_action + a2_list[t]
            temp_A2 = Q[t, a_index] - torch.sum(pi_a2_list[t] * temp_Q2)
            A2_list.append(temp_A2)

        a1_loss = torch.FloatTensor([0.0])
        for t in range(T):
            a1_loss = a1_loss + A1_list[t].item() * torch.log(pi_a1_list[t][0, a1_list[t]])
        
        a1_loss = -a1_loss / T + np.mean(adv1_int)
        #a1_loss =np.mean(adv1_int)
        a1_optimizer.zero_grad()
        a1_loss.backward()
        a1_optimizer.step()

        a2_loss = torch.FloatTensor([0.0])
        for t in range(T):
            a2_loss = a2_loss + A2_list[t].item() * torch.log(pi_a2_list[t][0, a2_list[t]])
        a2_loss = -a2_loss / T + np.mean(adv2_int)
        #a2_loss = np.mean(adv2_int)
        a2_optimizer.zero_grad()
        a2_loss.backward()
        a2_optimizer.step()
        
        r1_loss = self.c_loss_fn(self.rnd1.target_forward(obs_for_rnd1).detach(),self.rnd1.predictor_forward(obs_for_rnd1))
        rnd1_optimizer.zero_grad()
        r1_loss.backward()
        rnd1_optimizer.step()
        
        r2_loss = self.c_loss_fn(self.rnd2.target_forward(obs_for_rnd2).detach(),self.rnd2.predictor_forward(obs_for_rnd2))
        rnd2_optimizer.zero_grad()
        r2_loss.backward()
        rnd2_optimizer.step()


    
def game(mode):
    torch.set_num_threads(1)
    env = EnvFindGoals()
    max_epi_iter = 10
    max_MC_iter = 200
    agent = COMA(N_action=5)
    train_curve = []
    warm_up_run = 200
    obs1_norm = RunningMeanStd()
    obs2_norm = RunningMeanStd()
    reward_rms1 = RunningMeanStd()
    reward_rms2 = RunningMeanStd()
    discounted_reward1 = RewardForwardFilter(gamma=0.999)
    discounted_reward2 = RewardForwardFilter(gamma=0.999)
    env.reset()
    o1_list = []
    o2_list = []
    #Normalize Obs for each agent
    for i in range (warm_up_run):
        obs1 = env.get_agt1_obs()
        obs2 = env.get_agt2_obs()
        o1_list.append(obs1)
        o2_list.append(obs2)
    obs1_norm.update(o1_list)
    obs2_norm.update(o2_list)
    
    for epi_iter in range(max_epi_iter):
        env.reset()
        o1_list = []
        a1_list = []
        pi_a1_list = []
        o2_list = []
        a2_list = []
        pi_a2_list = []
        r_list = []
        r1_int_list = []
        r2_int_list = []
        total_int_rwd1_list = []
        total_int_rwd2_list = []
        adv1_int = []
        adv2_int = []
        dones_list = []
        v1_list,v2_list=[],[]
        acc_r = 0
        
        for MC_iter in range(max_MC_iter):
            #env.render()
            obs1 = env.get_agt1_obs()
            obs2 = env.get_agt2_obs()
            o1_list.append(obs1)
            o2_list.append(obs2)
            #print("obs1 =",obs1)
            #print("intrinsic reward =",agent.compute_intrinsic_reward1(obs1))
            action1, pi_a1, action2, pi_a2,v1,v2 = agent.get_action(((obs1-obs1_norm.mean)/obs1_norm.var).clip(-5, 5), ((obs2-obs2_norm.mean)/obs2_norm.var).clip(-5, 5))
            int_rwd1 = agent.compute_intrinsic_reward1(((obs1-obs1_norm.mean)/obs1_norm.var).clip(-5, 5))
            int_rwd2 = agent.compute_intrinsic_reward2(((obs2-obs2_norm.mean)/obs2_norm.var).clip(-5, 5))
            #print("int_rwd1 =",int_rwd1)
            #print("int_rwd2 =",int_rwd2)
            v1_list.append(v1.data.numpy()[0])
            v2_list.append(v2.data.numpy()[0])
            #print(v1.data.numpy()[0])
            r1_int_list.append(int_rwd1)
            r2_int_list.append(int_rwd2)
            a1_list.append(action1)
            pi_a1_list.append(pi_a1)
            a2_list.append(action2)
            pi_a2_list.append(pi_a2)
            [reward_1, reward_2], done = env.step([action1, action2])
            #print(reward_1, reward_2)
            if done == False:
                dones_list.append(0)
            else:
                dones_list.append(1)
            acc_r = acc_r + reward_1
            r_list.append(reward_1)
            ###Test using reward_2 instead
            #acc_r = acc_r + reward_2
            #r_list.append(reward_2)
            
            if done:
                break
        obs1_norm.update(o1_list)
        obs2_norm.update(o2_list)
        """
        for i in reversed(r1_int_list):
            r1_int_temp = discounted_reward1.update(i)
        for j in reversed(r2_int_list):
            r2_int_temp = discounted_reward2.update(j)
        """
        #print("mean =",reward_rms1.mean)
        mean1, std1, count1 = np.mean(r1_int_list), np.std(r1_int_list), len(r1_int_list)
        mean2, std2, count2 = np.mean(r2_int_list), np.std(r2_int_list), len(r2_int_list)
        #mean1, std1, count1 = np.mean(r1_int_temp), np.std(r1_int_temp), len(r1_int_temp)
        #mean2, std2, count2 = np.mean(r2_int_temp), np.std(r2_int_temp), len(r2_int_temp)
        reward_rms1.update_from_moments(mean1, std1 ** 2, count1)
        reward_rms2.update_from_moments(mean2, std2 ** 2, count2)
        #print("var =",reward_rms1.var)
        #adv1_int = (r1_int_list-reward_rms1.mean)/np.sqrt(reward_rms1.var)
        #adv2_int = (r2_int_list-reward_rms2.mean)/np.sqrt(reward_rms2.var)
        #May be not correct way of approx advantages
        
        total_int_rwd1_list = r1_int_list/np.sqrt(reward_rms1.var)
        total_int_rwd2_list = r2_int_list/np.sqrt(reward_rms2.var)

        dones_list = np.stack(np.expand_dims(dones_list, axis=1)).transpose()
        total_int_rwd1_list = np.stack(total_int_rwd1_list).transpose()
        total_int_rwd2_list = np.stack(total_int_rwd2_list).transpose()

        v1_list = np.stack(v1_list).transpose()
        v2_list = np.stack(v2_list).transpose()
        #print(len(total_int_rwd1_list),len(dones_list),len(v1_list))
        #target_int1, adv1_int = make_train_data_me(total_int_rwd1_list,dones_list,v1_list,0.999,MC_iter)
        #target_int2, adv2_int = make_train_data_me(total_int_rwd2_list,dones_list,v2_list,0.999,MC_iter)
        #print(target_int1,adv1_int)
        if mode == 'intrinsic':
            adv1_int = total_int_rwd1_list - reward_rms1.mean/reward_rms1.var
            adv2_int = total_int_rwd2_list - reward_rms2.mean/reward_rms2.var
        else:
            adv1_int,adv2_int=0,0
        
        #print("adv1_int=",adv1_int)
        
        if epi_iter % 10 == 0:
            train_curve.append(acc_r/MC_iter)
        print('Episode', epi_iter, 'reward', acc_r/MC_iter)
        agent.train(o1_list, a1_list, pi_a1_list, o2_list, a2_list, pi_a2_list, r_list,adv1_int,adv2_int)
    plt.plot(train_curve, linewidth=1, label='COMA')
    #plt.show()
    env.reset()
    return plt
    
if __name__ == '__main__':
    loop = 5
    plot= {}
    modes = ['intrinsic','normal']
    for i in range(loop):
        for mode in modes:
            plot[str(i)+"_"+str(mode)] = game(mode=mode)
            plot[str(i)+"_"+str(mode)].title(str(i)+"_"+str(mode))
            plot[str(i)+"_"+str(mode)].show()
            
