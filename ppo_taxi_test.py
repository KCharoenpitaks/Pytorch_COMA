import torch
import torch.nn as nn
from torch.distributions import Categorical
import gym
import numpy as np
import matplotlib.pyplot as plt
from frozen_lake import FrozenLakeEnv
from blackjack import BlackjackEnv
from taxi import TaxiEnv
from utils import *
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.intrinsic_rewards = []
        self.is_terminals = []
    
    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.intrinsic_rewards[:]
        del self.is_terminals[:]

def get_intrinsic_rewards(AI,states,agents,n_agents,multiple):
    rewards_intrinsic = {}
    if (AI == "COMA") or (AI == "CAC") or (AI == "COMA2"):
        for agent_id in range(n_agents):
            rewards_intrinsic[agent_id] = agents.RND_net[agent_id].RND_diff(states[agent_id])*multiple
    else:
        for agent_id in range(n_agents):
            rewards_intrinsic = agents.RND_net.RND_diff(states)*multiple
    return rewards_intrinsic

def get_intrinsic_rewards2(AI,states,actions,agents,n_agents,multiple):
    rewards_intrinsic = {}
    if (AI == "COMA") or (AI == "CAC") or (AI == "COMA2"):
        for agent_id in range(n_agents):
            rewards_intrinsic[agent_id] = agents.RND_net2[agent_id].RND_diff(states[agent_id],actions[agent_id]).cpu()*multiple
    else:
        for agent_id in range(n_agents):
            rewards_intrinsic = agents.RND_net2.RND_diff(states,actions)*multiple
    return rewards_intrinsic

def get_intrinsic_rewards3(AI,states,actions,agents,n_agents,rewards,multiple):
    rewards_intrinsic = {}
    if (AI == "COMA") or (AI == "CAC") or (AI == "COMA2"):
        for agent_id in range(n_agents):
            rewards_intrinsic[agent_id] = agents.rwd_prediction[agent_id].diff(states[agent_id],actions[agent_id]).cpu()*multiple
    else:
        for agent_id in range(n_agents):
            rewards_intrinsic = agents.rwd_prediction.diff(states,actions,rewards)*multiple
    return rewards_intrinsic

def get_intrinsic_rewards4(AI,states,actions,agents,n_agents,episode_reward,timesteps,multiple,discount):
    rewards_intrinsic = {}
    episode_reward = max(episode_reward,1)
    if (AI == "COMA") or (AI == "CAC") or (AI == "COMA2"):
        for agent_id in range(n_agents):
            rewards_intrinsic[agent_id] = np.random.normal(0,0.1)*np.power(discount,timesteps)*multiple*np.log(episode_reward)
    else:
        for agent_id in range(n_agents):
            rewards_intrinsic = np.random.normal(0,0.1)*np.power(discount,timesteps)*multiple*np.log(episode_reward)
    return rewards_intrinsic

def get_intrinsic_rewards5(AI,states,agents,n_agents,multiple,bucket_size):
    rewards_intrinsic = {}
    count = {}
    if (AI == "COMA") or (AI == "CAC") or (AI == "COMA2"):
        for agent_id in range(n_agents):
            agents[agent_id].memcount.add(str(divide_count(states,bucket_size)))
            count[agent_id] = agents[agent_id].memcount.count(str(divide_count(states,bucket_size)))
            rewards_intrinsic[agent_id] = 0.1/np.sqrt(count[agent_id]).cpu()*multiple
    else:
        for agent_id in range(n_agents):
            agents.memcount.add(str(divide_count(states,bucket_size)))
            count = agents.memcount.count(str(divide_count(states,bucket_size)))
            rewards_intrinsic = 0.1/np.sqrt(count)*multiple
    return rewards_intrinsic
#=======================================================================
class MemoryCount():
    def __init__(self, max_size=10000):
        self.max_size = max_size
        self.buffer2 = {}
    
    def add(self, experience):
        try:
            self.buffer2[experience] += 1
        except:
            self.buffer2[experience] = 1
        
    def delete(self):
        self.buffer2 = {}
    
    def count(self, b):
        return self.buffer2[b]
    def leng(self):
        return len(self.buffer2)
#=================================================================
def divide_count(state, bucket_size):
    out = state//bucket_size
    return out

class RNDforPPO(nn.Module):
    def __init__(self, state_dim,action_dim, n_latent_var):
        super(RNDforPPO, self).__init__()
        self.affine = nn.Linear(state_dim, n_latent_var)
        self.MseLoss = nn.MSELoss()
        
        self.RND_NN_layer = nn.Sequential(
                nn.Linear(state_dim, n_latent_var),
                nn.Tanh(),
                nn.Linear(n_latent_var, n_latent_var),
                nn.Tanh(),
                nn.Linear(n_latent_var, 32),
                )
        self.Predictor_NN_layer = nn.Sequential(
                nn.Linear(state_dim, n_latent_var),
                nn.Tanh(),
                nn.Linear(n_latent_var, n_latent_var),
                nn.Tanh(),
                nn.Linear(n_latent_var, 32),
                )
    
    def forward_RND(self, state):
        state = torch.from_numpy(state).float()
        state = state
        value = self.RND_NN_layer(state)
        return torch.squeeze(value)
    
    def predictor_RND(self, state):
        state = torch.from_numpy(state).float()
        #state= state
        #print(state)
        value = self.Predictor_NN_layer(state)
        return torch.squeeze(value)
    
    def RND_diff(self,state):
        #print(state)
        #state = np.array(state)
        predictor = self.predictor_RND(state)
        forward = self.forward_RND(state)
        diff = self.MseLoss(forward,predictor)
        return diff
    
class RNDforPPO2(nn.Module):
    def __init__(self, state_dim,action_dim , n_latent_var):
        super(RNDforPPO2, self).__init__()
        #self.affine = nn.Linear(state_dim, n_latent_var)
        self.MseLoss = nn.MSELoss()
        self.action_dim=action_dim


        self.RND_NN_layer = nn.Sequential(
                nn.Linear(state_dim, n_latent_var),
                nn.Tanh(),
                nn.Linear(n_latent_var, n_latent_var),
                nn.Tanh(),
                nn.Linear(n_latent_var, 32),
                )
        self.Predictor_NN_layer = nn.Sequential(
                nn.Linear(int(state_dim)+int(action_dim), n_latent_var),
                nn.Tanh(),
                nn.Linear(n_latent_var, n_latent_var),
                nn.Tanh(),
                nn.Linear(n_latent_var, 32),
                )
                
    
    def forward_RND(self, state):
        state = torch.from_numpy(state).float()
        state = state.cpu()
        value = self.RND_NN_layer(state)
        return torch.squeeze(value)
    
    def predictor_RND(self, state,action):
        
        #print(action)
        #print(self.action_dim)
        action = to_categorical(action,self.action_dim)
        action = torch.from_numpy(action).float()
        #action = torch.tensor(action.clone().detach()).float()
        action = torch.squeeze(action,0)

        
        state = torch.from_numpy(state).float()
        state= state.cpu()

        state_action = torch.cat((state, action), -1)

        value = self.Predictor_NN_layer(state_action.cpu())
        return torch.squeeze(value).cpu()
    
    def RND_diff(self,state,action):
        action = np.array(action)
        predictor = self.predictor_RND(state,action)
        forward = self.forward_RND(state)
        diff = self.MseLoss(forward,predictor)
        return diff
    
def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='uint8')[y]

class Reward_prediction(nn.Module):
    def __init__(self, state_dim,action_dim , n_latent_var):
        super(Reward_prediction, self).__init__()
        #self.affine = nn.Linear(state_dim, n_latent_var)
        self.MseLoss = nn.MSELoss()
        self.action_dim=action_dim

        self.Predictor_NN_layer = nn.Sequential(
                nn.Linear(int(state_dim)+int(action_dim), n_latent_var),
                nn.Tanh(),
                nn.Linear(n_latent_var, n_latent_var),
                nn.Tanh(),
                nn.Linear(n_latent_var, 1),
                )
    
    def predictor_RND(self, state,action):

        action = to_categorical(action,self.action_dim)
        action = torch.from_numpy(action).float()
        #action = torch.tensor(action.clone().detach()).float()
        action = torch.squeeze(action,0)

        
        state = torch.from_numpy(state).float()
        state= state.cpu()
        state_action = torch.cat((state, action), -1)
        value = self.Predictor_NN_layer(state_action.cpu())
        return torch.squeeze(value).cpu()
    
    def diff(self,state,action,reward):
        action = np.array(action)
        predictor = self.predictor_RND(state,action)
        reward = torch.tensor(reward)
        diff = self.MseLoss(reward.float().detach(),predictor.float().detach())
        #print(diff)
        return diff

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, n_latent_var):
        super(ActorCritic, self).__init__()
        self.affine = nn.Linear(state_dim, n_latent_var)
        
        # actor
        self.action_layer = nn.Sequential(
                nn.Linear(state_dim, n_latent_var),
                nn.Tanh(),
                nn.Linear(n_latent_var, n_latent_var),
                nn.Tanh(),
                nn.Linear(n_latent_var, action_dim),
                nn.Softmax(dim=-1)
                )
        
        # critic
        self.value_layer = nn.Sequential(
                nn.Linear(state_dim, n_latent_var),
                nn.Tanh(),
                nn.Linear(n_latent_var, n_latent_var),
                nn.Tanh(),
                nn.Linear(n_latent_var, 1)
                )
        
    def forward(self):
        raise NotImplementedError
        
    def act(self, state, memory):
        #print(state)
        state = torch.from_numpy(state).float().to(device) 
        action_probs = self.action_layer(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        
        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(dist.log_prob(action))
        
        return action.item()
    
    def evaluate(self, state, action):
        action_probs = self.action_layer(state)
        dist = Categorical(action_probs)
        
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        
        state_value = self.value_layer(state)
        
        return action_logprobs, torch.squeeze(state_value), dist_entropy
        
class PPO:
    def __init__(self, state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.policy = ActorCritic(state_dim, action_dim, n_latent_var).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)
        self.policy_old = ActorCritic(state_dim, action_dim, n_latent_var).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()
        self.RND_net = RNDforPPO(state_dim,action_dim, n_latent_var).to(device)
        self.RND_net_optimizer = torch.optim.Adam(self.RND_net.parameters(),
                                              lr=lr, betas=betas)
        
        self.RND_net2 = RNDforPPO2(state_dim,action_dim, n_latent_var).to(device)
        self.RND_net_optimizer2 = torch.optim.Adam(self.RND_net2.parameters(),
                                              lr=lr, betas=betas)
        
        self.rwd_prediction =Reward_prediction(state_dim,action_dim, n_latent_var).to(device)
        self.rwd_optimizer = torch.optim.Adam(self.rwd_prediction.parameters(),
                                              lr=lr, betas=betas)
        
        self.memory = Memory()
        self.memcount = MemoryCount(max_size = 10000)
        self.MseLoss2 = nn.MSELoss()
        self.MseLoss3 = nn.MSELoss()
        self.MseLoss4 = nn.MSELoss()
        self.MseLoss5 = nn.MSELoss()
    
    def update(self, memory,adv_int):   
        # Monte Carlo estimate of state rewards:
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        # Normalizing the rewards:
        rewards = torch.tensor(rewards).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        
        # convert list to tensor
        old_states = torch.stack(memory.states).to(device).detach()
        old_actions = torch.stack(memory.actions).to(device).detach()
        old_logprobs = torch.stack(memory.logprobs).to(device).detach()
        RND_Net_values = self.RND_net.forward_RND(old_states.cpu().data.numpy())
        #RND_Net_values.detach()
        RND_Net_values2 = self.RND_net2.forward_RND(old_states.cpu().data.numpy())
        #RND_Net_values2.detach()
        RND_predictor_values = self.RND_net.predictor_RND(old_states.cpu().data.numpy())
        RND_predictor_values2 = self.RND_net2.predictor_RND(old_states.cpu().data.numpy(),old_actions)       
        rwd_predictor_value = self.rwd_prediction.predictor_RND(old_states.cpu().data.numpy(),old_actions)
        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Evaluating old actions and values :
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            
            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach())
                
            # Finding Surrogate Loss:
            advantages = rewards - state_values.detach()
            surr1 = ratios * (advantages + torch.tensor(adv_int).float())
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * (advantages + torch.tensor(adv_int).float())
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy 

            loss3 = self.MseLoss3(RND_Net_values.detach(), RND_predictor_values)
            loss4 = self.MseLoss3(RND_Net_values2.detach().cpu(), RND_predictor_values2.cpu())
            loss5 = self.MseLoss5(rwd_predictor_value.float(),torch.tensor(memory.rewards).float().detach())
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
            self.RND_net_optimizer.zero_grad()
            loss3.backward(retain_graph=True)          
            self.RND_net_optimizer.step()
            
            self.RND_net_optimizer2.zero_grad()
            loss4.backward(retain_graph=True)    
            self.RND_net_optimizer2.step()
            
            self.rwd_optimizer.zero_grad()
            loss5.backward(retain_graph=True)
            self.rwd_optimizer.step()
        
        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())
        
def game(N_episodes, AI_type,Intrinsic_type):
    ############## Hyperparameters ##############

    env = TaxiEnv()
    #memory = Memory(max_size=300)

    #n_episodes = number_of_episodes
    #n_actions = env.action_space.n
    #intrinsic = intrinsic
    #print(n_actions)
    #n_agents = 1
    #n_episodes = number_of_episodes
    #state_size = env.observation_space.n
    
    
    #env_name = "LunarLander-v2"
    # creating environment
    state_dim = env.observation_space.n
    action_dim = env.action_space.n
    render = False
    solved_reward = 230         # stop training if avg_reward > solved_reward
    log_interval = 20           # print avg reward in the interval
    max_episodes = N_episodes        # max training episodes
    max_timesteps = 250         # max timesteps in one episode
    n_latent_var = 64           # number of variables in hidden layer
    update_timestep = 2000     # update policy every n timesteps
    lr = 0.002
    betas = (0.9, 0.999)
    gamma = 0.99                # discount factor
    K_epochs = 4                # update policy for K epochs
    eps_clip = 0.2              # clip parameter for PPO
    random_seed = None
    samp_rewards = []
    avg_rewards = []
    best_avg_reward = -np.inf
    n_agents = 1
    #############################################
    
    if random_seed:
        torch.manual_seed(random_seed)
        env.seed(random_seed)
    
    memory = Memory()
    ppo = PPO(state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip)
    print(lr,betas)
    
    # logging variables
    running_reward = 0
    avg_length = 0
    timestep = 0
    avg_reward = 0
    ppo.memcount.delete()
    state_size = env.observation_space.n
    reward_rms = RunningMeanStd()
    obs_rms = RunningMeanStd()
    norm_step = 5000
    #Pre Run
    next_obs = []
    for norm_step in range (norm_step):
        action_norm = np.random.randint(0,action_dim)
        state_norm, reward_norm, done_norm, _ = env.step(action_norm)
        state_norm = to_categorical(state_norm,state_size) #optional
        next_obs.append(state_norm)
    obs_rms.update(next_obs)
    
    # training loop
    for i_episode in range(1, max_episodes+1):
        state = env.reset()
        state = to_categorical(state,state_size)
        done = False
        t= 0
        episode_reward = 0
        intrinsic_rewards = 0
        reward = 0
        #for t in range(max_timesteps):
        #while not done:
        while t <= max_timesteps:
            timestep += 1
            t +=1
            
            # Running policy_old:
            action = ppo.policy_old.act(state, memory)
            state, reward, done, _ = env.step(action)
            state = to_categorical(state,state_size)

            #========================================================
            if ((AI_type == "PPO"or AI_type == "A2C") and Intrinsic_type == "1"):
                intrinsic_rewards = get_intrinsic_rewards(AI_type,state,ppo,n_agents,10)
                intrinsic_rewards = intrinsic_rewards.data.numpy()
                #print("intrinsic_rewards1",intrinsic_rewards)
            elif ((AI_type == "PPO"or AI_type == "A2C") and Intrinsic_type == "2"):
                intrinsic_rewards = get_intrinsic_rewards2(AI_type,state,action,ppo,n_agents,10)
                intrinsic_rewards = intrinsic_rewards.data.numpy()
                #print("intrinsic_rewards2",intrinsic_rewards)
            
            elif ((AI_type == "PPO"or AI_type == "A2C") and Intrinsic_type == "3"):
                intrinsic_rewards = get_intrinsic_rewards3(AI_type,state,action,ppo,n_agents,reward,1) 
                intrinsic_rewards = intrinsic_rewards.data.numpy()
                #print("intrinsic_rewards3",intrinsic_rewards)
            elif ((AI_type == "PPO"or AI_type == "A2C") and Intrinsic_type == "4"):
                intrinsic_rewards = get_intrinsic_rewards4(AI_type,state,action,ppo,n_agents,reward,t,1,0.99) 
            
            elif ((AI_type == "PPO"or AI_type == "A2C") and Intrinsic_type == "5"):
                intrinsic_rewards = get_intrinsic_rewards5(AI_type,state,ppo,n_agents,1,16)    
                #print("intrinsic_rewards5",intrinsic_rewards)
            else:
                intrinsic_rewards = 0
            reward_sum = reward #+ intrinsic_rewards
            #===========================================================
            memory.rewards.append(reward_sum)
            #temp_int = memory.intrinsic_rewards.data.numpy()
            #temp_int = memory.intrinsic_rewards
            #print(temp_int)
            memory.intrinsic_rewards.append(intrinsic_rewards)
            memory.is_terminals.append(done)
            """
            try:
                mean1, std1, count1 = np.mean(temp_int), np.std(temp_int), len(temp_int)
                reward_rms.update_from_moments(mean1, std1 ** 2, count1)
                adv_int = (memory.intrinsic_rewards-reward_rms.mean)/np.sqrt(reward_rms.var)
            except:
                adv_int = 0
                """

            # update if its time
            if timestep % update_timestep == 0:
                temp_int = memory.intrinsic_rewards
                mean1, std1, count1 = np.mean(temp_int), np.std(temp_int), len(temp_int)
                reward_rms.update_from_moments(mean1, std1 ** 2, count1)
                adv_int = (temp_int)/np.sqrt(reward_rms.var)
                ppo.update(memory,adv_int)
                memory.clear_memory()
                timestep = 0
            
            
            running_reward += reward
            episode_reward += reward
            if render:
                env.render()
            #if done:
                #break
                
        avg_length += t
        
        # stop training if avg_reward > solved_reward
        if running_reward > (log_interval*solved_reward):
            print("########## Solved! ##########")
            #torch.save(ppo.policy.state_dict(), './PPO_{}.pth'.format(env_name))
            #break
            
        # logging
        if i_episode % log_interval == 0:
            avg_length = int(avg_length/log_interval)
            running_reward = int((running_reward/log_interval))
            
            print('Episode {} \t avg length: {} \t reward: {}'.format(i_episode, avg_length, running_reward))
            running_reward = 0
            avg_length = 0
            
        samp_rewards.append(episode_reward)
        if (i_episode >= 100):
            # get average reward from last 100 episodes
            avg_reward = np.mean(samp_rewards[-100:])
            # append to deque
            avg_rewards.append(avg_reward)
            # update best average reward
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward

        print("Total reward in episode {} = {}".format(i_episode, episode_reward))
        print("Best_avg_reward =", np.round(best_avg_reward,3),"Average_rewards =", np.round(avg_reward,3))
    #env.save_replay()
    env.close()

    return avg_rewards, best_avg_reward,samp_rewards,"0"

def main():
    AI1 = ["PPO"]#,"REINFORCE","A2C","PPO","COMA","COMA2"
    N_episodes = 1500
    for AI in AI1:
        avg_reward, best_avg_reward,samp_rewards,stats = game(N_episodes, AI,"0") 
        plot("Taxi",avg_reward, best_avg_reward,samp_rewards, AI,'b')
        avg_reward, best_avg_reward,samp_rewards,stats = game(N_episodes, AI, "1") 
        plot("Taxi",avg_reward, best_avg_reward,samp_rewards, AI,'g')
        avg_reward, best_avg_reward,samp_rewards,stats = game(N_episodes, AI, "2") 
        plot("Taxi",avg_reward, best_avg_reward,samp_rewards, AI,'r')
        avg_reward, best_avg_reward,samp_rewards,stats = game(N_episodes, AI, "3") 
        plot("Taxi",avg_reward, best_avg_reward,samp_rewards, AI,'c')
        avg_reward, best_avg_reward,samp_rewards,stats = game(N_episodes, AI, "4") 
        plot("Taxi",avg_reward, best_avg_reward,samp_rewards, AI,'m')
        avg_reward, best_avg_reward,samp_rewards,stats = game(N_episodes, AI, "5") 
        plot("Taxi",avg_reward, best_avg_reward,samp_rewards, AI,'y')
        
    
def plot(name,avg_reward, best_avg_reward,samp_rewards, AI,color):
    t = np.arange(1,len(avg_reward)+1,1) 
    plt.figure('Episode Rwards of '+ AI+' in' + name)
    plt.plot(t, avg_reward,color)
    plt.xlabel('Episode')
    plt.ylabel('Average 100 episodes Rewards of ' + AI)
    plt.title(AI)
    plt.show()    

if __name__ == '__main__':
    main()
    
