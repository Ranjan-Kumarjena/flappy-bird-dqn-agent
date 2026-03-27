import argparse
from pickletools import optimize

import flappy_bird_gymnasium
import gymnasium as gym 
import torch
from dqn import DQN
import itertools
import yaml
import torch
import torch.nn as nn
import random
import torch.optim as optim
import os

 

from Experience_replay import ReplayMemory

if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
    
else:
    device = "cpu" # runtime => change runtime => t4 gpu
RUNS_DIR = "runs"
os.makedirs(RUNS_DIR,exist_ok=True)


class Agent:
    def __init__(self,params_set):
        self.pram_set = params_set
        with open("parameters.yaml","r") as f:
            all_params_set = yaml.safe_load(f)
            params = all_params_set[params_set]
        self.alpha = params["alpha"]
        self.gamma = params["gamma"]
        
        self.epsilon_init = params["epsilon_init"]
        
        self.epsilon_min = params["epsilon_min"]
        
        self.epsilon_decay = params["epsilon_decay"]
        
        self.network_sync_ratte = params["network_sync_ratte"]
        self.mini_batch_size = params["mini_batch_size"]
        self.reward_threshold = params["reward_threshold"]
        self.replay_memory_size = params["replay_memory_size"]
        
        self.loss_fn =nn.MSELoss()
        self.optimizer = None
        
        
        self.LOG_FILE = os.path.join(RUNS_DIR,f"{self.pram_set}.log")
        self.MODEL_FILE = os.path.join(RUNS_DIR,f"{self.pram_set}.pt")

    def run(self,is_training=True,render=False):
         
    
    
        env=gym.make("FlappyBird-v0",render_mode="human" if render else None)
    
        num_states=env.observation_space.shape[0] # input dimension 
        num_action=env.action_space.n # output dimension 
    
    
        policy_dqn=DQN(num_states,num_action).to(device)
        state,_=env.reset()
    
        if is_training:
            memory = ReplayMemory(self.replay_memory_size) # we use dynamic  value instade of static value or fixed value 
            epsilon = self.epsilon_init
            
            target_dqn=DQN(num_states,num_action).to(device)
            # copy the weight and bias from policy_dqn to target_dqn
            target_dqn.load_state_dict(policy_dqn.state_dict())
            steps=0
            self.optimizer = optim.Adam(policy_dqn.parameters(),lr=self.alpha)
            best_rewards= float("-inf")
            
        else:
            # best policy load
            policy_dqn.load_state_dict(torch.load(self.MODEL_FILE))
            policy_dqn.eval()
            

        for episode in itertools.count():
            
            state,_= env.reset()
            state = torch.tensor(state,dtype=torch.long)
            
            
            
            episode_reward = 0
            terminated=False
        
            while (not terminated and episode_reward < self.reward_threshold) :
                if is_training and random.random() < epsilon:
                     action = env.action_space.sample()  # explore
                     action = torch.tensor(action , dtype=torch.long,device=device)
                else:
                    with torch.no_grad():
                        action = policy_dqn(state.unsqueeze(dim=0).float()).squeeze().argmax() # exploit
                    
                
            
            # next action:
            # (feed the observation to your agent here )
               
    
    # Processing : terminated ==> done
                next_state,reward,terminated, _,_=env.step(action.item())
                episode_reward +=reward
                
                # create tensor 
                reward = torch.tensor(reward , dtype=torch.float , device=device)
                next_state= torch.tensor(state,dtype=torch.float,device=device)
                
                
        
                if is_training:
                    
                
                    memory.append((state,action,next_state,reward,terminated))
                  
                    steps+=1
                state= next_state
                
            print(f"for episode = {episode+1} with total reards {episode_reward} and epsilon = {epsilon}")
        
            if is_training:               
    # epsilon decay
                 epsilon = max(epsilon * self.epsilon_decay ,self.epsilon_min)
                 
                 if episode_reward > best_rewards:
                     log_msg=f"best rewards ={episode_reward} for episode ={episode+1}"
                     
                     with open(self.LOG_FILE,"a") as f:
                         f.write(log_msg +"\n")
                         
                     torch.save(policy_dqn.state_dict(),self.MODEL_FILE)
                     
                     best_reawrd= episode_reward
                 
                 
            if is_training and len(memory)> self.mini_batch_size:
                # get sample 
                mini_batch = memory.sample(self.mini_batch_size)
                
                self.optimize(mini_batch,policy_dqn,target_dqn)
                
                # sync the network
                if steps > self.network_sync_ratte:
                    target_dqn.load_state_dict(policy_dqn.state_dict())
                    steps=0
    
    # env.close()  ==>manually stop 
        
    def optimize(self,mini_batch,policy_dqn,target_dqn):
        # getbatch of  experiences
      states,actions,next_states,rewards,terminations= zip(*mini_batch) 
      
      states=torch.stack(states)
      actions=torch.stack(actions)
      next_states=torch.stack(next_states)
      rewards=torch.stack(rewards)
      terminations = torch.tensor(terminations).float().to(device)
      # calculated target Q-value - if terminations = TRue ==> zero
          
      with torch.no_grad():
            target_q=rewards +(1-terminations) * self.gamma * target_dqn(next_states).max(dim=1)[0]  # y_true actual value 
       # claculated y_pred i.e Q -value from curent policy             
      current_q=policy_dqn(states).gather(dim=1,index=actions.unsqueeze(dim=1)).squeeze()
      
            
    # compute the loss
      loss=self.loss_fn(current_q,target_q)
      # optimize model       
      self.optimizer.zero_grad()
      loss.backward()
      self.optimizer.step()
                
if __name__== "__main__":
    # parse command line inputs 
    
    parser = argparse.ArgumentParser(description="Train or test model.")    
    parser.add_argument("hyperparametres",help='')  
    parser.add_argument('--train',help='Training mode',action='store_true')  
    args=parser.parse_args()
    dql=Agent(params_set=args.hyperparametres)
    
    if args.train:
        dql.run(is_training=True)
    else:
        dql.run(is_training=False,render=True)
      
