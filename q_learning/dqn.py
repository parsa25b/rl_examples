import argparse
import gymnasium as gym
from model import Model
import torch
from itertools import count
import numpy as np
from eval_policy import eval_policy

# GLOBAL VARIABLES
NUM_EPISODES = 4000
TARGET_UPDATE = 10
TEST_INTERVAL = 100
PRINT_INTERVAL = 10

class DQNAgent():
    def __init__(
        self, 
        env: gym.Env,
        gamma: float,
        eps_greedy: float,
        learning_rate: float,
        batch_size: int
        ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Using device:", self.device)
        
        self.env = env
        self.gamma = gamma
        self.eps_greedy = eps_greedy
        self.learning_rate = learning_rate
        self.batch_size = batch_size
                
        action_size = env.action_space.n
        state_size = env.observation_space.shape[0]

        self.model = Model(state_size, action_size).to(self.device)
        self.target = Model(state_size, action_size).to(self.device)
        self.target.load_state_dict(self.model.state_dict())
        self.target.eval()
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        

    def choose_action(self, state: np.ndarray, test_mode: bool=False) -> torch.Tensor:
        state = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(0)
        if test_mode or torch.rand(1).item() > EPS_EXPLORATION:
            with torch.no_grad():
                action = self.model.select_action(state)
        else:
            action = self.env.action_space.sample()
            action = torch.tensor([[action]], device=self.device, dtype=torch.long)
        
        return action
    
    def optimize_model(
        self, 
        state: np.ndarray, 
        action: np.ndarray, 
        next_state: np.ndarray,
        reward: np.ndarray, 
        done: np.ndarray
    ) -> None:
        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        action = torch.tensor([[action]], device=self.device)
        next_state = torch.tensor(next_state, dtype=torch.float32, device=self.device).unsqueeze(0)
        reward = torch.tensor([reward], device=self.device)
        done = torch.tensor([float(done)], device=self.device)
               
        # Compute Q values and targets
        q_values = self.model(state).gather(1, action)
        with torch.no_grad():
            next_q_values = self.target(next_state).max(1)[0].unsqueeze(1)
            target_q_values = reward + (1 - done) * GAMMA * next_q_values
        
        # Compute loss
        loss = torch.nn.functional.mse_loss(q_values, target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

def train_rl(args):
    env = gym.make(args.env)
    agent = DQNAgent(env, args.gamma, args.eps_greedy, args.learning_rate, args.batch_size)
    best_score = -float('inf')
    
    for i_episode in range(NUM_EPISODES):
        episode_total_reward = 0
        state, _ = env.reset()
        
        for t in count():
            action = agent.choose_action(state).cpu().numpy()[0][0]
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode_total_reward += reward
            
            agent.optimize_model(state, action, next_state, reward, terminated)
            state = next_state
            if (terminated or truncated):
                if i_episode % PRINT_INTERVAL == 0:
                    print('[Episode {:4d}/{}] [Steps {:4d}] [reward {:.1f}]'
                        .format(i_episode, NUM_EPISODES, t, episode_total_reward))
                break
            
        if i_episode % TARGET_UPDATE == 0:
            agent.target.load_state_dict(agent.model.state_dict())
            
        if i_episode % TEST_INTERVAL == 0:
            print('-'*10)
            score = eval_policy(policy=agent.model, env=args.env, render=False)
            if score > best_score:
                best_score = score
                torch.save(agent.model.state_dict(), "best_model_{}.pt".format(args.env))
                print('saving model.')
            print("[TEST Episode {}] [Average Reward {}]".format(i_episode, score))
            print('-'*10)
                
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="CartPole-v1", help="Gym environment name")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--eps-greedy", type=float, default=0.1, help="Epsilon for epsilon-greedy policy")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate for optimizer")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for training")
    args = parser.parse_args()
    
    train_rl(args)
    
if __name__ == "__main__":
    main()