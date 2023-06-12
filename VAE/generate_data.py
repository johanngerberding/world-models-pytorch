"""
Data Generation Script for the VAE training
""" 
import os 
import argparse 
import gymnasium as gym 
import numpy as np 
from multiprocessing import Pool 


def rollout(data):
    data_dir, seq_len, rollouts = data 
    os.makedirs(data_dir)
    env = gym.make("CarRacing-v2")

    for i in range(rollouts):
        env.reset()
        # get random actions
        actions_rollout = [env.action_space.sample() for _ in range(seq_len)]
        observations_rollout = []
        rewards_rollout = [] 
        dones_rollout = [] 

        t = 0 
        while True: 
            action = actions_rollout[t]
            t += 1

            obs, reward, done, truncated, _ = env.step(action) 
            observations_rollout += [obs]
            rewards_rollout += [reward]
            dones_rollout += [done]

            if done or truncated: 
                print(f"End of rollout {i} | {t} frames") 
                np.savez(
                    os.path.join(data_dir, f"rollout_{i}"), 
                    observations=np.array(observations_rollout), 
                    rewards=np.array(rewards_rollout), 
                    actions=np.array(actions_rollout), 
                    terminals=np.array(dones_rollout),
                )
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    parser.add_argument('--rollouts', help="number of rollouts", type=int, default=10_000)
    parser.add_argument('--threads', help="number of threads", type=int, default=20)
    parser.add_argument('--seq_len', help="sequence length", type=int, default=1000)
    parser.add_argument('--dir', help="output directory", type=str, default="data/vae")
    args = parser.parse_args()
    
    os.makedirs(args.dir) 
    reps = args.rollouts // args.threads + 1

    p = Pool(args.threads)
    work = [
        (os.path.join(args.dir, f"thread_{i}"), args.seq_len, reps) for i in range(args.threads)
    ] 
    print(work)
    p.map(rollout, tuple(work))
