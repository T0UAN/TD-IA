import gymnasium as gym
import random
import numpy as np
import matplotlib.pyplot as plt

def update_q_table(Q, s, a, r, sprime, alpha, gamma):
    """
    This function should update the Q function for a given pair of action-state
    following the q-learning algorithm, it takes as input the Q function, the pair action-state,
    the reward, the next state sprime, alpha the learning rate and gamma the discount factor.
    Return the same input Q but updated for the pair s and a.
    """
    old_value = Q[s, a]
    next_max = np.max(Q[sprime])
    new_value = (1 - alpha) * old_value + alpha * (r + gamma * next_max)
    Q[s, a] = new_value
    return Q


def epsilon_greedy(Q, s, epsilone):
    """
    This function implements the epsilon greedy algorithm.
    Takes as unput the Q function for all states, a state s, and epsilon.
    It should return the action to take following the epsilon greedy algorithm.
    """
    if random.uniform(0, 1) < epsilone:
        return env.action_space.sample()
    else:
        return np.argmax(Q[s])


if __name__ == "__main__":

    env = gym.make("Taxi-v3", render_mode=None)
    env.reset()
    Q = np.zeros([env.observation_space.n, env.action_space.n])

    alpha = 0.7 
    gamma = 0.95 
    epsilon = 0.1 
    n_epochs = 2000 
    max_itr_per_epoch = 100 
    rewards = []

    print("entrainement")

    for e in range(n_epochs):
        r = 0
        S, _ = env.reset()
        for _ in range(max_itr_per_epoch):
            A = epsilon_greedy(Q=Q, s=S, epsilone=epsilon)
            Sprime, R, done, _, info = env.step(A)
            r += R
            
            Q = update_q_table(
                Q=Q, s=S, a=A, r=R, sprime=Sprime, alpha=alpha, gamma=gamma
            )
            S = Sprime
            
            if done:
                break

        rewards.append(r)
        
        if e % 200 == 0:
            print(f"épisode #{e} : r = {r}")


    print("récompense moyenne = ", np.mean(rewards))
    print("Entrainement terminé.\n")
    
    env.close()

#Phase de test
    print("\ntest sur 5 parties")
    env = gym.make("Taxi-v3", render_mode="human")
    
    nb_parties = 5 # Nombre de parties
    
    for i in range(nb_parties):
        S, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0

        while not done:
            A = np.argmax(Q[S])
            
            S, R, done, truncated, info = env.step(A)
            total_reward += R
            steps += 1
        
        print(f"partie finie en {steps} étapes. Score : {total_reward}")
    
    env.close()
    print("fin du test")