import os
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
import gym
from model_free_method.qlearning import QLearningAgent
from model_free_method.expected_value_sarsa_epsilon_annealing import EVSarsaAgent
from pandas import  Series


"""
    !!!
    Q-learning in the wild + reducing epsilon
    !!!
"""


agent = QLearningAgent(alpha=0.5,epsilon=0.25,discount=0.99,
                       get_legal_actions = lambda s: range(n_actions))
env = gym.make("Taxi-v2")
print(env)
n_actions = env.action_space.n
print(n_actions)

def play_and_train(env, agent, t_max=10 ** 4):
    """This function should
    - run a full game, actions given by agent.getAction(s)
    - train agent using agent.update(...) whenever possible
    - return total reward"""
    total_reward = 0.0
    s = env.reset()

    for t in range(t_max):
        a = agent.get_action(s)  # <get agent to pick action given state s>

        next_s, r, done, _ = env.step(a)

        # <train (update) agent for state s>
        agent.update(s, a, r, next_s)

        s = next_s
        total_reward += r
        if done: break

    return total_reward


rewards = []
for i in range(5000):
    rewards.append(play_and_train(env, agent))
    if i %1000 ==0:
        clear_output(True)
        print("mean reward",np.mean(rewards[-100:]))
        plt.plot(rewards)
        plt.show()


"""
    !!!
    Expected value SARSA + epsilon reducing
    !!!
"""


agent_sarsa = EVSarsaAgent(alpha=0.5,epsilon=0.25,discount=0.99,
                       get_legal_actions = lambda s: range(n_actions))


def play_and_train(env, agent, t_max=10 ** 4):
    """This function should
    - run a full game, actions given by agent.getAction(s)
    - train agent using agent.update(...) whenever possible
    - return total reward"""
    total_reward = 0.0
    s = env.reset()

    for t in range(t_max):
        a = agent.get_action(s)

        next_s, r, done, _ = env.step(a)
        agent.update(s, a, r, next_s)

        s = next_s
        total_reward += r
        if done: break

    return total_reward


moving_average = lambda ts, span=100: ewma(Series(ts), min_periods=span // 10, span=span).values

rewards_sarsa = []

for i in range(5000):
    rewards_sarsa.append(play_and_train(env, agent_sarsa))
    # Note: agent.epsilon stays constant

    if i % 1000 == 0:
        clear_output(True)
        print('EVSARSA mean reward =', np.mean(rewards_sarsa[-100:]))
        plt.title("epsilon = %s" % agent_sarsa.epsilon)
        plt.plot(moving_average(rewards_sarsa), label='ev_sarsa')
        plt.grid()
        plt.legend()
        plt.ylim(-500, 0)
        plt.show()



# env = gym.wrappers.Monitor(gym.make("CartPole-v0"), directory="videos", force=True)
# sessions = [play_and_train(env, agent) for _ in range(100000)]