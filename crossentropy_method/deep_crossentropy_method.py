import numpy as np
import matplotlib.pyplot as plt
import gym.envs.classic_control
from sklearn.neural_network import MLPClassifier


env = gym.make("CartPole-v0").env  #if you see "<classname> has no attribute .env", remove .env or update gym

env.reset()
n_actions = env.action_space.n


# create agent
agent = MLPClassifier(hidden_layer_sizes=(20, 20),
                      activation='tanh',
                      warm_start=True,
                      max_iter=1 #make only 1 iteration on each .fit(...)
                      )

# initialize agent to the dimension of state an amount of actions
agent.fit([env.reset()]*n_actions, range(n_actions))
print([env.reset()]*n_actions)
print(n_actions)
print(env.reset())


def generate_session(t_max=1000):
    states, actions = [], []
    total_reward = 0

    s = env.reset()
    print(s)
    for t in range(t_max):

        # a vector of action probabilities in current state
        probs = agent.predict_proba([s])[0]

        a = np.random.choice(2, 1, p=probs)[0]

        new_s, r, done, info = env.step(a)

        # record sessions like you did before
        states.append(s)
        actions.append(a)
        total_reward += r

        s = new_s
        if done: break
    return states, actions, total_reward


def select_elites(states_batch, actions_batch, rewards_batch, percentile=50):
    """
    Select states and actions from games that have rewards >= percentile
    :param states_batch: list of lists of states, states_batch[session_i][t]
    :param actions_batch: list of lists of actions, actions_batch[session_i][t]
    :param rewards_batch: list of rewards, rewards_batch[session_i][t]

    :returns: elite_states,elite_actions, both 1D lists of states and respective actions from elite sessions

    Please return elite states and actions in their original order
    [i.e. sorted by session number and timestep within session]

    If you're confused, see examples below. Please don't assume that states are integers (they'll get different later).
    """

    reward_threshold = np.percentile(rewards_batch, percentile)

    elite_states = [s for i in range(len(states_batch)) if rewards_batch[i] >= reward_threshold for s in
                    states_batch[i]]
    elite_actions = [a for i in range(len(actions_batch)) if rewards_batch[i] >= reward_threshold for a in
                     actions_batch[i]]

    return elite_states, elite_actions


def show_progress(batch_rewards, log, percentile, reward_range=[-990, +10]):
    """
    A convenience function that displays training progress.
    No cool math here, just charts.
    """

    mean_reward, threshold = np.mean(batch_rewards), np.percentile(batch_rewards, percentile)
    log.append([mean_reward, threshold])

    print("mean reward = %.3f, threshold=%.3f" % (mean_reward, threshold))
    # plt.figure(figsize=[8, 4])
    # plt.subplot(1, 2, 1)
    # plt.plot(list(zip(*log))[0], label='Mean rewards')
    # plt.plot(list(zip(*log))[1], label='Reward thresholds')
    # plt.legend()
    # plt.grid()
    #
    # plt.subplot(1, 2, 2)
    # plt.hist(batch_rewards, range=reward_range);
    # plt.vlines([np.percentile(batch_rewards, percentile)], [0], [100], label="percentile", color='red')
    # plt.legend()
    # plt.grid()
    #
    # plt.show()


n_sessions = 100
percentile = 70
log = []

for i in range(30):
    print('Step - ', i+1)
    # generate new sessions
    sessions = [generate_session() for _ in range(n_sessions)]

    batch_states, batch_actions, batch_rewards = map(np.array, zip(*sessions))

    elite_states, elite_actions = select_elites(batch_states, batch_actions, batch_rewards, percentile)
    #     print(elite_states[:3])
    #     print(elite_actions[:3])

    agent.fit(elite_states, elite_actions)

    show_progress(batch_rewards, log, percentile, reward_range=[0, np.max(batch_rewards)])

    if np.mean(batch_rewards) > 190:
        print("You Win! You may stop training now via KeyboardInterrupt.")

env = gym.wrappers.Monitor(gym.make("CartPole-v0"), directory="videos", force=True)
sessions = [generate_session() for _ in range(1000)]
env.close()