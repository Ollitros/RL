import numpy as np
import tensorflow as tf
import gym.wrappers


def get_cumulative_rewards(rewards,  # rewards at each step
                           gamma=0.99  # discount for reward
                           ):
    """
    take a list of immediate rewards r(s,a) for the whole session
    compute cumulative rewards R(s,a) (a.k.a. G(s,a) in Sutton '16)
    R_t = r_t + gamma*r_{t+1} + gamma^2*r_{t+2} + ...

    The simple way to compute cumulative rewards is to iterate from last to first time tick
    and compute R_t = r_t + gamma*R_{t+1} recurrently

    You must return an array/list of cumulative rewards with as many elements as in the initial rewards.
    """

    cumulative_rewards = np.empty_like(rewards)
    cumulative_rewards = cumulative_rewards.astype(float)
    cumulative_rewards[-1] = rewards[-1]

    for index in range(len(rewards) - 2, -1, -1):
        discount = cumulative_rewards[index + 1] * gamma
        reward = rewards[index]
        cumulative_rewards[index] = discount + reward

    return cumulative_rewards   # <array of cumulative rewards>


def train_step(_states, _actions, _rewards):
    """given full session, trains agent with policy gradient"""
    _cumulative_rewards = get_cumulative_rewards(_rewards)
    update.run({states:_states, actions: _actions, cumulative_rewards:_cumulative_rewards})


# utility function to pick action in one given state
def get_action_proba(s):
    return policy.eval({states: [s]})[0]


def generate_session(t_max=1000):
    """play env with REINFORCE agent and train at the session end"""

    # arrays to record session
    states, actions, rewards = [], [], []

    s = env.reset()

    for t in range(t_max):

        # action probabilities array aka pi(a|s)
        action_probas = get_action_proba(s)

        a = np.random.choice(range(n_actions), p=action_probas)  # <pick random action using action_probas>

        new_s, r, done, info = env.step(a)

        # record session history to train later
        states.append(s)
        actions.append(a)
        rewards.append(r)

        s = new_s
        if done:
            break

    train_step(states, actions, rewards)

    return sum(rewards)


env = gym.make("CartPole-v0")

env.reset()
n_actions = env.action_space.n
state_dim = env.observation_space.shape
print(state_dim)

# create input variables. We only need <s,a,R> for REINFORCE
states = tf.placeholder('float32', (None,) + state_dim, name="states")
actions = tf.placeholder('int32', name="action_ids")
cumulative_rewards = tf.placeholder('float32', name="cumulative_returns")


# <define network graph using raw tf or any deep learning library>
layer1 = tf.contrib.layers.fully_connected(states, 100, activation_fn=tf.nn.relu)
layer2 = tf.contrib.layers.fully_connected(layer1, 50, activation_fn=tf.nn.relu)
logits = tf.layers.dense(layer2, n_actions)  # <linear outputs (symbolic) of your network>

policy = tf.nn.softmax(logits)
log_policy = tf.nn.log_softmax(logits)


# get probabilities for parti
indices = tf.stack([tf.range(tf.shape(log_policy)[0]), actions], axis=-1)
log_policy_for_actions = tf.gather_nd(log_policy, indices)


# policy objective as in the last formula. please use mean, not sum.
# note: you need to use log_policy_for_actions to get log probabilities for actions taken
J = tf.reduce_mean(log_policy_for_actions * cumulative_rewards)


# regularize with entropy
entropy = - tf.reduce_mean(policy * log_policy)  # <compute entropy. Don't forget the sign!>


# all network weights
all_weights = [v for v in tf.trainable_variables()]  # <a list of all trainable weights in your network>


# weight updates. maximizing J is same as minimizing -J. Adding negative entropy.
loss = -J - (0.1 * entropy)
update = tf.train.AdamOptimizer().minimize(loss, var_list=all_weights)


assert len(get_cumulative_rewards(range(100))) == 100
assert np.allclose(get_cumulative_rewards([0,0,1,0,0,1,0],gamma=0.9),[1.40049, 1.5561, 1.729, 0.81, 0.9, 1.0, 0.0])
assert np.allclose(get_cumulative_rewards([0,0,1,-2,3,-4,0],gamma=0.5), [0.0625, 0.125, 0.25, -1.5, 1.0, -4.0, 0.0])
assert np.allclose(get_cumulative_rewards([0,0,1,2,3,4,0],gamma=0), [0, 0, 1, 2, 3, 4, 0])
print("looks good!")


s = tf.InteractiveSession()
s.run(tf.global_variables_initializer())

for i in range(100):

    rewards = [generate_session() for _ in range(100)]  # generate new sessions

    print("mean reward:%.3f" % (np.mean(rewards)))

    if np.mean(rewards) > 300:
        print("You Win!")
        break


env = gym.wrappers.Monitor(gym.make("CartPole-v0"), directory="videos", force=True)
sessions = [generate_session() for _ in range(100)]
env.close()
