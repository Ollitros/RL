from IPython.core import display
import matplotlib.pyplot as plt
import numpy as np
import gym
from pomdp_based.atari_util import PreprocessAtari
import tensorflow as tf
from keras.layers import Conv2D, Dense, Flatten
from pomdp_based.env_pool import EnvPool


def make_env():
    env = gym.make("KungFuMasterDeterministic-v0")
    env = PreprocessAtari(env, height=42, width=42,
                          crop = lambda img: img[60:-30, 15:],
                          dim_order = 'tensorflow',
                          color=False, n_frames=1)
    return env


def evaluate(agent, env, n_games=1):
    """Plays an entire game start to end, returns session rewards."""

    game_rewards = []
    for _ in range(n_games):
        # initial observation and memory
        observation = env.reset()
        prev_memories = agent.get_initial_state(1)

        total_reward = 0
        while True:
            new_memories, readouts = agent.step(prev_memories, observation[None, ...])
            action = agent.sample_actions(readouts)

            observation, reward, done, info = env.step(action[0])

            total_reward += reward
            prev_memories = new_memories
            if done: break

        game_rewards.append(total_reward)
    return game_rewards


def sample_batch(rollout_length=10):
    prev_mem = pool.prev_memory_states
    rollout_obs, rollout_actions, rollout_rewards, rollout_mask = pool.interact(rollout_length)

    feed_dict = {
        observations_ph: rollout_obs,
        actions_ph: rollout_actions,
        rewards_ph: rollout_rewards,
        mask_ph: rollout_mask,
    }
    for placeholder, value in zip(initial_memory_ph, prev_mem):
        feed_dict[placeholder] = value
    return feed_dict


# Simple agent for fully-observable MDP
class FeedforwardAgent:
    def __init__(self, name, obs_shape, n_actions, reuse=False):
        """A simple actor-critic agent"""

        with tf.variable_scope(name, reuse=reuse):
            # Note: number of units/filters is arbitrary, you can and should change it at your will
            self.conv0 = Conv2D(32, (3, 3), strides=(2, 2), activation='elu')
            self.conv1 = Conv2D(32, (3, 3), strides=(2, 2), activation='elu')
            self.conv2 = Conv2D(32, (3, 3), strides=(2, 2), activation='elu')
            self.flatten = Flatten()
            self.hid = Dense(128, activation='elu')
            self.logits = Dense(n_actions)
            self.state_value = Dense(1)

            # prepare a graph for agent step
            _initial_state = self.get_initial_state(1)
            self.prev_state_placeholders = [tf.placeholder(m.dtype,
                                                           [None] + [m.shape[i] for i in range(1, m.ndim)])
                                            for m in _initial_state]
            self.obs_t = tf.placeholder('float32', [None, ] + list(obs_shape))
            self.next_state, self.agent_outputs = self.symbolic_step(self.prev_state_placeholders, self.obs_t)

    def symbolic_step(self, prev_state, obs_t):
        """Takes agent's previous step and observation, returns next state and whatever it needs to learn (tf tensors)"""

        nn = self.conv0(obs_t)
        nn = self.conv1(nn)
        nn = self.conv2(nn)
        nn = self.flatten(nn)
        nn = self.hid(nn)
        logits = self.logits(nn)
        state_value = self.state_value(nn)

        # feedforward agent has no state
        new_state = []

        return new_state, (logits, state_value)

    def get_initial_state(self, batch_size):
        """Return a list of agent memory states at game start. Each state is a np array of shape [batch_size, ...]"""
        # feedforward agent has no state
        return []

    def step(self, prev_state, obs_t):
        """Same as symbolic state except it operates on numpy arrays"""
        sess = tf.get_default_session()
        feed_dict = {self.obs_t: obs_t}
        for state_ph, state_value in zip(self.prev_state_placeholders, prev_state):
            feed_dict[state_ph] = state_value
        return sess.run([self.next_state, self.agent_outputs], feed_dict)

    def sample_actions(self, agent_outputs):
        """pick actions given numeric agent outputs (np arrays)"""
        logits, state_values = agent_outputs
        policy = np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)
        return [np.random.choice(len(p), p=p) for p in policy]


env = make_env()

obs_shape = env.observation_space.shape
n_actions = env.action_space.n

print("Observation shape:", obs_shape)
print("Num actions:", n_actions)
print("Action names:", env.env.env.get_action_meanings())

s = env.reset()
for _ in range(100):
    s, _, _, _ = env.step(env.action_space.sample())

# plt.title('Game image')
# plt.imshow(env.render('rgb_array'))
# plt.show()
#
# plt.title('Agent observation')
# plt.imshow(s.reshape([42,42]))
# plt.show()
#

tf.reset_default_graph()
sess = tf.InteractiveSession()

n_parallel_games = 5
gamma = 0.99

agent = FeedforwardAgent("agent", obs_shape, n_actions)

sess.run(tf.global_variables_initializer())

state = [env.reset()]
_, (logits, value) = agent.step(agent.get_initial_state(1), state)
print("action logits:\n", logits)
print("state values:\n", value)


# Let's play!
env_monitor = gym.wrappers.Monitor(env, directory="kungfu_videos", force=True)
rw = evaluate(agent, env_monitor, n_games=30,)
env_monitor.close()
print (rw)


# Training on parallel games
pool = EnvPool(agent, make_env, n_parallel_games)

# for each of n_parallel_games, take 10 steps
rollout_obs, rollout_actions, rollout_rewards, rollout_mask = pool.interact(10)

print("Actions shape:", rollout_actions.shape)
print("Rewards shape:", rollout_rewards.shape)
print("Mask shape:", rollout_mask.shape)
print("Observations shape: ",rollout_obs.shape)

