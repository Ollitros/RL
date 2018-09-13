from value_based.mdp import MDP
from value_based.mdp import has_graphviz
from value_based.mdp import plot_graph, plot_graph_with_state_values, plot_graph_optimal_strategy_and_state_values
from value_based.mdp import FrozenLakeEnv
from IPython.display import display
from IPython.display import clear_output
from time import sleep
import numpy as np
import matplotlib.pyplot as plt


def get_action_value(mdp, state_values, state, action, gamma):

    Q = 0
    next_states = mdp.get_next_states(state, action)
    print('next_states', next_states)
    # print(state_values)

    for next_state in next_states:
      probability = next_states[next_state]
      reward = mdp.get_reward(state, action, next_state)
      print(next_state, probability, reward)
      Q += probability * (reward + gamma * state_values[next_state])
      print('Q', Q)

    return Q


def get_state_value(mdp, state_values, state, gamma):

    if mdp.is_terminal(state): return 0

    actions = mdp.get_possible_actions(state)
    print('actions',actions)
    state_value = 0
    for action in actions:
      print('action', action)
      action_value = get_action_value(mdp, state_values, state, action, gamma)
      print('action_value', action_value)
      if action_value > state_value:
        state_value = action_value

    return state_value


def get_optimal_action(mdp, state_values, state, gamma=0.9):

    if mdp.is_terminal(state):
        return None

    actions = mdp.get_possible_actions(state)
    print("GET POSSIBLE ACTIONS: ", actions)
    optimal_action = None
    optimal_action_value = - float("inf")
    for action in actions:
        action_value = get_action_value(mdp, state_values, state, action, gamma)
        if action_value >= optimal_action_value:
            optimal_action_value = action_value
            optimal_action = action

    print('optimal action', optimal_action)
    return optimal_action


def value_iteration(mdp, state_values=None, gamma=0.9, num_iter=1000, min_difference=1e-5):
  state_values = state_values or {s: 0 for s in mdp.get_all_states()}
  new_state_values = {}
  for i in range(num_iter):

    # Compute new state values using the functions you defined above. It must be a dict {state : new_V(state)}
    for s in state_values:
      new_state_values[s] = get_state_value(mdp, state_values, s, gamma)
    assert isinstance(new_state_values, dict)

    # Compute difference
    diff = max(abs(new_state_values[s] - state_values[s]) for s in mdp.get_all_states())
    print("iter %4i   |   diff: %6.5f   |   V(start): %.3f " % (i, diff, new_state_values[mdp._initial_state]))

    state_values = dict(new_state_values)
    if diff < min_difference:
      print("Terminated")
      break

  return state_values


def draw_policy(mdp, state_values, gamma):
    plt.figure(figsize=(3, 3))
    h, w = mdp.desc.shape
    states = sorted(mdp.get_all_states())
    V = np.array([state_values[s] for s in states])
    Pi = {s: get_optimal_action(mdp, state_values, s, gamma) for s in states}
    plt.imshow(V.reshape(w, h), cmap='gray', interpolation='none', clim=(0, 1))
    ax = plt.gca()
    ax.set_xticks(np.arange(h) - .5)
    ax.set_yticks(np.arange(w) - .5)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    Y, X = np.mgrid[0:4, 0:4]
    a2uv = {'left': (-1, 0), 'down': (0, -1), 'right': (1, 0), 'up': (-1, 0)}
    for y in range(h):
        for x in range(w):
            plt.text(x, y, str(mdp.desc[y, x].item()),
                     color='g', size=12, verticalalignment='center',
                     horizontalalignment='center', fontweight='bold')
            a = Pi[y, x]
            if a is None: continue
            u, v = a2uv[a]
            plt.arrow(x, y, u * .3, -v * .3, color='m', head_width=0.1, head_length=0.1)
    plt.grid(color='b', lw=2, ls='-')
    plt.show()


def start_example():
    transition_probs = {
      's0':{
        'a0': {'s0': 0.5, 's2': 0.5},
        'a1': {'s2': 1}
      },
      's1':{
        'a0': {'s0': 0.7, 's1': 0.1, 's2': 0.2},
        'a1': {'s1': 0.95, 's2': 0.05}
      },
      's2':{
        'a0': {'s0': 0.4, 's1': 0.6},
        'a1': {'s0': 0.3, 's1': 0.3, 's2':0.4}
      }
    }
    rewards = {
      's1': {'a0': {'s0': +5}},
      's2': {'a1': {'s0': -1}}
    }


    # mdp = MDP(transition_probs, rewards, initial_state='s0')
    #
    # print('initial state =', mdp.reset())
    # next_state, reward, done, info = mdp.step('a1')
    # print('next_state = %s, reward = %s, done = %s' % (next_state, reward, done))
    # print("mdp.get_all_states =", mdp.get_all_states())
    # print("mdp.get_possible_actions('s1') = ", mdp.get_possible_actions('s1'))
    # print("mdp.get_next_states('s1', 'a0') = ", mdp.get_next_states('s1', 'a0'))
    # print("mdp.get_reward('s1', 'a0', 's0') = ", mdp.get_reward('s1', 'a0', 's0'))
    # print("mdp.get_transition_prob('s1', 'a0', 's0') = ", mdp.get_transition_prob('s1', 'a0', 's0'))
    #
    # print("Graphviz available:", has_graphviz)
    #
    # display(plot_graph(mdp))


    mdp = MDP(transition_probs, rewards, initial_state='s0')

    test_Vs = {s: i for i, s in enumerate(sorted(mdp.get_all_states()))}
    assert np.allclose(get_action_value(mdp, test_Vs, 's2', 'a1', 0.9), 0.69)
    assert np.allclose(get_action_value(mdp, test_Vs, 's1', 'a0', 0.9), 3.95)

    test_Vs_copy = dict(test_Vs)
    assert np.allclose(get_state_value(mdp, test_Vs, 's0', 0.9), 1.8)
    assert np.allclose(get_state_value(mdp, test_Vs, 's2', 0.9), 0.69)
    assert test_Vs == test_Vs_copy, "please do not change state_values in get_new_state_value"


    # parameters
    gamma = 0.9  # discount for MDP
    num_iter = 100  # maximum iterations, excluding initialization
    min_difference = 0.001  # stop VI if new values are this close to old values (or closer)

    # initialize V(s)
    state_values = {s: 0 for s in mdp.get_all_states()}

    if has_graphviz:
        display(plot_graph_with_state_values(mdp, state_values))

    new_state_values = {}
    print("state values: ", state_values)
    for i in range(num_iter):

        # Compute new state values using the functions you defined above. It must be a dict {state : new_V(state)}
        for s in state_values:
            new_state_values[s] = get_state_value(mdp, state_values, s, gamma)
        assert isinstance(new_state_values, dict)

        # Compute difference
        print('new_state_values', new_state_values)
        print('state_values', state_values)
        diffs = [abs(new_state_values[s] - state_values[s]) for s in mdp.get_all_states()]
        # print(diffs)
        diff = max(abs(new_state_values[s] - state_values[s]) for s in mdp.get_all_states())
        print("iter %4i   |   diff: %6.5f   |   " % (i, diff), end="")
        print('   '.join("V(%s) = %.3f" % (s, v) for s, v in state_values.items()), end='\n\n')
        state_values = dict(new_state_values)
        print(state_values)


        if diff < min_difference:
            print("Terminated")
            break

    print("Final state values:", state_values)

    assert abs(state_values['s0'] - 8.032) < 0.01
    assert abs(state_values['s1'] - 11.169) < 0.01
    assert abs(state_values['s2'] - 8.921) < 0.01

    assert get_optimal_action(mdp, state_values, 's0', gamma) == 'a1'
    assert get_optimal_action(mdp, state_values, 's1', gamma) == 'a0'
    assert get_optimal_action(mdp, state_values, 's2', gamma) == 'a0'

    # Measure agent's average reward

    s = mdp.reset()
    rewards = []
    for _ in range(10000):
        s, r, done, _ = mdp.step(get_optimal_action(mdp, state_values, s, gamma))
        rewards.append(r)

    print("Average reward: ", np.mean(rewards))

    assert (0.85 < np.mean(rewards) < 1.0)


    ### START THE GAME 4x4 ###


    mdp = FrozenLakeEnv(slip_chance=0)
    mdp.render()

    state_values = value_iteration(mdp)

    s = mdp.reset()
    mdp.render()

    for t in range(100):
        a = get_optimal_action(mdp, state_values, s, gamma)
        print(a, end='\n\n')
        s, r, done, _ = mdp.step(a)
        mdp.render()
        if done:
            break
    state_values = {s: 0 for s in mdp.get_all_states()}

    for i in range(10):
        print("after iteration %i" % i)
        state_values = value_iteration(mdp, state_values, num_iter=1)
        draw_policy(mdp, state_values, gamma)
        # please ignore iter 0 at each step



    ### START THE GAME 8x8 ###


    mdp = FrozenLakeEnv(map_name='8x8', slip_chance=0.1)
    state_values = {s: 0 for s in mdp.get_all_states()}

    for i in range(30):
      clear_output(True)
      print("after iteration %i" % i)
      state_values = value_iteration(mdp, state_values, num_iter=1)
      draw_policy(mdp, state_values, gamma)
      sleep(0.5)
    # please ignore iter 0 at each step

    mdp = FrozenLakeEnv(slip_chance=0)
    state_values = value_iteration(mdp)

    total_rewards = []
    for game_i in range(1000):
        s = mdp.reset()
        rewards = []
        for t in range(100):
            s, r, done, _ = mdp.step(get_optimal_action(mdp, state_values, s, gamma))
            rewards.append(r)
            if done: break
        total_rewards.append(np.sum(rewards))

    print("average reward: ", np.mean(total_rewards))
    assert (1.0 <= np.mean(total_rewards) <= 1.0)
    print("Well done!")


    #
    # Measure agent's average reward
    #


    mdp = FrozenLakeEnv(slip_chance=0.1)
    state_values = value_iteration(mdp)

    total_rewards = []
    for game_i in range(1000):
        s = mdp.reset()
        rewards = []
        for t in range(100):
            s, r, done, _ = mdp.step(get_optimal_action(mdp, state_values, s, gamma))
            rewards.append(r)
            if done:
                break
        total_rewards.append(np.sum(rewards))

    print("average reward: ", np.mean(total_rewards))
    assert (0.8 <= np.mean(total_rewards) <= 0.95)
    print("Well done!")

    # Measure agent's average reward
    mdp = FrozenLakeEnv(slip_chance=0.25)
    state_values = value_iteration(mdp)

    total_rewards = []
    for game_i in range(1000):
        s = mdp.reset()
        rewards = []
        for t in range(100):
            s, r, done, _ = mdp.step(get_optimal_action(mdp, state_values, s, gamma))
            rewards.append(r)
            if done:
                break
        total_rewards.append(np.sum(rewards))

    print("average reward: ", np.mean(total_rewards))
    assert (0.6 <= np.mean(total_rewards) <= 0.7)
    print("Well done!")

    # Measure agent's average reward
    mdp = FrozenLakeEnv(slip_chance=0.2, map_name='8x8')
    state_values = value_iteration(mdp)

    total_rewards = []
    for game_i in range(1000):
        s = mdp.reset()
        rewards = []
        for t in range(100):
            s, r, done, _ = mdp.step(get_optimal_action(mdp, state_values, s, gamma))
            rewards.append(r)
            if done:
                break
        total_rewards.append(np.sum(rewards))

    print("average reward: ", np.mean(total_rewards))
    assert (0.6 <= np.mean(total_rewards) <= 0.8)
    print("Well done!")


if __name__ == "__main__":

    start_example()