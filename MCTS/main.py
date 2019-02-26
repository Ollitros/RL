from MCTS.with_snapshots import WithSnapshots
from IPython.display import clear_output
from itertools import count
from pickle import dumps,loads
import gym
import numpy as np
import matplotlib.pyplot as plt

#make env
env = WithSnapshots(gym.make("CartPole-v0"))
env.reset()
n_actions = env.action_space.n


class Node:
    """ a tree node for MCTS """

    # metadata:
    parent = None  # parent Node
    value_sum = 0.  # sum of state values from all visits (numerator)
    times_visited = 0  # counter of visits (denominator)

    def __init__(self, parent, action, ):
        """
        Creates and empty node with no children.
        Does so by commiting an action and recording outcome.

        :param parent: parent Node
        :param action: action to commit from parent Node

        """

        self.parent = parent
        self.action = action
        self.children = set()  # set of child nodes

        # get action outcome and save it
        res = env.get_result(parent.snapshot, action)
        self.snapshot, self.observation, self.immediate_reward, self.is_done, _ = res

    def is_leaf(self):
        return len(self.children) == 0

    def is_root(self):
        return self.parent is None

    def get_mean_value(self):
        return self.value_sum / self.times_visited if self.times_visited != 0 else 0

    def ucb_score(self, scale=10, max_value=1e100):
        """
        Computes ucb1 upper bound using current value and visit counts for node and it's parent.

        :param scale: Multiplies upper bound by that. From hoeffding inequality, assumes reward range to be [0,scale].
        :param max_value: a value that represents infinity (for unvisited nodes)

        """

        if self.times_visited == 0:
            return max_value

        # compute ucb-1 additive component (to be added to mean value)
        # hint: you can use self.parent.times_visited for N times node was considered,
        # and self.times_visited for n times it was visited

        U = np.sqrt(np.log(self.parent.times_visited) / self.times_visited)  # <your code here>

        return self.get_mean_value() + scale * U

    # MCTS steps

    def select_best_leaf(self):
        """
        Picks the leaf with highest priority to expand
        Does so by recursively picking nodes with best UCB-1 score until it reaches the leaf.

        """
        if self.is_leaf():
            return self

        children = self.children
        child_values = {child: child.ucb_score() for child in children}

        best_child = max(child_values, key=child_values.get)  # <select best child node in terms of node.ucb_score()>

        return best_child.select_best_leaf()

    def expand(self):
        """
        Expands the current node by creating all possible child nodes.
        Then returns one of those children.
        """

        assert not self.is_done, "can't expand from terminal state"

        for action in range(n_actions):
            self.children.add(Node(self, action))

        return self.select_best_leaf()

    def rollout(self, t_max=10 ** 4):
        """
        Play the game from this state to the end (done) or for t_max steps.

        On each step, pick action at random (hint: env.action_space.sample()).

        Compute sum of rewards from current state till
        Note 1: use env.action_space.sample() for random action
        Note 2: if node is terminal (self.is_done is True), just return 0

        """

        # set env into the appropriate state
        env.load_snapshot(self.snapshot)
        obs = self.observation
        is_done = self.is_done

        # <your code here - rollout and compute reward>
        rollout_reward = 0
        for i in range(t_max):
            _, r, is_done, _ = env.step(env.action_space.sample())

            if is_done:
                return 0

            rollout_reward += r

        return rollout_reward

    def propagate(self, child_value):
        """
        Uses child value (sum of rewards) to update parents recursively.
        """
        # compute node value
        my_value = self.immediate_reward + child_value

        # update value_sum and times_visited
        self.value_sum += my_value
        self.times_visited += 1

        # propagate upwards
        if not self.is_root():
            self.parent.propagate(my_value)

    def safe_delete(self):
        """safe delete to prevent memory leak in some python versions"""
        del self.parent
        for child in self.children:
            child.safe_delete()
            del child


class Root(Node):
    def __init__(self, snapshot, observation):
        """
        creates special node that acts like tree root
        :snapshot: snapshot (from env.get_snapshot) to start planning from
        :observation: last environment observation
        """

        self.parent = self.action = None
        self.children = set()  # set of child nodes

        # root: load snapshot and observation
        self.snapshot = snapshot
        self.observation = observation
        self.immediate_reward = 0
        self.is_done = False

    @staticmethod
    def from_node(node):
        """initializes node as root"""
        root = Root(node.snapshot, node.observation)
        # copy data
        copied_fields = ["value_sum", "times_visited", "children", "is_done"]
        for field in copied_fields:
            setattr(root, field, getattr(node, field))
        return root


assert isinstance(env,WithSnapshots)

def plan_mcts(root,n_iters=10):
    """
    builds tree with monte-carlo tree search for n_iters iterations
    :param root: tree node to plan from
    :param n_iters: how many select-expand-simulate-propagete loops to make
    """
    for _ in range(n_iters):

        node = root.select_best_leaf() #<select best leaf>

        if node.is_done:
            node.propagate(0)

        else: #node is not terminal
            #<expand-simulate-propagate loop>
            child_node = node.expand()
            child_reward = child_node.rollout()
            node.propagate(child_reward)


root_observation = env.reset()
root_snapshot = env.get_snapshot()
root = Root(root_snapshot, root_observation)

#plan from root:
plan_mcts(root, n_iters=1000)

total_reward = 0  # sum of rewards
test_env = loads(root_snapshot)  # env used to show progress

for i in count():

    # get best child

    # <select child with highest mean reward>

    best_value = 0
    for child in root.children:
        if child.get_mean_value() >= best_value:
            best_value = child.get_mean_value()
            best_child = child

    # take action
    s, r, done, _ = test_env.step(best_child.action)

    # show image
    clear_output(True)
    plt.title("step %i" % i)
    plt.imshow(test_env.render('rgb_array'))

    total_reward += r
    if done:
        print("Finished with reward = ", total_reward)
        break

    # discard unrealized part of the tree [because not every child matters :(]
    for child in root.children:
        if child != best_child:
            child.safe_delete()

    # declare best child a new root
    root = Root.from_node(best_child)

    print(root.is_leaf())

    if root.is_leaf():
        plan_mcts(root, n_iters=1000)

    # assert not root.is_leaf(), "We ran out of tree! Need more planning! Try growing tree right inside the loop."

    # you may want to expand tree here
    # <your code here>