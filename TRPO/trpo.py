import numpy as np
import torch, torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import gym
import scipy.signal
from numpy.linalg import inv
import time
from itertools import count
from collections import OrderedDict

cuda = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Assume that we are on a CUDA machine, then this should print a CUDA device:

print(cuda)

env = gym.make("Acrobot-v1")
env.reset()
observation_shape = env.observation_space.shape
n_actions = env.action_space.n
print("Observation Space", env.observation_space)
print("Action Space", env.action_space)



class TRPOAgent(nn.Module):
    def __init__(self, state_shape, n_actions, hidden_size=32):
        '''
        Here you should define your model
        You should have LOG-PROBABILITIES as output because you will need it to compute loss
        We recommend that you start simple:
        use 1-2 hidden layers with 100-500 units and relu for the first try
        '''
        nn.Module.__init__(self)

        # <your network here>
        self.fc1 = nn.Linear(state_shape[0], 128)
        self.fc2 = nn.Linear(128, hidden_size)
        self.fc3 = nn.Linear(hidden_size, n_actions)
        # self.model = None

    def forward(self, states):
        """
        takes agent's observation (Variable), returns log-probabilities (Variable)
        :param state_t: a batch of states, shape = [batch_size, state_shape]
        """

        # Use your network to compute log_probs for given state
        x = F.relu(self.fc1(states))
        x = F.relu(self.fc2(x))
        x = F.log_softmax(self.fc3(x))
        log_probs = x
        # log_probs = self.model(states)
        return log_probs

    def get_log_probs(self, states):
        '''
        Log-probs for training
        '''

        return self.forward(states)

    def get_probs(self, states):
        '''
        Probs for interaction
        '''

        return torch.exp(self.forward(states))

    def act(self, obs, sample=True):
        '''
        Samples action from policy distribution (sample = True) or takes most likely action (sample = False)
        :param: obs - single observation vector
        :param sample: if True, samples from \pi, otherwise takes most likely action
        :returns: action (single integer) and probabilities for all actions
        '''

        probs = self.get_probs(Variable(torch.FloatTensor([obs]))).data.numpy()

        if sample:
            action = int(np.random.choice(n_actions, p=probs[0]))
        else:
            action = int(np.argmax(probs))

        return action, probs[0]


def get_flat_params_from(model):
    params = []
    for param in model.parameters():
        params.append(param.data.view(-1))

    flat_params = torch.cat(params)
    return flat_params


def set_flat_params_to(model, flat_params):
    prev_ind = 0
    for param in model.parameters():
        flat_size = int(np.prod(list(param.size())))
        param.data.copy_(
            flat_params[prev_ind:prev_ind + flat_size].view(param.size()))
        prev_ind += flat_size


def get_cummulative_returns(r, gamma=1):
    """
    Computes cummulative discounted rewards given immediate rewards
    G_i = r_i + gamma*r_{i+1} + gamma^2*r_{i+2} + ...
    Also known as R(s,a).
    """
    r = np.array(r)
    assert r.ndim >= 1
    return scipy.signal.lfilter([1], [1, -gamma], r[::-1], axis=0)[::-1]


def rollout(env, agent, max_pathlength=2500, n_timesteps=50000):
    """
    Generate rollouts for training.
    :param: env - environment in which we will make actions to generate rollouts.
    :param: act - the function that can return policy and action given observation.
    :param: max_pathlength - maximum size of one path that we generate.
    :param: n_timesteps - total sum of sizes of all pathes we generate.
    """
    paths = []

    total_timesteps = 0
    while total_timesteps < n_timesteps:
        obervations, actions, rewards, action_probs = [], [], [], []
        obervation = env.reset()
        for _ in range(max_pathlength):
            action, policy = agent.act(obervation)
            obervations.append(obervation)
            actions.append(action)
            action_probs.append(policy)
            obervation, reward, done, _ = env.step(action)
            rewards.append(reward)
            total_timesteps += 1
            if done or total_timesteps==n_timesteps:
                path = {"observations": np.array(obervations),
                        "policy": np.array(action_probs),
                        "actions": np.array(actions),
                        "rewards": np.array(rewards),
                        "cumulative_returns":get_cummulative_returns(rewards),
                       }
                paths.append(path)
                break
    return paths


def get_loss(agent, observations, actions, cummulative_returns, old_probs):
    """
    Computes TRPO objective
    :param: observations - batch of observations
    :param: actions - batch of actions
    :param: cummulative_returns - batch of cummulative returns
    :param: old_probs - batch of probabilities computed by old network
    :returns: scalar value of the objective function
    """
    batch_size = observations.shape[0]
    log_probs_all = agent.get_log_probs(observations)
    probs_all = torch.exp(log_probs_all)

    probs_for_actions = probs_all[torch.arange(0, batch_size, out=torch.LongTensor()), actions]
    old_probs_for_actions = old_probs[torch.arange(0, batch_size, out=torch.LongTensor()), actions]

    # Compute surrogate loss, aka importance-sampled policy gradient
    Loss = torch.mean(((probs_for_actions / old_probs_for_actions) * cummulative_returns), dim=0,
                      keepdim=True)  # <compute surrogate loss>
    Loss = -Loss

    assert Loss.shape == torch.Size([1])
    return Loss


def get_kl(agent, observations, actions, cummulative_returns, old_probs):
    """
    Computes KL-divergence between network policy and old policy
    :param: observations - batch of observations
    :param: actions - batch of actions
    :param: cummulative_returns - batch of cummulative returns (we don't need it actually)
    :param: old_probs - batch of probabilities computed by old network
    :returns: scalar value of the KL-divergence
    """
    # print(observations.shape)
    batch_size = observations.shape[0]
    log_probs_all = agent.get_log_probs(observations)
    probs_all = torch.exp(log_probs_all)

    # Compute Kullback-Leibler divergence (see formula above)
    # Note: you need to sum KL and entropy over all actions, not just the ones agent took
    old_log_probs = torch.log(old_probs + 1e-10)

    # print(log_probs_all.shape)
    # print(old_log_probs.shape)

    kl = torch.mean(torch.sum(old_probs * (old_log_probs - log_probs_all), 1), dim=0,
                    keepdim=True)  # <compute kullback-leibler>

    # print(kl.shape)

    assert kl.shape == torch.Size([1])
    assert (kl > -0.0001).all() and (kl < 10000).all()
    return kl


def get_entropy(agent, observations):
    """
    Computes entropy of the network policy
    :param: observations - batch of observations
    :returns: scalar value of the entropy
    """

    observations = Variable(torch.FloatTensor(observations))

    batch_size = observations.shape[0]
    log_probs_all = agent.get_log_probs(observations)
    probs_all = torch.exp(log_probs_all)

    # print(probs_all.shape)

    entropy = torch.sum(-probs_all * log_probs_all) / batch_size
    entropy = entropy.unsqueeze(0)
    # print(entropy.shape)

    assert entropy.shape == torch.Size([1])
    return entropy


def linesearch(f, x, fullstep, max_kl):
    """
    Linesearch finds the best parameters of neural networks in the direction of fullstep contrainted by KL divergence.
    :param: f - function that returns loss, kl and arbitrary third component.
    :param: x - old parameters of neural network.
    :param: fullstep - direction in which we make search.
    :param: max_kl - constraint of KL divergence.
    :returns:
    """
    max_backtracks = 10
    loss, _, = f(x)
    for stepfrac in .5**np.arange(max_backtracks):
        xnew = x + stepfrac * fullstep
        new_loss, kl = f(xnew)
        actual_improve = new_loss - loss
        if kl.data.numpy()<=max_kl and actual_improve.data.numpy() < 0:
            x = xnew
            loss = new_loss
    return x


def conjugate_gradient(f_Ax, b, cg_iters=10, residual_tol=1e-10):
    """
    This method solves system of equation Ax=b using iterative method called conjugate gradients
    :f_Ax: function that returns Ax
    :b: targets for Ax
    :cg_iters: how many iterations this method should do
    :residual_tol: epsilon for stability
    """
    p = b.clone()
    r = b.clone()
    x = torch.zeros(b.size())
    rdotr = torch.sum(r*r)
    for i in range(cg_iters):
        z = f_Ax(p)
        v = rdotr / (torch.sum(p*z) + 1e-8)
        x += v * p
        r -= v * z
        newrdotr = torch.sum(r*r)
        mu = newrdotr / (rdotr + 1e-8)
        p = r + mu * p
        rdotr = newrdotr
        if rdotr < residual_tol:
            break
    return x


def update_step(agent, observations, actions, cummulative_returns, old_probs, max_kl):
    """
    This function does the TRPO update step
    :param: observations - batch of observations
    :param: actions - batch of actions
    :param: cummulative_returns - batch of cummulative returns
    :param: old_probs - batch of probabilities computed by old network
    :param: max_kl - controls how big KL divergence may be between old and new policy every step.
    :returns: KL between new and old policies and the value of the loss function.
    """

    # Here we prepare the information
    observations = Variable(torch.FloatTensor(observations))
    actions = torch.LongTensor(actions)
    cummulative_returns = Variable(torch.FloatTensor(cummulative_returns))
    old_probs = Variable(torch.FloatTensor(old_probs))

    # Here we compute gradient of the loss function
    loss = get_loss(agent, observations, actions, cummulative_returns, old_probs)
    grads = torch.autograd.grad(loss, agent.parameters())
    loss_grad = torch.cat([grad.view(-1) for grad in grads]).data

    def Fvp(v):
        # Here we compute Fx to do solve Fx = g using conjugate gradients
        # We actually do here a couple of tricks to compute it efficiently

        kl = get_kl(agent, observations, actions, cummulative_returns, old_probs)

        grads = torch.autograd.grad(kl, agent.parameters(), create_graph=True)
        flat_grad_kl = torch.cat([grad.view(-1) for grad in grads])

        kl_v = (flat_grad_kl * Variable(v)).sum()
        grads = torch.autograd.grad(kl_v, agent.parameters())
        flat_grad_grad_kl = torch.cat([grad.contiguous().view(-1) for grad in grads]).data

        return flat_grad_grad_kl + v * 0.1

    # Here we solveolve Fx = g system using conjugate gradients
    stepdir = conjugate_gradient(Fvp, -loss_grad, 10)

    # Here we compute the initial vector to do linear search
    shs = 0.5 * (stepdir * Fvp(stepdir)).sum(0, keepdim=True)

    lm = torch.sqrt(shs / max_kl)
    fullstep = stepdir / lm[0]

    neggdotstepdir = (-loss_grad * stepdir).sum(0, keepdim=True)

    # Here we get the start point
    prev_params = get_flat_params_from(agent)

    def get_loss_kl(params):
        # Helper for linear search
        set_flat_params_to(agent, params)
        return [get_loss(agent, observations, actions, cummulative_returns, old_probs),
                get_kl(agent, observations, actions, cummulative_returns, old_probs)]

    # Here we find our new parameters
    new_params = linesearch(get_loss_kl, prev_params, fullstep, max_kl)

    # And we set it to our network
    set_flat_params_to(agent, new_params)

    return get_loss_kl(new_params)



agent = TRPOAgent(observation_shape, n_actions)


# Check if log-probabilities satisfies all the requirements
log_probs = agent.get_log_probs(Variable(torch.FloatTensor([env.reset()])))
assert isinstance(log_probs, Variable) and log_probs.requires_grad, "qvalues must be a torch variable with grad"
assert len(log_probs.shape) == 2 and log_probs.shape[0] == 1 and log_probs.shape[1] == n_actions
sums = torch.sum(torch.exp(log_probs), dim=1)
assert (0.999<sums).all() and (1.001>sums).all()

# Demo use
print("sampled:", [agent.act(env.reset()) for _ in range(5)])
print("greedy:", [agent.act(env.reset(),sample=False) for _ in range(5)])

# simple demo on rewards [0,0,1,0,0,1]
get_cummulative_returns([0,0,1,0,0,1], gamma=0.9)

paths = rollout(env, agent, max_pathlength=5, n_timesteps=100)
print (paths[-1])
assert (paths[0]['policy'].shape==(5, n_actions))
assert (paths[0]['cumulative_returns'].shape==(5,))
assert (paths[0]['rewards'].shape==(5,))
assert (paths[0]['observations'].shape==(5,)+observation_shape)
assert (paths[0]['actions'].shape==(5,))
print('It\'s ok')

# This code validates conjugate gradients
A = np.random.rand(8, 8)
A = np.matmul(np.transpose(A), A)

def f_Ax(x):
    return torch.matmul(torch.FloatTensor(A), x.view((-1, 1))).view(-1)

b = np.random.rand(8)

w = np.matmul(np.matmul(inv(np.matmul(np.transpose(A), A)), np.transpose(A)), b.reshape((-1, 1))).reshape(-1)
print (w)
print (conjugate_gradient(f_Ax, torch.FloatTensor(b)).numpy())

max_kl = 0.01  # this is hyperparameter of TRPO. It controls how big KL divergence may be between old and new policy every step.
numeptotal = 0  # this is number of episodes that we played.

start_time = time.time()


for i in range(10):

    print("\n********** Iteration %i ************" % i)

    # Generating paths.
    print("Rollout")
    paths = rollout(env, agent)
    print("Made rollout")

    # Updating policy.
    observations = np.concatenate([path["observations"] for path in paths])
    actions = np.concatenate([path["actions"] for path in paths])
    returns = np.concatenate([path["cumulative_returns"] for path in paths])
    old_probs = np.concatenate([path["policy"] for path in paths])

    loss, kl = update_step(agent, observations, actions, returns, old_probs, max_kl)

    # Report current progress
    episode_rewards = np.array([path["rewards"].sum() for path in paths])

    stats = OrderedDict()
    numeptotal += len(episode_rewards)
    stats["Total number of episodes"] = numeptotal
    stats["Average sum of rewards per episode"] = episode_rewards.mean()
    stats["Std of rewards per episode"] = episode_rewards.std()
    stats["Time elapsed"] = "%.2f mins" % ((time.time() - start_time) / 60.)
    stats["KL between old and new distribution"] = kl.data.numpy()
    stats["Entropy"] = get_entropy(agent, observations).data.numpy()
    stats["Surrogate loss"] = loss.data.numpy()
    for k, v in stats.items():
        print(k + ": " + " " * (40 - len(k)) + str(v))
    i += 1