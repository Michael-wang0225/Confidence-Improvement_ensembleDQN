import numpy as np
import random
from collections import namedtuple, deque
import warnings

from models import QNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim
import dgl

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64  # minibatch size
GAMMA = 0.99  # discount factor
TAU = 1e-3  # for soft update of target parameters
LR = 5e-4  # learning rate
UPDATE_EVERY = 4  # how often to update the network
number_of_nets = 10 # number of ensemble member
safety_threshold = 2


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ensembleDQNAgent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, memory, train_policy, test_policy, seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.loss = []
        # initialize 10 ensemble members, so 10 local networks and 10 target networks.
        self.qnetworks_local = [ QNetwork(state_size, action_size, seed).to(device) for i in range(number_of_nets)]
        self.qnetworks_target = [ QNetwork(state_size, action_size, seed).to(device) for i in range(number_of_nets)]
        self.optimizer = [ optim.Adam(self.qnetworks_local[i].parameters(), lr=LR) for i in range(number_of_nets)]
        # print(self.optimizer)

        self.enough_samples = False

        # Replay memory
        self.memory = memory
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        # Initialize select_action policy
        self.policy=train_policy
        self.test_policy=test_policy

    def step(self, state, action, reward, done, coef_of_var):
        # Save experience in replay memory
        self.memory.append(state, action, reward, done, coef_of_var)
        if self.enough_samples == False:
            has_enough_data = [False for i in range(number_of_nets)]
            flag = True
            for net in range(number_of_nets):
                # print('check if enough data available')
                if len(self.memory.index_refs[net]) >= BATCH_SIZE:
                    has_enough_data[net] = True
            for x in has_enough_data: # check in every net, there are enough data,only when all nets are satisfied, flag will be finally setted with 1 (True*True==1)
                flag=flag*x
            if flag==1:
                self.enough_samples=True
                print('enough data in memory, start learning... \n')
        else:
            # Learn every UPDATE_EVERY time steps.
            self.t_step = (self.t_step + 1) % UPDATE_EVERY
            if self.t_step == 0 and self.enough_samples:
                # If enough samples are available in memory, get random subset and learn
                for active_net in range(number_of_nets):
                    self.learn_single_net(active_net, GAMMA)


    def act_train(self, state, active_model):
        """Returns action index and coefficient for given state as per Epsilon-greedy policy.
        
        Params
        ======
            state (array_like): current state
            active_model(int): one of the ensemble member, which is chosen as active
        """
        # First, select action using epsilon greedy policy
        ##############################################################
        state = torch.tensor(state).float().unsqueeze(0).to(device)
        self.qnetworks_local[active_model].eval()
        with torch.no_grad():
            action_values = self.qnetworks_local[active_model](state)

        # Epsilon-greedy action selection
        action = self.policy.select_action(action_values)
        ###############################################################
        # Then, calculate coef_of_var
        q_values_all_nets = []
        for net in range(number_of_nets):
            self.qnetworks_local[net].eval()
            with torch.no_grad():
                action_values = self.qnetworks_local[net](state).cpu().data.numpy().flatten()
                q_values_all_nets.append(action_values)

        q_values_all_nets = np.array(q_values_all_nets)
        assert q_values_all_nets.shape == (number_of_nets, self.action_size)

        mean_q_values = np.mean(q_values_all_nets)
        std_q_values = np.std(q_values_all_nets, axis=0)
        coef_of_var_all_actions = std_q_values / np.abs(mean_q_values) # This is the coef_of_var for all actions
        coef_of_var = coef_of_var_all_actions[action] # This is the coef_of_var for the chosen action
        #################################################################

        return action,coef_of_var



    def act_eval(self,state):
        """Returns action index and Information about the Q-values of the chosen action for given state as per current policy.

          Params
          ======
              state (array_like): current state
          """
        q_values_all_nets= []

        state = torch.tensor(state).float().unsqueeze(0).to(device)
        for net in range(number_of_nets):
            self.qnetworks_local[net].eval()
            with torch.no_grad():
                action_values = self.qnetworks_local[net](state).cpu().data.numpy().flatten()
                q_values_all_nets.append(action_values)

        q_values_all_nets=np.array(q_values_all_nets)
        assert q_values_all_nets.shape ==(number_of_nets, self.action_size)
        # Ensemble_test_policy action selection
        info={}
        action, policy_info= self.test_policy.select_action(q_values_all_nets=q_values_all_nets)
        info['q_values_all_nets']= q_values_all_nets
        # info['mean']= np.mean(q_values_all_nets[:, :],axis=0)
        info['mean'] = np.mean(q_values_all_nets, axis=0)
        # print('mean of q_values_all_nets is',info['mean'])
        info['standard_deviation'] = np.std(q_values_all_nets, axis=0)
        # print('standard_deviation is',info['standard_deviation'])
        info['coefficient_of_variation'] = np.std(q_values_all_nets, axis=0) /np.abs(info['mean'])

        # print('coefficient_of_variation is',info['coefficient_of_variation'])
        info.update(policy_info) # fallback action: True or False
        # print('fallback action is',info['fallback action'])

        return action, info

    def learn_single_net(self, active_net, gamma):
        experiences, importance_sample_weights = self.memory.sample(active_net, BATCH_SIZE)
        assert len(experiences) == BATCH_SIZE

        # Start by extracting the necessary parameters (we use a vectorized implementation).
        state_batch = []
        reward_batch = []
        action_batch = []
        done_batch = []
        next_state_batch = []
        for e in experiences:
            state_batch.append(e.state)
            next_state_batch.append(e.next_state)
            reward_batch.append(e.reward)
            action_batch.append(e.action)
            done_batch.append(1. if e.done else 0.)

        state_batch=torch.FloatTensor(state_batch).to(device)
        next_state_batch=torch.FloatTensor(next_state_batch).to(device)
        reward_batch=torch.FloatTensor(reward_batch).to(device)
        action_batch=torch.FloatTensor(action_batch).to(device)
        done_batch=torch.FloatTensor(done_batch).to(device)

        action_batch= action_batch.unsqueeze(1).type(torch.int64)
        done_batch = done_batch.unsqueeze(1)
        reward_batch = reward_batch.unsqueeze(1)
        state_batch = state_batch.squeeze(1)
        next_state_batch = next_state_batch.squeeze(1)

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetworks_target[active_net](next_state_batch).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states
        Q_targets = reward_batch + (gamma * Q_targets_next * (1 - done_batch))

        # Get expected Q values from local model
        self.qnetworks_local[active_net].train()
        Q_expected = self.qnetworks_local[active_net](state_batch).gather(1, action_batch)


        # Compute loss
        loss = self.weighted_loss(importance_sample_weights, Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer[active_net].zero_grad()
        loss.backward()
        self.optimizer[active_net].step()
        self.loss.append(loss.item())
        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetworks_local[active_net], self.qnetworks_target[active_net], TAU)


    def weighted_loss(self, weights, inputs, targets):
        t = torch.abs(inputs - targets)
        zi = torch.where(t < 1, 0.5 * t ** 2, t - 0.5)
        loss = (weights * zi).sum()
        return loss


    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed, number_of_nets=number_of_nets, adding_prob=0.5):
        """Initialize a ReplayBuffer object.
        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.batch_size = batch_size

        self.seed = random.seed(seed)

        self.limit= buffer_size
        self.number_of_nets = number_of_nets
        self.adding_prob = adding_prob
        self.index_refs = [[] for i in range(self.number_of_nets)]
        self.alpha = 1.0
        self.beta = 1.0

        self.actions = deque(maxlen=buffer_size)
        self.rewards = deque(maxlen=buffer_size)
        self.dones = deque(maxlen=buffer_size)
        self.states = deque(maxlen=buffer_size)
        self.coef_of_vars = deque(maxlen=buffer_size)

    def append(self, state, action, reward, done, coef_of_var):


        # #####################################################################################################
        # intent of self.index_refs will be created
        if self.nb_entries < self.limit:   # One more entry will be added after this loop
            # There should be enough experiences before the chosen sample to fill the window length + 1
            if self.nb_entries > 2:
                for i in range(self.number_of_nets):
                    if np.random.rand() < self.adding_prob:
                        self.index_refs[i].append(self.nb_entries)

        # Append state, action, reward, done, coef_of_var into memory
        ####################################################################################################
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.coef_of_vars.append(coef_of_var)
        if len(self.states) >= self.batch_size:
            for i in range(len(self.states)):
                if self.states[i]==state and self.actions[i]==action:# if state-action pair exited already in memory, coef_of_var should be updated
                    self.coef_of_vars[i]=coef_of_var
                    # del self.states[i]
                    # del self.actions[i]
                    # del self.rewards[i]
                    # del self.dones[i]
                    # del self.coef_of_vars[i]



    def sample_batch_idxs(self, net, batch_size):
        """
        Sample random replay memory indexes from the list of replay memory indexes of the specified ensemble member.

        Args:
            net (int): Index of ensemble member
            batch_size (int): Size of the batch

        Returns:
            A list of replay memory indexes.
        """
        memory_size = len(self.index_refs[net])
        assert memory_size > 2
        if batch_size > memory_size:
            warnings.warn("Less samples in memory than batch size.")
        ######################################################################
        #因为不是所有的index都加进来了，所以需要把加进去的index先选出来，再通过index把相应的coef_of_var提取出来，加入priorities里
        priorities = []
        for i in range(memory_size):
            priorities.append(self.coef_of_vars[self.index_refs[net][i]]+1) #to avoid probability=0, so add one
        priorities = np.array(priorities)
        ######################################################################
        sample_transition_prob = np.power(priorities, self.alpha) / sum(np.power(priorities, self.alpha))
        ref_idxs = np.random.choice(memory_size, size=batch_size, p=sample_transition_prob)
        batch_idxs = [self.index_refs[net][idx] for idx in ref_idxs]
        assert len(batch_idxs) == batch_size

        importance_sample_weights = np.power(memory_size * sample_transition_prob, -self.beta)
        importance_sample_weights = torch.from_numpy(importance_sample_weights).float().to(device)

        return batch_idxs, importance_sample_weights



    def sample(self, net, batch_size):
        """
        Returns a randomized batch of experiences for an ensemble member
        Args:
            net (int): Index of ensemble member
            batch_size (int): Size of the batch

        Returns:
            A list of random experiences
        """
        # It is not possible to tell whether the first state in the memory is terminal, because it
        # would require access to the "terminal" flag associated to the previous state. As a result
        # we will never return this first state (only using `self.terminals[0]` to know whether the
        # second state is terminal).
        # In addition we need enough entries to fill the desired window length.
        # print(self.nb_entries)
        assert self.nb_entries >= 3, 'not enough entries in the memory'

        # This is to be understood as a transition: Given `state0`, performing `action`
        # yields `reward` and results in `state1`, which might be `terminal`.
        Experience = namedtuple('Experience', 'state, action, reward, next_state, done')

        # Sample random indexes for the specified ensemble member
        batch_idxs, importance_sample_weights = self.sample_batch_idxs(net, batch_size)

        assert np.min(batch_idxs) >= 2
        assert np.max(batch_idxs) < self.nb_entries
        assert len(batch_idxs) == batch_size

        # Create experiences
        experiences = []
        for idx in batch_idxs:
            terminal0 = self.dones[idx - 2]
            while terminal0:
                # Skip this transition because the environment was reset here. Select a new, random
                # transition and use this instead. This may cause the batch to contain the same
                # transition twice.
                index, _ = self.sample_batch_idxs(net, 1)
                idx=index[0]
                terminal0 = self.dones[idx - 2]
            assert 2 <= idx < self.nb_entries

            # This code is slightly complicated by the fact that subsequent observations might be
            # from different episodes. We ensure that an experience never spans multiple episodes.
            # This is probably not that important in practice but it seems cleaner.
            state0 = [self.states[idx - 1]]
            action = self.actions[idx - 1]
            reward = self.rewards[idx - 1]
            terminal1 = self.dones[idx - 1]

            # Okay, now we need to create the follow-up state. This is state0 shifted on timestep
            # to the right. Again, we need to be careful to not include an observation from the next
            # episode if the last state is terminal.
            state1 = [np.copy(x) for x in state0[1:]]
            state1.append(self.states[idx])

            assert len(state0) == 1
            assert len(state1) == len(state0)
            experiences.append(Experience(state=state0, action=action, reward=reward,
                                          next_state=state1, done=terminal1))
        assert len(experiences) == batch_size
        return experiences, importance_sample_weights



    @property
    def nb_entries(self):
        """Return number of observations

        Returns:
            Number of states
        """
        return len(self.states)

class Epsilon_greedy_policy():
    """Implement the epsilon greedy policy

       Eps Greedy policy either:

       - takes a random action with probability epsilon
       - takes current best action with prob (1 - epsilon)
       """

    def __init__(self, action_size):
        self.eps=1.
        self.action_size=action_size
        self.eps_end = 0.01
        self.eps_decay = 0.9995

    def select_action(self, action_values):

        if random.random() > self.eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def epsilon_decay(self):
        self.eps = max(self.eps_end, self.eps_decay * self.eps)


class Ensemble_test_policy():
    """
    Policy used by the ensemble method during testing episodes.

    During testing episodes, the policy chooses the action with the highest mean Q-value
    If safety_threshold is set, only actions with a coefficient of variation below the set value are considered.
    If no action is considered safe, the fallback action safe_action is used.

    Args:
        safety_threshold (float): Maximum coefficient of variation that is considered safe.
        fallback_action (int): Fallback action if all actions are considered unsafe.
    """
    def __init__(self, saftety_threshold=None, fallback_action=None):
        self.safety_threshold=saftety_threshold
        self.fallback_action=fallback_action
        if self.safety_threshold is not None:
            assert (self.fallback_action is not None)

    def select_action(self,q_values_all_nets):
        """
        Selects action by highest mean, possibly subject to safety threshold.

        Args:
            q_values_all_nets (ndarray): Array with Q-values of all the actions for all the ensemble members.

        Returns:
            tuple: containing:
                int: chosen action
                dict: if the safety threshold is active: info if the fallback action was used or not,
                      otherwise: empty
        """
        mean_q_values=np.mean(q_values_all_nets,axis=0) # axis=0 按列计算mean
        if self.safety_threshold is None:
            return np.argmax(mean_q_values), {}
        else:
            std_q_values=np.std(q_values_all_nets,axis=0)
            coef_of_var=std_q_values/np.abs(mean_q_values)
            sorted_q_indexes=mean_q_values.argsort()[::-1] #每个值代表着对应导致最大q值的action到导致最小q值的action的index
            i=0
            while i<len(coef_of_var) and coef_of_var[sorted_q_indexes[i]] > self.safety_threshold:
                i+=1
            if i == len(coef_of_var):  # No action is considered safe - use fallback action
                return self.fallback_action, {'fallback action': True}
            else:
                return sorted_q_indexes[i], {'fallback action': False}
