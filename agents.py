import numpy as np
import random
from collections import namedtuple, deque
import warnings

from models import QNetwork, ModelWithPrior

import torch
import torch.nn.functional as F
import torch.optim as optim
from SumTree import SumTree
import time
import math
import csv


GAMMA = 0.99  # discount factor
TAU = 1e-3  # for soft update of target parameters
LR = 5e-4  # learning rate
UPDATE_EVERY = 4  # how often to update the network
number_of_nets = 10 # number of ensemble member
cv_update_frequency = 500



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ensembleDQNAgent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, memory, train_policy, test_policy, seeds):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seeds[0])


        self.normal_buffer = False
        self.confidence_based = False
        self.prioritized = False

        self.loss = []
        # initialize 10 ensemble members, so 10 local networks and 10 target networks.
        self.qnetworks_local = [QNetwork(state_size=state_size, action_size=action_size, seed=seeds[i]).to(device)for i in range(number_of_nets)]
        self.qnetworks_target = [QNetwork(state_size=state_size, action_size=action_size, seed=seeds[i]).to(device) for i in range(number_of_nets)]

        # self.qnetworks_local = [ ModelWithPrior(state_size=state_size, action_size=action_size, seed= seeds[i]).to(device) for i in range(number_of_nets)]
        # self.qnetworks_target = [ ModelWithPrior(state_size=state_size, action_size=action_size, seed= seeds[i]).to(device) for i in range(number_of_nets)]
        self.optimizer = [ optim.Adam(self.qnetworks_local[i].parameters(), lr=LR) for i in range(number_of_nets)]
        # print(self.optimizer)

        self.enough_samples = False

        # Replay memory
        self.memory = memory
        if self.memory.type =='Replay_Buffer':
            self.normal_buffer=True
            print('\nusing '+ self.memory.type)
        elif self.memory.type == 'Confidence_Based_Replay_Buffer':
            self.confidence_based=True
            print('\nusing '+ self.memory.type)
        elif self.memory.type =='Prioritized_Replay_Buffer':
            self.prioritized= True
            print('\nusing '+ self.memory.type)
        self.batch_size = self.memory.batch_size
        print('Buffer size is',self.memory.limit)
        print('Batch Size is',self.memory.batch_size)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        # Initialize update step for updating cv value
        self.update_step = 0
        # Initialize select_action policy
        self.policy=train_policy
        self.test_policy=test_policy

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
                q_values_all_nets.append(action_values[action])# only append the q_values of the chosen action

        q_values_all_nets = np.array(q_values_all_nets)

        mean_q_values = np.mean(q_values_all_nets)
        std_q_values = np.std(q_values_all_nets, axis=0)
        coef_of_var = std_q_values / np.abs(mean_q_values)  # This is the coef_of_var for the chosen action
        #################################################################

        return action, coef_of_var

    def act_eval(self, state):
        """Returns action index and Information about the Q-values of the chosen action for given state as per current policy.

          Params
          ======
              state (array_like): current state
          """
        q_values_all_nets = []

        state = torch.tensor(state).float().unsqueeze(0).to(device)
        for net in range(number_of_nets):
            self.qnetworks_local[net].eval()
            with torch.no_grad():
                action_values = self.qnetworks_local[net](state).cpu().data.numpy().flatten()
                q_values_all_nets.append(action_values)

        q_values_all_nets = np.array(q_values_all_nets)
        assert q_values_all_nets.shape == (number_of_nets, self.action_size)
        # Ensemble_test_policy action selection
        info = {}
        action, policy_info = self.test_policy.select_action(q_values_all_nets=q_values_all_nets)
        info['q_values_all_nets'] = q_values_all_nets[action]
        # print(info['q_values_all_nets'])
        # info['mean']= np.mean(q_values_all_nets[:, :],axis=0)
        info['mean'] = np.mean(q_values_all_nets[action], axis=0)
        # print('mean of q_values_all_nets is',info['mean'])
        info['standard_deviation'] = np.std(q_values_all_nets[action], axis=0)
        # print('standard_deviation is',info['standard_deviation'])
        info['coefficient_of_variation'] = np.std(q_values_all_nets[action], axis=0) / np.abs(info['mean'])

        # print('coefficient_of_variation is',info['coefficient_of_variation'])
        info.update(policy_info)  # fallback action: True or False
        # print('fallback_action is',info['fallback_action'])

        return action, info

    def calulate_TD_error(self, state, next_state, action, reward, done, active_model):

        state=torch.FloatTensor(state).to(device)
        next_state=torch.FloatTensor(next_state).to(device)
        Q_target_next = self.qnetworks_target[active_model](next_state).max().cpu().data.numpy()

        # Q_target_next = self.qnetworks_target[active_model](next_state).detach().max(1)[0].unsqueeze(1)
        # print(done)
        # print(reward)
        if done:
            Q_target = reward
        else:
            Q_target = (reward + (GAMMA * Q_target_next))
        # print('Q_target_next is', Q_target_next)
        # print('reward is',reward)
        # print('Q_target is',Q_target)
        Q_expected = self.qnetworks_local[active_model](state)[action].cpu().data.numpy()
        # print('Q_expected is',Q_expected)
        error = abs(Q_expected - Q_target)
        # print('error is',error)
        return error


    def step(self, state, next_state, action, reward, done, is_collided, coef_of_var, active_model):

        # if self.update_step % 1000 ==0:
        #     print('\n{:d} steps passed...\n'.format(self.update_step))
        if self.prioritized:
            # calculate TD error
            error= self.calulate_TD_error(state, next_state, action, reward, done, active_model)
            # Save experience in replay memory
            self.memory.append(state, action, reward, done, coef_of_var, error)
        elif self.confidence_based:
            self.memory.append(state, action, reward, done, is_collided, coef_of_var)
        else:
            self.memory.append(state, action, reward, done, coef_of_var)
        if self.enough_samples == False:
            has_enough_data = [False for i in range(number_of_nets)]
            flag = True
            for net in range(number_of_nets):
                # print('check if enough data available')
                if len(self.memory.index_refs[net]) >= self.batch_size:
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
                self.update_step +=1
                # If enough samples are available in memory, get random subset and learn
                for active_net in range(number_of_nets):
                    self.learn_single_net(active_net, GAMMA)

    def learn_single_net(self, active_net, gamma):
        # time_forward_start=time.clock()
        if self.prioritized or self.confidence_based:
            experiences, importance_sample_weights, batch_tree_idxs = self.memory.sample(active_net, self.batch_size)
        else:
            experiences = self.memory.sample(active_net, self.batch_size)
        assert len(experiences) == self.batch_size

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

        if self.confidence_based or self.prioritized:
            loss = ((torch.FloatTensor(importance_sample_weights).to(device)) * F.mse_loss(Q_expected, Q_targets)).mean()
            if self.prioritized: # update priority
                errors = torch.abs(Q_targets - Q_expected).squeeze(1).cpu().data.numpy()
                # print('errors:',errors)
                # print('len(errors):',len(errors))
                # update priority
                for i in range(self.batch_size):
                    tree_idx = batch_tree_idxs[i]
                    self.memory.update(active_net, tree_idx, errors[i])
            # else:
                # time_forward_start=time.clock()
                ############################################################################################################## cv update process
                # if self.update_step % cv_update_frequency == 0: # cv update frequency
                #     self.update_cv_based_priority(active_net, batch_tree_idxs, state_batch, action_batch)
                ##############################################################################################################
                    # print('priority updated...')
                # time_forward_end=time.clock()
                # print('cv update time is:',time_forward_end-time_forward_start)
        else:
            loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer[active_net].zero_grad()
        loss.backward()
        self.optimizer[active_net].step()
        self.loss.append(loss.item())
        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetworks_local[active_net], self.qnetworks_target[active_net], TAU)

        # time_forward_end=time.clock()
        # print('one loop time is:',time_forward_end-time_forward_start)


    def update_cv_based_priority(self, active_net, batch_tree_idxs, state_batch, action_batch):
        for i in range(len(batch_tree_idxs)):
            state = state_batch[i]
            action = action_batch[i]
            new_cv_value = self.forward_cv_value(state, action)
            # speed_difference = np.abs(state[0] - state[1])  # state[0] = ego speed, state[1] = 最近的vehicle speed
            # distance_difference = math.sqrt((state[5] - state[7]) ** 2 + (state[6] - state[8]) ** 2)  # state{5} ego x axis, state[6] ego y axis, state[7] = vehicle x axis, state[8] = vehicle y axis
            # self._get_risk_priority(speed_difference, distance_difference, is_collided) + coef_of_var
            tree_idx = batch_tree_idxs[i]
            self.memory.update(active_net, tree_idx, new_cv_value)

    def forward_cv_value(self, state, action):
        """Returns the coefficient of variation for given state and action.

        """
        q_values_all_nets = []
        state = state.clone().detach().float().unsqueeze(0).to(device)
        # state = torch.tensor(state).float().unsqueeze(0).to(device)
        for net in range(number_of_nets):
            # time_forward_start=time.clock()
            self.qnetworks_local[net].eval()
            with torch.no_grad():
                action_values = self.qnetworks_local[net](state).cpu().data.numpy().flatten()
                q_values_all_nets.append(action_values[action])
            # time_forward_end=time.clock()
            # print('one forward time is:',time_forward_end-time_forward_start)

        q_values_all_nets = np.array(q_values_all_nets)

        mean_q_values = np.mean(q_values_all_nets)
        std_q_values = np.std(q_values_all_nets, axis=0)
        coef_of_var = std_q_values / np.abs(mean_q_values) # This is the coef_of_var for the chosen action

        return coef_of_var

    # def weighted_loss(self, weights, inputs, targets):
    #     t = torch.abs(inputs - targets)
    #     zi = torch.where(t < 1, 0.5 * t ** 2, t - 0.5)
    #     loss = (weights * zi).sum()
    #     return loss


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


class Confidence_Based_Replay_Buffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seeds, number_of_nets=number_of_nets, adding_prob=0.5):
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

        self.seed = random.seed(seeds[0])

        self.type ='Confidence_Based_Replay_Buffer'

        self.limit= buffer_size
        self.number_of_nets = number_of_nets
        self.adding_prob = adding_prob
        self.index_refs = [[] for i in range(self.number_of_nets)]
        self.tree = [SumTree(int(self.limit / 2)) for i in range(self.number_of_nets)]

        self.PER_e = 0.01  # Hyperparameter that we use to avoid some experiences to have 0 probability of being taken
        self.PER_a = 0.6  # Hyperparameter that we use to make a tradeoff between taking only experience with high priority and sampling randomly
        self.beta = 0.4  # importance-sampling, from initial value increasing to 1
        self.beta_increment_per_sampling = 0.001


        self.actions = deque(maxlen=buffer_size)
        self.rewards = deque(maxlen=buffer_size)
        self.dones = deque(maxlen=buffer_size)
        self.states = deque(maxlen=buffer_size)
        # self.coef_of_vars = deque(maxlen=buffer_size)
        self.priorities = deque(maxlen=buffer_size)
    def _get_priority(self, error):
        return (np.abs(error) + self.PER_e) ** self.PER_a

    def _get_risk_priority(self, speed_difference, distance_difference,is_collided):
        if is_collided== True:
            # print('collided, priority append 500')
            # print('distance_diff is',distance_difference)
            return 500
        elif distance_difference <=6:
            # print('risky, priority append', 10 * speed_difference + 50 / (distance_difference))
            return 10*speed_difference + 50/distance_difference
        else:
            # print('not risky, priority append', 0.01*speed_difference + 1/(distance_difference))
            return 0.01*speed_difference + 10/distance_difference


    def append(self, state, action, reward, done, is_collided, coef_of_var):


        #####################################################################################################
        # intent of self.index_refs will be created
        if self.nb_entries < self.limit:   # One more entry will be added after this loop
            # There should be enough experiences before the chosen sample to fill the window length + 1, windows length: How many historic states to include (1 uses only current state)
            if self.nb_entries > 2:
                for i in range(self.number_of_nets):
                    if np.random.rand() < self.adding_prob:
                        self.index_refs[i].append(self.nb_entries-1)
                        self.tree[i].add(self.priorities[self.nb_entries - 1])

        # Append state, action, reward, done, coef_of_var into memory
        ####################################################################################################
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        # self.coef_of_vars.append(coef_of_var)
        ######################################################################################
        # version 01: if cv bigger than 1, append 2. Otherwise append the original cv values
        # if coef_of_var <=1:
        #     self.priorities.append(self._get_priority(coef_of_var))
        # else:
        #     self.priorities.append(self._get_priority(2))
        ######################################################################################
        # version 02: directly append all the original cv values, but every 5000 steps the priority will be updated
        # cv without collided
        speed_difference = np.abs(state[0]-state[1]) # state[0] = ego speed, state[1] = 最近的vehicle speed
        distance_difference = math.sqrt((state[5]-state[7])**2 + (state[6] - state[8])**2) # state{5} ego x axis, state[6] ego y axis, state[7] = vehicle x axis, state[8] = vehicle y axis
        # self.priorities.append(self._get_risk_priority(speed_difference,distance_difference,is_collided))
        self.priorities.append(self._get_priority(self._get_risk_priority(speed_difference, distance_difference, is_collided)+coef_of_var))
        # print(self.priorities)
        # self.priorities.append(self._get_priority(coef_of_var))
        ####################################################################################################
        # cv with collided
        # if is_collided== True:
        #     self.priorities.append(self._get_priority(coef_of_var + 100))
        # else:
        #     self.priorities.append(self._get_priority(coef_of_var))

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
        # #因为不是所有的index都加进来了，所以需要把加进去的index先选出来，再通过index把相应的coef_of_var提取出来，加入priorities里
        # priorities = []
        # for i in range(memory_size):
        #     # priorities.append(self.coef_of_vars[self.index_refs[net][i]]+1) #to avoid probability=0, so add one
        #     if 0 <= self.coef_of_vars[self.index_refs[net][i]] < 0.1:
        #         priorities.append(1)
        #     elif 0.1 <= self.coef_of_vars[self.index_refs[net][i]] < 0.2:
        #         priorities.append(1.2)
        #     elif 0.2 <= self.coef_of_vars[self.index_refs[net][i]] < 0.3:
        #         priorities.append(1.4)
        #     elif 0.3 <= self.coef_of_vars[self.index_refs[net][i]] < 0.4:
        #         priorities.append(1.6)
        #     elif 0.4 <= self.coef_of_vars[self.index_refs[net][i]] < 0.5:
        #         priorities.append(1.8)
        #     elif 0.5 <= self.coef_of_vars[self.index_refs[net][i]] < 0.6:
        #         priorities.append(2)
        #     elif 0.6 <= self.coef_of_vars[self.index_refs[net][i]] < 0.7:
        #         priorities.append(2.2)
        #     elif 0.7 <= self.coef_of_vars[self.index_refs[net][i]] < 0.8:
        #         priorities.append(2.4)
        #     elif 0.8 <= self.coef_of_vars[self.index_refs[net][i]] < 0.9:
        #         priorities.append(2.6)
        #     elif 0.9 <= self.coef_of_vars[self.index_refs[net][i]] < 1:
        #         priorities.append(2.8)
        #     elif 1 <= self.coef_of_vars[self.index_refs[net][i]] < 10:
        #         priorities.append(3)
        #     elif 10 <= self.coef_of_vars[self.index_refs[net][i]] < 100:
        #         priorities.append(5)
        #     else:
        #         priorities.append(8)
        # priorities = np.array(priorities)
        ######################################################################
        # self.sample_transition_prob = np.power(priorities, self.alpha) / sum(np.power(priorities, self.alpha))
        # # print('sample_transition_prob is',sample_transition_prob)
        # # print('toal probability is',sum(sample_transition_prob))
        # ref_idxs = np.random.choice(memory_size, size=batch_size, p=self.sample_transition_prob)
        # batch_idxs = [self.index_refs[net][idx] for idx in ref_idxs]
        # assert len(batch_idxs) == batch_size
        #
        # self.beta = np.min([1., self.beta * self.beta_increment_per_sampling])
        #
        # importance_sample_weights = torch.from_numpy(np.power(memory_size * self.sample_transition_prob, -self.beta)).float().to(device)
        segment = self.tree[net].total() / batch_size
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])
        batch_idxs = []
        batch_tree_idxs = []
        priorities = []
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            p, dataIdx, idx = self.tree[net].get(s)
            priorities.append(p)
            # print('dataIdx is',dataIdx)
            # print('len(index_refs) is',len(self.index_refs[net]))
            batch_idxs.append(self.index_refs[net][dataIdx])
            # print('len(batch_idxs) is',len(batch_idxs))
            batch_tree_idxs.append(idx)

        sampling_probabilities = priorities / self.tree[net].total()
        is_weight = np.power(self.tree[net].n_entries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()

        return batch_idxs, is_weight, batch_tree_idxs



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
        batch_idxs, importance_sample_weights, batch_tree_idxs = self.sample_batch_idxs(net, batch_size)

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
                index, _, _ = self.sample_batch_idxs(net, 1)
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
        return experiences, importance_sample_weights, batch_tree_idxs

    def update(self, net, tree_idx, error):
        # time_forward_start=time.clock()
        p = self._get_priority(error)
        self.tree[net].update(tree_idx, p)
        # time_forward_end=time.clock()
        # print('update time is:',time_forward_end-time_forward_start)



    @property
    def nb_entries(self):
        """Return number of observations

        Returns:
            Number of states
        """
        return len(self.states)

class Replay_Buffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seeds, number_of_nets=number_of_nets, adding_prob=0.5):
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

        self.seed = random.seed(seeds[0])

        self.type='Replay_Buffer'

        self.limit= buffer_size
        self.number_of_nets = number_of_nets
        self.adding_prob = adding_prob
        self.PER_e = 0.01  # Hyperparameter that we use to avoid some experiences to have 0 probability of being taken
        self.PER_a = 0.6  # Hyperparameter that we use to make a tradeoff between taking only experience with high priority and sampling randomly
        self.index_refs = [[] for i in range(self.number_of_nets)]

        self.actions = deque(maxlen=buffer_size)
        self.rewards = deque(maxlen=buffer_size)
        self.dones = deque(maxlen=buffer_size)
        self.states = deque(maxlen=buffer_size)
        # self.coef_of_vars = deque(maxlen=buffer_size)
        self.priorities = deque(maxlen=buffer_size)

    def _get_priority(self, error):
        return (np.abs(error) + self.PER_e) ** self.PER_a

    def append(self, state, action, reward, done, coef_of_var):

        if self.nb_entries < self.limit:   # One more entry will be added after this loop
            # There should be enough experiences before the chosen sample to fill the window length + 1
            if self.nb_entries > 2:
                for i in range(self.number_of_nets):
                    if np.random.rand() < self.adding_prob:
                        self.index_refs[i].append(self.nb_entries)

        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        # self.coef_of_vars.append(coef_of_var)
        self.priorities.append(self._get_priority(coef_of_var))

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
        ref_idxs = np.random.randint(0, memory_size, batch_size)
        batch_idxs = [self.index_refs[net][idx] for idx in ref_idxs]
        assert len(batch_idxs) == batch_size
        return batch_idxs

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
        batch_idxs = self.sample_batch_idxs(net, batch_size)

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
                idx = self.sample_batch_idxs(net, 1)[0]
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
        return experiences



    @property
    def nb_entries(self):
        """Return number of observations

        Returns:
            Number of states
        """
        return len(self.states)


class Prioritized_Replay_Buffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seeds, number_of_nets=number_of_nets, adding_prob=0.5):
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

        self.seed = random.seed(seeds[0])

        self.type='Prioritized_Replay_Buffer'

        self.limit= buffer_size
        self.number_of_nets = number_of_nets
        self.adding_prob = adding_prob
        self.index_refs = [[] for i in range(self.number_of_nets)]
        self.tree = [SumTree(int(self.limit/2)) for i in range(self.number_of_nets)]

        self.PER_e = 0.01 # Hyperparameter that we use to avoid some experiences to have 0 probability of being taken
        self.PER_a = 0.6 # Hyperparameter that we use to make a tradeoff between taking only experience with high priority and sampling randomly
        self.beta = 0.4 # importance-sampling, from initial value increasing to 1
        self.beta_increment_per_sampling = 0.001

        # # 定义采样和权重指数参数
        # self.alpha = 1.0
        # self.beta = 0.01
        # self.beta_increment_per_sampling = 1.00001 # After 460519 steps, beta will be incresed to 1.0

        # self.sample_transition_prob = []

        self.actions = deque(maxlen=buffer_size)
        self.rewards = deque(maxlen=buffer_size)
        self.dones = deque(maxlen=buffer_size)
        self.states = deque(maxlen=buffer_size)
        self.coef_of_vars = deque(maxlen=buffer_size)
        self.priorities = deque(maxlen=buffer_size)

    def _get_priority(self, error):
        return (np.abs(error) + self.PER_e) ** self.PER_a

    def append(self, state, action, reward, done, coef_of_var, error):

        if self.nb_entries <= self.limit:   # One more entry will be added after this loop
            # There should be enough experiences before the chosen sample to fill the window length + 1
            if self.nb_entries > 2: # self.nb_entries = len(self.states)
                for i in range(self.number_of_nets):
                    if np.random.rand() < self.adding_prob:
                        self.index_refs[i].append(self.nb_entries-1) # self.nb_entries = len(self.states)
                        # print('priority:',self.priorities)
                        # print('adding priority',self.priorities[self.nb_entries-1])
                        self.tree[i].add(self.priorities[self.nb_entries-1])
                        # self.tree[i].add(self.priorities[self.index_refs[i][-1]])

        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.coef_of_vars.append(coef_of_var)
        self.priorities.append(self._get_priority(error)) # priority = self._get_priority(error)
        # for i in range(number_of_nets):
        #     if self.index_refs[i]:
        #         self.tree[i].add(self.priorities[self.index_refs[i][-1]])

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
        # ######################################################################
        # #因为不是所有的index都加进来了，所以需要把加进去的index先选出来，再通过index把相应的self.default_priority提取出来，加入priorities里
        # priorities = []
        # for i in range(memory_size):
        #     priorities.append(self.priorities[self.index_refs[net][i]])
        # priorities = np.array(priorities)
        # ######################################################################
        # self.sample_transition_prob = np.power(priorities, self.alpha) / sum(np.power(priorities, self.alpha))
        # ref_idxs = np.random.choice(memory_size, size=batch_size, p=self.sample_transition_prob)
        # batch_idxs = [self.index_refs[net][idx] for idx in ref_idxs]
        # assert len(batch_idxs) == batch_size
        #
        # self.beta = np.min([1., self.beta * self.beta_increment_per_sampling])
        #
        # importance_sample_weights = torch.from_numpy(np.power(memory_size * self.sample_transition_prob, -self.beta)).float().to(device)

        segment = self.tree[net].total() / batch_size
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])
        batch_idxs = []
        batch_tree_idxs = []
        priorities = []
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            p, dataIdx,idx = self.tree[net].get(s)
            priorities.append(p)
            # print('dataIdx is',dataIdx)
            # print('len(index_refs) is',len(self.index_refs[net]))
            batch_idxs.append(self.index_refs[net][dataIdx])
            # print('len(batch_idxs) is',len(batch_idxs))
            batch_tree_idxs.append(idx)

        sampling_probabilities = priorities / self.tree[net].total()
        # print('len(index_refs) is',len(self.index_refs[net]))
        # print('sum(priorities)',sum(priorities))
        # print('self.tree[net].total():',self.tree[net].total())
        # print('priorities:',priorities)
        # print('len(priorities):',len(priorities))
        # print('sampling_probabilities:',sampling_probabilities)
        # print('len(sampling_probabilities):', len(sampling_probabilities))
        # print('sum(sampling_probabilities):',sum(sampling_probabilities))
        is_weight = np.power(self.tree[net].n_entries * sampling_probabilities, -self.beta)
        # print('is_weight:',is_weight)
        is_weight /= is_weight.max()
        # print(is_weight)

        return batch_idxs , is_weight, batch_tree_idxs

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
        batch_idxs, importance_sample_weights, batch_tree_idxs = self.sample_batch_idxs(net, batch_size)

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
                index, _, _ = self.sample_batch_idxs(net, 1)
                idx = index[0]
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
        return experiences, importance_sample_weights, batch_tree_idxs

    def update(self, net, tree_idx, error):
        p = self._get_priority(error)
        self.tree[net].update(tree_idx, p)



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
        self.eps_decay = 0.9995 # After 9208 episodes epsilon will be decayed to 0.01
        # self.eps_decay = 0.999  # After 4602 episodes epsilon will be decayed to 0.01
        # self.eps_decay = 0.995  # After 918 episodes epsilon will be decayed to 0.01
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
                return self.fallback_action, {'fallback_action': True}
            else:
                return sorted_q_indexes[i], {'fallback_action': False}
