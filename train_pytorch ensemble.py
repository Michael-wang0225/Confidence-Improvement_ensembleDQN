import pandas as pd

from sumo_env import SumoEnv
from agents import ensembleDQNAgent, Confidence_Based_Replay_Buffer, Replay_Buffer,Prioritized_Replay_Buffer, Epsilon_greedy_policy, Ensemble_test_policy
from utils import get_options, set_path, write_logs, plot_scores
import os
import torch
import numpy as np
from collections import deque
import datetime
import csv

if __name__ == "__main__":
    options = get_options("train")
    agent_type = options.agent
    path = set_path(agent_type)
    # start sumo env
    env = SumoEnv(mode="train", log_path=path["log_path"], nogui=options.nogui)
    state = env.start()
    start_time = datetime.datetime.now() # to record running time
    # initialise
    episodes = [i for i in range(100000)]
    scores = []
    scores_window = deque(maxlen=100)
    num_collision_window = deque(maxlen=100)
    total_step = 0
    num_collision = 0
    actions = [-2.5, -1.5, 0.0, 1.5, 2.5]
    BUFFER_SIZE = 100000  # buffer size
    BATCH_SIZE = 64  # minibatch size
    number_of_nets = 10
    safety_threshold = 2
    fallback_action = 0
    # Replay memory
    memory_CBRB = Confidence_Based_Replay_Buffer(action_size=len(actions), buffer_size =BUFFER_SIZE, batch_size=BATCH_SIZE, seed=0)
    memory_RB = Replay_Buffer(action_size=len(actions), buffer_size =BUFFER_SIZE, batch_size=BATCH_SIZE, seed=0)
    memory_PERB = Prioritized_Replay_Buffer(action_size=len(actions), buffer_size =BUFFER_SIZE, batch_size=BATCH_SIZE, seed=0)
    # Identify policy
    train_policy=Epsilon_greedy_policy(action_size=len(actions))
    test_policy=Ensemble_test_policy(saftety_threshold=None, fallback_action=None)
    # generate models
    agent=ensembleDQNAgent(state_size=len(state), action_size=len(actions), memory=memory_CBRB,train_policy=train_policy, test_policy=test_policy, seed=0)

    write_logs(file_path=path["agent_log_path"],
               data=["total_step", "episode", "step", "state", "action", "next state", "reward", "return", "epsilon", "done","is_collided", "num_collision", "coef_of_var", "average score","success_rate"])
    write_logs(file_path=path["loss_log_path"],
               data=["loss"])
    print("\nstart training " + agent_type + " agent:\n")
    # run episode
    for episode in episodes:
        active_model = np.random.randint(number_of_nets) # choose a arbitrary active model from models [0,10)
        score = 0
        state = env.reset()
        is_collided = False
        step = 0

        while True:
            action, coef_of_var= agent.act_train(state, active_model)
            next_state, reward, done, is_collided = env.step(actions[action])
            agent.step(state, next_state, action, reward, done, is_collided, coef_of_var, active_model)
            state = next_state
            score += reward
            step += 1
            total_step +=1
            if is_collided == True:
                num_collision += 1
            ################################################################## test the priority distribution,save in a additional csv file
            # if total_step >= 100000 and (total_step % 2000) ==0: # 每2000 step 记录一次priority情况，不然csv文件过于大
            #     print(total_step)
            #     with open("F:\\priority csv\\priority_CBRB_update_step_ 500.csv", 'a+', newline='') as f:
            #         mywrite = csv.writer(f)
            #         mywrite.writerow(agent.memory.priorities)
            ###################################################################
            # save agent and loss logs
            write_logs(file_path=path["agent_log_path"],
                       data=[total_step, episode, step, state, actions[action], next_state, reward, score, agent.policy.eps, done, is_collided, num_collision, coef_of_var])
            if len(agent.loss):
                write_logs(file_path=path["loss_log_path"], data=[agent.loss[-1]])
            if done:
                break

        scores_window.append(score)
        num_collision_window.append(num_collision)
        # ################################################################## test the priority distribution,save in a additional csv file
        if (episode+1) % 5000 == 0:  # 每5000 episode 记录一次priority情况，不然csv文件过于大
            print('priority in episode {:d} saved',episode+1)
            with open("F:\\priority csv\\priority_CBRB_cv_update_1000_with_collided.csv", 'a+', newline='') as f:
                mywrite = csv.writer(f)
                mywrite.writerow(agent.memory.priorities)
        # ###################################################################
        if episode <=99:
            write_logs(file_path=path["agent_log_path"],
                       data=[total_step, episode, step, state, actions[action], next_state, reward, score, agent.policy.eps,
                             done, is_collided, num_collision, coef_of_var, np.mean(scores_window), 1-(num_collision_window[episode]-num_collision_window[0])/100])
        else:
            write_logs(file_path=path["agent_log_path"],
                       data=[total_step, episode, step, state, actions[action], next_state, reward, score,
                             agent.policy.eps,
                             done, is_collided, num_collision, coef_of_var, np.mean(scores_window),
                             1 - (num_collision_window[99] - num_collision_window[0]) / 100])
        agent.policy.epsilon_decay() # let epsilon decay from 1 to 0.01
        # show recent mean score; when the scores reach 350, save weights and end training
        if episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode +1, np.mean(scores_window)))
            for i in range(number_of_nets):
                torch.save(agent.qnetworks_local[i].state_dict(),
                           os.path.join(path["model_path"], "checkpoint_episode_" + str(episode) + "_score_{:.2f}".format(
                               np.mean(scores_window)) + "_ensemble_"+ str(i) +".pth"))

        if np.mean(scores_window) > 250 and episode >= 100:
            print('\nSave weights in {:d} episodes!\tAverage Score: {:.2f}'.format(episode +1,
                                                                                         np.mean(scores_window)))
            for i in range(number_of_nets):
                torch.save(agent.qnetworks_local[i].state_dict(), os.path.join(path["model_path"], "High_score_checkpoint_episode_"+ str(episode) + "_score_{:.2f}".format(
                               np.mean(scores_window)) +"_ensemble_"+ str(i) +".pth"))

        if episode >= 99 and (1 - (num_collision_window[99] - num_collision_window[0]) / 100) >= 0.93:
            print('\nSave weights in {:d} episodes!\tSuccess rate per 100 episodes: {:.2f}'.format(episode + 1,
                                                                                   1 - (num_collision_window[99] - num_collision_window[0]) / 100))
            for i in range(number_of_nets):
                torch.save(agent.qnetworks_local[i].state_dict(), os.path.join(path["model_path"], "High_success_rate_checkpoint_episode_"+ str(episode) + "_success_rate_{:.2f}".format(
                               1 - (num_collision_window[99] - num_collision_window[0]) / 100) +"_ensemble_"+ str(i) +".pth"))

        if episode >=40000:
            print('\nTrained {:d} episodes'.format(episode +1))
            break

    print("\nTraining ended.")

    end_time = datetime.datetime.now()
    time_cost = end_time - start_time
    print('cost time: ',str(time_cost).split('.')[0])

    # plot_scores([i for i in range(episode + 1)], scores, average=100, path=path["graph_path"])
    env.close()
