from sumo_env import SumoEnv
from agents import ensembleDQNAgent, Confidence_Based_Replay_Buffer, Replay_Buffer,Prioritized_Replay_Buffer, Epsilon_greedy_policy, Ensemble_test_policy
import numpy as np
import torch
from utils import get_options, set_path, write_logs, plot_scores
import os
import datetime
from models import QNetwork

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    options = get_options("eval")
    agent_type = options.data_folder[:11]
    path = set_path(data_name=options.data_folder, model_name=options.model)
    # start sumo env
    env = SumoEnv(mode="eval", log_path=path["eval_path"], nogui=options.nogui)
    state=env.start()
    start_time = datetime.datetime.now()  # to record running time
    # initialise
    episodes = [i for i in range(2000)]
    actions = [-2.5, -1.5, 0.0, 1.5, 2.5]
    number_of_nets = 10
    BUFFER_SIZE = 100000  # buffer size
    BATCH_SIZE = 64  # minibatch size
    seeds = [i for i in range(number_of_nets)]
    score = 0
    scores = []
    num_collision = 0
    total_step = 0
    # Replay memory
    memory_CBRB = Confidence_Based_Replay_Buffer(action_size=len(actions), buffer_size=BUFFER_SIZE,batch_size=BATCH_SIZE, seeds=seeds)
    memory_RB = Replay_Buffer(action_size=len(actions), buffer_size=BUFFER_SIZE, batch_size=BATCH_SIZE, seeds=seeds)
    memory_PERB = Prioritized_Replay_Buffer(action_size=len(actions), buffer_size=BUFFER_SIZE, batch_size=BATCH_SIZE,seeds=seeds)
    # Identify policy
    train_policy = Epsilon_greedy_policy(action_size=len(actions))
    test_policy = Ensemble_test_policy(saftety_threshold=None, fallback_action=None)
    # generate models
    agent = ensembleDQNAgent(state_size=len(state), action_size=len(actions), memory=memory_CBRB, train_policy=train_policy, test_policy=test_policy, seeds=seeds)
    # load weights
    for i in range(number_of_nets):
        agent.qnetworks_local[i].load_state_dict(torch.load(os.getcwd() +"\\results\\ensembleDQN_model\\models\\checkpoint_episode_40000_score_-405.43_ensemble_"+ str(i) + ".pth"))
        # agent.qnetworks_local[i].load_state_dict(torch.load(path["model_path"] + str(i) + ".pth"))
    # initial arbitary weights
    # agent.qnetworks_local= [QNetwork(state_size=len(state), action_size=len(actions), seed=seeds[i]).to(device) for i in range(number_of_nets)]



    write_logs(file_path=path["agent_log_path"],
               data=["total_step","episode", "step", "state", "action", "next state", "reward", "return", "collided", "done", "q_values_all_nets", "mean of q_values_all_nets", "standard_deviation", "coefficient_of_variation","fallback_action"])
    write_logs(file_path=path["score_log_path"],
               data=["episode", "return"])
    print("\nstart evaluation of " + agent_type + " agent:\n")
    # print('cv+risk new map, High_success_rate_checkpoint_episode_33625_success_rate_0.8900_ensemble')
    # run evaluation
    for episode in episodes:
        score = 0
        state = env.reset()
        is_collided = False
        step = 0

        while True:
            step += 1
            total_step += 1
            action, action_info = agent.act_eval(state)
            next_state, reward, done, is_collided = env.step(actions[action])
            score += reward
            state = next_state
            if is_collided == True:
                print('撞车了')
                num_collision += 1
            write_logs(file_path=path["agent_log_path"],
                       data=[total_step, episode, step, state, actions[action], next_state, reward, score, is_collided, done, action_info['q_values_all_nets'], action_info['mean'], action_info['standard_deviation'], action_info['coefficient_of_variation']])

            if done:
                scores.append(score)
                print("episode: {}, score: {:.2f}".format(episode, score))
                write_logs(file_path=path["score_log_path"],
                           data=[episode, score])
                break

    print("\nRun {} test episodes, collision rate is {:.2%}, success rate is {:.2%}, mean score is {:.2f}.".format(
        len(episodes), num_collision / len(episodes),
        1 - num_collision / len(episodes), np.mean(scores)))
    write_logs(file_path=path["score_log_path"],
               data=["total_episode", "success_rate", "mean_score"])
    write_logs(file_path=path["score_log_path"],
               data=[len(episodes), "{:.2%}".format(1 - num_collision / len(episodes)),
                     round(np.mean(scores), 2)])
    print("\nEvaluation ended.")
    # plot_scores(episodes, scores, path=path["graph_path"])

    end_time = datetime.datetime.now()
    time_cost = end_time - start_time
    print('cost time: ',str(time_cost).split('.')[0])

    env.close()
