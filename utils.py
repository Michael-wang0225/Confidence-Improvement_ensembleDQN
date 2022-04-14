import optparse
import csv
import os
import time
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from more_itertools import chunked

ABS_PATH = os.path.abspath(os.path.dirname(__file__))


# get the parameter from the terminal
def get_options(mode="train"):
    optParser = optparse.OptionParser()
    optParser.add_option("--nogui", action="store_true",
                         default=False, help="run the commandline version of sumo")
    if mode == "train":
        optParser.add_option("-a", "--agent", dest="agent", default="ensembleDQN", metavar="agent_type",
                             help="agent type options: 1.DQN 2.ensembleDQN")
    if mode == "eval":
        optParser.add_option("-f", "--folder", dest="data_folder", default="ensembleDQN_model",
                             metavar="folder_name",
                             help="the folder name of data, e.g. DQN_01234567")
        optParser.add_option("-m", "--model", dest="model", default="checkpoint_episode_9900_score_110.80_ensemble_",
                             metavar="model_name",
                             help="the name of model to evaluate, e.g. checkpoint0_score_123.pth") #输入训练好的weights
    options, args = optParser.parse_args()
    return options


# set file paths in a dic for all
def set_path(agent_type=None, data_name=None, model_name=None):
    path_dic = {}
    results_path = os.path.join(ABS_PATH, "results")
    if agent_type:
        path_dic["result_path"] = os.path.join(results_path, agent_type + "_" + time.strftime("%Y%m%d-%H%M%S"))
        path_dic["model_path"] = os.path.join(path_dic["result_path"], "models")
        path_dic["log_path"] = os.path.join(path_dic["result_path"], "logs")
        path_dic["agent_log_path"] = os.path.join(path_dic["log_path"], "agent_log.csv")
        path_dic["loss_log_path"] = os.path.join(path_dic["log_path"], "loss_log.csv")
        path_dic["graph_path"] = os.path.join(path_dic["log_path"], "scores.png")
        os.makedirs(path_dic["result_path"])
        os.makedirs(path_dic["model_path"])
        os.makedirs(path_dic["log_path"])
    else:
        path_dic["data_path"] = os.path.join(results_path, data_name) #在results下面的一个叫做ensembleDQN_model的文件夹
        path_dic["model_path"] = os.path.join(path_dic["data_path"], "models", model_name) #在ensembleDQN_model的文件夹下有个models，models里有checkpiont.pth，记得把训练好的weights放在里面
        path_dic["eval_path"] = os.path.join(path_dic["data_path"], "eval_" + time.strftime("%Y%m%d-%H%M%S"))#在ensembleDQN_model的文件夹下创建一个eval_2022xxxx的文件夹
        path_dic["agent_log_path"] = os.path.join(path_dic["eval_path"], "agent_log.csv")
        path_dic["score_log_path"] = os.path.join(path_dic["eval_path"], "score_log.csv")
        path_dic["graph_path"] = os.path.join(path_dic["eval_path"], "scores.png")
        os.makedirs(path_dic["eval_path"])
    return path_dic


# write logs
def write_logs(file_path, data):
    with open(file_path, "a+") as f:
        csv_file = csv.writer(f)
        csv_file.writerow(data)


# plot scores, average is the number used to calculate mean
def plot_scores(episodes=[], scores=[], average=1, path=None):
    plt.figure("Scores of the Agent")
    plt.clf()
    # set int locator
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    scoresGraph, = plt.plot(episodes, scores, color="red", label='scores')
    plt.title("Scores of the Agent", fontsize=15, pad=10)
    plt.xlabel("Episode")
    plt.ylabel("Scores")
    # show original data in red line and mean of data in blue line
    if average != 1:
        # mean of every average(e.g. average=100) data points
        scoresGraphp, = plt.plot(episodes[int(average / 2)::average],#这是横坐标,开始为50，步长为100
                                 [sum(x) / len(x) for x in chunked(scores, average)],# chunked()把scores拆成一个个average长度的数组，这是纵坐标
                                 color="blue", label="mean per " + str(average) + " episodes")
        plt.legend(handles=[scoresGraph, scoresGraphp], bbox_to_anchor=(0.7, 1.0), loc="lower left", fancybox=True,
                   shadow=True, ncol=2)
        plt.title("Scores of the Agent", fontsize=15, pad=35)
    if path:
        plt.savefig(path)
    plt.show()
