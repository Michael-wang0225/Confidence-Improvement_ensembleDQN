# add API for sumo
import traci
import sumolib
from sumolib import checkBinary

# add path, get parameters from terminal
import os
import sys
from utils import ABS_PATH

import random
from scipy.spatial import distance
import math

# generate network files
def generate_env(mode="eval"):
    print("checking environment...")
    env_path = os.path.join(ABS_PATH, "intersection", mode + "_env")
    randomTrips_path = os.path.join(ABS_PATH, "randomTrips.py")
    if mode == "eval":
        command_trips_file = "cd " + env_path + " && python " + randomTrips_path + " -n intersection.net.xml -l " \
                                                                                   "-e 150000 -p 2 --binomial 4 -o intersection.trips.xml"
        command_route_file = "cd " + env_path + " && duarouter -n intersection.net.xml -t intersection.trips.xml -o intersection.rou.xml " \
                                                "--randomize-flows true --departlane random --arrivallane random --random --ignore-errors"
    if mode == "train":
        command_trips_file = "cd " + env_path + " && python " + randomTrips_path + " -n intersection.net.xml -l " \
                                                                                   "-e 10000000 -p 2 --binomial 4 -o intersection.trips.xml"
        command_route_file = "cd " + env_path + " && duarouter -n intersection.net.xml -t intersection.trips.xml -o intersection.rou.xml " \
                                                "--randomize-flows true --departlane random --arrivallane random --seed 30 --ignore-errors"

    if not os.path.exists(os.path.join(env_path, "intersection.trips.xml")):
        print("generating xml files...")
        os.system(command_trips_file)
    if not os.path.exists(os.path.join(env_path, "intersection.rou.xml")):
        os.system(command_route_file)
    print("finished checking.")
    return env_path


class SumoEnv:
    def __init__(self, mode="eval", log_path=".", nogui=False):
        # self.agent_type = agent_type
        self.mode = mode
        self.log_path = log_path
        self.nogui = nogui

        self.state_size = 64
        self.net_path = generate_env(self.mode)
        self.net = sumolib.net.readNet(os.path.join(self.net_path, "intersection.net.xml"))

        self.time_step = 0.1
        self.ego_index = -1
        self.ego_name = "ego" + str(self.ego_index)
        self.state = [0] * self.state_size
        self.reward = 0
        self.max_step = 400
        self.current_step = 0
        self.done = False

        self.vt = 0
        self.pt = [0, 0]
        self.angle = 0
        self.action = 0

        self.intersect_lane_dic = {}

        self.veh_type_param = {}
        motorcycle = {"length": 2.2, "width": 0.9, "height": 1.5,"a_max":6}
        passenger = {"length": 5, "width": 1.8, "height": 1.5,"a_max":2.6}
        ego = {"length": 5, "width": 1.8, "height": 1.5,"a_max":2.6}
        truck = {"length": 7.1, "width": 2.4, "height": 2.4,"a_max":1.3}
        bus = {"length": 12, "width": 2.5, "height": 3.4,"a_max":1.2}

        self.veh_type_param["m"] = motorcycle
        self.veh_type_param["p"] = passenger
        self.veh_type_param["t"] = truck
        self.veh_type_param["b"] = bus
        self.veh_type_param["g"] = ego

    def start(self):
        if 'SUMO_HOME' in os.environ:
            tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
            sys.path.append(tools)
        else:
            sys.exit("please declare environment variable 'SUMO_HOME'")
        # with/without gui
        # options = get_options()
        if self.nogui:
            sumoBinary = checkBinary('sumo')
        else:
            sumoBinary = checkBinary('sumo-gui')
        # start, set configs: log, time step, set collision stop time, remove cars in collision...
        traci.start(
            [sumoBinary, "--log", os.path.join(self.log_path, "sumo_log.txt"), "--step-length", str(self.time_step),
             "--collision.mingap-factor", "0", "--collision.check-junctions", "true",
             "--collision.stoptime", str(self.time_step), "--collision.action", "remove",
             "-c", os.path.join(self.net_path, "intersection.sumocfg")])  # set the right path
        return self.state

    def step(self, action):
        self.apply_action(action)
        # step
        traci.simulationStep()
        # update done, state, reward
        self.update()
        return self.state, self.reward, self.done, self.is_collided()

    def reset(self):
        # remove last vehicle if it is not removed
        if self.ego_name in traci.vehicle.getIDList():
            traci.vehicle.remove(self.ego_name)
        # reset parameters
        self.ego_index += 1
        self.ego_name = "ego" + str(self.ego_index)
        self.done = False
        self.current_step = 0
        self.vt = 0
        self.pt = [0, 0]
        self.angle = 0

        # keep less than 10 cars in the scene
        id_list = traci.vehicle.getIDList()
        id_list_diff = len(id_list) - 30
        while id_list_diff > 0:
            traci.vehicle.remove(id_list[id_list_diff])
            id_list_diff -= 1

        # search routes for ego
        lenOfEdges = len(self.net.getEdges())
        while True:
            start = self.net.getEdges()[random.randint(0, lenOfEdges - 1)].getID()
            end = self.net.getEdges()[random.randint(0, lenOfEdges - 1)].getID()
            route = traci.simulation.findRoute(fromEdge=start, toEdge=end, vType="DEFAULT_VEHTYPE").edges
            # check route from start to end: 1.from beginning 2.start!=end 3.route exists
            if start[0] == '-' or start == end or not route:
                continue
            else:
                break

        # add route, ego
        traci.route.add(routeID="ego_route" + str(self.ego_index), edges=route)
        traci.vehicle.add(routeID="ego_route" + str(self.ego_index), vehID=self.ego_name, typeID="DEFAULT_VEHTYPE",
                          departSpeed="random")
        traci.vehicle.setColor(color=(0, 255, 0, 255), vehID=self.ego_name)  # set green color for ego
        traci.vehicle.setSpeedMode(self.ego_name, 32)  # set the mode "all checks off", controlled only from traci
        # step until ego in the scene
        while self.ego_name not in traci.vehicle.getIDList():
            traci.simulationStep()
        return self.get_state()

    def close(self):
        traci.close()

    def apply_action(self, acceleration):
        # kinematic formula: vt = v0 + a * delta_t
        self.vt = traci.vehicle.getSpeed(self.ego_name) + acceleration * self.time_step
        traci.vehicle.setSpeed(self.ego_name, self.vt)

        self.action = acceleration

        # calc ego features in case it successfully reached the goal
        if traci.vehicle.getDistance(self.ego_name) > 150:
            self.angle = traci.vehicle.getAngle(self.ego_name)
            edge_id = traci.vehicle.getRoadID(self.ego_name)
            self.pt = list(traci.vehicle.getPosition(self.ego_name))

            self.pt[0]=self.pt[0]+self.vt*math.sin(math.radians(self.angle))*self.time_step+0.5 * acceleration*math.sin(math.radians(self.angle)) * self.time_step ** 2
            self.pt[1]=self.pt[1]+self.vt*math.cos(math.radians(self.angle))*self.time_step+0.5 * acceleration*math.sin(math.radians(self.angle)) * self.time_step ** 2
            #print("{},{}".format(self.pt[0],self.pt[1]))
            """
            if edge_id == "-L2" or edge_id == "-L3":
                for i in range(2):
                    if abs(self.pt[i]) != 1.6:
                        break
                self.pt[i] = self.pt[i] + self.vt * self.time_step + 0.5 * acceleration * self.time_step ** 2
            if edge_id == "-L1" or edge_id == "-L4":
                for i in range(2):
                    if abs(self.pt[i]) != 1.6:
                        break
                self.pt[i] = self.pt[i] - self.vt * self.time_step - 0.5 * acceleration * self.time_step ** 2
"""
    def get_state(self):
        # exception: when successfully reached, calc the ego state separately
        if self.ego_name not in traci.vehicle.getIDList():
            # get 8 nearest vehicles within 50m
            vehicles = [
                [vehicle,
                 distance.euclidean(traci.vehicle.getPosition(vehicle), self.pt)]
                for vehicle in traci.vehicle.getIDList() if
                distance.euclidean(traci.vehicle.getPosition(vehicle), self.pt) < 50]
            vehicles = sorted(vehicles, key=(lambda x: x[1]))[0:7]
            """
            try:
                vehicles.pop(0)
            except IndexError:
                pass
"""
            # speed of ego and other 8 vehicles as state, when lack add -200
            velocity_state = [traci.vehicle.getSpeed(vehicle[0]) for vehicle in vehicles]
            velocity_state.insert(0, self.vt)
            while len(velocity_state) < 8:
                velocity_state.append(-200)

            position_state = [p for vehicle in vehicles for p in traci.vehicle.getPosition(vehicle[0])]
            position_state.insert(0, self.pt[1])
            position_state.insert(0, self.pt[0])
            while len(position_state) < 16:
                position_state.append(-200)

            angle_state = [traci.vehicle.getAngle(vehicle[0]) for vehicle in vehicles]
            angle_state.insert(0, self.angle)
            while len(angle_state) < 8:
                angle_state.append(-200)

            type_state = [
                [self.veh_type_param[vehicle[0][1]]["length"], self.veh_type_param[vehicle[0][1]]["width"],
                 self.veh_type_param[vehicle[0][1]]["height"], self.veh_type_param[vehicle[0][1]]["a_max"]] for
                vehicle in vehicles]
            type_state.insert(0,[self.veh_type_param[self.ego_name[1]]["length"], self.veh_type_param[self.ego_name[1]]["width"],self.veh_type_param[self.ego_name[1]]["height"], self.veh_type_param[self.ego_name[1]]["a_max"]])
            type_state = [j for i in type_state for j in i]
            while len(type_state) < 32:
                type_state.append(-200)

            self.state = velocity_state + position_state + angle_state +type_state

        else:
            # get 8 nearest vehicles within 50m
            vehicles = [
                [vehicle,
                 distance.euclidean(traci.vehicle.getPosition(vehicle), traci.vehicle.getPosition(self.ego_name))]
                for vehicle in traci.vehicle.getIDList() if
                distance.euclidean(traci.vehicle.getPosition(vehicle),
                                   traci.vehicle.getPosition(self.ego_name)) < 50]

            vehicles = sorted(vehicles, key=(lambda x: x[1]))[0:8]
            # speed of ego and other 4 vehicles as state, when lack add -200
            velocity_state = [traci.vehicle.getSpeed(vehicle[0]) for vehicle in vehicles]
            while len(velocity_state) < 8:
                velocity_state.append(-200)

            position_state = [p for vehicle in vehicles for p in traci.vehicle.getPosition(vehicle[0])]
            while len(position_state) < 16:
                position_state.append(-200)

            angle_state = [traci.vehicle.getAngle(vehicle[0]) for vehicle in vehicles]
            while len(angle_state) < 8:
                angle_state.append(-200)

            type_state = [
                [self.veh_type_param[vehicle[0][1]]["length"], self.veh_type_param[vehicle[0][1]]["width"],
                 self.veh_type_param[vehicle[0][1]]["height"],self.veh_type_param[vehicle[0][1]]["a_max"]] for vehicle in vehicles]
            type_state = [j for i in type_state for j in i]
            while len(type_state) < 32:
                type_state.append(-200)

            self.state = velocity_state + position_state + angle_state+type_state

        return self.state

    def get_reward(self):
        # if collision, reward -2000-v^2; if success, reward 1000
        if self.done:
            self.reward = -2000 - traci.vehicle.getSpeed(self.ego_name) ** 2 if self.is_collided() else 1000
            return self.reward
        # speed reward: 0 reward when 8.33 ~ 11.11 m/s(30 ~ 40 km/h); out of the range negative reward
        ego_speed = traci.vehicle.getSpeed(self.ego_name)
        speed_reward = 1 * (ego_speed - 8.33) if ego_speed < 8.33 else (
            0 if ego_speed < 11.11 else 4 * (11.11 - ego_speed))
        # action reward
        action_reward = -0.1 if self.action != 0 else 0

        self.reward = speed_reward + action_reward
        return self.reward

    def update(self):
        self.done = self.is_done()
        self.get_state()
        self.get_reward()

    def is_done(self):
        return self.ego_name not in traci.vehicle.getIDList() or self.is_collided()  # reached or collided

    def is_collided(self):
        return True if self.ego_name in traci.simulation.getCollidingVehiclesIDList() else False


if __name__ == "__main__":
    from utils import plot_scores

    # start sumo env
    env = SumoEnv()
    env.start()
    # initialise
    episodes = [i for i in range(2000)]
    scores = []
    actions = [-2.5, -1.5, 0.0, 1.5, 2.5]
    # run episode
    for episode in episodes:
        observation = env.reset()
        score = 0
        while 1:
            # simple rule based model: 15m/s
            ego_speed = observation[0]  # get ego speed
            # process information and take action
            if abs(ego_speed) <= 15:  # if ego speed is under 15 m/s, speed up, acceleration 1.5m/s^2; else -1.5 m/s^2
                action = 1.5
            else:
                action = -1.5
            # step and add score
            observation, reward, done, _ = env.step(action)
            score += reward
            if done:
                break
        scores.append(score)
    # plot mean scores every 100 episode
    plot_scores(episodes, scores, average=100)
    env.close()
