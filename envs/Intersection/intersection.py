import sys
import os
import random
import numpy as np
import gym
from natsort import natsorted

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

import traci
import traci.constants as tc
from envs.Intersection.ego import EgoVehicle
from envs.Intersection.vehicle import Vehicle


class IntersectionEnv(gym.Env):
    """SUMO GYM WRAPPER"""
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.mode = 0
        self.random = 1
        self.Ts = 0.1
        MAX_VEHICLES = 4
        INFO_EGO = 2
        INFO_VEHICLES = 4
        self.OBS_DIM = MAX_VEHICLES * INFO_VEHICLES + INFO_EGO
        self.sensingDist = 100

        index = 0
        if not self.random:
            random.seed(42)
        self.traci_handle = self.reset_sumo(index)
        self.inert_pos_start = np.array([0, 20.7])
        self.ego = None
        self.manuals = dict()
        self.K = 5  # action scaling time
        self.sum_safe = 0

    def step(self, action):
        info = dict()
        self.sum_safe = 0
        # K times
        for i in range(0, self.K):

            pos_ego, vel = self.ego.step(action)

            for man in self.manuals:
                self.manuals[man]['obj'].step(self.Ts / self.K, self.traci_handle)

            # move everyone to new position in sumo
            for i, man in enumerate(self.manuals):
                pos = self.manuals[man]['obj'].state[0]
                if pos < 100:
                    vel_man = self.manuals[man]['obj'].state[1]
                    self.traci_handle.vehicle.setSpeed(man, vel_man)
                    road_id, _, lane_id = self.traci_handle.simulation.convertRoad(pos, 0)
                    self.traci_handle.vehicle.moveToXY(man, road_id, lane_id, pos, 0, 90)
                else:
                    self.traci_handle.vehicle.remove(man)
                    self.manuals[list(self.manuals.keys())[i + 1]]['obj'].lead = None

            self.traci_handle.vehicle.setSpeed('ego', vel)
            road_id, _, lane_id = self.traci_handle.simulation.convertRoad(0, pos_ego)
            self.traci_handle.vehicle.moveToXY('ego', road_id, lane_id, 0, -pos_ego, 180)

            self.traci_handle.simulationStep()

            self.update_manuals()
            # get new measurement
            foes = self.ego.get_observation(self.traci_handle)
            safe, _ = self.ego.monitor.is_safe(foes, self.ego.state)
            self.sum_safe = self.sum_safe + int(not safe)

        isDone = pos_ego>10.7
        info['safe'] = safe
        info['speed'] = self.ego.v_ref

        return isDone

    def reset(self):

        info = dict()
        self.sum_safe = 0

        for man in self.manuals:
            self.traci_handle.vehicle.remove(man)
        self.manuals = dict()
        self.traci_handle.simulationStep(self.traci_handle.simulation.getTime() + 100)
        start_speed = random.uniform(10, 12)

        self.ego = EgoVehicle('ego', np.array([-self.inert_pos_start[1], start_speed, 0]), self.traci_handle)
        self.traci_handle.vehicle.setSpeed("ego", start_speed)
        self.traci_handle.vehicle.moveToXY('ego', 'v1', 0, 0, self.inert_pos_start[1], 180)

        traci.vehicle.subscribeContext('ego', tc.CMD_GET_VEHICLE_VARIABLE, self.sensingDist,
                                       list([tc.VAR_POSITION, tc.VAR_SPEED, tc.VAR_ACCELERATION]))

        self.traci_handle.simulationStep()

        self.update_manuals()
        # get new measurements and observation model
        foes = self.ego.get_observation(self.traci_handle)
        safe, _ = self.ego.monitor.is_safe(foes, self.ego.state)

        info['safe'] = safe
        info['speed'] = self.ego.v_ref

        return 1

    def update_manuals(self):
        vehicles = list(self.traci_handle.vehicle.getIDList())
        del vehicles[0]
        vehicles = natsorted(vehicles)

        toDel = list()
        for veh in self.manuals:
            if veh not in vehicles:
                toDel.append(veh)
        for veh in toDel:
            del self.manuals[veh]

        for veh in vehicles:

            if veh not in self.manuals:
                position = self.traci_handle.vehicle.getPosition(veh)[0]
                speed = self.traci_handle.vehicle.getSpeed(veh)
                accel = self.traci_handle.vehicle.getAcceleration(veh)
                lead = self.traci_handle.vehicle.getLeader(veh, 100)
                rho = 1
                maxbrake = -3
                minbrake = -4
                maxacc = 2
                if lead is not None:
                    new_veh = Vehicle(veh, np.array([position, speed, accel]), self.traci_handle, 1, lead[0])
                else:
                    new_veh = Vehicle(veh, np.array([position, speed, accel]), self.traci_handle, -1)
                self.manuals[veh] = dict()
                self.manuals[veh]['obj'] = new_veh
                self.manuals[veh]['params'] = list([rho, maxbrake, minbrake, maxacc])
                self.manuals[veh]['safety'] = ''

    def render(self, mode='human', close=False):
        ...

    def reset_sumo(self, index):
        c_path = "envs/Intersection/sumo_cfg/config.sumocfg"
        traci.start(["sumo-gui", "-c", c_path, "--seed", str(index), "--random", str(self.random)], 8873 + index,
                    label='sim' + str(index))
        handle = traci.getConnection('sim' + str(index))
        handle.simulationStep()
        view = handle.gui.getIDList()
        handle.gui.setSchema(view[0], 'custom_1')
        handle.gui.setZoom(view[0], 700000)
        handle.gui.trackVehicle(view[0], 'ego')
        return handle

    def collision_check(self):
        if self.traci_handle.simulation.getCollidingVehiclesNumber() > 0:
            if 'ego' in self.traci_handle.simulation.getCollidingVehiclesIDList():
                return 1
        return 0
