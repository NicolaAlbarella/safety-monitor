import numpy as np
from safety.safety_monitor import SafetyMonitor
from random import random

class Vehicle:

    def __init__(self, idin, state, tr, rot, lead=None):
        self.id = idin
        self.state = state
        self.lead = lead
        tr.vehicle.setSpeedMode(self.id, 0)
        self.monitor = SafetyMonitor(rot)
        # control params
        self.aggressive = tr.vehicle.getTypeID(idin) == 'aggressive'
        self.s0 = 2 * random() + 2 + 5
        self.a = 2
        self.b = random() + 1
        self.T = random() + 2
        self.v_ref = 3 * random() + 13
        self.mode = 'pre'

        self.foes = dict()


    def step(self, dT, tr):
        assert tr.vehicle.getSpeed(self.id) == self.state[1]
        self.get_observation(tr)
        safe, state = self.monitor.is_safe(self.foes, self.state)


        if not self.aggressive:
            if (state == 'dontcare' or state == '1long1') and safe:
                if self.lead is not None:
                    dist_lead = tr.vehicle.getPosition(self.lead)[0] - self.state[0]
                    vel = tr.vehicle.getSpeed(self.lead)
                    acc = self.get_inputs(dist_lead, vel)
                else:
                    acc = self.get_inputs(1, -1)
            elif (state == '1lat' or state == '1long0' and safe) or not safe:
                dist_ego = -(tr.vehicle.getPosition('ego')[1]) - self.state[0]
                if self.lead is not None:
                    dist_lead = tr.vehicle.getPosition(self.lead)[0] - self.state[0]
                    if dist_lead > dist_ego > 0:
                        vel = tr.vehicle.getSpeed('ego')
                        acc = self.get_inputs(dist_ego, vel)
                    else:
                        vel = tr.vehicle.getSpeed(self.lead)
                        acc = self.get_inputs(dist_lead, vel)
                else:
                    vel = tr.vehicle.getSpeed('ego')
                    acc = self.get_inputs(dist_ego, vel)
        else:
            if safe:
                if self.lead is not None:
                    dist_lead = tr.vehicle.getPosition(self.lead)[0] - self.state[0]
                    vel = tr.vehicle.getSpeed(self.lead)
                    acc = self.get_inputs(dist_lead, vel)
                else:
                    acc = self.get_inputs(1, -1)
            else:
                dist_ego = -(tr.vehicle.getPosition('ego')[1]) - self.state[0]
                if self.lead is not None:
                    dist_lead = tr.vehicle.getPosition(self.lead)[0] - self.state[0]
                    if dist_ego < dist_lead:
                        vel = tr.vehicle.getSpeed('ego')
                        acc = self.get_inputs(dist_ego, vel)
                    else:
                        vel = tr.vehicle.getSpeed(self.lead)
                        acc = self.get_inputs(dist_lead, vel)
                else:
                    vel = tr.vehicle.getSpeed('ego')
                    acc = self.get_inputs(dist_ego, vel)

        if not safe:
            acc = -5

        new_v = self.state[1] + dT * acc
        new_v = np.clip(new_v, 0, 100)
        new_pos = self.state[0] + dT * self.state[1]
        self.state = np.array([new_pos, new_v, acc])

    def get_observation(self, tr):
        position = tr.vehicle.getPosition('ego')[1]
        speed = tr.vehicle.getSpeed('ego')
        acc = tr.vehicle.getAcceleration('ego')
        if 'ego' in self.foes:
            self.foes['ego'][0:3] = list([-position, speed, acc])
        else:
            self.foes['ego'] = list([-position, speed, acc, 2, -3, -4, 2, '1lat'])

    def get_inputs(self, dist, vel_lead):
        v = self.state[1]
        acc = self.a * (1 - (v / self.v_ref) ** 4)

        if vel_lead >= 0:
            Dv = v - vel_lead
            s_star = self.s0 + v * self.T + v * Dv / (2 * np.sqrt(self.a * self.b))
            acc = acc - self.a * (s_star / dist) ** 2

        return np.clip(acc, -5, 2)
