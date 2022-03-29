from envs.Intersection.vehicle import Vehicle
import numpy as np

import traci.constants as tc


class EgoVehicle(Vehicle):

    def __init__(self, idin, state, tr):
        super().__init__(idin, state, tr, -1)
        self.Ts = 0.02  # mpc sample time
        self.tau = 5
        self.control = None
        self.counter = 0

    def step(self, vin):

        if self.counter == 5:
            self.control = None
            self.counter = 0
        safe, foe = self.monitor.get_state()
        if self.control is None:
            self.control = 0.1*(vin-self.state[1])
        accelerate = self.control if safe else -5
        self.integrate_dynamics(accelerate)
        self.counter = self.counter + 1
        return self.state[0], self.state[1]

    def get_observation(self, tr):
        sumo_out = tr.vehicle.getContextSubscriptionResults('ego')
        sumo_out.pop('ego', None)
        vehicles = sumo_out.keys()

        toDel = list()
        for veh in self.foes:
            if veh not in vehicles:
                toDel.append(veh)
        for veh in toDel:
            del self.foes[veh]

        for veh in vehicles:
            position = sumo_out[veh][tc.VAR_POSITION][0]
            speed = sumo_out[veh][tc.VAR_SPEED]
            accel = sumo_out[veh][tc.VAR_ACCELERATION]
            if veh in self.foes:
                self.foes[veh][0:3] = list([-position, speed, accel])
            else:
                rho = 2
                maxbrake = -3
                minbrake = -4
                maxacc = 2
                self.foes[veh] = list([-position, speed, accel, rho, maxbrake, minbrake, maxacc, '1lat'])

        self.foes = {k: v for k, v in sorted(self.foes.items(), key=lambda item: item[1])}
        return self.foes


    def integrate_dynamics(self, cmd):
        s = self.state[0]
        v = self.state[1]
        a = self.state[2]

        v_next = v + self.Ts * a
        if v_next < 0:
            v_next = 0
            a_next = 0
        else:
            a_next = a + self.Ts * self.tau * (-a + cmd)
        s_next = s + + self.Ts * v
        self.state = np.array([s_next, v_next, a_next])


