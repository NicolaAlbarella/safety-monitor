import numpy as np


# Python implementation of RSS for intersection geometries
# Definitions  16-17-18 from the original paper
# Definition 17 pnt 1 is not implemented (intersection priorities)
class SafetyMonitor:

    def __init__(self, rot):
        # kind of a rotation matrix used for coordinate transformation
        # +1 or -1 for my intersection
        self.rot = rot  
        # ego vehicle parameters: reaction time, minbrake, maxbrake, maxacc
        self.params = np.array([2, -3, -4, 2])  
        self.car_len = 0
        # position init intersection
        self.p_in = - 5.7  
        # position end intersection 
        self.p_out = 2 + 5 + self.car_len  

        # outputs
        self.apply_bound = False
        self.lastState = ''

    # checks if situation is safe 
    # paramters of other vehicles (foes) are saved in variable foes 
    # foes is a dictionary of lists e.g. 
    # {'foe1' : [pos, speed, accel, reaction time, minbrake, maxbrake, maxacc, lastState], 'foe2': ...}
    # ego is a list [pos, speed, accel]
    def is_safe(self, foes, ego):
        ego = np.append(ego, self.params)
        distance_ego_in = self.p_in - ego[0]  

        bound = list([False]) * foes.__len__()
        # for each foe
        for i, foe in enumerate(foes):
            
            # evaluates if it is on the intersection path
            if self.is_intersecting(foes[foe], ego):
                distance_foe_in = self.p_in - self.rot * foes[foe][0]  

                # check if ego is in front  (Definition 16)
                is_in_front = ego[0] > self.rot * foes[foe][0]

                # is it safe according to Definition 17 pnt 2?
                if is_in_front:
                    distance = np.max([distance_foe_in - distance_ego_in - self.car_len, 0])
                    safe, safeDistance = check_longitudinal(ego, foes[foe], distance)
                else:
                    distance = np.max([distance_ego_in - distance_foe_in - self.car_len, 0])
                    safe, safeDistance = check_longitudinal(foes[foe], ego, distance)

                #  if not safe check for Definition 17 pnt 3
                if not safe:
                    safe = check_lateral(ego, foes[foe], self.p_in, self.p_out, self.rot)
                    new_state = str(int(safe)) + 'lat'
                else:
                    new_state = str(int(safe)) + 'long' + str(int(is_in_front))

                #  get last safe state of the object
                last_state = foes[foe][7]

                if safe:
                    # if it is safe update its last state
                    bound[i] = False
                    foes[foe][7] = new_state
                else:
                    #  otherwise, do not update the state and calculate
                    #  proper response (Definition 18)
                    if last_state == '':
                        # no last state available
                        bound[i] = True
                    elif last_state == '1long1':
                        # last state is safe and ego in front 
                        bound[i] = False
                    elif last_state == '1long0':
                        # last state is safe and foe in front
                        bound[i] = True
                    elif last_state == '1lat':
                        # last state is safe for lateral check (no one in front)
                        bound[i] = True
            else:
                bound[i] = False
                foes[foe][7] = 'dontcare'

        # evaluate Most Important Object
        MIO = next((i for i, x in enumerate(bound) if x is True), 0)
        self.apply_bound = any(bound)
        self.lastState = foes[list(foes.keys())[MIO]][7]
        return not self.apply_bound, self.lastState

    def get_state(self):
        return not self.apply_bound, self.lastState

    def is_intersecting(self, foe, ego):
        return ego[0] < self.p_out and self.rot * foe[0] < self.p_out


def check_lateral(ego, foe, pin, pout, rot):
    #  lateral is always unsafe
    #  i.e. just check for arriving times in intersection (Definition 17 pnt 3)

    p_ego = ego[0]
    v_ego = ego[1]
    rho_ego = ego[3]
    maxbrake_ego = ego[4]
    minbrake_ego = ego[5]
    maxacc_ego = ego[6]
    p_object = rot * foe[0]  # to my coordinate system
    v_object = foe[1]
    rho_object = foe[3]
    maxbrake_object = foe[4]
    minbrake_object = foe[5]
    maxacc_object = foe[6]
    maxv = 20
    # minimum time

    isSafe = False
    t_in_ego = calculate_time_to_cover_distance(v_ego, maxv, rho_ego, maxacc_ego, minbrake_ego,
                                                np.abs(p_ego - pin))
    t_out_ego = calculate_time_to_cover_distance(v_ego, maxv, rho_ego, maxbrake_ego, maxbrake_ego,
                                                 np.abs(p_ego - pout))

    t_in_object = calculate_time_to_cover_distance(v_object, maxv, rho_object, maxacc_object, minbrake_object,
                                                   np.abs(p_object - pin))
    t_out_object = calculate_time_to_cover_distance(v_object, maxv, rho_object, maxbrake_object, maxbrake_object,
                                                    np.abs(p_object - pout))

    if (t_in_ego > t_out_object) or (t_in_object > t_out_ego) or \
            (np.isinf(t_in_ego) and np.isinf(t_in_object)):
        isSafe = True

    return isSafe


def check_longitudinal(leading, following, distance):
    #  check longitudinal safe distance between leader and follower
    safeDistance = calculate_safe_longitudinal_distance(leading, following)
    return distance > safeDistance, safeDistance


def calculate_safe_longitudinal_distance(leading, following):
    #  calculate the longitudinal safe distance based on kinematic prediction
    v = following[1]
    maxv = 20
    rho = following[3]
    minbrake = following[5]
    maxacc = following[6]

    following_distance = calculate_distance_after_braking(v, maxv, rho, maxacc, minbrake)

    v = leading[1]
    maxbrake = leading[4]
    leading_distance = calculate_stopping_distance(v, maxbrake)
    safe_distance = following_distance - leading_distance

    return np.max([safe_distance, 0])


def calculate_distance_after_braking(v, maxv, rho, maxacc, minbrake):
    #  calculate distance after braking pattern
    #  first accelerate during reaction time, then brake

    p1, v1 = calculate_accelerated_limited_movement(v, maxv, maxacc, rho)
    braking_distance = calculate_stopping_distance(v1, minbrake)
    return p1 + braking_distance


def calculate_time_to_cover_distance(v, maxv, rho, a_until_rho, a_after_rho, distance):
    #  calculate time to reach a destination

    assert distance >= 0, 'negative distance'
    # position and speed after response
    p1, v1 = calculate_accelerated_limited_movement(v, maxv, a_until_rho, rho)
    if p1 >= distance:
        # if vehicle passed the distance before rho
        requiredTime = calculate_time_for_distance(v, maxv, a_until_rho, distance)
    elif v1 == 0:
        # distance not reached and speed zero, will never reach the distance
        requiredTime = np.inf
    else:
        stopping_distance = calculate_stopping_distance(v1, a_after_rho)

        if p1 + stopping_distance > distance:
            # stopped too far
            distance_diff = distance - p1
            # maxv does not matter now
            requiredTime = calculate_time_for_distance(v1, maxv, a_after_rho, distance_diff)
            requiredTime = requiredTime + rho
        else:
            # cannot reach that distance
            requiredTime = np.inf

    return requiredTime


def calculate_accelerated_limited_movement(v, maxv, acc, duration):
    #  calculate uniform acceleration movement accounting for maximum velocity
    assert duration >= 0, 'negative duration'
    assert v >= 0, 'negative speed'
    # assert acc != 0, 'null acceleration'

    v1 = calculate_speed_in_acc_movement(v, acc, duration)

    v1 = np.max([v1, 0])  # ignore negative speeds
    limit = np.max([maxv, v])  # if v > maxv update the limit
    v1 = np.min([v1, limit])
    if v1 > limit:
        #  this way is more stable
        #  if over the limit saturate and re-evaluate the duration of the acceleration
        v1 = limit
        if acc != 0:
            accelDuration = (v1 - v) / acc
            res_duration = duration - accelDuration
    else:
        res_duration = 0

    assert res_duration >= 0, 'negative duration'
    return v * duration + 0.5 * acc * duration * duration + (v1 - v) * res_duration, v1


def calculate_speed_in_acc_movement(v, acc, duration):
    #  calculate speed in uniform accelerated movement
    assert duration >= 0, 'negative duration'
    return v + acc * duration


def calculate_time_for_distance(v, maxv, acc, distance):
    #  calculate time to reach a destination
    assert v >= 0, 'negative speed'
    if acc == 0 or (v >= maxv and acc > 0):
        if v == 0:
            # no acceleration and stopped. cannot reach
            requiredTime = np.inf
        else:
            # not accelerating (or saturating velocity)
            requiredTime = distance / v
    else:
        # acc is not zero, need more calc
        requiredTime = calc_required_time(v, acc, distance)

        if acc > 0:
            # need saturate for maxv
            acceleration_time = (maxv - v) / acc

            if requiredTime > acceleration_time:
                # saturate now
                d, v = calculate_accelerated_limited_movement(v, maxv, acc, acceleration_time)
                if v != maxv or d > distance:
                    raise Exception("something is wrong")
                else:
                    requiredTime = calc_required_time(v, acc, d)
                    requiredTime = requiredTime + (distance - d) / maxv

    return requiredTime


def calc_required_time(v, acc, distance):
    #  solving the quadratic formula for time
    fistPart = - v / acc
    Delta_sq = np.sqrt(fistPart * fistPart + 2 * distance / acc)
    t1 = fistPart + Delta_sq
    t2 = fistPart - Delta_sq

    if t2 > 0:
        return t2
    else:
        return t1


def calculate_stopping_distance(v, deceleration):
    #  calculate uniform deceleration (final stopping distance)

    assert np.sign(v) != np.sign(deceleration)
    if deceleration == 0:
        if v != 0:
            # cannot stop
            return False
        else:
            # already stopped
            return 0

    return v * v / (2 * (-deceleration))

