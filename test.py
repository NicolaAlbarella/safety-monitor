from envs.Intersection.intersection import IntersectionEnv
import time


def main():
    env_config = dict()
    env_config['mode'] = 1
    env_config['random'] = 1

    env = IntersectionEnv()
    env.reset()

    while 1:
        isDone = env.step(15)
        if isDone:
            env.reset()
        time.sleep(0.02)



if __name__ == "__main__":
    main()
