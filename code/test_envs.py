from environments import TwoAZeroObsOneStepEnv, TwoARandomObsOneStepEnv, LineWorldEasyEnv, LineWorldMirrorEnv

def test1():
    env = TwoAZeroObsOneStepEnv()

    for i in range(5):
        print(f"Episodio {i+1}")
        obs, _ = env.reset()
        terminated = False
        truncated  = False
        while not terminated and not truncated:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Acción: {action}, Reward: {reward}, Observation:{obs}\n")


def test2():
    env = TwoARandomObsOneStepEnv()

    for i in range(5):
        print(f"Episodio {i+1}")
        obs, _ = env.reset()
        print(f"Estado inicial: {obs}")
        terminated = False
        truncated  = False
        while not terminated and not truncated:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Acción: {action}, Reward: {reward}, Observation:{obs}\n")

def test3():
    env = LineWorldEasyEnv()

    obs, _ = env.reset()
    terminated = False
    truncated  = False
    while not terminated and not truncated:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Acción: {action}, Reward: {reward}, Observation:{obs}\n")


def test4():
    env = LineWorldMirrorEnv()

    obs, _ = env.reset()
    terminated = False
    truncated  = False
    while not terminated and not truncated:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Acción: {action}, Reward: {reward}, Observation:{obs}\n")


test4()