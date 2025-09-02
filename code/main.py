import torch
import numpy as np
from torch import nn
from typing import List
import torch.optim as optim
from environments import TwoAZeroObsOneStepEnv, TwoARandomObsOneStepEnv, LineWorldEasyEnv, LineWorldMirrorEnv
import torch.nn.functional as F
import matplotlib.pyplot as plt
import gymnasium as gym
import utils
from reinforce import build_mlp, train_policy

def train_and_test_env(environment_name: str, episodes: int, obs: np.ndarray):
    """
    Entrena una política en el entorno especificado y luego prueba la política aprendida.
    Args:
        environment_name (str): Nombre del entorno a entrenar.
        episodes (int): Número de episodios para entrenar.
        obs (np.ndarray): Observación inicial para probar la política aprendida.
    """
    # --------------------- ENTORNOS ---------------------
    envs = {
        "TwoAZeroObsOneStep": TwoAZeroObsOneStepEnv,
        "TwoARandomObsOneStepEnv": TwoARandomObsOneStepEnv,
        "LineWorldEasyEnv": LineWorldEasyEnv,
        "LineWorldMirrorEnv": LineWorldMirrorEnv,
        "CartPole-v1" : lambda: gym.make("CartPole-v1"),
        "Acrobot-v1": lambda: gym.make("Acrobot-v1")
    }

    # ---------------- ENTRENAMIENTO ----------------
    env = envs[environment_name]()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    gamma = 0.99 
    lr_policy = 0.01 
    lr_value = 3e-3

    # Redes
    policy_net = build_mlp([state_dim, 32, action_dim])
    value_net = build_mlp([state_dim, 16, 1])            # la salida es un escalar (valor del estado)

    policy_optimizer = optim.Adam(policy_net.parameters(), lr=lr_policy)
    value_opt = optim.Adam(value_net.parameters(), lr=lr_value)

    train_losses, value_losses, rewards_per_episode = train_policy(env, policy_net, value_net, policy_optimizer, value_opt, episodes=episodes, gamma=gamma, batch_size=5)

    print("\nEntrenamiento completado. Probando la política aprendida...\n")

    # --------------------- TEST ---------------------
    obs = torch.tensor(obs, dtype=torch.float32)                  # Observación trivial obs, la recibimos como parámetro
    logits = policy_net(obs)                                      # Hacemos forward pass para obtener los logits
    probs = F.softmax(logits, dim=-1).detach().numpy().flatten()  # Softmax para convertir logits en probabilidades

    # --------------------- GRÁFICOS ---------------------
    utils.plot_action_probs(probs, environment_name)
    utils.plot_loss_curve(train_losses, environment_name)
    utils.plot_loss_curve(value_losses, environment_name + " (value loss)")
    utils.plot_rewards(rewards_per_episode, environment_name)

    # --------------------- SIMULACIÓN ---------------------
    # if "CartPole" in environment_name:
    #     env = gym.make("CartPole-v1", render_mode="human") 
    #     utils.simulate_policy(env, policy_net, episodes=3, render=True, sleep_time=0.5)

    # if "Acrobot" in environment_name:
    #     env = gym.make("Acrobot-v1", render_mode="human") 
    #     utils.simulate_policy(env, policy_net, episodes=3, render=True, sleep_time=0.5)




if __name__ == "__main__":
    # Entrenar y probar en TwoAZeroObsOneStepEnv
    # train_and_test_env("TwoAZeroObsOneStep", episodes=150, obs=np.array([0.0]))

    # Entrenar y probar en TwoARandomObsOneStepEnv
    # train_and_test_env("TwoARandomObsOneStepEnv", episodes=150, obs=np.array([1.0, 0.0]))

    # Entrenar y probar en LineWorldEasyEnv
    # train_and_test_env("LineWorldEasyEnv", episodes=150, obs=np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]))

    # Entrenar y probar en LineWorldMirrorEnv
    # train_and_test_env("LineWorldMirrorEnv", episodes=150, obs=np.array([1.0, 0.0, 0.0, 0.0]))

    # Entrenar y probar en CartPole-v1
    # train_and_test_env("CartPole-v1", episodes=500, obs=np.array([0.0, 0.0, 0.0, 0.0]))

    # Entrenar y probar en Acrobot-v1
    train_and_test_env("Acrobot-v1", episodes=500, obs=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
