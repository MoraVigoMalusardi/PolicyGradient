import torch
import numpy as np
import torch.optim as optim
from environments import TwoAZeroObsOneStepEnv, TwoARandomObsOneStepEnv, LineWorldEasyEnv, LineWorldMirrorEnv
import torch.nn.functional as F
import gymnasium as gym
import utils
from reinforce import build_mlp, train_policy
from utils import CONVERGENCE

def train_and_test_env(environment_name: str, 
                       episodes: int, 
                       obs: np.ndarray, 
                       batch_size: int, 
                       early_stop: bool = False):
    """
    Entrena una política en el entorno especificado y luego prueba la política aprendida.
    Args:
        environment_name (str): Nombre del entorno a entrenar.
        episodes (int): Número de episodios para entrenar.S
        obs (np.ndarray): Observación inicial para probar la política aprendida.
        batch_size (int): Tamaño del batch (numero de trayectorias antes de actualizar la política).
        early_stop (bool): Si True, usa criterio de convergencia para detener el entrenamiento.
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
    value_net = build_mlp([state_dim, 16, 1])                     # la salida es un escalar (valor del estado)

    policy_optimizer = optim.Adam(policy_net.parameters(), lr=lr_policy)
    value_opt = optim.Adam(value_net.parameters(), lr=lr_value)

    if early_stop:
        early_stop_cfg = {**CONVERGENCE[environment_name]}
    else:
        early_stop_cfg = None

    train_losses, value_losses, rewards_per_episode = train_policy(env, policy_net, value_net, policy_optimizer, value_opt, episodes=episodes, gamma=gamma, batch_size=10, use_baseline=True, normalize_advantages=True, early_stop_cfg=early_stop_cfg)

    print("\nEntrenamiento completado. Probando la política aprendida...\n")

    # --------------------- TEST ---------------------
    obs = torch.tensor(obs, dtype=torch.float32)                  # Observación trivial obs, la recibimos como parámetro
    logits = policy_net(obs)                                      # Hacemos forward pass para obtener los logits
    probs = F.softmax(logits, dim=-1).detach().numpy().flatten()  # Softmax para convertir logits en probabilidades

    # --------------------- GRÁFICOS ---------------------
    #utils.plot_action_probs(probs, environment_name)
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

import random

def set_seed(seed):
    """ Fija una seed para poder reproducir resultados """
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def run_once(env_ctor, env_name, episodes, batch_size, use_baseline, seed):
    """ Corre una vez el entrenamiento de REINFORCE en un entorno dado.
    Args:
        env_ctor: funcion que crea el entorno
        env_name (str): nombre del entorno (para guardar graficos)
        episodes (int): cantidad de episodios para entrenar
        batch_size (int): tamaño de batch (cantidad de trayectorias antes de actualizar la politica)
        use_baseline (bool): si es True, usa baseline
        seed (int): semilla para reproducibilidad
    """
    set_seed(seed)
    env = env_ctor()

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy_net = build_mlp([state_dim, 32, action_dim])
    value_net  = build_mlp([state_dim, 32, 1]) if use_baseline else build_mlp([state_dim, 1, 1])

    pol_opt = optim.Adam(policy_net.parameters(), lr=3e-3)
    val_opt = optim.Adam(value_net.parameters(),  lr=3e-3) if use_baseline else optim.Adam(value_net.parameters(), lr=1e-6)

    pol_losses, val_losses, rewards = train_policy(
        env, policy_net, value_net, pol_opt, val_opt,
        episodes=episodes, gamma=0.99, batch_size=batch_size,
        normalize_advantages=True, use_baseline=use_baseline
    )
    return {"rewards": rewards, "pol_losses": pol_losses, "val_losses": val_losses}


def compare_base_and_no_base(env_name, episodes=500, batch_size=10):    
    """ 
    Compara el desempeño de REINFORCE con y sin baseline en un entorno dado.

    Args:   
        env_name (str): puede ser CartPole-v1 o Acrobot-v1
        episodes (int): cantidad de episodios para entrenar
        batch_size (int): tamaño de batch (cantidad de trayectorias antes de actualizar la política)
    """
    envs = {
        "CartPole-v1": lambda: gym.make("CartPole-v1"),
        "Acrobot-v1":  lambda: gym.make("Acrobot-v1"),
    }
    env_ctor = envs[env_name]

    with_base = run_once(env_ctor, env_name, episodes, batch_size, use_baseline=True, seed=0)
    no_base   = run_once(env_ctor, env_name, episodes, batch_size, use_baseline=False, seed=0)

    # curvas de rewards
    rewards_base = with_base["rewards"]
    rewards_noba = no_base["rewards"]

    # grafico de comparacion
    utils.plot_two_curves(
        rewards_base, rewards_noba,
        title=f"REINFORCE en {env_name}: con vs sin baseline",
        label_A="Con baseline", label_B="Sin baseline",
        x_label="Episodios", y_label="Recompensa promedio por episodio",
        out_path=f"plots/compare_baseline/{env_name}.png"
    )


def compare_and_plot(env_name, episodes=500, batch_size=10, seeds=(0,1,2,3,4)):
    """ Compara el desempeño de REINFORCE con y sin baseline en un entorno dado.
    Args:   
        env_name (str): puede ser CartPole-v1 o Acrobot-v1
        episodes (int): cantidad de episodios para entrenar
        batch_size (int): tamaño de batch (cantidad de trayectorias antes de actualizar la política)
        seeds (tuple): semillas para repetir la experiencia y promediar resultados
    """
    envs = {
        "CartPole-v1": lambda: gym.make("CartPole-v1"),
        "Acrobot-v1":  lambda: gym.make("Acrobot-v1"),
    }
    env_ctor = envs[env_name]

    with_base, no_base = [], []
    for s in seeds:
        with_base.append(run_once(env_ctor, env_name, episodes, batch_size, True,  s))
        no_base.append(run_once(env_ctor, env_name, episodes, batch_size, False, s))

    # curvas de rewards
    curves_base = [r["rewards"] for r in with_base]
    curves_noba = [r["rewards"] for r in no_base]

    # grafico de comparacion
    utils.plot_mean_std_superposed(
        curves_base, curves_noba,
        title=f"REINFORCE en {env_name}: con vs sin baseline",
        label_A="Con baseline", label_B="Sin baseline",
        out_path=f"plots/compare_baseline/compare_return_{env_name}.png",
        window=50
    )

if __name__ == "__main__":
    # Entrenar y probar en TwoAZeroObsOneStepEnv
    # train_and_test_env("TwoAZeroObsOneStep", episodes=150, obs=np.array([0.0]), batch_size=10, early_stop=False)

    # Entrenar y probar en TwoARandomObsOneStepEnv
    # train_and_test_env("TwoARandomObsOneStepEnv", episodes=150, obs=np.array([1.0, 0.0]), batch_size=10, early_stop=True)

    # Entrenar y probar en LineWorldEasyEnv
    # train_and_test_env("LineWorldEasyEnv", episodes=150, obs=np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]), batch_size=10, early_stop=True)

    # Entrenar y probar en LineWorldMirrorEnv
    # train_and_test_env("LineWorldMirrorEnv", episodes=150, obs=np.array([1.0, 0.0, 0.0, 0.0]), batch_size=10, early_stop=True)

    # Entrenar y probar en CartPole-v1
    # train_and_test_env("CartPole-v1", episodes=500, obs=np.array([0.0, 0.0, 0.0, 0.0]), batch_size=10, early_stop=True)

    # Entrenar y probar en Acrobot-v1
    # train_and_test_env("Acrobot-v1", episodes=500, obs=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), batch_size=10, early_stop=True )

    #compare_and_plot("CartPole-v1", episodes=500, batch_size=10, seeds=(0,1,2,3,4))
    # compare_and_plot("Acrobot-v1", episodes=500, batch_size=10, seeds=[0, 1, 2])
    
    #compare_base_and_no_base("CartPole-v1", episodes=500, batch_size=10)
    #compare_base_and_no_base("Acrobot-v1", episodes=500, batch_size=10)

