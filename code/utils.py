# ----------------- UTILS PARA GRAFICOS -----------------

import matplotlib.pyplot as plt
import numpy as np


def plot_loss_curve(train_losses, environment_name: str):
    """Genera y guarda la curva de pérdida por episodio."""
    plt.figure()
    plt.plot(train_losses)
    plt.xlabel("Episodio")
    plt.ylabel("Pérdida")
    plt.title(f"Curva de pérdida por episodio en {environment_name}")
    plt.savefig(f"plots/losses/loss_curve_{environment_name}.png")
    plt.close()

def plot_action_probs(probs, environment_name: str):
    """Genera y guarda un gráfico de barras con las probabilidades de acción."""
    plt.figure()
    plt.bar(range(len(probs)), probs)
    plt.ylabel("Probabilidad")
    plt.xlabel("Acciones")
    plt.xticks(range(len(probs)))
    plt.title(f"Probabilidades de cada acción en {environment_name}")
    plt.savefig(f"plots/action_probs/action_probs_{environment_name}.png")
    plt.close()

def plot_rewards(rewards_per_episode, environment_name: str, avg_random: float = None):
    """Genera y guarda la curva de recompensa promedio por episodio.
       Si se pasa avg_random, dibuja línea horizontal de referencia.
    """
    plt.figure()
    plt.plot(rewards_per_episode, label="Agente")
    if avg_random is not None:
        plt.axhline(y=avg_random, color="red", linestyle="--", label="Agente aleatorio")
    plt.xlabel("Episodio")
    plt.ylabel("Recompensa")
    plt.title(f"Recompensa promedio por episodio en {environment_name}")
    plt.legend()
    plt.savefig(f"plots/rewards/rewards_{environment_name}.png")
    plt.close()


# ----------------- UTILS PARA SIMULACIÓN -----------------
import time
import torch
import torch.nn.functional as F
from reinforce import get_policy

def simulate_policy(env, policy_net, episodes: int = 5, render: bool = True, sleep_time: float = 1):
    """
    Simula la política aprendida en el entorno.
    Args:
        env: entorno de gymnasium ya creado
        policy_net: red entrenada
        episodes (int): cantidad de episodios de simulación
        render (bool): si True, muestra animación con render()
        sleep_time (float): tiempo entre frames
    """
    for ep in range(episodes):
        obs, _ = env.reset()
        done, truncated = False, False
        total_reward = 0

        while not (done or truncated):
            if render:
                env.render()
                time.sleep(sleep_time)

            s_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                dist = get_policy(policy_net, s_t)
                action = dist.probs.argmax().item()  # acción más probable

            obs, reward, done, truncated, _ = env.step(action)
            total_reward += reward
            time.sleep(sleep_time)

        print(f"Episodio {ep+1}: recompensa total = {total_reward}")


    env.close()
