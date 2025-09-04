# ----------------- UTILS PARA GRAFICOS -----------------

import matplotlib.pyplot as plt
import numpy as np


def plot_loss_curve(train_losses, environment_name: str):
    """Genera y guarda la curva de perdida por episodio."""

    plt.figure()
    plt.plot(train_losses)
    plt.xlabel("Episodio")
    plt.ylabel("Perdida")
    plt.title(f"Curva de perdida por episodio en {environment_name}")
    plt.savefig(f"plots/losses/loss_curve_{environment_name}.png")
    plt.close()

def plot_action_probs(probs, environment_name: str):
    """Genera y guarda un grafico de barras con las probabilidades de accion."""

    plt.figure()
    plt.bar(range(len(probs)), probs)
    plt.ylabel("Probabilidad")
    plt.xlabel("Acciones")
    plt.xticks(range(len(probs)))
    plt.title(f"Probabilidades de cada accion en {environment_name}")
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


def moving_average(x, w):
    """ 
    Calcula la media movil de ventana w sobre los valores de la lista x
    Para cada punto i, reemplaza x[i] por el promedio de los w puntos 
    anteriores (incluyendo el valor de x[i])
    """
    
    w = min(w, len(x))
    return np.convolve(x, np.ones(w)/w, mode="valid") if len(x) else np.array([])


def pad_to_same_length(curves):
    """
    Dada una lista de listas (curvas) las rellena con NaNs para que todas tengan la misma longitud
    y devuelve un array 2D (num_curvas, L) 
    """
    L = max(len(c) for c in curves)                                # longitud de la mas larga
    out = []
    for c in curves:
        if len(c) < L:
            c = np.concatenate([c, np.full(L-len(c), np.nan)])     # relleno con NaNs
        out.append(c)
    return np.vstack(out)


def plot_mean_std_superposed(curves_A, curves_B, title, label_A, label_B, out_path, window=50):
    """
    Para cada seed tenemos una serie de recompensas: seed1 --> r_1, r_2, ..., r_N, seed2 --> r_1, r_2, ..., r_M, etc.

    1. La media movil de ventana w reemplaza cada punto (r_i) por el promedio de sus w vecinos anteriores:
        r_i  = (r_{i-w} + ... + r_{i-1} + r_i) / w
    
    2. Despues de suavizar para cada seed, las listas podriann (casi siempre si) no medir lo mismo.
         Entonces las igualamos rellenando con NaNs hasta la longitud de la mas larga.
    
    3. Calculamos para cada episodio, la media y std comparando a trves de las ejecuciones (con una seed distinta c/u).
    """

    # Suavizamos por seed 
    A = [moving_average(c, window) for c in curves_A]
    B = [moving_average(c, window) for c in curves_B]

    # Alineamos las longitudes de las listas suavizadas 
    A = pad_to_same_length(A) # (num_seeds, L)
    B = pad_to_same_length(B) # (num_seeds, L)

    # Promedio y std por seed (osea por ejecucion de entrenamientos distintos)
    Am, As = np.nanmean(A, axis=0), np.nanstd(A, axis=0)  # para cada episodio guarda el promedio de todas las ejecuciones en ese episodio
    Bm, Bs = np.nanmean(B, axis=0), np.nanstd(B, axis=0)

    # Eje x (cantidad de episodios. Cada seed tiene = cant. de episodios, por lo que Am tiene longitud = cant. episodios)
    xA = np.arange(len(Am)) 
    xB = np.arange(len(Bm))

    plt.figure()
    plt.plot(xA, Am, label=label_A)
    plt.fill_between(xA, Am-As, Am+As, alpha=0.2)  # esto es para la sombrita alrededor de la curva. Marca la varianza alrededor de la media 
    plt.plot(xB, Bm, label=label_B)
    plt.fill_between(xB, Bm-Bs, Bm+Bs, alpha=0.2)
    plt.xlabel("Episodio")
    plt.ylabel("Rewards (con media movil)")
    plt.title(title)
    plt.legend()
    plt.savefig(out_path)
    plt.close()


def plot_two_curves(curve_A, curve_B, title, label_A, label_B, x_label, y_label, out_path):
    """ Grafica las curvas A y B """
    plt.figure()
    plt.plot(curve_A, label=label_A)
    plt.plot(curve_B, label=label_B)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.savefig(out_path)
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
