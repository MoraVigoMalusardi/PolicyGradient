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


def build_mlp(
    layer_sizes: List[int],
    activation: nn.Module = nn.Tanh,
    output_activation: nn.Module = nn.Identity
) -> nn.Sequential:
    
    """ Construye un multi-layer perceptron (MLP)."""
    layers = []
    for i in range(len(layer_sizes) - 1):
        act = activation if i < len(layer_sizes) - 2 else output_activation
        layers += [nn.Linear(layer_sizes[i], layer_sizes[i + 1]), act()]
    return nn.Sequential(*layers)

def get_policy(policy_net: nn.Module, obs: torch.Tensor) -> torch.distributions.Categorical:
    """Computa el forward de la NN y obtiene la distribución de probabilidad de la política."""
    # logits = f_θ(s_t)
    # La red policy_net produce para cada estado s_t un vector de logits
    # (valores reales sin normalizar) para cada acción posible a ∈ A
    # Matemáticamente: f_θ(s_t) = logits_t ∈ ℝ^{|A|}
    logits = policy_net(obs)
    # π_θ(a_t | s_t) = softmax(logits_t)
    # Construimos la distribución categórica a partir de los logits
    # La softmax convierte los logits en probabilidades
    return torch.distributions.Categorical(logits=logits)

def compute_returns(rewards, gamma):
    """ Computa los retornos descontados """
    # Inicializamos G = 0, que representa G_{T+1} = 0 (después del último paso)
    G, returns = 0, []
    # Recorremos las recompensas en orden inverso: r_T, r_{T-1}, ..., r_0
    for r in reversed(rewards):
        # G_t = r_t + γ * G_{t+1}
        # Esto calcula los retornos acumulados descontados
        G = r + gamma * G
        # Insertamos G_t al inicio de la lista para obtener {G_0, ...,G_T} (porque lo estamos calculando al revés)
        returns.insert(0, G)
    # Convertimos la lista a tensor para poder usarla en PyTorch
    return torch.tensor(returns, dtype=torch.float32)

def compute_loss(policy_net, obs, acts, retunrs):
    """Computa la pérdida del gradiente de política."""
    # log_probs = log π_θ(a_t | s_t)
    # Calculamos log π_θ(a_t | s_t) para cada paso t del episodio
    # obs = {s_t}, acts = {a_t}
    log_probs = get_policy(policy_net, obs).log_prob(acts)
    # L(θ) = - Σ_t log π_θ(a_t | s_t) * retunrs
    # returs: retornos acumulados para cada t
    # El signo negativo es porque queremos maximizar la suma pero
    # los optimizadores minimizan, por eso usamos (-) en la pérdida
    return -(log_probs * retunrs).sum()

def train_policy(
    env,
    policy_net: nn.Module,
    optimizer: optim.Optimizer,
    episodes: int = 150,
    gamma: float = 0.99,
    batch_size: int = 1,
):
    losses = []
    rewards_per_episode = [] 

    for episode in range(episodes):
        batch_obs, batch_acts, batch_returns = [], [], []
        total_reward = 0 

        # Recolectamos N trayectorias
        for _ in range(batch_size):
            obs_buf, act_buf, rew_buf = [], [], []
            s, _ = env.reset()
            terminated, truncated = False, False
            # Generar una trayectoria
            while not (terminated or truncated):
                obs_buf.append(s)
                s_t = torch.tensor(s, dtype=torch.float32).unsqueeze(0) 
                dist = get_policy(policy_net, s_t)  # π_θ(a_t | s_t)
                a = dist.sample()      # a_t ~ π_θ(a_t | s_t)
                s, r, terminated, truncated, _ = env.step(a.item()) # Ejecuta a_t y transiciona de estado
                act_buf.append(a.item())
                rew_buf.append(r)
                total_reward += r   # Es mejor tomar las recompensas descontadas no?

            batch_obs.extend(obs_buf)
            batch_acts.extend(act_buf)
            discounted_returns = compute_returns(rew_buf, gamma)
            batch_returns.extend(discounted_returns)  

        # Preparar tensores
        obs = torch.tensor(np.array(batch_obs), dtype=torch.float32)
        acts = torch.tensor(batch_acts, dtype=torch.int64)
        returns = torch.tensor(batch_returns, dtype=torch.float32)

        # Calcular pérdida y actualizar parámetros
        loss = compute_loss(policy_net, obs, acts, returns)
        losses.append(loss.item())
        optimizer.zero_grad() # pone a cero los gradientes de todos los parámetros asociados a ese optimizador.
        loss.backward()       # Calcula los gradientes de la función de pérdida con respecto a los parámetros de la red neuronal.
        optimizer.step()      # actualiza los parametros de la red utilizando los gradientes calculados por loss.backward()
        rewards_per_episode.append(total_reward) 

        if (episode + 1) % 5  == 0:
            print(f"Episodio {episode+1}, Recompensa total: {sum(rew_buf)}")

    env.close()
    return losses, rewards_per_episode

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
    lr = 0.01     # learning rate

    # Red de política
    policy_net = build_mlp([state_dim, 16, action_dim])
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)

    train_losses, rewards_per_episode = train_policy(env, policy_net, optimizer, episodes=episodes, gamma=gamma, batch_size=5)

    print("\nEntrenamiento completado. Probando la política aprendida...\n")

    # --------------------- TEST ---------------------
    # Observación trivial obs, la recibimos como parámetro
    obs = torch.tensor(obs, dtype=torch.float32)
    # Hacemos forward pass para obtener los logits
    logits = policy_net(obs)
    # Softmax para convertir logits en probabilidades
    probs = F.softmax(logits, dim=-1).detach().numpy().flatten()  # shape: (action_dim,)

    # --------------------- GRÁFICOS ---------------------
    utils.plot_action_probs(probs, environment_name)
    utils.plot_loss_curve(train_losses, environment_name)
    utils.plot_rewards(rewards_per_episode, environment_name)

    # --------------------- SIMULACIÓN ---------------------
    # if "CartPole" in environment_name:
    #     env = gym.make("CartPole-v1", render_mode="human") 
    #     simulate_policy(env, policy_net, episodes=3, render=True, sleep_time=0.5)

    # if "Acrobot" in environment_name:
    #     env = gym.make("Acrobot-v1", render_mode="human") 
    #     simulate_policy(env, policy_net, episodes=3, render=True, sleep_time=0.5)



import time

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
    train_and_test_env("CartPole-v1", episodes=500, obs=np.array([0.0, 0.0, 0.0, 0.0]))

    # Entrenar y probar en Acrobot-v1
    # train_and_test_env("Acrobot-v1", episodes=500, obs=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
