import torch
import numpy as np
from torch import nn
from typing import List
import torch.optim as optim
import torch.nn.functional as F
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


def compute_loss(policy_net, obs, acts, rewards_or_advantages):

    """Computa la pérdida del gradiente de política."""
    # log_probs = log π_θ(a_t | s_t)
    # Calculamos log π_θ(a_t | s_t) para cada paso t del episodio
    # obs = {s_t}, acts = {a_t}
    probs = get_policy(policy_net, obs)
    log_probs = probs.log_prob(acts)
    policy_loss = -(log_probs * rewards_or_advantages).mean()     # cambie de .sum() a .mean() porque no cambia la direccion del gradiente, pero asi no depende de la cantidad de pasos del episodio
    # L(θ) = - Σ_t log π_θ(a_t | s_t) * retunrs
    # CON BASELINE se transforma en:
    # L(θ) = - Σ_t log π_θ(a_t | s_t) * (retunrs - b(s_t)) = - Σ_t log π_θ(a_t | s_t) * advantages
    # returs: retornos acumulados para cada t
    # El signo negativo es porque queremos maximizar la suma pero
    # los optimizadores minimizan, por eso usamos (-) en la pérdida
    return policy_loss

def train_policy(
    env,
    policy_net: nn.Module,
    value_net: nn.Module,
    policy_optimizer: optim.Optimizer,
    value_opt: optim.Optimizer,
    episodes: int = 150,
    gamma: float = 0.99,
    batch_size: int = 1,
    normalize_advantages: bool = True,
    use_baseline: bool = True,
    early_stop_cfg: dict = None
):
    """
    Entrena la poltica usando el algoritmo REINFORCE. Puede o no usar baseline.
    Args:
        env: entorno de gymnasium ya creado
        policy_net: red neuronal que representa la politica
        value_net: red neuronal que estima la funcion de valor (V(s)) (baseline)
        policy_optimizer: optimizador para la política
        value_opt: optimizador para la red de valor
        episodes (int): cantidad de episodios para entrenar
        gamma (float): factor de descuento
        batch_size (int): cantidad de trayectorias por episodio, antes de actualizar la politica
        normalize_advantages (bool): si True, normaliza las ventajas a media 0 y varianza 1
        use_baseline (bool): si True, usa baseline (red de value funtion)
        early_stop_cfg (dict): diccionaio con los parametros para early stop
    """
    policy_losses, value_losses = [], []
    avg_rewards_per_episode = [] 
    rewards_per_trayectory = []  # para criterio de convergencia
    episode_steps   = []         # para criterio de convergencia

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
                total_reward += r 

            rewards_per_trayectory.append(float(sum(rew_buf)))      # rewards de esta trayectoria
            episode_steps.append(len(rew_buf))                      # cantidad de pasos de esta trayectoria
            discounted_returns = compute_returns(rew_buf, gamma)
            batch_obs.extend(obs_buf)
            batch_acts.extend(act_buf)
            batch_returns.extend(discounted_returns)  

        # tensores del batch
        obs = torch.tensor(np.array(batch_obs), dtype=torch.float32)
        acts = torch.tensor(batch_acts, dtype=torch.int64)
        returns = torch.tensor(batch_returns, dtype=torch.float32)

        if use_baseline:
            # ---- forward de valor y ventajas ----
            values = value_net(obs).squeeze(-1)                          # shape (N,)
            rewards_or_advantages = returns - values.detach()            
            if normalize_advantages:
                rewards_or_advantages = (rewards_or_advantages - rewards_or_advantages.mean()) / (rewards_or_advantages.std() + 1e-8)
        else:
            values = torch.zeros_like(returns)                           # solo para logging
            rewards_or_advantages = returns

        # ---- actualizar la politica ----
        policy_loss = compute_loss(policy_net, obs, acts, rewards_or_advantages)
        policy_optimizer.zero_grad() # pone a cero los gradientes de todos los parámetros asociados a ese optimizador.
        policy_loss.backward()       # Calcula los gradientes de la función de pérdida con respecto a los parámetros de la red neuronal.
        policy_optimizer.step()      # actualiza los parametros de la red utilizando los gradientes calculados por loss.backward()
        
        if use_baseline:
            # ---- actualizar la red de valor ----
            value_loss = F.mse_loss(values, returns)
            value_opt.zero_grad()
            value_loss.backward()
            value_opt.step()
        else:
            value_loss = torch.tensor(float("nan"))                          # solo para logging

        policy_losses.append(policy_loss.item())
        value_losses.append(value_loss.item())
        avg_rewards_per_episode.append(total_reward/batch_size)              # promedio por trayectoria

        # -------- EARLY STOP por convergencia --------
        if early_stop_cfg:
            converged = utils.check_convergence(early_stop_cfg["env_name"], early_stop_cfg, rewards_per_trayectory, episode_steps)
            if converged:
                print(f"** Convergencia alcanzada en el episodio {episode+1} **")
                break
            
        if (episode + 1) % 5  == 0:
            print(f"Episodio {episode+1} | Recompensa total: {total_reward} | "
                  f"Policy Loss={policy_losses[-1]:.3f} | ValueF Loss={value_losses[-1]:.3f}")

    env.close()
    return policy_losses, value_losses, avg_rewards_per_episode

