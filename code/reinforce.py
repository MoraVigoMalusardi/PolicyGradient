import torch
import numpy as np
from torch import nn
from typing import List
import torch.optim as optim
from environments import TwoAZeroObsOneStepEnv
import torch.nn.functional as F
import matplotlib.pyplot as plt


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
    for episode in range(episodes):
        batch_obs, batch_acts, batch_returns = [], [], []

        # Recolectamos N trayectorias
        for _ in range(batch_size):
            obs_buf, act_buf, rew_buf = [], [], []
            s, _ = env.reset()
            terminated, truncated = False, False
            # Generar una trayectoria
            while not (terminated or truncated):
                obs_buf.append(s)
                s_t = torch.tensor(s, dtype=torch.float32)
                dist = get_policy(policy_net, s_t)  # π_θ(a_t | s_t)
                a = dist.sample()      # a_t ~ π_θ(a_t | s_t)
                s, r, terminated, truncated, _ = env.step(a.item()) # Ejecuta a_t y transiciona de estado
                act_buf.append(a.item())
                rew_buf.append(r)

            batch_obs.extend(obs_buf)
            batch_acts.extend(act_buf)
            batch_returns.extend(compute_returns(rew_buf, gamma))  

        # Preparar tensores
        obs = torch.tensor(batch_obs, dtype=torch.float32)
        acts = torch.tensor(batch_acts, dtype=torch.int64)
        returns = torch.tensor(batch_returns, dtype=torch.float32)

        # Calcular pérdida y actualizar parámetros
        loss = compute_loss(policy_net, obs, acts, returns)
        optimizer.zero_grad() # pone a cero los gradientes de todos los parámetros asociados a ese optimizador.
        loss.backward()       # Calcula los gradientes de la función de pérdida con respecto a los parámetros de la red neuronal.
        optimizer.step()      # actualiza los parametros de la red utilizando los gradientes calculados por loss.backward()

        if (episode + 1) % 5  == 0:
            print(f"Episodio {episode+1}, Recompensa total: {sum(rew_buf)}")

    env.close()

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
        # Agregar otros entornos desps
    }

    # ---------------- ENTRENAMIENTO ----------------
    env = envs[environment_name]()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    gamma = 0.0  # cómo descuenta las recompensas futuras
    lr = 0.01     # learning rate

    # Red de política
    policy_net = build_mlp([state_dim, 16, action_dim])
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)

    train_policy(env, policy_net, optimizer, episodes=episodes, gamma=gamma, batch_size=5)

    print("\nEntrenamiento completado. Probando la política aprendida...\n")

    # --------------------- TEST ---------------------
    # Observación trivial obs, la recibimos como parámetro
    obs = torch.tensor(obs, dtype=torch.float32)
    # Hacemos forward pass para obtener los logits
    logits = policy_net(obs)
    # Softmax para convertir logits en probabilidades
    probs = F.softmax(logits, dim=-1).detach().numpy().flatten()  # shape: (action_dim,)

    plt.bar(range(len(probs)), probs)
    plt.xlabel("Acciones")
    plt.ylabel("Probabilidad")
    plt.title("Probabilidades de cada accion según la política aprendida")
    plt.show()


if __name__ == "__main__":
    # Entrenar y probar en TwoAZeroObsOneStepEnv
    train_and_test_env("TwoAZeroObsOneStep", episodes=150, obs=np.array([0.0]))
