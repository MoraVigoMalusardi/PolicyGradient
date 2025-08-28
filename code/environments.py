import numpy as np
import gymnasium as gym

"""
En gymnasium, nuestros ambientes custom va a heredar de gymnasium.Env, 
que marca la estructura que todos los entornos tienen que seguir. 
Una de las cosas obligatorias es definir los observation_space y action_space, 
que básicamente declaran qué entradas (acciones) y salidas (observaciones) 
son válidas para este entorno, como vimos anteriormente.
"""

class TwoAZeroObsOneStepEnv(gym.Env):
    """
    Observación: constante 0 
    Duración: un solo paso 
    Acciones: dos posibles 
    Recompensa: +1 o -1, depende únicamente de la acción tomada, hay una acción “buena” 
    y una “mala” 

    """
    def __init__(self):
        super().__init__()
        self.observation_space = gym.spaces.Box(low=0.0, high=0.0, shape=(1,), dtype=int) # Aunque la observación es constante, devolver un vector np.array([0]) es más estándar si después vamos a usar redes neuronales, porque muchas esperan vectores como entrada.
        """
        - low (Union[SupportsFloat, np.ndarray]): Lower bounds of the intervals.
        - high (Union[SupportsFloat, np.ndarray]): Upper bounds of the intervals.
        - shape (Optional[Sequence[int]]): This only needs to be specified if both low and high are scalars and determines the shape of the space. Otherwise, the shape is inferred from the shape of low or high.
        - dtype: The dtype of the elements of the space. If this is an integer type, the Box is essentially a discrete space.
        """
        self.action_space = gym.spaces.Discrete(2) # Dos acciones posibles: 0 y 1
        self.rewards = {0: -1, 1: 1}

    # Internal function for current observation.
    def _get_obs(self):
      """Convert internal state to observation format.

      Returns
        np array: current observation. Always [0]
      """
      return np.array([0], dtype=np.float32)
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        return self._get_obs(), {} # el diccionario es como la info que ponen ellos en el jupiter. Lo tengo que dejar aunq no lo use?
    
    def step(self, action):
        assert self.action_space.contains(action), "Acción inválida"
        
        reward = self.rewards[action]
        terminated = True

        return self._get_obs(), reward, terminated, False, {} 
    
    def close(self):
        pass


    
class TwoARandomObsOneStepEnv(gym.Env):
    """
    Observación: puede empezar en S1 o S2 aleatoriamente
    Duración: un solo paso 
    Acciones: dos posibles 
    Recompensa: si arranca en el estado 0, la acción “buena” es 1, si arranca en el estado 1, la acción “buena” es 0
    """
    def __init__(self):
        super().__init__()
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=int) 
        self.action_space = gym.spaces.Discrete(2) # Dos acciones posibles: 0 y 1
        self.rewards = {0:-1, 1:1}

    # Internal function for current observation.
    def _get_obs(self):
        """
        Convert internal state to observation format.

        Returns:
            np.ndarray: current observation 
        """
        return np.array(np.eye(2)[self.state], dtype=np.float32) 
      
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.random.choice([0, 1]) # Estado inicial aleatorio entre 0 y 1
        return self._get_obs(), {} # el diccionario es como la info que ponen ellos en el jupiter. Lo tengo que dejar aunq no lo use?
    
    def step(self, action):
        assert self.action_space.contains(action), "Acción inválida"
        reward = self.rewards[(self.state+action)%2]
        terminated = True
        return self._get_obs(), reward, terminated, False, {}

    def close(self):
        pass

    
class LineWorldEasyEnv(gym.Env):
    """
    Observacion: Pasillo lineal de 6 casilleros (0, 1, 2, 3, 4, 5) 
                 La observación es la posición actual [1–6] 
    Acciones: dos posibles (moverse a la izquierda o a la derecha) 
    Recompensa: Recompensa de +1 al alcanzar el objetivo en el extremo derecho; 0 en todos los otros casos 
    Posición inicial en el extremo izquierdo 
    El episodio termina al alcanzar el objetivo
    """
    def __init__(self):
        super().__init__()
        self.observation_space = gym.spaces.Box(low=0, high=5, shape=(6,), dtype=np.float32)  # Posiciones posibles: 0, 1, 2, 3, 4, 5
        self.action_space = gym.spaces.Discrete(2) # Dos acciones posibles: 0 (izquierda) y 1 (derecha)
        self.current_step = 0
        self.state = 0 # Posicion inicial en el extremo izquierdo
        self.rewards = [0, 1]

    # Internal function for current observation.
    def _get_obs(self):
      """Convert internal state to observation format.

      Returns:
        array: current observation, one-hot encoded position
      """
      return np.array(np.eye(2)[self.state], dtype=np.float32)
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0 # Reiniciamos a step 0 
        self.state = 0 # volvemos a posicion inicial en el extremo izquierdo
        return self._get_obs(), {} # el diccionario es como la info que ponen ellos en el jupiter. Lo tengo que dejar aunq no lo use?
    
    def step(self, action):
        assert self.action_space.contains(action), "Acción inválida"
    
        # Actualizamos la posición según la acción tomada
        if action == 0 and self.state > 0: # Moverse a la izquierda
            self.state -= 1
        elif action == 1 and self.state < 5: # Moverse a la derecha
            self.state += 1
        
        # Definimos la recompensa según la posición actual
        if self.state == 5:
            reward = self.rewards[1]
            terminated = True
        else:
            reward = self.rewards[0]
            terminated = False

        self.current_step += 1
        return self._get_obs(), reward, terminated, False, {}

    def close(self):
        pass
        

class LineWorldMirrorEnv(gym.Env):
    """
    Pasillo de 4 estados: tres intermedios más el objetivo 

    Observacion: posicion actual [0-3]

    Acciones: las acciones posibles son “izquierda” (0) y “derecha” (1). En el segundo estado (1) las acciones 
              están invertidas (derecha lleva a la izquierda y viceversa).

    Recompensa: -1 por cada paso

    Duracion: el episodio termina al llegar al objetivo (estado 3)
    """
    def __init__(self):
        super().__init__()
        self.observation_space = gym.spaces.Box(low=0, high=3, shape=(4,), dtype=np.float32)  # Posiciones posibles: 0, 1, 2, 3
        self.action_space = gym.spaces.Discrete(2) # Dos acciones posibles: 0 (izquierda) y 1 (derecha)
        self.current_step = 0
        self.state = 0 # posicion inicial en el extremo izquierdo

    # Internal function for current observation.
    def _get_obs(self):
      """Convert internal state to observation format.

      Returns:
        array: current position as a np array [1,0,0,0], [0,1,0,0], ..., [0,0,0,1]
      """
      return np.array(np.eye(2)[self.state], dtype=np.float32)
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0 # Reiniciamos a step 0 
        self.state = 0
        return self._get_obs(), {} 
    
    def step(self, action):
        assert self.action_space.contains(action), "Acción inválida"
    
        if self.state == 1: # En el estado 1 las acciones están invertidas
            action = (action + 1) % 2 # (0+1)%2=1 y (1+1)%2=0

        if action == 0 and self.state > 0: # Moverse a la izquierda
            self.state -= 1
        elif action == 1 and self.state < 3: # Moverse a la derecha
            self.state += 1
        
        reward = -1 # Recompensa de -1 por cada paso
        self.current_step += 1
        terminated = (self.state == 3)
        return self._get_obs(), reward, terminated, False, {}
    
    def close(self):
        pass