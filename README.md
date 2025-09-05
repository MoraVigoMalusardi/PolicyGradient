## Introduccion

Este repositorio implementa y evalua el algoritmo REINFORCE (gradiente de politica) con y sin baseline aprendido, sobre seis entornos: cuatro custom y dos de Gymnasium. Los objetivos principales fueron:

- Entrenar una política paramétrica $\pi_\theta (a|s)$
- Incorporar una red de valor $V_\phi (s)$ como baseline para reducir la varianza y acelerar la convergencia.
- Definir criterios de convergencia por entorno.
- Estudiar el desempeno del agente entrenado en cada entorno, comparando el caso sin y con baseline.

## Entornos

### Entornos custom

- Todos los ambientes custom va a heredar de gymnasium.Env, que marca la estructura que todos los entornos tienen que seguir.
- Exponen:
    - observation_space
    - action_space
- Implementan:
    - reset → (obs, info): resetea el entorno y devuelve la observacion inicial y un diccionario info (vacio en nuestros casos).
    - step → (obs, reward, terminated, truncated, info): recibe una accion y devuelve: la siguiente observacion, la recompensa, si el episodio termino (terminated o truncated) y el diccionario de info.
- Los estados son arreglos one-hot, con un 1 en la posicion del estado actual.

#### 1. TwoAZeroObsOneStepEnv
En este entorno hay un unico estado.
- Obs: constante [[0]].
- Acciones: 2 (0 o 1).
- Duración: 1 paso.
- Reward: +1 si acción “buena”, −1 si “mala”.

#### 2. TwoARandomObsOneStepEnv
En este entorno hay dos estados.
- Obs: estado incial aleatorio entre [[1,0]] y [[0,1]].
- Acciones: 2 (0 o 1).
- Duración: 1 paso.
- Reward: +1 si acción “buena”, −1 si “mala”. La acción “buena” depende del estado inicial.

#### 3. LineWorldEasyEnv
En este entorno hay 6 estados consecutivos.
- Obs: estado inicial fijo [[1,0,0,0,0,0]].
- Acciones: 2 (0=izquierda, 1=derecha).
- Duración: hasta alcanzar el objetivo. 
- Reward: +1 al alcanzar el objetivo ([[0,0,0,0,0,1]]), 0 en otro caso.

#### 4. LineWorldMirrorEnv
En este entorno hay 4 estados consecutivos.
- Obs: estado inicial fijo [[1,0,0,0]].
- Acciones: 2 (0=izquierda, 1=derecha). En el segundo estado las acciones están invertidas (derecha lleva a la izquierda y viceversa).
- Duración: hasta alcanzar el objetivo ([[0,0,0,1]]).
- Reward: -1 por paso.

