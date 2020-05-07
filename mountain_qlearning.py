'''
Simple Q-Learning

'''
import gym
import numpy as np

env = gym.make("MountainCar-v0")

#alpha = 1 fast learning rate & > 1 Slow lrearning rate (values 1-0)
_learning_rate = 0.1
_discount = 0.95
_episodes = 1000
_display_episodes = 1

_discrete_os_size = [20] * len(env.observation_space.high)
discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / _discrete_os_size

epsilon = 0.5
_Start_epsilon_decaying = 1
_end_epsilon_decaying = _episodes // 2

epsilon_decay_value = epsilon / (_end_epsilon_decaying - _Start_epsilon_decaying )
q_table = np.random.uniform(low= -2, high=0, size = (_discrete_os_size +  [env.action_space.n]))

def get_discrete_state(state):
    discrete_state = ( state - env.observation_space.low) / discrete_os_win_size
    return tuple(discrete_state.astype(np.int))

for episode in range(_episodes):
    if episode % _display_episodes == 0:
        render = True
    else:
         render = False
    
    discrete_state = get_discrete_state(env.reset())
    done = False
    
    while not done:        
        if np.random.random() > epsilon:
            action = np.argmax(q_table[discrete_state])            
        else:
            action = np.random.randint(0, env.action_space.n)
            
        new_state, reward, done, _ = env.step(action)
        new_discrete_state = get_discrete_state(new_state)
        print("EPISODE:", episode, "| ACTION:", action , "| REWARD:", reward)
        
        if render:
            env.render()
        if not done:
            max_future_q = np.max(q_table[new_discrete_state])
            current_q = q_table[discrete_state + (action, )]
            new_q = (1 - _learning_rate) * current_q + _learning_rate* (reward + _discount * max_future_q)
            q_table[discrete_state + (action, )] = new_q
            
        elif new_state[0] >= env.goal_position:
            print("Sucessfully Completed on: ",episode)
            q_table[discrete_state +  (action,)] = 0
            
        discrete_state = new_discrete_state  
        
    if _end_epsilon_decaying >= episode >= _Start_epsilon_decaying:
        epsilon -= epsilon_decay_value        
env.close()    
