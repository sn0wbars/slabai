import gym
import numpy as np
import mlp
name = "FrozenLake-v0"
env = gym.make(name)
discount = 0.99
e = 0.5
max_ep = 5000
l_rate = 0.2

n = mlp.network((2,20,1), l_rate)

def prediction():
    posible_act = 0
    q = []
    while(env.action_space.contains(posible_act)):
        q.append(n.feedforward((pos, posible_act)))
        posible_act += 1
    q_max = max(q)
    return (q_max, q.index(q_max))

# I use 2-layer network as q-function in q-learning algorithm
# Q(st,a) = r_imm + gamma*[ max(a{t+1}) Q(s{t+1}, a{t+1})]
for episode in range(max_ep):
    pos = env.reset()
    action = prediction()[1]
    q_max = prediction()[0]
    for t in range(300):
        #env.render()
        new_pos, reward, done, info = env.step(action)

        output = reward + discount*q_max - t*0.01
        state = (pos, action)
        n.train((state, output))
        
        q_max, action = prediction()
        if (np.random.rand(1)) < e: 
            action = env.action_space.sample()
        pos = new_pos
        if done:
            e = 1./((episode/50) + 2)
            if (episode > 45000):
                print("Episode {} finished after {} timesteps, reward = {}".format(episode, t+1, reward))
            break