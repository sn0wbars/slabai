import sys, tty, termios, gym
fd = sys.stdin.fileno()

name = "FrozenLake-v0"
env = gym.make(name)
observation = env.reset()
for t in range(3000):
    env.render()
    old_settings = termios.tcgetattr(fd)
    tty.setraw(sys.stdin.fileno())
    char = sys.stdin.read(1)
    if char == 'a':
        action = 0
    elif char == "s":
        action = 1
    elif char == "d":
        action = 2
    elif char == "w":
        action = 3
    else:
        break
    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    observation, reward, done, info = env.step(action)
    if done:
        print("Episode finished after {} timesteps".format(t+1))
        break

termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
