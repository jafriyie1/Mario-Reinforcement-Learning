import gym

env = gym.make("SuperMarioBros-1-1-v0")
obs = env.reset()
done = False
t = 0

while not done:
    action = env.action_space.sample() # This chooses a random action_space
    print(action)
    observation, reward, done, _ = env.step(action) # feedback from environment
    t+=1
    if not t%100:
        print(t)
