import gym

env = gym.make("SuperMarioBros-1-1-v0")
env.reset()
observation, reward, done, info = env.step(env.action_space.sample())

print("observation.shape = {}".format(observation.shape))
print("reward = {}".format(reward))
print("done = {}".format(done))
print("info = {}".format(info))
print("action_space = {}".format(env.action_space))

print(env.action_space.sample())

'''Info
observation.shape = (224, 256, 3)
reward = 2.4551
done = False
info = {'level': 0, 'distance': 40,
'score': 0, 'coins': 0, 'time': 400,
'player_status': 0, 'life': 3,
'scores': [2.4551, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
'total_reward': 2.4551, 'locked_levels': [False, True, True, True, True, True, True, True,
True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]}
action_space = MultiDiscrete6
'''

'''
MultiDiscrete6 represents the controls
by using array of arrays that represents
min and max value of each control
'''
