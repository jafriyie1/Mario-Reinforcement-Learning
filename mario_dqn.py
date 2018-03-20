import random
import gym
import numpy as np
import pickle
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Activation, Flatten
from keras.optimizers import Adam
from skimage.color import rgb2gray


EPISODES = 1000

class DQNAgent:
    def __init__(self, state_size,button_map,action_size):
        # This initial
        self.state_size = state_size
        self.action_size = action_size
        self.button_map = button_map
        self.memory = deque()
        ###Parameters
        self.gamma=0.95 # discount rate
        self.epsilon = 1.0 # exploration rate
        self.epsilon_min = 0.001 # minimum exploration rate
        self.epsilon_decay = 0.999
        self.learning_rate = 0.01 # used for gradient descent
        self.model = self._build_model_two()

    def _build_model(self):
        # Neural network for Deep-Q Learning model
        # Will update model with more sophisticated
        # CNN
        model = Sequential()
        model.add(Dense(24, input_shape=self.state_size, activation="relu"))
        model.add(Dense(24, activation="relu"))
        model.add(Dense(self.action_size, activation="linear"))
        model.compile(loss='mse',
                        optimizer=Adam(lr=self.learning_rate))

        return model

    def _build_model_two(self):
        # The convolutional network that will take a frame
        # from the emulator and output a predicted control scheme
        model = Sequential()
        model.add(Conv2D(32, 8, 8, subsample=(4,4),border_mode='same',input_shape=state_size))
        model.add(Activation('relu'))
        model.add(Conv2D(64, 4, 4, subsample=(2,2), border_mode='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(64, 3, 3, subsample=(1,1), border_mode='same'))
        model.add(Activation('relu'))
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dense(6, activation="sigmoid"))

        adam = Adam(lr=self.learning_rate)
        model.compile(loss='categorical_crossentropy',optimizer=adam)
        return model

    def remember(self,state,action,reward,next_state,done):
        # Stores past information into a list for later recall
        # state is the frame, action is the control scheme
        # reward is the distance traveled in the x direction
        # the more distance traveled, the more the reward will be
        # next_state is the next frame
        # done is a boolean on whether or not the game is finished
        # for one episdoe
        self.memory.append((state,action,reward,next_state,done))



    def act(self, state):
        # Tells the agent (Mario character) what action it should take
        # Since we are using a MultiDiscrete (Button Configuration) of size
        # 6, we will want to generate a random array
        # of bits i.e. [0,0,0,1,1,0]
        # MultiDiscrete is used for represented buttons
        if np.random.rand() <= self.epsilon:
            #use for random button input
            move = np.random.randint(2, size=6)
            self.button_map.append(move)
            return move
            '''item = np.random.randint(2, size=6)
            temp = []
            size = len(self.button_map) -1
            for i in range(len(self.button_map)):
                if np.all(item, self.button_map[i]):
                    print("NOT NEW BUTTON CONFIG")
                    t = np.random.randint(len(self.button_map), size=1)
                    return self.button_map[t]
                    break

            self.button_map.append(item)
            return item'''
            #from button
            '''i = np.random.randint(len(self.button_map), size=1)
            print(i)
            return self.button_map[i[0]]'''
        act_value = self.model.predict(state)
        #y_classes = self.model.predict_classes(state)
        print("ACTVALUE")
        print(act_value)
        p = act_value[0]
        print(p)
        t = [p>=0.5]
        t = np.array(t)[0]
        print(t)
        #return t.astype(int)
        return self.button_map[0]

    def replay(self, batch_size):
        # Picks random instance from the memory list and batches
        # over it (trains over the past instance).
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                # We will break up the preceeding code to define
                # our loss function
                target = (reward + self.gamma * \
                        np.max(self.model.predict(next_state)[0]))
                target_f = self.model.predict(state)
                '''print("TARGET F")
                print(target_f.shape)
                print(np.argmax(target_f, axis=1))
                print("ACTION")
                print(action)
                print("TARGET")
                print(target)
                print("SHAPE")
                print(target_f[0][action][1])'''
                target_f[0][action] = target
                self.model.fit(state, target_f, epochs=1, verbose=0)
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

    def load(self,name):
        #loads weights from keras h5 file
        self.model.load_weights(name)

    def load_list(self,name):
        f = open(name ,"rb+")

        self.button_map = pickle.load(f)
        f.close()

    def save(self,name):
        #save weights from keras h5 file
        self.model.save_weights(name)

    def save_list(self,name):
        with open(name, "wb") as f:
            pickle.dump(self.button_map,f)


def preprocess(observation):
    return np.resize(rgb2gray(observation), (84,84))

if __name__ == "__main__":
    # Initialize gym environment
    # Creates the connection between the emulator and the
    # coding environment
    env = gym.make('SuperMarioBros-1-1-v0')

    button_maps = [

            [0,0,0,1,1,1], #12 - right run jump
            [0,0,0,0,0,0],
            [1,0,0,0,0,0], #1 - up only (to climb vine)
            [0,1,0,0,0,0], #2 - left only
            [0,0,1,0,0,0], #3 - down only (duck, down pipe)
            [0,0,0,1,0,0], #4 - right only
            [0,0,0,0,1,0], #5 - run only
            [0,0,0,0,0,1], #6 - jump only
            [0,1,0,0,1,0], #7 - left run
            [0,1,0,0,0,1], #8 - left jump
            [0,0,0,1,1,0], #9 - right run
            [0,0,0,1,0,1], #10 - right jump
            [0,1,0,0,1,1], #11 - left run jump
            [0,0,1,0,0,1], #13 - down jump

    ]

    button_maps2 = [
        [0,0,0,0,0,0], #0 - no button
    ]

    state_size = (84,84,1)
    action_size = env.action_space.shape
    # create agent
    agent = DQNAgent(state_size, button_maps,action_size)
    agent.load("mario_dqn_temp4.h5")
    #agent.load_list("button_maps.pkl")

    print(type(state_size))
    print(state_size)
    print(action_size)

    done = False
    batch_size = 64
    state = env.reset()
    state = preprocess(state)
    state = np.reshape(state, state_size)
    state = np.expand_dims(state,axis=0)
    #state = np.expand_dims(state,axis=0)


    # Iterate the game
    for e in range(EPISODES):
        print(e)
        print("New episode")

        total_score = 0
        # time_t represents each frame of the game

        while not done:
            #env.render()
            #print(type(state))
            # Decide action
            action = agent.act(state)
            #print((action))

            # Advance the game to the next frame
            next_state, reward, done, info = env.step(action)

            total_score = info["total_reward"]
            #print(next_state.shape)
            reward = reward if not done else -10
            next_state = preprocess(next_state)
            next_state = np.reshape(next_state, state_size)
            next_state = np.expand_dims(next_state,axis=0)
            #print(next_state.shape)
            #next_state = np.expand_dims(state,axis=0)
            #print(next_state.shape)
            # Remember the previous state, action, reward, and done
            agent.remember(state,action,reward,next_state,done)

            # make next_state the current state
            state = next_state
            #print(done)

            # done becomes True when the game is finished
            if done:
                #print("It's done!")
                done = False
                with open("performance5.txt", "a") as text_file:
                    print("episode: {}/{}, score: {} e: {:.2}"
                        .format(e,EPISODES,total_score,agent.epsilon),
                        file=text_file)
                break

        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
            print("finished")

        if e % 10 == 0:
            agent.save("mario_dqn_temp5.h5")
            agent.save_list("button_maps3.pkl")
