# Mario-Reinforcement-Learning
A reinforcement learning project to create an agent that can play Super Mario Bros.

Hello! This is a project where I am using the Deep Q Learning Algorithm [1] to train an agent to Super Mario Bros. 
I was only able to start this project by looking at the following materials: 

[1] https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf

Setting up Mario Gym Environment: https://becominghuman.ai/getting-mario-back-into-the-gym-setting-up-super-mario-bros-in-openais-gym-8e39a96c1e41

General Explanation on Deep Q Learning Algorithm: https://keon.io/deep-q-learning/

As of right now the project is not finished, there is still a lot for me to do. I will continue to update this repo as I work on this project.

###################################


User’s Manual Intro Section 
	
In order to use the Mario RL program, one must use a computer that has a MacOS operating system. The next order of business is to install the FCEUX emulator. This can be done by using the Homebrew package system. If one does not have Homebrew installed please run the following in a terminal window: 
	
	/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
	
Once Homebrew is installed, FCEUX can now easily be installed. Please use the following command: 

	brew install fceux

Now that FCEUX is installed, installing the next packages will be more streamlined. 

User’s Manual Continued
	In order to gain access to the program, one should clone the repository for the program from Github. Go to the link below, click on the clone or download button and then choose the “Download ZIP” option. You can also clone the repo as well. 
	

After the folder has been unzipped, use the “cd” command in the terminal to point to the unzipped folder. After this is done, run the following command to install the necessary packages: 
	
	pip install tensorflow
	pip install -r requirements.txt
	
	
We are almost done. Now follow this article that will install the Super Mario Bros ROM so that it can be interfaced with the program: 
	
	https://becominghuman.ai/getting-mario-back-into-the-gym-setting-up-super-mario-bros-in-openais-gym-8e39a96c1e41

Once all of that is done you can now run the code! Run the following command and you can now see the program in action (Note: the program does not take any human input, but one can view how well Mario is performing). 
	
	Python mario_rl.py 
