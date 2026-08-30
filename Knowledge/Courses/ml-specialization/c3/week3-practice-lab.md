### Deep Q-Learning - Lunar Lander

Import Packages

- numpy
- deque → data structure for memory buffer
- namedtuple → stores the experience tuples
- gym → collection of environments that can be used to test RL algs
- PIL.Image and pyvirtualdisplay → render the Lunar Lander environment
- Keras
- utils → module that contains helper functions

    ```python
    import time
    from collections import deque, namedtuple

    import gym
    import numpy as np
    import PIL.Image
    import tensorflow as tf
    import utils

    from pyvirtualdisplay import Display
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import Dense, Input
    from tensorflow.keras.losses import MSE
    from tensorflow.keras.optimizers import Adam
    ```

    ```python
    # Set up a virtual display to render the Lunar Lander environment.
    Display(visible=0, size=(840, 480)).start();

    # Set the random seed for TensorFlow
    tf.random.set_seed(utils.SEED)
    ```

Hyperparameters

- set the hyperparameters

    ```python
    MEMORY_SIZE = 100_000     # size of memory buffer
    GAMMA = 0.995             # discount factor
    ALPHA = 1e-3              # learning rate  
    NUM_STEPS_FOR_UPDATE = 4  # perform a learning update every C time steps
    ```

The Lunar Lander Environment

- will be using  [OpenAI's Gym Library](https://gymnasium.farama.org/)
- the gym library provides a wide variety of environments for RL
- an environment represents a problem or task to be solved
- solve the Lunar lander environment using RL
- the goal of the LL environment is to land the lunar lander safely on the landing pad on the surface of the moon
- the landing pad is designated by two flag poles and its center is at coordinates (0,0) but the lander is also allowed to land outside of the landing pad
- the lander starts at the top center of the environment with a random initial force applied to its center of mass and has infinite fuel
- the environment is considered solved when 200 points are achieved
- Action Space
    - the agent has four discrete actions available
        - do nothing
        - fire right rngine
        - fire main engine
        - fire left engine
    - each action has a corresponding numerical value
        - Do nothing = 0
        - fire right engine = 1
        - fire main engine = 2
        - fire left engine = 3
- Observation Space
    - The agent’s observation space consists of a state vector with eight variables
        - its (x,y) coordinates, the landing pad is always at coordinates (0,0)
        - its linear velocities (x_dot, y_dot)
        - its angle theta
        - its angular velocity theta_dot
        - two booleans, l and r, that represent whether each leg is in contact with the ground or not
- Rewards
    - after every step, a reward is granted
    - the total reward of an episode is the sum of the rewards for all the steps within that episode
    - for each step, the reward
        - is increased/decreased the closer/further the lander is to the landing pad
        - is increased/decreased the slower/faster the lander is moving
        - is decreased the more the lander is tilted (angle not horizontal)
        - is increased by 10 points for each leg that is in contact with the ground
        - is decreased by 0.03 points each frame a side engine is firing
        - is decreased by 0.3 points each frame the main engine is firing
    - the episode receives an additional reward of 100 or 100 points for crashing or landing safely respectively
- Episode Termination
    - an episode ends (the environment enters a terminal state) if
        - the lunar lander crasher (the body of the lunar lander come in contact with the surface of the moon)
        - the absolute value of the lander’s x-coordinate is greater than 1 (it goes beyond the left or right border)

Load the Environment

- start by loading the LunarLander-v2 environment from the gym library by using the .make() method
- LunarLander-v2 is the latest version of the Lunar Lander Environment

    ```python
    env = gym.make('LunarLander-v2')
    ```

- once the environment is loaded, use the .reset() method to reset the environment to the initial state
- the lander starts at the top center of the environment and the first frame is rendered using the .render() method

    ```python
    env.reset()
    PIL.Image.fromarray(env.render(mode='rgb_array'))
    ```

- in order to build the NN later on, the size of the state vector and the number of valid actions need to be known
- get this information from the environment by using the .observation_space.shape and action_space.n methods

    ```python
    state_size = env.observation_space.shape
    num_actions = env.action_space.n

    print('State Shape:', state_size)
    print('Number of actions:', num_actions)
    ```

Interacting with the Gym Environment

- the Gym library implements the standard ‘agent-environment loop’ formalism
- in the standard ‘agent-environment loop’ formalism, an agent interacts with the environment in descrete time steps t = 0, 1, 2, …
- at each time step t, the agent uses a policy pi to select an action A_t based on its observation of the environment’s state S_t
- the agent receives a numerical reward Rt and on the next time step, moves to a new state S_(t+1)
- Exploring the Environment’s Synamics
    - in open Ai’s Gym environments, use the .step() method to run a single time step of the environment’s dynamics
    - in the version of gym that is being used, the .step() method accepts and action and return four values
        - observation (object): an environment-specific object representing the observation of the environment, in the lunar lander environment this corresponds to a numpy array containing the positions and velocities of the lander
        - reward (float): amount of reward returned as a result of taking the given action, in the lunar lander environment this corresponds to a float of type numpy.float64
        - done (boolean): when done is True, it indicates the episode has terminated and it’s time to reset the environment
        - info (dictionary): diagnostic information useful for debugging

        ```python
        # Reset the environment and get the initial state.
        current_state = env.reset()
        ```

    - once the environment is reset the agent can start taking actions in the environment by using the .step() method
    - note that the agent can only take one action per time step
    - in the cell below, select different actions and see how the returned values change depending on the action taken
    - remember that in this environment the agent has four discrete actions available
        - do nothing = 0
        - fire right engine = 1
        - fire main engine = 2
        - fire left engine = 3

        ```python
        # Select an action
        action = 0

        # Run a single time step of the environment's dynamics with the given action.
        next_state, reward, done, _ = env.step(action)

        # Display table with values.
        utils.display_table(current_state, action, next_state, reward, done)

        # Replace the `current_state` with the state after the action is taken
        current_state = next_state
        ```

    - when training, the agent is allowed to make many consecutive actions during an episode

Deep Q-Learning

- in cases where both the state and action space are discrete, estimate of the action-value function iteratively by using the Bellman equation

- this iterative method converges to the optimal action-value function Q*(s,a) as i →  inf
- the agent just needs to gradually explore the state-action space and keep updating the estimate of Q(s,a) until it converges to the optimal action-value function Q*(s,a)
- in cases where the state space is continuous, it becomes practically impossible to explore the entire state-action space
- consequently, this also makes it practically impossible to gradually estimate Q(s,a) until it converges to Q*(s,a)
- In the Deep Q-Learning, the problem is solved by using a NN to estimate the action-value function Q(s,a) ~ Q*(s,a)
- this NN is called a Q-Network and can be trained by adjusting its weights at each iteration to minimize the mean-squared error in the Bellman equation
- unfortunately, using NN in reinforcement learning to estimate action_Value functions has proven to be highly unstable
- luckily, there are techniques that can be employed to avoid instabilities
- these consist of using a Target Network and Experience Replay
- Target Network
    - train the Q-Network by adjusting the weights at each iteration to minimize the mean-squared error in the Bellman equation, where the target values are given by

    - where w are the weights of the Q-Network
    - this means to adjust the weights w at each iteration to minimize the following error

    - this forms a problem because the y target is changing on every iteration
    - having a constantly moving target can lead to oscillations and instabilities
    - to avoid this, create a separate NN for generating the y targets
    - this separate NN is called the target Q_hat-Network and it will have the same architecture as the original Q-Network
    - by using the target Q_hat-Network, the above error becomes

    - where w- and w are the weights of the target Q_hat-Network and Q-Network
    - In practice, use the following alg: every C time steps, use the Q_hat-Network to generate the y targets and update the weights of the target Q_hat-Network using the weights of the Q-Network
    - we will update the weights w- of the target Q_hat-Network using a soft update
    - update the weights w- using the following rule:

    - where tao << 1
    - by using the soft update, the target values, y, change slowly, which greatly improves the stability of our learning alg
- Exercise 1
    - create the Q and target Q_hat networks and set the optimizer
    - remember that the Deep Q-Network (DQN) is a NN that approximates the action-value function Q(s,a) ~ Q*(s<a)
    - it does this be learning how to map states to Q values
    - to solve the LL environment, employ a DQN with the following architecture
        - an Input layer that takes state_size a sinput
        - a Dense layer with 64 units and a relu activation function
        - a Dense layer with 64 units and a relu activation function
        - a Dense layer with num)actions units and a linear activation function
    - in create the Q-Network and the target Q_hat-Network using the model architecture above
    - both the Q-Network and the target Q-Hat Network have the same architecture
    - set Adam as the optimizer with a learning rate equal to ALPHA

    ```python
    # UNQ_C1
    # GRADED CELL

    # Create the Q-Network
    q_network = Sequential([
        ### START CODE HERE ### 
        Input(state_size),
        Dense(64, activation='relu'),
        Dense(64, activation='relu'),
        Dense(num_actions)
        ### END CODE HERE ### 
        ])

    # Create the target Q^-Network
    target_q_network = Sequential([
        ### START CODE HERE ### 
        Input(state_size),
        Dense(64, activation='relu'),
        Dense(64, activation='relu'),
        Dense(num_actions)     
        ### END CODE HERE ###
        ])

    ### START CODE HERE ### 
    optimizer = Adam(ALPHA)
    ### END CODE HERE ###
    ```

- Experience Replay
    - when an agent interacts with the environment, the states, actions, and rewards the agent experiences are sequential by nature
    - if the agent tries to learn from these consecutive experiences it can run into problems due to the strong correlations between them
    - to avoid this, employ a technique known as experience replay to generate uncorrelated experiences for training the agent
    - experience replay consists of storing the agent’s experiences (the states, actions, and rewards the agent receives) in a memory buffer and then sampling a random mini-batch of experiences from the buffer to do the learning
    - the experience tuples (S_t, A_t, R_t, S_(t+1)) will be added to the memory buffer at each time step as the agent interacts with the environment
    - store the experiences as named tuples

        ```python
        # Store experiences as named tuples
        experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        ```

    - experience replay avoids the problematic correlations, oscillations and instabilities
    - in addition, experience replay also allows the agent to potentially use the same experience in multiple weight updates, which increases data efficiency

Deep Q-Learning Alg with Experience Replay

- Exercise 2

    - implement line 12 of the alg above
    - compute the loss between the y targets and the Q(s,a) values
    - complete the compute_loss function by setting the y target equal to

    - note
        - the compute_loss function takes in a mini-batch of experience tuples
        - this minibatch of experience tuples is unpacked to extract the states, actions, rewards, next_states and done_vals
        - keep in mind that these variables are TensorFlow Tensors whose size will depend on the mini-batch size
        - for example, if the mini-batch size is 64 then both rewards and done_vals will be TensorFlow Tensors with 53 elements
        - using if/else statements to set the y targets will not work when the variables are tensors with many elements
        - however, notice that the done_vals can be used to implement the above in a single line of code
        - to do this, recall that the done variable is a boolean variable that takes the value True when an episode terminates at step j+1 and it is False otherwise
        - taking into account that a Boolean value of True has the numerical value of 1 and a Boolean value of False has the numerical value of 0, use the factor (1 - done_vals) to implement the above in a single line of code
        - notice that (1 - done_vals) has a value of 0 when done_vals is True and a value of 1 when the done_vals is False
    - compute the loss by calculating the MSE between the y_targets and the q_values
    - use MSE from Keras

        ```python
        # UNQ_C2
        # GRADED FUNCTION: calculate_loss

        def compute_loss(experiences, gamma, q_network, target_q_network):
            """ 
            Calculates the loss.

            Args:
              experiences: (tuple) tuple of ["state", "action", "reward", "next_state", "done"] namedtuples
              gamma: (float) The discount factor.
              q_network: (tf.keras.Sequential) Keras model for predicting the q_values
              target_q_network: (tf.keras.Sequential) Keras model for predicting the targets

            Returns:
              loss: (TensorFlow Tensor(shape=(0,), dtype=int32)) the Mean-Squared Error between
                    the y targets and the Q(s,a) values.
            """

            # Unpack the mini-batch of experience tuples
            states, actions, rewards, next_states, done_vals = experiences

            # Compute max Q^(s,a)
            max_qsa = tf.reduce_max(target_q_network(next_states), axis=-1)

            # Set y = R if episode terminates, otherwise set y = R + γ max Q^(s,a).
            ### START CODE HERE ### 
            y_targets = rewards + (gamma * max_qsa * (1-done_vals))
            ### END CODE HERE ###

            # Get the q_values and reshape to match y_targets
            q_values = q_network(states)
            q_values = tf.gather_nd(q_values, tf.stack([tf.range(q_values.shape[0]),
                                                        tf.cast(actions, tf.int32)], axis=1))

            # Compute the loss
            ### START CODE HERE ### 
            loss = MSE(y_targets, q_values) 
            ### END CODE HERE ### 

            return loss
        ```

Update the Network Weights

- use the agent_learn function to implement lines 12-13 of the alg
- the agent_learn function will update the weights of the Q and target Q_hat networks using ta custom training loop
- need to retrieve the gradients via a tf.GradientTape instance
- then call optimizer.apply_gradients() to update the weights of the Q-Network
- also using the @tf.function decorator to increase performance
- without this decorator, the training will take twice as long
- documentation: [TensorFlow documentation](https://www.tensorflow.org/guide/function)
- the last line of the function updates the weights of the target Q_hat Network using a [soft update](https://dgcezzlgcmyt.labs.coursera.org/notebooks/C3_W3_A1_Assignment.ipynb#6.1)

    ```python
    @tf.function
    def agent_learn(experiences, gamma):
        """
        Updates the weights of the Q networks.

        Args:
          experiences: (tuple) tuple of ["state", "action", "reward", "next_state", "done"] namedtuples
          gamma: (float) The discount factor.

        """

        # Calculate the loss
        with tf.GradientTape() as tape:
            loss = compute_loss(experiences, gamma, q_network, target_q_network)

        # Get the gradients of the loss with respect to the weights.
        gradients = tape.gradient(loss, q_network.trainable_variables)

        # Update the weights of the q_network.
        optimizer.apply_gradients(zip(gradients, q_network.trainable_variables))

        # update the weights of target q_network
        utils.update_target_network(q_network, target_q_network)
    ```

Train the Agent

- now ready to train the agent to solve the LL environment
- implement the alg
    - line 1:
        - initialize the memory_buffer with a capacity of N = MEMORY_SIZE
        - using a deque as the data structure for the memory_buffer
    - line 2:
        - skip this line since q_network is already initialized
    - line 3:
        - initialize the target_q_network by setting its weights to be equal to those of the q_network
    - line 4:
        - start the outer loop
        - notice that M = num_episodes = 2000
        - this number is reasonable because the agent should be able to solve the LL environment in less than 2000 episodes using the default parameters
    - line 5:
        - use the .reset() method to reset the environment to the initial state and get the initial state
    - line 6:
        - start the inner loop
        - set T = max_num_timesteps = 1000
        - this means that the episode will automatically terminate if the episode hasn’t terminated after 1000 time steps
    - line 7:
        - the agent observes the current state and chooses an action using an epsilon-greedy policy
        - the agent starts out using the value of epsilon = 1 which yields an epsilon-greedy poliby that is equivalent to the equiprobable random policy
        - this means that at the beginning of the training the agent is just going to take random actions regardless of the observed state
        - as training progresses, decrease the value of epsilon slowly towards a minimum value using a given epsilon-decay rate
        - we want this minimum value to be close to zero because a value of epsilon = 0 will yield an epsilon-greedy policy that is equivalent to the greedy policy
        - this means that towards the end of training, the agent will lean towards selecting the action that is believes (based on its past experiences) will maximize q(s,a)
        - set the minimum epsilon value to be 0.01 and not exactly 0 because there should always remain a little chance for exploration during training
    - line 8
        - use the .step() method to take the given action in the environment and get the reward and the next_state
    - line 9
        - store the experience(state, action, reward, next_state, done) tuple in the memory_buffer
        - the done variable is also stored to keep track of episode termination
    - line 10
        - check if the conditions are met to perform a learning update
        - do this by using the check_update_conditions function
        - this function checks if C = NUM_STEPS_FOR_UPDATE = 4 times steps have occured and if the memory_buffer has enough experience tuples to fill a mini-batch
        - if the mini-batch size is 64, then the memory_buffer should have more than 64 experience tuples in order to pass the latter condition
        - if the conditions are met, then the check_update_conditions function will return a value of True, otherwise it will return a value of False
    - lines 11 - 14
        - the update variable is True then perform a learning update
        - the learning update consists of sampling a random mini-batch of experience tuples from the memory_buffer, setting the y targets, performing gradient descent, and updating the weights of the networks
        - use the agent_learn function
    - line 15
        - at the end of each iteration of the inner loop, set next_state as the new state so that the loop can start again from the new state
        - in addition, check if the episode has reached a terminal state (if done = True)
        - if a terminal state has been reached, then break out of the inner loop
    - line 16
        - at the end of each iteration of the outer loop we update the value of epsilon, and check if the environment has been solved
        - consider that the environment has been solved if the agent receives an average of 200 points in the last 100 episodes
        - if the environment has not been solved, continue the outer loop and start a new episode
    - note the included extra variables to keep track of the total number of points the agent received in each episode
    - this will help determine if the agent has solved the environment and it will also show how the agent performed during training
    - use the time module to measure how long the training takes

        ```python
        start = time.time()

        num_episodes = 2000
        max_num_timesteps = 1000

        total_point_history = []

        num_p_av = 100    # number of total points to use for averaging
        epsilon = 1.0     # initial ε value for ε-greedy policy

        # Create a memory buffer D with capacity N
        memory_buffer = deque(maxlen=MEMORY_SIZE)

        # Set the target network weights equal to the Q-Network weights
        target_q_network.set_weights(q_network.get_weights())

        for i in range(num_episodes):

            # Reset the environment to the initial state and get the initial state
            state = env.reset()
            total_points = 0

            for t in range(max_num_timesteps):

                # From the current state S choose an action A using an ε-greedy policy
                state_qn = np.expand_dims(state, axis=0)  # state needs to be the right shape for the q_network
                q_values = q_network(state_qn)
                action = utils.get_action(q_values, epsilon)

                # Take action A and receive reward R and the next state S'
                next_state, reward, done, _ = env.step(action)

                # Store experience tuple (S,A,R,S') in the memory buffer.
                # We store the done variable as well for convenience.
                memory_buffer.append(experience(state, action, reward, next_state, done))

                # Only update the network every NUM_STEPS_FOR_UPDATE time steps.
                update = utils.check_update_conditions(t, NUM_STEPS_FOR_UPDATE, memory_buffer)

                if update:
                    # Sample random mini-batch of experience tuples (S,A,R,S') from D
                    experiences = utils.get_experiences(memory_buffer)

                    # Set the y targets, perform a gradient descent step,
                    # and update the network weights.
                    agent_learn(experiences, GAMMA)

                state = next_state.copy()
                total_points += reward

                if done:
                    break

            total_point_history.append(total_points)
            av_latest_points = np.mean(total_point_history[-num_p_av:])

            # Update the ε value
            epsilon = utils.get_new_eps(epsilon)

            print(f"\rEpisode {i+1} | Total point average of the last {num_p_av} episodes: {av_latest_points:.2f}", end="")

            if (i+1) % num_p_av == 0:
                print(f"\rEpisode {i+1} | Total point average of the last {num_p_av} episodes: {av_latest_points:.2f}")

            # We will consider that the environment is solved if we get an
            # average of 200 points in the last 100 episodes.
            if av_latest_points >= 200.0:
                print(f"\n\nEnvironment solved in {i+1} episodes!")
                q_network.save('lunar_lander_model.h5')
                break

        tot_time = time.time() - start

        print(f"\nTotal Runtime: {tot_time:.2f} s ({(tot_time/60):.2f} min)")

        '''
        Episode 100 | Total point average of the last 100 episodes: -132.04
        Episode 200 | Total point average of the last 100 episodes: -89.677
        Episode 300 | Total point average of the last 100 episodes: -45.00
        Episode 400 | Total point average of the last 100 episodes: 124.20
        Episode 455 | Total point average of the last 100 episodes: 200.05

        Environment solved in 455 episodes!

        Total Runtime: 535.67 s (8.93 min)
        '''
        ```

    - plot the total point history along with the moving average to see how the agent improved during training

        ```python
        # Plot the total point history along with the moving average
        utils.plot_history(total_point_history)
        ```

See the Trained Agent In Action

- use the utils.create_video function to create a video of the agent interacting with the environment using the trained Q-Network
- the utils.create_video function uses the imagio library to create the video
- this library produces some warnings that can be distracting

    ```python
    # Suppress warnings from imageio
    import logging
    logging.getLogger().setLevel(logging.ERROR)
    ```

- create a video of the agent interacting with the LL environment using the trained Q_network
- the video is saved to the videos folder with the given filename
- use the utils.embed_mp4 function to embed the video in the Jupyter Notebook with no downloading
- the lunar lander starts with a random initial force applied to the center of mass
- every time the cell is run, a different video will play
- if the agent was trained properly, it should be able to land the lunar lander in the landing pad every time, regardless of initial force applied to its center of mass

    ```python
    filename = "./videos/lunar_lander.mp4"

    utils.create_video(filename, env, q_network)
    utils.embed_mp4(filename)
    ```

References

- Mnih, V., Kavukcuoglu, K., Silver, D. et al. Human-level control through deep reinforcement learning. Nature 518, 529–533 (2015).
- Lillicrap, T. P., Hunt, J. J., Pritzel, A., et al. Continuous Control with Deep Reinforcement Learning. ICLR (2016).
- Mnih, V., Kavukcuoglu, K., Silver, D. et al. Playing Atari with Deep Reinforcement Learning. arXiv e-prints. arXiv:1312.5602 (2013).