What is Reinforcement Learning?

- Autonomous Helicopter
    - How to fly it?
- Reinforcement Learning
    - position of helicopter → how to move control sticks
    - state s → action a
    - very difficult to give the exact right input through supervised learning because its difficult to get a dataset for x for the exact action y
    - reward function rather than the optimal action
        - positive reward: helicopter flying well +1
        - negative reward: helicopter flying poorly (crashing) -1000
- Applications
    - controlling robots
    - factory optimization
    - financial trading
    - playing games

Mars rover example

- Mars rover example
    - rover can be in any of six positions → state
    - day ends → terminal state / nothing more happens
    - robot is in a state s, gets to choose an action a, gets some reward R(s), and as a result it gets to some new state s’

The Return in reinforcement learning

- how do you know if a set of rewards is better or worse than a different set of rewards
- Return
    - the return is the sum of the reward at the steps but weighted by the discount factor

    - common choice of the discount factor is close to one
    - also know as interest rate
- Example of Return
    - return depends on the actions taken

Making decisions: Policies in reinforcement learning

- Policy
    - always go nearer, always go left …
    - create a policy pi function that takes action a at a state
- The goal of reinforcement learning
    - find a policy pi that gives an action (a = pi(s)) to take in every state (s) so as to maximize the return

Review of key concepts

- states
- actions
- rewards
- discount factor
- return
- policy

- Markov Decision Process (MDP)
    - the future only depends on the current state and not on how you got to the state
    - agent gets to choose action a which changes the environment/world then observe the state s and get a reward R

State-action value function definition

- State action value function
    - Q(s, a) = return if you
        - start in state s
        - take action a (once)
        - then behave optimally after that

        - known as q function as well
- Picking actions
    - the best possible return from state s is max Q(s,a)
    - the best possible action in state s is the action a that gives max Q(s,a)
    - Q* → optimal Q function is the state-action value function

Bellman Equation

- computing the state action value function Q(s,a) gives a good action from every state
- s : current state
- R(s) = reward of current state
- a : current action
- s’ : state you get to after taking action a
- a’ : action that you take in state s’

- Explanation of Bellman Equation
    - best possible return from state s’ is maxQ(s’,a’)
    - R(s) is known as the immediate reward

Random (stochastic) environment

- Stochastic Environment
    - chance at another state
- Expected Return
    - interesting not in maximizing return but the average value of the sums of the sum of discounted rewards
    - expected meaning average

Example of continuous state space applications

- Discrete vs Continuous State
    - rover can be anywhere on a line of kilometer → discrete state
    - controlling and truck → continuous state

Lunar lander

- actions
    - do nothing
    - left thruster
    - main thruster
    - right thruster

    - l and r are if the left or right legs are touching the ground
- Reward Function
    - getting to landing pad: 100 -140
    - additional reward for moving toward/away from pad
    - crash: -100
    - soft landing: +100
    - leg grounded: +10
    - fire main engine: -0.3
    - fir side thruster: -0.03
- Lunar Lander Problem
    - learn a policy pi that, given the state, picks action a = pi(s) so as to maximize the return

Learning the state-value function

- key idea is training a NN to compute or to approximate the state action value function Q(s,a), and that in turn will pick good actions
- Deep Reinforcement Learning
    - in a state s, use NN to compute Q(s, nothing), Q(s, left), Q(s, right)
    - in the lander example, eight numbers describe the state
    - four possible actions: nothing, left, main engine, right
    - encode using one-hot feature vector

- Bellman Equation
    - how to come up with the training set for the NN
    - try taking random actions if no policy set
    - (s, a, R(s), s’) tuples
    - s and a are x
    - R(s) + gamma max Q(s’,a’)
    - when you don’t know the Q function, take a random guess
- Learning Alg
    - initialize NN randomly as a guess of Q(s,a)
    - repeat
        - take actions in the lunar lander → get tuples (s, a, R(s), s’)
        - store 10,000 most recent (s, a, R(s), s’) tuples
        - replay buffer → storing the most recent 10,000 tuples
        - train NN
            - create training set of 10,000 examples using
            - x = (s,a) and y = R(s) + gamma max Q(s’,a’)
            - train Qnew such the Qnew(s,a) ~ y
        - set Q = Qnew
        - called DQN alg Deep Q Network

Alg refinement: Improved NN Architecture

- Deep reinforcement learning
    - inefficient to carry out inference for all actions at current state
    - instead more efficient to train a NN to output all four values

    - in state s, input s to NN
    - pick the action a that maximizes Q(s,a)

Alg refinement: Epsilon-Greedy Policy

- Learning Alg
    - don’t know what’s the best actions to take
- How to choose actions while still learning
    - in some state s
    - option 1:
        - pick the action a that maximizes Q(s,a)
    - option 2:
        - with probability 0.95, pick the action a that maximizes Q(s,a) / ‘greedy’, ‘exploitation’
        - with probability 0.05, pick an action a randomly / ‘exploration’
    - if the alg initialized Q(s,a) randomly and the alg thinks that one action is never a good idea, then the NN will never try that action
    - small random probability so the alg can overcome its initial preconceptions
    - epsilon-greedy policy → epsilon = 0.05
    - start epsilon out high and gradually decrease
- more finicky with choice of hyperparameter
- may take more time to learn with wrong choices

Alf refinement: Mini-batch and soft Updates

- How to choose actions while still learning?
    - if there is too much data
    - don’t use all training example per each iteration of the loop
    - pick a subset for each iteration m’
- Mini-batch
    - so an average of averages

    - top to bottom in batches or random
    - not reliably and somewhat noisy but gets the job done faster

- Learning Alg
    - instead of using all 10,000 tuple examples, just use a subset of 1,000, for example
    - set Q = Qnew step then might be unlucky and make an abrupt change
    - soft updates help prevent one unlucky step update of the network
- Soft Update
    - instead of the normal update of Q = Qnew, update the weights and biases as follows

    - a more gradual change
    - less likely for the alg to oscillate

The state of reinforcement learning

- Limitations of RL
    - much easier to get RL working in a simulation
    - far fewer application than supervised and unsupervised learning
    - exciting research direction with potential for future applications