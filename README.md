**Notes about Reinforcement Learning**

**Definition**

Reinforcement Learning is a type of machine learning where an agent learns to make decisions by taking actions in an environment and receiving rewards or penalties, gradually improving its choices through trial and error to maximize total reward

**Components**
1. Agent
2. Environment
3. State
4. Action
5. Reward

**Parameters**
1. Alpha    :  [Learning rate] Agent uses this factor to make adjustments in Q-Table
2. Gamma    :  [Reward Fulfillment] How much the agent cares about short term reward or wait for a long term reward
3. Epsilon  : [Exploit vs Explore] Probability that decides if the agent should explore a new decision or stick to known safe decision
4. Q-Table  : [State Table] The table which maps different states, actions, rewards, etc. It guides the agent to maximize reward on the current decision in such a way that the reward is maximized on the next decision. Greedy Approach.

**Use Cases**
1. Robot: Navigating Maze
2. Healthcare: Patient responses to new medications
3. NN: Searching for new activation functions





