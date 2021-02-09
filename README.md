# Car Racing Problem

We are trying to solve the CarRacing-v0 problem, a continuous control task to learn from pixels, a top-down racing environment : 
- State consists of 96x96 pixels (RGB buffer).
- Reward is -0.1 every frame and +1000/N for every track tile visited (N = nbr of tiles).
- Episode finishes when all tiles are visited.

What make this problem challenging for Q-Learning Algorithms is that actions are continuous instead of being discrete.
That is, instead of using two discrete actions like -1 or +1, we have to select from infinite actions ranging from -1 to +1.

## Deep Q-Network Approach (DQL-CarRacing)

<b>This approach requires a discrete action space</b>, For that reason,
we created a custom environment that wraps the CarRacing-v0 and instead of taking a continues action state,
it takes a discrete one, composed of 1000 possible actions.

As an agent takes actions and moves through an environment, it learns to map the observed state of the environment to an action. <br>
An agent will choose an action in a given state based on a "Q-value", which is a weighted reward based on the expected highest long-term reward.

## Deep Deterministic Policy Gradient Approach (DDPG-CarRacing)

<b>This approach can operate over continuous action spaces</b>, as it is based on DPG, and uses Experience Replay and slow-learning target networks from DQN.

As an agent takes actions and moves through an environment, it learns to map the observed state of the environment to two possible outputs:
- Actor: This takes as input the state of our environment and returns a probability value for each action in its action space.
- Critic: This takes as input the state of our environment and returns an estimate of total rewards in the future.

