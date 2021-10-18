"""
# Proximal Policy Optimization - PPO in PyTorch

This is a minimalistic implementation of [Proximal Policy Optimization - PPO](https://arxiv.org/abs/1707.06347)
 clipped version for Atari Breakout game on OpenAI Gym.
This has less than 250 lines of code.
It runs the game environments on multiple processes to sample efficiently.
Advantages are calculated using [Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438).

**The code for this tutorial is available at
[Github labml/rl_samples](https://github.com/lab-ml/rl_samples).**
And the web version of the tutorial is available
[on my blog](http://blog.varunajayasiri.com/ml/ppo_pytorch.html).

If someone reading this has any questions or comments
 please find me on Twitter, **[@vpj](https://twitter.com/vpj)**.
"""


class Game:
    """
    ## <a name="game-environment"></a>Game environment
    This is a wrapper for OpenAI gym game environment.
    We do a few things here:

    1. Apply the same action on four frames and get the last frame
    2. Convert observation frames to gray and scale it to (84, 84)
    3. Stack four frames of the last four actions
    4. Add episode information (total reward for the entire episode) for monitoring
    5. Restrict an episode to a single life (game has 5 lives, we reset after every single life)

    #### Observation format
    Observation is tensor of size (4, 84, 84). It is four frames
    (images of the game screen) stacked on first axis.
    i.e, each channel is a frame.
    """
