.. _quickstart:

===============
Getting Started
===============

Making an environment
------------------------

Here is a quick example of how to create an environment:

.. jupyter-execute::

  import gym
  import highway_env
  from matplotlib import pyplot as plt
  %matplotlib inline

  env = gym.make('highway-v0')
  env.reset()
  for _ in range(3):
      action = env.action_type.actions_indexes["IDLE"]
      obs, reward, done, truncated, info = env.step(action)
      env.render()

  plt.imshow(env.render(mode="rgb_array"))
  plt.show()

All the environments
~~~~~~~~~~~~~~~~~~~~
Here is the list of all the environments available and their descriptions:

.. toctree::
  :maxdepth: 1

  environments/highway
  environments/merge
  environments/roundabout
  environments/parking
  environments/intersection
  environments/racetrack

.. _configuration:

Configuring an environment
---------------------------

The :ref:`observations <observations>`, :ref:`actions <actions>`, :ref:`dynamics <dynamics>` and :ref:`rewards <rewards>`
of an environment are parametrized by a configuration, defined as a
:py:attr:`~highway_env.envs.common.abstract.AbstractEnv.config` dictionary.
After environment creation, the configuration can be accessed using the
:py:attr:`~highway_env.envs.common.abstract.AbstractEnv.config` attribute.

.. jupyter-execute::

  import pprint

  env = gym.make("highway-v0")
  pprint.pprint(env.config)

For example, the number of lanes can be changed with:

.. jupyter-execute::

  env.config["lanes_count"] = 2
  env.reset()
  plt.imshow(env.render(mode="rgb_array"))
  plt.show()

.. note::

    The environment must be :py:meth:`~highway_env.envs.common.abstract.AbstractEnv.reset` for the change of configuration
    to be effective.


Training an agent
-------------------

Reinforcement Learning agents can be trained using libraries such as `eleurent/rl-agents <https://github.com/eleurent/rl-agents>`_,
`openai/baselines <https://github.com/openai/baselines>`_ or `Stable Baselines3 <https://github.com/DLR-RM/stable-baselines3>`_.

Here is an example of SB3's DQN implementation trained on ``highway-fast-v0`` with its default kinematics observation and an MLP model.

.. |highway_dqn|  image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/github/eleurent/highway-env/blob/master/scripts/sb3_highway_dqn.ipynb

|highway_dqn|

.. code-block:: python

  import gym
  import highway_env
  from stable_baselines3 import DQN

  env = gym.make("highway-fast-v0")
  model = DQN('MlpPolicy', env,
                policy_kwargs=dict(net_arch=[256, 256]),
                learning_rate=5e-4,
                buffer_size=15000,
                learning_starts=200,
                batch_size=32,
                gamma=0.8,
                train_freq=1,
                gradient_steps=1,
                target_update_interval=50,
                verbose=1,
                tensorboard_log="highway_dqn/")
  model.learn(int(2e4))
  model.save("highway_dqn/model")

  # Load and test saved model
  model = DQN.load("highway_dqn/model")
  while True:
    done = truncated = False
    obs, info = env.reset()
    while not (done or truncated):
      action, _states = model.predict(obs, deterministic=True)
      obs, reward, done, truncated, info = env.step(action)
      env.render()

A full run takes about 25mn on my laptop (fps=14). The following results are obtained:

.. figure:: https://raw.githubusercontent.com/eleurent/highway-env/gh-media/docs/media/highway_fast_dqn.png

   Training curves, for 5 random seeds.

.. figure:: https://raw.githubusercontent.com/eleurent/highway-env/gh-media/docs/media/highway_fast_dqn.gif

   Video of an episode run with the trained policy.

.. note::

    There are several ways to get better performances. For instance, `SB3 provides only vanilla Deep Q-Learning and has no extensions such as Double-DQN, Dueling-DQN and Prioritized Experience Replay <https://stable-baselines3.readthedocs.io/en/master/modules/dqn.html#notes>`_.
    However, `eleurent/rl-agents <https://github.com/eleurent/rl-agents>`_'s implementation of DQN does provide those extensions, which yields better results. Improvements can also be obtained by changing the observation type or the model, see the :ref:`FAQ <faq>`.


Examples on Google Colab
-------------------------

Several scripts and notebooks to train driving policies on `highway-env` are available `on this page <https://github.com/eleurent/highway-env/tree/master/scripts>`_.
Here are a few of them:

.. |highway_dqn_cnn|  image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/github/eleurent/highway-env/blob/master/scripts/sb3_highway_dqn_cnn.ipynb
.. |planning_hw|  image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/github/eleurent/highway-env/blob/master/scripts/highway_planning.ipynb
.. |parking_mb|  image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/github/eleurent/highway-env/blob/master/scripts/parking_model_based.ipynb
.. |parking_her|  image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/github/eleurent/highway-env/blob/master/scripts/parking_her.ipynb
.. |dqn_social|  image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/github/eleurent/highway-env/blob/master/scripts/intersection_social_dqn.ipynb

- | Highway with image observations and a CNN model |highway_dqn_cnn|
  | Train SB3's DQN on `highway-fast-v0` , but using :ref:`image observations <Grayscale Image>` and a CNN model for the value function.
- | Trajectory Planning on Highway |planning_hw|
  | Plan a trajectory on `highway-v0` using the `OPD` :cite:`Hren2008` implementation from `eleurent/rl-agents <https://github.com/eleurent/rl-agents>`_.
- | A Model-based Reinforcement Learning tutorial on Parking |parking_mb|
  | A tutorial written for `RLSS 2019 <https://rlss.inria.fr/>`_ and demonstrating the principle of model-based reinforcement learning on the `parking-v0` task.
- | Parking with Hindsight Experience Replay |parking_her|
  | Train a goal-conditioned `parking-v0` policy using the `HER` :cite:`Andrychowicz2017` implementation from `stable-baselines <https://github.com/hill-a/stable-baselines>`_.
- | Intersection with DQN and social attention |dqn_social|
  | Train an `intersection-v0` crossing policy using the social attention architecture :cite:`Leurent2019social` and the DQN implementation from `eleurent/rl-agents <https://github.com/eleurent/rl-agents>`_.