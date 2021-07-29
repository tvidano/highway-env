# Environment
import gym
import sys
import timeit
from stable_baselines3 import PPO
import os

if 'highway_env' not in sys.modules:
    try:
        import highway_env
    except ImportError:
        sys.path.append(os.getcwd())
        import highway_env

# ==================================
#        Main script
# ==================================

if __name__ == "__main__":
    env = highway_env.envs.collision_env.CollisionEnv()

    # Recommended Environment Hypertuning Parameters:
    env.configure({
        "duration": 15,  # [s]
        "road_friction": 1.0,
        "stopping_vehicles_count": 5,
        "time_to_intervene": 6, # [s]
        "time_after_collision": 0, # [s]
        "vehicles_density": 2,
        "vehicles_count": 25,
        "control_time_after_avoid": 4, #[s]
        "imminent_collision_distance": 7,  # [m] 
        "reward_type": "penalty_dense"
    })

    # Uncomment to check environment with OpenAi Gym:
    # check_env(env)


    # Batch simulation parameters
    totalruns = 100  # number of runs, obviously
    render_env = True  # whether to render the car
    report_every = 10  # how often to report running progress Ex. every 5th run
    do_training = False # whether to train a new model or use a saved one
    model_name = 'PPO' # choose from:  'baseline' = deterministic hard braking, no steering always
                                        #   'PPO' = implements trained PPO if available, otherwise trains a PPO
                                        #   'A2C' = implements trained A2C if available, otherwise trains an A2C
    debug = False # runs only 1 episode and plots outputs on baseline policy

    model_path = model_name.lower() + "_collision"
    modifier = '2'
    model_path += modifier

    reward_stats = []
    num_mitigated = 0
    num_crashed = 0
    num_no_interaction = 0
    num_offroad = 0

    model = PPO("MlpPolicy", env, learning_rate=0.003, n_steps=2048, batch_size=64, n_epochs=20, verbose=2)
    if do_training:
        print("Training " + model_path)
        start = timeit.default_timer()
        model.learn(total_timesteps=50000, )
        model.save(model_path)
        stop = timeit.default_timer()
        print("Training took", stop - start, "seconds.")
    model.load(model_path)
    print(f'Loaded {model_name} from {os.path.join(os.getcwd(), model_path + ".zip")}\n')

