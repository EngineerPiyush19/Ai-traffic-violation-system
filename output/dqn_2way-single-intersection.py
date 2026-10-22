import os
import sys

import gymnasium as gym
from stable_baselines3.dqn.dqn import DQN


if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")
import traci

from sumo_rl import SumoEnvironment


if __name__ == "__main__":
    env = SumoEnvironment(
        net_file="sumo_rl/nets/2way-single-intersection/single-intersection.net.xml",
        route_file="sumo_rl/nets/2way-single-intersection/single-intersection-vhvh.rou.xml",
        out_csv_name="outputs/2way-single-intersection/dqn",
        single_agent=True,
        use_gui=False,
        num_seconds=5000,
    )

    # --- print the numbers the camera script needs ---
    ts = env.traffic_signals[env.ts_ids[0]]
    print("lanes:", ts.lanes)
    print("lane lengths:", [ts.lanes_length[l] for l in ts.lanes])
    print("green phases:", ts.num_green_phases)
    print("obs shape:", env.observation_space.shape)

    model = DQN(
        env=env,
        policy="MlpPolicy",
        learning_rate=0.001,
        learning_starts=0,
        train_freq=1,
        target_update_interval=500,
        exploration_initial_eps=0.05,
        exploration_final_eps=0.01,
        verbose=1,
    )
    model.learn(total_timesteps=100000)

    # --- save the trained model ---
    os.makedirs("outputs/2way-single-intersection", exist_ok=True)
    model.save("outputs/2way-single-intersection/dqn_model")
    print("model saved")