from IA2CC import IA2CC
from environment import MultiAgentGridEnv
from utils import evaluate, load_json


GRID_FILE = 'grid_world2.json'

env = MultiAgentGridEnv(
    grid_file=GRID_FILE,
    coverage_radius=4,
    max_steps_per_episode=150,
    num_agents=4,
)


# NN pararmeters
critic_input_size = env.get_state_size()
actor_input_size = env.get_obs_size()
actor_output_size = env.get_total_actions()

model = IA2CC(actor_input_size=actor_input_size,
              actor_output_size=actor_output_size,
              critic_input_size=critic_input_size,
              num_agents=env.num_agents)


weights = [1, 2]

for weight in weights:
    path = f'rewardweight/config{weight}'

    model.load_actors(f'model/{path}')
    model_stats = load_json(f'model/{path}/model_stats.json')

    evaluate(model, model_stats=model_stats, path=path)
