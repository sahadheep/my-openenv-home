from tasks import get_easy_task
from interface import HomeAction

env = get_easy_task()
obs = env.reset()
print(f"Starting Temp: {obs.current_temp}")

# Take a fake step (Turn on AC)
new_obs, reward = env.step(HomeAction(action_code=1))
print(f"New Temp: {new_obs.current_temp}, Reward: {reward}")

# Lightweight sanity checks
assert isinstance(new_obs.current_temp, float)
assert isinstance(new_obs.is_human_home, bool)
assert isinstance(new_obs.energy_price, float)