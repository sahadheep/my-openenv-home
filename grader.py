from tasks import get_easy_task, get_medium_task, get_hard_task
from interface import HomeAction
from policy import choose_action


def grade_environment(task_func):
    env = task_func()
    total_reward = 0.0
    obs = env.reset()

    # Simulate 24 hours (24 steps)
    for _ in range(24):
        action_code = choose_action(obs)
        obs, reward = env.step(HomeAction(action_code=action_code))
        total_reward += reward

    # Normalize score between 0.0 and 1.0
    # Expected reward band is roughly [-1.5, 1.0] per step.
    min_reward, max_reward = -1.5, 1.0
    normalized = (total_reward / 24.0 - min_reward) / (max_reward - min_reward)
    final_score = max(0.0, min(1.0, normalized))
    return final_score


if __name__ == "__main__":
    print(f"Easy Task Score: {grade_environment(get_easy_task):.3f}")
    print(f"Medium Task Score: {grade_environment(get_medium_task):.3f}")
    print(f"Hard Task Score: {grade_environment(get_hard_task):.3f}")