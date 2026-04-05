from environment import SmartHomeEnv


def get_easy_task(seed=42) -> SmartHomeEnv:
    return SmartHomeEnv(difficulty="easy", seed=seed)


def get_medium_task(seed=42) -> SmartHomeEnv:
    return SmartHomeEnv(difficulty="medium", seed=seed)


def get_hard_task(seed=42) -> SmartHomeEnv:
    return SmartHomeEnv(difficulty="hard", seed=seed)