import pytest

from environment import SmartHomeEnv
from interface import HomeAction, HomeObservation
from policy import choose_action


def test_invalid_action_code_raises_validation_error():
    env = SmartHomeEnv(difficulty="easy", seed=42)
    env.reset()
    with pytest.raises(ValueError):
        env.step(HomeAction(action_code=99))


def test_easy_mode_has_constant_price_and_human_home():
    env = SmartHomeEnv(difficulty="easy", seed=1)
    obs = env.reset()

    for _ in range(10):
        obs, _ = env.step(HomeAction(action_code=0))
        assert obs.energy_price == 0.1
        assert obs.is_human_home is True


def test_hard_mode_price_changes_over_time():
    env = SmartHomeEnv(difficulty="hard", seed=42)
    obs = env.reset()
    prices = [obs.energy_price]

    for _ in range(8):
        obs, _ = env.step(HomeAction(action_code=0))
        prices.append(obs.energy_price)

    assert len(set(prices)) > 1


def test_reward_penalizes_energy_use_when_discomfort_matches():
    env_stay = SmartHomeEnv(difficulty="easy", seed=10)
    env_cool = SmartHomeEnv(difficulty="easy", seed=10)
    env_stay.reset()
    env_cool.reset()

    # Keep both states equally uncomfortable so only energy usage differs.
    env_stay.temp = 30.0
    env_stay.human_home = True
    env_cool.temp = 30.0
    env_cool.human_home = True

    _, reward_stay = env_stay.step(HomeAction(action_code=0))
    _, reward_cool = env_cool.step(HomeAction(action_code=1))

    assert reward_stay > reward_cool


def test_seeded_env_is_deterministic_for_same_actions():
    env_a = SmartHomeEnv(difficulty="medium", seed=123)
    env_b = SmartHomeEnv(difficulty="medium", seed=123)

    obs_a = env_a.reset()
    obs_b = env_b.reset()
    assert obs_a == obs_b

    actions = [0, 1, 0, 2, 0, 0, 1]
    for action_code in actions:
        obs_a, reward_a = env_a.step(HomeAction(action_code=action_code))
        obs_b, reward_b = env_b.step(HomeAction(action_code=action_code))
        assert obs_a == obs_b
        assert reward_a == reward_b


def test_policy_thresholds_home_and_away():
    hot_home = HomeObservation(current_temp=24.0, is_human_home=True, energy_price=0.2)
    mild_away = HomeObservation(current_temp=24.0, is_human_home=False, energy_price=0.2)
    cold_away = HomeObservation(current_temp=17.0, is_human_home=False, energy_price=0.2)

    assert choose_action(hot_home) == 1
    assert choose_action(mild_away) == 0
    assert choose_action(cold_away) == 2
