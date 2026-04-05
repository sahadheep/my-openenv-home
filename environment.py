import random
from interface import HomeObservation, HomeAction


class SmartHomeEnv:
    def __init__(self, difficulty="easy", seed=None):
        if difficulty not in ("easy", "medium", "hard"):
            raise ValueError("difficulty must be one of: easy, medium, hard")

        self.difficulty = difficulty
        self._rng = random.Random(seed)
        self.step_count = 0
        self.reset()

    def reset(self):
        # Start close to a realistic indoor setpoint with small variance.
        self.step_count = 0
        self.temp = self._rng.uniform(21.5, 25.5)
        self.human_home = True

        if self.difficulty == "easy":
            self.price = 0.10
        elif self.difficulty == "medium":
            self.price = self._rng.uniform(0.15, 0.35)
        else:
            self.price = self._rng.uniform(0.10, 0.80)

        return HomeObservation(current_temp=self.temp, is_human_home=self.human_home, energy_price=self.price)

    def step(self, action: HomeAction):
        self.step_count += 1

        # Occupancy changes in medium/hard modes.
        if self.difficulty == "easy":
            self.human_home = True
        elif self.difficulty == "medium":
            self.human_home = self._rng.random() > 0.15
        else:
            self.human_home = self._rng.random() > 0.30

        # Medium has mild drift, hard has fast-changing real-time prices.
        if self.difficulty == "medium":
            self.price = max(0.10, min(0.60, self.price + self._rng.uniform(-0.03, 0.03)))
        elif self.difficulty == "hard":
            self.price = self._rng.uniform(0.10, 0.80)

        outside_temp = self._rng.uniform(28.0, 34.0) if self.difficulty == "hard" else self._rng.uniform(27.0, 31.0)

        # 1. Update Physics
        if action.action_code == 1:  # AC
            self.temp -= 1.0
        elif action.action_code == 2:  # Heat
            self.temp += 1.0

        # Temp naturally drifts toward outside air.
        drift = 0.15 if self.difficulty == "easy" else (0.25 if self.difficulty == "medium" else 0.35)
        self.temp += (outside_temp - self.temp) * drift

        # 2. Calculate Reward
        reward = 0.0
        comfort_low, comfort_high = (21.0, 23.0) if self.human_home else (18.0, 28.0)
        if comfort_low <= self.temp <= comfort_high:
            reward += 1.0
        else:
            reward -= 0.5

        # Extra penalty for discomfort while residents are home.
        if self.human_home and not (21.0 <= self.temp <= 23.0):
            reward -= 0.5

        if action.action_code != 0:
            reward -= self.price

        obs = HomeObservation(
            current_temp=round(self.temp, 2),
            is_human_home=self.human_home,
            energy_price=round(self.price, 3),
        )
        return obs, round(reward, 3)