import argparse
import json
import os
from tasks import get_easy_task, get_medium_task, get_hard_task
from interface import HomeAction
from policy import choose_action

# These are the variables the hackathon system looks for (from your screenshot)
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN")


def select_task(difficulty: str):
    if difficulty == "easy":
        return get_easy_task
    if difficulty == "medium":
        return get_medium_task
    return get_hard_task


def emit_log(payload, json_mode=False, log_file=None):
    if json_mode:
        line = json.dumps(payload, separators=(",", ":"))
        print(line)
        if log_file:
            log_file.write(line + "\n")
            log_file.flush()
        return

    message_type = payload.get("type")
    if message_type == "step":
        print(
            f"STEP {payload['step']}: action={payload['action']}, "
            f"temp={payload['temp']}, home={payload['home']}, "
            f"price={payload['price']}, reward={payload['reward']}"
        )
    elif message_type == "summary":
        print(f"TOTAL_REWARD: {payload['total_reward']}")


def run_inference(difficulty="easy", steps=24, seed=42, json_mode=False, jsonl_path=None):
    """Run the environment using a deterministic, energy-aware baseline policy."""
    print("START")
    env = select_task(difficulty)(seed=seed)
    obs = env.reset()
    log_file = open(jsonl_path, "a", encoding="utf-8") if jsonl_path else None

    total_reward = 0.0
    try:
        emit_log(
            {
                "type": "run_config",
                "difficulty": difficulty,
                "steps": steps,
                "seed": seed,
                "model": MODEL_NAME,
            },
            json_mode=json_mode,
            log_file=log_file,
        )

        for i in range(steps):
            action = choose_action(obs)
            obs, reward = env.step(HomeAction(action_code=action))
            total_reward += reward

            emit_log(
                {
                    "type": "step",
                    "step": i + 1,
                    "action": action,
                    "temp": obs.current_temp,
                    "home": obs.is_human_home,
                    "price": obs.energy_price,
                    "reward": reward,
                },
                json_mode=json_mode,
                log_file=log_file,
            )

        emit_log(
            {"type": "summary", "total_reward": round(total_reward, 3)},
            json_mode=json_mode,
            log_file=log_file,
        )
    finally:
        if log_file:
            log_file.close()

    print("END")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SmartHome environment inference.")
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard"], default="easy")
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--json", action="store_true", help="Emit structured JSON logs to stdout.")
    parser.add_argument("--jsonl-path", type=str, default=None, help="Optional file path for JSONL logs.")
    args = parser.parse_args()
    run_inference(
        difficulty=args.difficulty,
        steps=args.steps,
        seed=args.seed,
        json_mode=args.json,
        jsonl_path=args.jsonl_path,
    )