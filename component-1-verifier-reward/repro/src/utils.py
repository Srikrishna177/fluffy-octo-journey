"""Shared utilities: seeding, JSON curve logging, param counting, timers."""
import json
import os
import random
import time

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class StepLogger:
    """Appends {step, ...metrics} dicts to a JSONL file and mirrors a line to a text log."""

    def __init__(self, jsonl_path: str, text_log_path: str):
        self.jsonl_path = jsonl_path
        self.text_log_path = text_log_path
        os.makedirs(os.path.dirname(jsonl_path), exist_ok=True)
        os.makedirs(os.path.dirname(text_log_path), exist_ok=True)
        # start fresh each run
        open(self.jsonl_path, "w").close()
        open(self.text_log_path, "w").close()

    def log(self, **kwargs):
        line = json.dumps(kwargs)
        with open(self.jsonl_path, "a") as f:
            f.write(line + "\n")
        text = " ".join(f"{k}={v}" for k, v in kwargs.items())
        with open(self.text_log_path, "a") as f:
            f.write(text + "\n")
        print(text, flush=True)


class WallClock:
    def __init__(self, budget_seconds: float):
        self.start = time.time()
        self.budget_seconds = budget_seconds

    def elapsed(self) -> float:
        return time.time() - self.start

    def expired(self) -> bool:
        return self.elapsed() >= self.budget_seconds

    def remaining(self) -> float:
        return max(0.0, self.budget_seconds - self.elapsed())
