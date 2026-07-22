"""Run the MAB button-pushing evaluation for one model config.

    python run.py configs/qwen3_4b.json

An evaluation is a set of independent trajectories (epochs). Each trajectory is a
sequence of `num_steps` decisions against one bandit instance. Trajectories are
independent, so we advance them in lockstep and batch one prompt per trajectory into
each vLLM call. Classic baselines (UCB/Greedy/Thompson) are run on the same instances
for reference.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

import numpy as np

from baselines import AGENTS
from history import build_prompt
from mab import BernoulliMAB, VerbalMAB, make_means
import metrics


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULTS = {
    # bandit instance
    "num_arms": 5,
    "difficulty": "hard",      # 'hard' (gap 0.2) or 'easy' (gap 0.5)
    "shuffle_arms": True,      # randomize which button is best, per epoch
    # prompting
    "history_type": "SH",      # SH (primary) | RH
    "instruction_type": "detailed",
    # sampling / logging
    "max_tokens": 20,
    "log_prompts": True,
}


def load_config(path: str) -> dict:
    cfg = dict(DEFAULTS)
    cfg.update(json.loads(Path(path).read_text()))
    return cfg


def epoch_instances(cfg: dict) -> tuple[List[List[float]], List[int]]:
    """One arm-mean vector per epoch. Means built from difficulty, optionally shuffled so
    the best button varies across epochs (removes positional bias)."""
    base = make_means(cfg["num_arms"], cfg["difficulty"])
    rng = np.random.default_rng(cfg["seed"])
    epoch_seeds = rng.integers(0, 2**32 - 1, size=cfg["num_epochs"])
    instances = []
    for es in epoch_seeds:
        means = list(base)
        np.random.default_rng(int(es)).shuffle(means)
        instances.append(means)
    return instances, [int(s) for s in epoch_seeds]


# ---------------------------------------------------------------------------
# LLM rollouts (lockstep, batched)
# ---------------------------------------------------------------------------

def run_llm(cfg: dict, instances, seeds, runner) -> List[List[dict]]:
    envs = [VerbalMAB(BernoulliMAB(means, cfg["num_steps"], seed=s), cfg["instruction_type"])
            for means, s in zip(instances, seeds)]
    trajs: List[List[dict]] = [[] for _ in envs]

    for _ in range(cfg["num_steps"]):
        prompts = [build_prompt(env.core.history, env.num_arms, cfg["history_type"],
                                cfg["instruction_type"]) for env in envs]
        responses = runner.chat(prompts, cfg)
        for env, prompt, response, traj in zip(envs, prompts, responses, trajs):
            rec = env.step(response)
            step = asdict(rec)
            if cfg["log_prompts"]:
                step["prompt"] = prompt
            traj.append(step)
    return trajs


# ---------------------------------------------------------------------------
# Classic baselines (same instances, cheap, pure-python)
# ---------------------------------------------------------------------------

def run_baseline(agent_key: str, cfg: dict, instances, seeds) -> List[List[dict]]:
    trajs = []
    for means, s in zip(instances, seeds):
        core = BernoulliMAB(means, cfg["num_steps"], seed=s + 1)
        agent = AGENTS[agent_key](core.num_arms, seed=s + 2)
        traj = []
        for step in range(cfg["num_steps"]):
            arm = agent.act()
            reward = core.sample_reward(arm, step)
            agent.update(arm, reward)
            best_mean = core.optimal_mean(step)
            exp = core.expected_reward(arm, step)
            traj.append({
                "step": step, "action": arm, "reward": reward,
                "expected_reward": exp, "regret": best_mean - exp,
                "best_mean": best_mean, "is_parse_failure": False,
            })
        trajs.append(traj)
    return trajs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

BASELINE_CONFIG = "configs/baseline.json"


def _write_results(cfg: dict, results: dict, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"Wrote {out_path}")
    for name, data in results["agents"].items():
        m = data["metrics"]
        print(f"  {name:16s} avg_reward={m['final_avg_reward']:.3f} "
              f"cum_regret={m['final_cumulative_regret']:.2f} "
              f"parse_fail={m['parse_failure_rate']:.3f}")


def run_model(config_path: str):
    cfg = load_config(config_path)
    instances, seeds = epoch_instances(cfg)

    from runners import VLLMRunner  # imported here so non-vLLM tooling can use the rest
    runner = VLLMRunner(cfg["model"])

    results: Dict[str, object] = {"config": cfg, "instances": instances, "seeds": seeds}
    llm_trajs = run_llm(cfg, instances, seeds, runner)
    results["agents"] = {
        cfg["model_name"]: {
            "trajectories": llm_trajs,
            "metrics": metrics.summarize(llm_trajs),
        }
    }

    out_path = Path(cfg["output"]) / f"results_{cfg.get('temperature', 1.0)}.json"
    _write_results(cfg, results, out_path)


def run_baselines():
    """Run the classic reference agents once on the shared bandit instances.

    Baselines depend only on the instance params (seed/num_epochs/num_arms/difficulty),
    which are identical across all model configs, so we run them separately instead of
    repeating them for every model."""
    cfg = load_config(BASELINE_CONFIG)
    instances, seeds = epoch_instances(cfg)

    results: Dict[str, object] = {"config": cfg, "instances": instances, "seeds": seeds}
    agents_out = {}
    for key in AGENTS:
        b_trajs = run_baseline(key, cfg, instances, seeds)
        agents_out[key] = {"trajectories": b_trajs, "metrics": metrics.summarize(b_trajs)}
    results["agents"] = agents_out

    out_path = Path(cfg["output"]) / "results.json"
    _write_results(cfg, results, out_path)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python run.py <config.json | baseline>")
        sys.exit(1)
    if sys.argv[1] == "baseline":
        run_baselines()
    else:
        run_model(sys.argv[1])
