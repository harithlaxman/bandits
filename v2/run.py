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

from baselines import AGENTS, CONTEXTUAL_AGENTS
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
    "switch_frac": None,       # fraction of horizon at which the optimal arm swaps (None = stationary)
    # non-stationary baselines
    "gamma": 0.99,             # D-UCB discount factor
    "window": 200,             # SW-UCB window size
    "sw_window": 300,          # SW-LinUCB window size (covertype)
    # prompting
    "history_type": "SH",      # SH (primary) | RH
    "instruction_type": "detailed",
    "window": 30,              # covertype LLM: sliding window of raw interactions in prompt
    # sampling / logging
    "max_tokens": 20,
    "log_prompts": True,
}


def load_config(path: str) -> dict:
    cfg = dict(DEFAULTS)
    cfg.update(json.loads(Path(path).read_text()))
    return cfg


def switch_step_of(cfg: dict) -> int | None:
    """Absolute step at which the optimal arm swaps, or None for a stationary instance."""
    sf = cfg.get("switch_frac")
    return int(sf * cfg["num_steps"]) if sf else None


def ns_suffix(cfg: dict) -> str:
    """'_ns' for a non-stationary run, '' otherwise — appended to result filenames."""
    return "_ns" if switch_step_of(cfg) is not None else ""


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
    switch = switch_step_of(cfg)
    envs = [VerbalMAB(BernoulliMAB(means, cfg["num_steps"], seed=s, switch_step=switch),
                      cfg["instruction_type"])
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

# Per-agent hyperparameters drawn from config (other agents take no extra args).
AGENT_KWARGS = {
    "ducb": lambda c: {"gamma": c["gamma"]},
    "swucb": lambda c: {"window": c["window"]},
}


def run_baseline(agent_key: str, cfg: dict, instances, seeds) -> List[List[dict]]:
    switch = switch_step_of(cfg)
    kw = AGENT_KWARGS.get(agent_key, lambda c: {})(cfg)
    trajs = []
    for means, s in zip(instances, seeds):
        core = BernoulliMAB(means, cfg["num_steps"], seed=s + 1, switch_step=switch)
        agent = AGENTS[agent_key](core.num_arms, seed=s + 2, **kw)
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


def run_covertype_llm(cfg: dict, seeds, runner) -> List[List[dict]]:
    """Lockstep batched rollouts on the Covertype bandit, one trajectory per epoch.
    Same env seeding as run_covertype(), so the model plays the identical row
    sequences as the LinUCB baseline."""
    from covertype import CovertypeBandit, VerbalCovertypeBandit, load_covertype
    from history import build_covertype_prompt

    X, y = load_covertype()
    switch = switch_step_of(cfg)
    envs = [VerbalCovertypeBandit(CovertypeBandit(X, y, cfg["num_steps"], seed=s + 1,
                                                  switch_step=switch),
                                  seed=s + 2)
            for s in seeds]
    trajs: List[List[dict]] = [[] for _ in envs]

    for _ in range(cfg["num_steps"]):
        prompts = [build_covertype_prompt(env.history, env.current_context_text(),
                                          env.num_arms, cfg["window"]) for env in envs]
        responses = runner.chat(prompts, cfg)
        for env, prompt, response, traj in zip(envs, prompts, responses, trajs):
            rec = env.step(response)
            step = asdict(rec)
            if cfg["log_prompts"]:
                step["prompt"] = prompt
            traj.append(step)
    return trajs


def run_model(config_path: str):
    cfg = load_config(config_path)

    from runners import VLLMRunner  # imported here so non-vLLM tooling can use the rest
    runner = VLLMRunner(cfg["model"])

    if cfg.get("env") == "covertype":
        rng = np.random.default_rng(cfg["seed"])
        seeds = [int(s) for s in rng.integers(0, 2**32 - 1, size=cfg["num_epochs"])]
        results: Dict[str, object] = {"config": cfg, "seeds": seeds}
        llm_trajs = run_covertype_llm(cfg, seeds, runner)
    else:
        instances, seeds = epoch_instances(cfg)
        results = {"config": cfg, "instances": instances, "seeds": seeds}
        llm_trajs = run_llm(cfg, instances, seeds, runner)
    results["agents"] = {
        cfg["model_name"]: {
            "trajectories": llm_trajs,
            "metrics": metrics.summarize(llm_trajs),
        }
    }

    out_path = Path(cfg["output"]) / f"results_{cfg.get('temperature', 1.0)}{ns_suffix(cfg)}.json"
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

    out_path = Path(cfg["output"]) / f"results{ns_suffix(cfg)}.json"
    _write_results(cfg, results, out_path)


COVERTYPE_CONFIG = "configs/covertype.json"

# Extra per-agent kwargs drawn from config (other contextual agents take none).
CONTEXTUAL_AGENT_KWARGS = {
    "sw_linucb": lambda c: {"window": c["sw_window"]},
}


def run_covertype(config_path: str = COVERTYPE_CONFIG):
    """Contextual baselines on the Covertype classification bandit. The optimal
    action always yields reward 1, so regret is simply 1 - reward per step."""
    from covertype import CovertypeBandit, load_covertype

    cfg = load_config(config_path)
    switch = switch_step_of(cfg)
    X, y = load_covertype()
    rng = np.random.default_rng(cfg["seed"])
    seeds = [int(s) for s in rng.integers(0, 2**32 - 1, size=cfg["num_epochs"])]

    results: Dict[str, object] = {"config": cfg, "seeds": seeds}
    agents_out = {}
    for key, cls in CONTEXTUAL_AGENTS.items():
        kw = CONTEXTUAL_AGENT_KWARGS.get(key, lambda c: {})(cfg)
        trajs = []
        for s in seeds:
            env = CovertypeBandit(X, y, cfg["num_steps"], seed=s + 1, switch_step=switch)
            agent = cls(env.num_arms, X.shape[1], alpha=cfg["linucb_alpha"],
                        lam=cfg["linucb_lam"], seed=s + 2, **kw)
            traj = []
            for step in range(cfg["num_steps"]):
                x = env.context(step)
                arm = agent.act(x)
                reward = env.reward(arm, step)
                agent.update(x, arm, reward)
                traj.append({
                    "step": step, "action": arm, "reward": reward,
                    "expected_reward": reward, "regret": 1.0 - reward,
                    "best_mean": 1.0, "is_parse_failure": False,
                })
            trajs.append(traj)
        agents_out[key] = {"trajectories": trajs, "metrics": metrics.summarize(trajs)}
    results["agents"] = agents_out

    _write_results(cfg, results, Path(cfg["output"]) / f"results{ns_suffix(cfg)}.json")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python run.py <config.json | baseline | covertype [config.json]>")
        sys.exit(1)
    if sys.argv[1] == "baseline":
        run_baselines()
    elif sys.argv[1] == "covertype":
        run_covertype(*sys.argv[2:3])
    else:
        run_model(sys.argv[1])
