import os
import argparse
import yaml
import wandb
import ablate_agent


def _load_train_config(config_path: str):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    params = cfg.get("parameters", {})
    # flatten to a dict of default values
    flat = {k: (v.get("value") if isinstance(v, dict) else v) for k, v in params.items()}
    return flat


def _build_sweep_from_train(train_cfg: dict, sweep_name: str | None = None) -> dict:
    entity = train_cfg.get("WANDB_ENTITY", None)
    project = train_cfg.get("WANDB_PROJECT", None)

    sweep = {
        "method": "grid",
        "metric": {"goal": "minimize", "name": "Validation/epoch/Loss"},
        "program": "ablate_agent.py",
        "run_cap": 2,  # ensure sweep ends after two runs
        # keep legacy top-level project to match config_sweep.yaml style
        "project": project,
        "parameters": {},
    }
    if sweep_name:
        sweep["name"] = str(sweep_name)

    # Carry over defaults as single values so the agent sees a complete config
    for k, v in train_cfg.items():
        sweep["parameters"][k] = {"value": v}

    # Insert our two-run toggle
    sweep["parameters"]["ABLATE"] = {"values": [True, False]}
    # Provide sweep display name to agent so it can name the ablation run
    if sweep_name:
        sweep["parameters"]["RUN_DISPLAY_NAME"] = {"value": sweep_name}

    return sweep, entity, project


def main():
    parser = argparse.ArgumentParser(description="Launch a 2-run ablation sweep")
    parser.add_argument("--name", "-n", type=str, default=None, help="Name for the W&B sweep")
    parser.add_argument("--config", "-c", type=str, default=os.environ.get("ABLATE_CONFIG", "config_train.yaml"), help="Path to base train config")
    args, unknown = parser.parse_known_args()

    # Load base config
    train_cfg = _load_train_config(args.config)

    # Apply CLI overrides like --LEARNING_RATE=1e-3 to train_cfg before building the sweep
    for arg in unknown:
        if not arg.startswith("--"):
            continue
        key_value = arg.lstrip("--").split("=", 1)
        key = key_value[0].upper()
        value = None if len(key_value) == 1 else key_value[1]
        if key in train_cfg and value is not None:
            orig = train_cfg[key]
            try:
                if isinstance(orig, bool):
                    v = value.lower() in ("true", "1", "yes")
                elif isinstance(orig, (list, tuple, dict)):
                    import ast as _ast
                    v = _ast.literal_eval(value)
                elif isinstance(orig, int):
                    v = int(value)
                elif isinstance(orig, float):
                    v = float(value)
                else:
                    v = type(orig)(value)
            except Exception:
                v = value
            train_cfg[key] = v

    sweep_cfg, entity, project = _build_sweep_from_train(train_cfg, sweep_name=args.name)

    # Create sweep and launch exactly two runs (max is 2 because grid of [True, False])
    sweep_id = wandb.sweep(sweep=sweep_cfg, entity=entity, project=project)
    # Pass config path for agents that want to read base config
    os.environ["ABLATE_CONFIG"] = os.path.abspath(args.config)
    # agent runs locally; count=2 ensures exactly the two configurations
    wandb.agent(
        sweep_id,
        function=ablate_agent.ablate_agent,
        count=2,
        project=project,
        entity=entity,
    )
    # No more agents will run and run_cap is reached: the sweep will settle as finished


if __name__ == "__main__":
    main()


