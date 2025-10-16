from rich.traceback import install

import main
import wandb

# Keep utilities import to align env setup expectations
from utilities import *  # noqa: F401,F403

install(show_locals=False)


def ablate_agent():
    with wandb.init():
        config = wandb.config
        config_dict = dict(config)
        # Prefix the auto-generated run name: baseline_... or <RUN_DISPLAY_NAME>_...
        is_ablate = bool(config_dict.get("ABLATE", False))
        auto_name = (wandb.run.name or "").strip()
        prefix = (config_dict.get("RUN_DISPLAY_NAME") or "ablation") if is_ablate else "baseline"
        prefix = str(prefix).strip().replace(" ", "_")
        try:
            if auto_name and not auto_name.startswith(prefix + "_"):
                wandb.run.name = f"{prefix}_{auto_name}"
                wandb.run.save()
        except Exception:
            pass
        # Delegate to the same unified pipeline used elsewhere
        main.run_pipeline(mode="train", config=config_dict)


if __name__ == "__main__":
    ablate_agent()


