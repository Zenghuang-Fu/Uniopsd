"""Evaluate either a saved distributed checkpoint or an exported model."""
import hydra

from .train import CONFIG_DIR, prepare_config


@hydra.main(config_path=CONFIG_DIR, config_name="alfworld_corr_3b", version_base=None)
def main(config):
    from verl.trainer.main_rlsd import run_rlsd
    run_rlsd(prepare_config(config, evaluation=True))


if __name__ == "__main__":
    main()
