"""Run the complete corr training pipeline with a selected task configuration."""
from pathlib import Path

import hydra
from omegaconf import OmegaConf

CONFIG_DIR = str(Path(__file__).resolve().parents[1] / "verl" / "trainer" / "config")


def prepare_config(config, evaluation=False):
    if config.algorithm.adv_estimator != "gigpo":
        raise ValueError("UniOPSD corr requires the GiGPO backbone")
    for field, expected in (("teacher", "peer"), ("form", "fusion"), ("c_mode", "corr")):
        if config.algorithm.rlsd[field] != expected:
            raise ValueError(f"UniOPSD requires algorithm.rlsd.{field}={expected}")
    if config.env.env_name == "search" and not config.env.search.search_url:
        raise ValueError("Set RETRIEVAL_ADDRESS or env.search.search_url to your retrieval service")
    if evaluation:
        config.trainer.val_only = True
        config.trainer.val_before_train = True
        config.trainer.save_freq = -1
        config.trainer.test_freq = -1
        config.trainer.total_epochs = 1
        config.trainer.total_training_steps = 1
        if config.trainer.resume_from_path:
            config.trainer.resume_mode = "resume_path"
    OmegaConf.resolve(config)
    return config


@hydra.main(config_path=CONFIG_DIR, config_name="alfworld_corr_3b", version_base=None)
def main(config):
    from verl.trainer.main_rlsd import run_rlsd
    run_rlsd(prepare_config(config))


if __name__ == "__main__":
    main()
