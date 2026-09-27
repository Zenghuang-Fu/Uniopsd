"""CPU checks for the public configurations and portable release entry points."""
import unittest

from hydra import compose, initialize_config_dir

from uniopsd.train import CONFIG_DIR, prepare_config


def configuration(task="alfworld", scale="3b"):
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        return compose(config_name=f"{task}_corr_{scale}")


class ReleaseTests(unittest.TestCase):
    def test_all_six_task_model_configs(self):
        for task in ("alfworld", "webshop", "search"):
            for scale in ("3b", "7b"):
                with self.subTest(task=task, scale=scale):
                    config = configuration(task, scale)
                    self.assertEqual(config.algorithm.rlsd.teacher, "peer")
                    self.assertEqual(config.algorithm.rlsd.c_mode, "corr")
                    self.assertEqual(config.algorithm.rlsd.form, "fusion")
                    self.assertEqual(config.trainer.logger, ["console"])
                    self.assertEqual(config.trainer.n_gpus_per_node, 8)
                    self.assertIn(scale.upper(), config.actor_rollout_ref.model.path)
                    self.assertEqual(config.trainer.total_training_steps, 150)
                    self.assertEqual(config.trainer.test_freq, 5)

    def test_search_requires_explicit_retriever_configuration(self):
        config = configuration("search")
        config.env.search.search_url = None
        with self.assertRaisesRegex(ValueError, "RETRIEVAL_ADDRESS"):
            prepare_config(config)

    def test_evaluation_resumes_without_training_or_saving(self):
        config = configuration()
        config.trainer.resume_from_path = "checkpoints/example/global_step_150"
        result = prepare_config(config, evaluation=True)
        self.assertTrue(result.trainer.val_only)
        self.assertTrue(result.trainer.val_before_train)
        self.assertEqual(result.trainer.resume_mode, "resume_path")
        self.assertEqual(result.trainer.save_freq, -1)
        self.assertEqual(result.trainer.test_freq, -1)

    def test_corr_entrypoint_rejects_other_teacher_modes(self):
        config = configuration()
        config.algorithm.rlsd.teacher = "skill"
        with self.assertRaisesRegex(ValueError, "teacher=peer"):
            prepare_config(config)

    def test_task_specific_validation_and_parallelism(self):
        search = configuration("search")
        webshop = configuration("webshop")
        self.assertFalse(search.actor_rollout_ref.rollout.val_kwargs.do_sample)
        self.assertTrue(webshop.actor_rollout_ref.rollout.val_kwargs.do_sample)
        self.assertEqual(search.algorithm.gigpo.similarity_thresh, 0.9)
        self.assertEqual(search.actor_rollout_ref.rollout.tensor_model_parallel_size, 1)
        self.assertEqual(webshop.actor_rollout_ref.rollout.tensor_model_parallel_size, 2)


if __name__ == "__main__":
    unittest.main()
