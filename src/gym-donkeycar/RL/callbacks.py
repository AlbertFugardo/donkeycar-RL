from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import TensorBoardOutputFormat
import os

class LapTimeCallback(BaseCallback):
    def _on_training_start(self):
        self.n_laps = 0
        output_formats = self.logger.output_formats
        # Save reference to tensorboard formatter object
        # note: the failure case (not formatter found) is not handled here, should be done with try/except.
        self.tb_formatter = next(formatter for formatter in output_formats if isinstance(formatter, TensorBoardOutputFormat))

    def _on_step(self) -> bool:
        lap_count = self.locals["infos"][0]["lap_count"]
        lap_time = self.locals["infos"][0]["last_lap_time"]

        if lap_count != self.n_laps and lap_time > 0:
            self.n_laps = lap_count
            self.tb_formatter.writer.add_scalar("time/lap_time", lap_time, self.num_timesteps)
            if lap_count == 1:
                self.tb_formatter.writer.add_scalar("time/first_lap_time", lap_time, self.num_timesteps)
            else:
                self.tb_formatter.writer.add_scalar("time/second_lap_time", lap_time, self.num_timesteps)
            self.tb_formatter.writer.flush()
            
class CheckpointCallback(BaseCallback):
    """
    Callback for saving a model every ``save_freq`` calls to ``env.step()``.
    It only saves model checkpoints.

    :param save_freq: Save checkpoints every ``save_freq`` call of the callback.
    :param save_path: Path to the folder where the model will be saved.
    :param verbose: Verbosity level: 0 for no output, 2 for indicating when saving model checkpoint
    """

    def __init__(
        self,
        save_freq: int,
        save_path: str,
        verbose: int = 2,
        save_replay_buffer: bool = False,
        replay_buffer_path: str = None,
        normalize: bool = False,
        normalize_path: str = None,
    ):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.last_best_test_time = float('inf')
        self.save_replay_buffer = save_replay_buffer
        self.replay_buffer_path = replay_buffer_path
        self.normalize = normalize
        self.normalize_path = normalize_path

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            save_path = self.save_path+"_"+str(self.n_calls)
            self.model.save(save_path)
            #model.save(config["model_path"])
            if self.verbose >= 2:
                print(f"Saving model checkpoint to {save_path}")
            if self.save_replay_buffer and self.replay_buffer_path is not None:
                replay_buffer_path = self.replay_buffer_path+"_"+str(self.n_calls)
                self.model.save_replay_buffer(replay_buffer_path+".pkl")
                print(f"Saving replay buffer to {replay_buffer_path}.pkl")
            if self.normalize and self.normalize_path is not None:
                normalize_path = self.normalize_path+"_"+str(self.n_calls)
                vec_normalize = self.model.get_vec_normalize_env()
                assert vec_normalize is not None
                vec_normalize.save(normalize_path+".pkl")
                print(f"Saving vec_normalize to {normalize_path}.pkl")
        if self.last_best_test_time != self.model.best_test_time:
            self.model.save(self.save_path+"_best")
            self.last_best_test_time = self.model.best_test_time

            print(f"Saving best model checkpoint to {self.save_path}_best, with test average time {self.last_best_test_time}")
            
            if self.save_replay_buffer and self.replay_buffer_path is not None:
                self.model.save_replay_buffer(self.replay_buffer_path+"_best.pkl")
                print(f"Saving replay buffer to {self.replay_buffer_path}_best.pkl")
            if self.normalize and self.normalize_path is not None:
                vec_normalize = self.model.get_vec_normalize_env()
                assert vec_normalize is not None
                vec_normalize.save(self.normalize_path+"_best.pkl")
                print(f"Saving vec_normalize to {self.normalize_path}_best.pkl")

        return True
    
class LearningRateCallback(BaseCallback):
    def _on_step(self) -> bool:
        if self.model.num_timesteps == 250000:
            for g in self.model.policy.optimizer.param_groups:
                g['lr'] = 5e-5
            print(f"Changing learning rate to {5e-5}")
        if self.model.num_timesteps == 350000:
            for g in self.model.policy.optimizer.param_groups:
                g['lr'] = 1e-5
            print(f"Changing learning rate to {1e-5}")
            
        return True
