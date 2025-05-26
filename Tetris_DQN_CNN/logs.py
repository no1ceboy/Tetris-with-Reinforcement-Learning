# logs.py
import tensorflow as tf
from typing import Any, Dict, Optional

class CustomTensorBoard:
    def __init__(self, log_dir: str = "logs/", **kwargs: Any):
        self.writer: tf.summary.SummaryWriter = tf.summary.create_file_writer(log_dir)
        print(f"TensorBoard logging to: {log_dir}")
    def set_model(self, model: Any) -> None: pass
    def log(self, step: int, **stats: float) -> None:
        with self.writer.as_default():
            for name, value in stats.items():
                if value is not None: tf.summary.scalar(name, value, step=step)
            self.writer.flush()
    def close(self) -> None: self.writer.close()