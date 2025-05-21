import tensorflow as tf
# from datetime import datetime # Not strictly needed if log_dir is timestamped in run.py
from typing import Any, Dict, Optional

class CustomTensorBoard:
    """
    Custom logger for writing metrics to TensorBoard.
    """
    def __init__(self, log_dir: str = "logs/", **kwargs: Any):
        """
        Initializes the TensorBoard SummaryWriter.

        Args:
            log_dir: The directory where TensorBoard logs will be saved.
                     It's assumed this path is made unique (e.g., timestamped) by the calling script.
            kwargs: Additional arguments (not used in this basic version).
        """
        self.log_dir: str = log_dir
        # Create a TensorFlow summary writer that writes to the specified log directory
        self.writer: tf.summary.SummaryWriter = tf.summary.create_file_writer(self.log_dir)
        # This print statement confirms the logging directory being used
        print(f"TensorBoard logging to: {self.log_dir}")


    def set_model(self, model: Any) -> None:
        """
        Placeholder method for Keras compatibility.
        The standard Keras TensorBoard callback uses this, but it's not strictly
        necessary for custom scalar logging as implemented here.
        """
        pass # No action needed for this custom logger

    def log(self, step: int, **stats: float) -> None:
        """
        Logs scalar statistics to TensorBoard.

        Args:
            step: The current training step or epoch number (acts as the x-axis on plots).
            stats: Keyword arguments where keys are metric names (e.g., 'avg_score')
                   and values are the corresponding metric values (e.g., 150.5).
        """
        with self.writer.as_default(): # Set this writer as the default for summary operations
            for name, value in stats.items(): # Iterate through all provided statistics
                if value is not None: # Only log the metric if its value is not None
                    # Write a scalar summary: creates a plot for 'name' with 'value' at 'step'
                    tf.summary.scalar(name, value, step=step)
            self.writer.flush() # Ensure all pending summaries are written to disk

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, float]] = None) -> None:
        """
        Mimics a Keras callback method, useful for logging at the end of an epoch or episode.
        This provides a convenient interface if migrating from Keras callbacks.
        """
        logs = logs or {} # Ensure logs is a dictionary, even if None is passed
        self.log(epoch, **logs) # Use the main log method

    def close(self) -> None:
        """Closes the TensorBoard SummaryWriter to free resources."""
        self.writer.close()