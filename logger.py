from typing import Any, Callable, Dict, Optional
import wandb
import torch
import torch.nn as nn

class WandbWriter:
    def __init__(self, project_name: str, sweep_config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize the WandbWriter class.

        Args:
            project_name (str): The name of the W&B project.
            sweep_config (Optional[Dict[str, Any]]): Configuration for W&B sweep. If provided, initializes a sweep.
        """
        if sweep_config:
            self.sweep_id = wandb.sweep(sweep_config, project=project_name)
        else:
            wandb.init(project=project_name)
        self.config = wandb.config

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """
        Log metrics to W&B.

        Args:
            metrics (Dict[str, Any]): Dictionary of metrics to log.
            step (Optional[int]): Step at which to log the metrics.
        """
        wandb.log(metrics, step=step)

    def log_plot(self, plot_name: str, figure: Any, step: Optional[int] = None) -> None:
        """
        Log a plot to W&B.

        Args:
            plot_name (str): Name of the plot.
            figure (Any): Figure object to log.
            step (Optional[int]): Step at which to log the plot.
        """
        wandb.log({plot_name: wandb.Image(figure)}, step=step)

    def log_model(self, model: nn.Module, model_name: str) -> None:
        """
        Save and log a model to W&B.

        Args:
            model (nn.Module): The model to save.
            model_name (str): The name of the saved model file.
        """
        torch.save(model.state_dict(), model_name)
        wandb.save(model_name)

    def finish(self) -> None:
        """
        Finish the W&B run.
        """
        wandb.finish()

    def run_sweep(self, train_function: Callable[[], None], count: int = 1) -> None:
        """
        Run a W&B sweep.

        Args:
            train_function (Callable[[], None]): The training function to run.
            count (int): Number of times to run the sweep.
        """
        wandb.agent(self.sweep_id, function=train_function, count=count)
