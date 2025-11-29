"""
Training script for temporal graph models using PyTorch Lightning CLI.

This script provides a command-line interface for training various temporal graph models
with support for different datasets and configurations.
"""

import os
from pathlib import Path

import yaml
from lightning.pytorch.cli import LightningCLI, SaveConfigCallback

__FILE_PATH__ = Path(__file__)
__CLI_PATH__ = __FILE_PATH__.parents[0].resolve()
__TGX_PATH__ = __FILE_PATH__.parents[1].resolve()
import sys

sys.path.append(str(__TGX_PATH__))


class CustomSaveConfigCallback(SaveConfigCallback):
    """Custom callback to save configuration in a specific format."""

    def setup(self, trainer, pl_module, stage):
        """Setup callback."""
        if stage == "fit" and trainer.logger is not None:
            config_path = Path(trainer.logger.log_dir) / "config.yaml"
            with open(config_path, "w") as f:
                yaml.dump(self.config, f, default_flow_style=False)


class TemporalGraphCLI(LightningCLI):
    """Custom Lightning CLI for temporal graph models."""

    def add_arguments_to_parser(self, parser) -> None:
        """Add custom arguments to the parser."""
        parser.add_argument('--fit-only', action='store_true',
                            help='Do not run test after training is complete.')
        
        parser.link_arguments('data.mode', 'model.init_args.evaluator.init_args.mode', apply_on='instantiate')
        
        super().add_arguments_to_parser(parser)


def main():
    """Main entry point for training script."""

    # Set environment variable for reproducibility
    os.environ["PL_SEED_WORKERS"] = "1"

    # Create CLI - DataModule parameters are automatically handled through configuration
    cli = TemporalGraphCLI(
        save_config_callback=CustomSaveConfigCallback,
        seed_everything_default=42,
        parser_kwargs={
            "parser_mode": "omegaconf"
        },
        run=True
    )
    trainer = cli.trainer

    if cli.subcommand in ('fit', 'validate') and not trainer.fast_dev_run and not getattr(cli.config.fit, 'fit_only', False):
        model = cli.model
        data_module = cli.datamodule
        trainer.test(model=model, datamodule=data_module, ckpt_path="best")

if __name__ == "__main__":
    main()