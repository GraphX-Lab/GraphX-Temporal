#!/bin/bash
set -euo pipefail

trainer_cfg="--trainer.logger.class_path lightning.pytorch.loggers.TensorBoardLogger --trainer.logger.init_args.save_dir ./logs/cl-ond/mooc/"

python -m tgx.cli.train --config tgx/baselines/cl_ond/mooc-pretrain.yaml fit --trainer.max_epochs 100 --trainer.devices 2,
python -m tgx.cli.train --config tgx/baselines/cl_ond/mooc-lp.yaml fit --trainer.max_epochs 100 --trainer.devices 2,