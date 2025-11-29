from rich import print
from rich.rule import Rule
from pathlib import Path
import subprocess

BASE_DIR = Path(__file__).resolve().parent
DATASETS = [
    "mooc",
    "wikipedia",
    "lastfm",
    "reddit",
    ]
MODES = [
    "transductive",
    "inductive"
    ]

SEED = 42

def run_cmd(cmd):
    print("Running command:")
    print("[italic]"+" ".join(cmd)+"[/italic]")
    subprocess.run(cmd, check=True)

def pretraining(dataset, mode=None, n_runs=3, device="0,"):
    # Pretraining
    for i in range(n_runs):
        seed = SEED + i
        print(Rule(f"[bold blue]Pretraining: {dataset}-{mode}-seed{seed}[/bold blue]"))
        # ckpt_dirpath = f"checkpoints/cl-ond/{dataset}/pretrain-{mode}/"
        cmd_pretrain = [
            "python", "-m", "tgx.cli.train",
            "--config", str(BASE_DIR / f"mooc-pretrain-inductive.yaml"),
            "fit",
            "--data.dataset", dataset,
            "--data.mode", mode,
            "--seed_everything", str(seed),
            "--trainer.devices", device,
            # "--trainer.callbacks[0].init_args.dirpath", ckpt_dirpath,
        ]
        run_cmd(cmd_pretrain)

def finetuning(dataset, task, mode, n_runs=3, device="0,"):
    # Finetuning
    # ckpt_dirpath = f"checkpoints/cl-ond/{dataset}/{task}-{mode}/"
    for i in range(n_runs):
        seed = SEED + i
        if i > 0:
            pretrained_ckpt = f"checkpoints/cl-ond/{dataset}/pretrain-{mode}/best-v{i}.ckpt"
        else:
            pretrained_ckpt = f"checkpoints/cl-ond/{dataset}/pretrain-{mode}/best.ckpt"
        print(Rule(f"[bold blue]Finetuning: {dataset}-{task}-{mode}-seed{seed}[/bold blue]"))
        cfg_name = f"mooc-{task}.yaml" if task == 'lc' else f"mooc-{task}-inductive.yaml"
        cmd_finetune = [
            "python", "-m", "tgx.cli.train",
            "--config", str(BASE_DIR / cfg_name),
            "fit",
            "--trainer.devices", device,
            "--data.dataset", dataset,
            "--data.mode", mode,
            "--seed_everything", str(seed),
            "--model.pretrained_ckpt_path", str(pretrained_ckpt),
            # "--trainer.callbacks[0].init_args.dirpath", ckpt_dirpath,
        ]
        run_cmd(cmd_finetune)

if __name__ == "__main__":
    n_runs = 3
    device = "0,"  # use 2 GPUs

    

    for dataset in DATASETS:
        # Link Prediction and Link Classification
        for mode in MODES:
            pretraining(dataset, mode, n_runs, device)
            finetuning(dataset, "lp", mode, n_runs, device)
            if mode == 'transductive' and dataset != 'lastfm':
                finetuning(dataset, "lc", mode, n_runs, device)
    