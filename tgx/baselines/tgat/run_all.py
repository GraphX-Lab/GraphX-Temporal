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
EDGE_DIM = {
    "mooc": 4,
    "wikipedia": 172,
    "lastfm": 2,
    "reddit": 172,
}
SEED = 42

def run_cmd(cmd):
    print("Running command:")
    print("[italic]"+" ".join(cmd)+"[/italic]")
    subprocess.run(cmd, check=True)


def finetuning(dataset, task, mode, n_runs=3, device="0,"):
    # Finetuning
    # ckpt_dirpath = f"checkpoints/cl-ond/{dataset}/{task}-{mode}/"
    for i in range(n_runs):
        seed = SEED + i
        print(Rule(f"[bold blue]Finetuning: {dataset}-{task}-{mode}-seed{seed}[/bold blue]"))
        cfg_name = f"mooc-{task}.yaml"
        # cfg_name = f"mooc-{task}.yaml" if task == 'lc' else f"mooc-{task}.yaml"
        cmd_finetune = [
            "python", "-m", "tgx.cli.train",
            "--config", str(BASE_DIR / cfg_name),
            "fit",
            "--trainer.devices", device,
            "--data.dataset", dataset,
            "--data.mode", mode,
            "--data.batch_size", str(4096*2),
            # "--model.evaluator.mode", mode,
            "--seed_everything", str(seed),
            "--model.init_args.edge_dim", str(EDGE_DIM.get(dataset, 0)),
            # "--trainer.callbacks[0].init_args.dirpath", ckpt_dirpath,
        ]
        run_cmd(cmd_finetune)

if __name__ == "__main__":
    n_runs = 3
    device = "0,"  # use 2 GPUs

    for dataset in DATASETS:
        # Link Prediction and Link Classification
        for mode in MODES:
            finetuning(dataset, "lp", mode, n_runs, device)
            if mode == 'transductive' and dataset != 'lastfm':
                finetuning(dataset, "lc", mode, n_runs, device)
    