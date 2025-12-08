from calendar import c
from pathlib import Path
import yaml
import pandas as pd
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.rule import Rule

LOG_DIR = Path("logs/")
console = Console()


def read_csv(csv_path: Path):
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    # all test metrics
    test_metrics = df.filter(like="test/").iloc[-1]
    return test_metrics


class SafeLoader(yaml.FullLoader):
    """自定义Loader，将特定的python/object标签转换为普通字典"""

    def __init__(self, stream):
        super(SafeLoader, self).__init__(stream)
        # 修正：使用 construct_mapping 获取内部字典
        self.add_constructor(
            "tag:yaml.org,2002:python/object:jsonargparse._namespace.Namespace",
            lambda loader, node: loader.construct_mapping(node),
        )
        self.add_constructor(
            "tag:yaml.org,2002:python/object/apply:pathlib.PosixPath",
            lambda loader, node: Path("/".join(loader.construct_sequence(node))),
        )


def read_yaml(config_path: Path):
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=SafeLoader)
    cfg = {
        "model": config["model"]["class_path"].split(".")[-1],
        "evaluator": config["model"]["init_args"]["evaluator"]["class_path"].split(".")[
            -1
        ],
        "mode": config["data"]["init_args"].get("mode", "transductive"),
        "dataset": config["data"]["init_args"]["dataset"],
    }
    return cfg


def read_records(log_dir: Path):
    records = []
    version_dirs = log_dir.rglob("version_*")
    for cfg_dir in version_dirs:
        if not cfg_dir.is_dir():
            continue
        config_path = cfg_dir / "config.yaml"
        csv_path = cfg_dir / "metrics.csv"
        cfg = read_yaml(config_path)
        metrics = read_csv(csv_path)
        if metrics is None:
            continue
        record = {
            "model": cfg["model"],
            "evaluator": cfg["evaluator"],
            "mode": cfg["mode"],
            "dataset": cfg["dataset"],
            "test/ap": metrics["test/ap"],
            "test/auroc": metrics["test/auroc"],
        }
            # "test/ap/new": metrics.get("test/ap/new", np.nan),
            # "test/auroc/new": metrics.get("test/auroc/new", np.nan),
        if "test/auroc/new" in metrics.index:
            record["test/auroc/new"] = metrics["test/auroc/new"]
        if "test/ap/new" in metrics.index:
            record["test/ap/new"] = metrics["test/ap/new"]
        records.append(record)
    df = pd.DataFrame(records)
    return df


def summarize_results(df: pd.DataFrame, precision: int = 2):
    group = df.groupby(["dataset", "evaluator", "mode", "model"])
    csv_lines = [
        "Model,Evaluator,Mode,Dataset,Test AP (mean ± std),Test AUROC (mean ± std),Runs"
    ]
    table = Table(title="Summary Results")
    table.add_column("Model", justify="center", style="cyan", no_wrap=True)
    table.add_column("Evaluator", justify="center", style="magenta")
    table.add_column("Mode", justify="center", style="green")
    table.add_column("Dataset", justify="center", style="yellow")
    table.add_column("Test AP\n(mean ± std)", justify="center", style="red")
    table.add_column("Test AUROC\n(mean ± std)", justify="center", style="blue")
    table.add_column("Runs", justify="center", style="white")

    pre_keys = None
    key_idx_make_sec = 0
    for (dataset, evaluator, mode, model), group_df in group:
        evaluator = evaluator.replace("Evaluator", "")

        keys = (dataset, evaluator, mode, model)
        if pre_keys is not None and keys[key_idx_make_sec] != pre_keys[key_idx_make_sec]:
            table.add_section()
        pre_keys = keys 

        ap_mean = group_df["test/ap"].mean()
        ap_std = group_df["test/ap"].std()
        if "test/auroc/new" in group_df.columns and mode == "inductive" and evaluator == "LinkPredictionEvaluator":
            auroc_mean = group_df["test/auroc/new"].mean()
            auroc_std = group_df["test/auroc/new"].std()
        else:
            auroc_mean = group_df["test/auroc"].mean()
            auroc_std = group_df["test/auroc"].std()
        n_runs = len(group_df)
        line = [
            str(model),
            str(evaluator),
            str(mode),
            str(dataset),
            f"{ap_mean * 100:.{precision}f} ± {ap_std * 100:.{precision}f}",
            f"{auroc_mean * 100:.{precision}f} ± {auroc_std * 100:.{precision}f}",
            str(n_runs),
        ]
        csv_lines.append(",".join(line))
        table.add_row(*line)

    console.print(table)
    console.print(Rule("CSV Format"))
    for line in csv_lines:
        console.print(line)


if __name__ == "__main__":
    df = read_records(LOG_DIR / "cl-ond")
    summarize_results(df, 4)
