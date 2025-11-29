import subprocess
import sys
import pytest

@pytest.mark.integration
@pytest.mark.parametrize(
    "dataset,mode,split_ratio",
    [
        ("mooc", "inductive", [0.1, 0.1]),
        ("mooc", "transductive", [0.1, 0.1]),
        ("mooc", "inductive", [0.3, 0.1, 0.1]),
        ("mooc", "transductive", [0.3, 0.1, 0.1]),

        ("reddit", "inductive", [0.1, 0.1]),
        ("reddit", "transductive", [0.1, 0.1]),
        ("reddit", "inductive", [0.3, 0.1, 0.1]),
        ("reddit", "transductive", [0.3, 0.1, 0.1]),

        ("mooc", "inductive", [0.1, 0.1]),
        ("mooc", "transductive", [0.1, 0.1]),
        ("mooc", "inductive", [0.3, 0.1, 0.1]),
        ("mooc", "transductive", [0.3, 0.1, 0.1]),

        ("wikipedia", "inductive", [0.1, 0.1]),
        ("wikipedia", "transductive", [0.1, 0.1]),
        ("wikipedia", "inductive", [0.3, 0.1, 0.1]),
        ("wikipedia", "transductive", [0.3, 0.1, 0.1]),
        # add more combinations as needed
    ],
    ids=lambda vals: f"{vals[0]}-{vals[1]}",
)
def test_train(dataset, mode, split_ratio):
    """
    Integration test: run a short training job for given dataset/mode/split_ratio.

    Run with:
        pytest -m integration
    """
    cmd = [
        sys.executable, "-m", "tgx.cli.train",
        "--config", "tgx/baselines/simple/gcn-mooc-lp.yaml",
        "fit",
        "--trainer.max_epochs=1",
        "--trainer.logger=False",
        f"--data.mode={mode}",
        f"--data.dataset={dataset}",
        f"--data.split_ratio={split_ratio}",

    ]

    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=600)
    # Print logs so pytest captures them
    print(proc.stdout)
    if proc.stderr:
        print(proc.stderr, file=sys.stderr)

    assert proc.returncode == 0, (
        f"Training process exited with code {proc.returncode}\n"
        f"stdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}"
    )