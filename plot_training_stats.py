import argparse
from pathlib import Path
import time

import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


PLOTS = [
    ("Loss/value", "01_loss_value.png", "Value Loss"),
    ("Loss/surrogate", "02_loss_surrogate.png", "Surrogate Loss"),
    ("Loss/entropy", "03_loss_entropy.png", "Entropy Loss"),
    ("Train/mean_reward", "04_train_mean_reward.png", "Mean Reward"),
    ("Episode/xy_dist_mean", "05_xy_dist_mean.png", "XY Distance Mean"),
    ("Episode/z_gap_mean", "06_z_gap_mean.png", "Z Gap Mean"),
    ("Episode/contact_force_mean", "07_contact_force_mean.png", "Contact Force Mean"),
]

DONE_PLOTS = [
    ("Episode/success_done_count", "success"),
    ("Episode/too_far_done_count", "too_far"),
    ("Episode/timeout_done_count", "timeout"),
]


def moving_average(values, window: int):
    if window <= 1 or len(values) < window:
        return values
    out = []
    acc = 0.0
    for i, v in enumerate(values):
        acc += v
        if i >= window:
            acc -= values[i - window]
        denom = min(i + 1, window)
        out.append(acc / denom)
    return out


def load_scalars(event_file: Path):
    ea = EventAccumulator(str(event_file))
    ea.Reload()
    return ea


def plot_single(tag, filename, title, ea, out_dir: Path, smooth: int):
    tags = ea.Tags().get("scalars", [])
    if tag not in tags:
        return False
    events = ea.Scalars(tag)
    x = [e.step for e in events]
    y = [e.value for e in events]
    y_s = moving_average(y, smooth)

    plt.figure(figsize=(8, 5))
    plt.plot(x, y, alpha=0.3, label="raw")
    plt.plot(x, y_s, linewidth=2, label=f"smooth={smooth}")
    plt.xlabel("Iteration")
    plt.ylabel(tag)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / filename, dpi=150)
    plt.close()
    return True


def plot_done_counts(ea, out_dir: Path, smooth: int):
    tags = ea.Tags().get("scalars", [])
    available = [(tag, label) for tag, label in DONE_PLOTS if tag in tags]
    if not available:
        return False

    plt.figure(figsize=(8, 5))
    for tag, label in available:
        events = ea.Scalars(tag)
        x = [e.step for e in events]
        y = [e.value for e in events]
        y_s = moving_average(y, smooth)
        plt.plot(x, y_s, linewidth=2, label=label)
    plt.xlabel("Iteration")
    plt.ylabel("Count")
    plt.title("Done Counts")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "08_done_counts.png", dpi=150)
    plt.close()
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--event_file", type=str, required=True)
    parser.add_argument("--smooth", type=int, default=5)
    parser.add_argument("--output_dir", type=str, default="debug_outputs/training_plots")
    args = parser.parse_args()

    event_file = Path(args.event_file)
    if not event_file.exists():
        raise FileNotFoundError(f"Event file not found: {event_file}")

    base_out = Path(args.output_dir)
    if not base_out.is_absolute():
        base_out = Path(__file__).resolve().parent / base_out
    out_dir = base_out / time.strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    ea = load_scalars(event_file)
    created = []

    for tag, filename, title in PLOTS:
        if plot_single(tag, filename, title, ea, out_dir, args.smooth):
            created.append(filename)

    if plot_done_counts(ea, out_dir, args.smooth):
        created.append("08_done_counts.png")

    summary = {
        "event_file": str(event_file),
        "smooth": args.smooth,
        "created_files": created,
        "available_scalar_tags": ea.Tags().get("scalars", []),
    }
    (out_dir / "summary.json").write_text(__import__("json").dumps(summary, indent=2), encoding="utf-8")

    print(f"Saved {len(created)} plot(s) to: {out_dir}")


if __name__ == "__main__":
    main()
