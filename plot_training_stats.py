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
    ("Episode/pre_align_gate_mean", "09_pre_align_gate_mean.png", "Pre-contact Alignment Gate"),
    ("Episode/soft_gate_mean", "10_soft_gate_mean.png", "Insertion Soft Gate"),
    ("Episode/precontact_xy_reward", "11_precontact_xy_reward.png", "Pre-contact XY Reward"),
    ("Episode/precontact_z_reward", "12_precontact_z_reward.png", "Pre-contact Z Reward"),
    ("Episode/misaligned_downward_penalty", "13_misaligned_downward_penalty.png", "Misaligned Downward Action Penalty"),
    (
        "Episode/misaligned_downward_progress_penalty",
        "14_misaligned_downward_progress_penalty.png",
        "Misaligned Downward Progress Penalty",
    ),
    ("Episode/fast_downward_penalty", "15_fast_downward_penalty.png", "Fast Downward Penalty"),
    ("Episode/action_z_mean", "16_action_z_mean_executed.png", "Executed Smoothed Action Z Mean"),
    ("Episode/action_downward_mean", "17_action_downward_mean_executed.png", "Executed Downward Action Mean"),
    ("Episode/action_xy_norm_mean", "18_action_xy_norm_mean_executed.png", "Executed Action XY Norm Mean"),
    ("Episode/raw_action_z_mean", "19_raw_action_z_mean.png", "Raw Actor Action Z Mean"),
    ("Episode/raw_action_z_near_neg1_ratio", "20_raw_action_z_near_neg1_ratio.png", "Raw Actor Z Near -1 Ratio"),
    ("Episode/raw_action_xy_norm_mean", "21_raw_action_xy_norm_mean.png", "Raw Actor Action XY Norm Mean"),
]

COMBINED_PLOTS = [
    (
        [("Episode/xy_dist_mean", "xy_dist"), ("Episode/z_gap_mean", "z_gap")],
        "22_xy_vs_z_gap.png",
        "XY Distance vs Z Gap",
        "Meters",
    ),
    (
        [("Episode/precontact_xy_reward", "xy_reward"), ("Episode/precontact_z_reward", "z_reward")],
        "23_precontact_xy_vs_z_reward.png",
        "Pre-contact XY Reward vs Z Reward",
        "Reward",
    ),
    (
        [
            ("Episode/misaligned_downward_penalty", "action_penalty"),
            ("Episode/misaligned_downward_progress_penalty", "progress_penalty"),
            ("Episode/fast_downward_penalty", "fast_penalty"),
        ],
        "24_downward_penalties.png",
        "Downward Penalty Components",
        "Penalty",
    ),
    (
        [("Episode/raw_action_z_mean", "raw_z"), ("Episode/action_z_mean", "executed_z")],
        "25_raw_vs_executed_action_z.png",
        "Raw vs Executed Action Z Mean",
        "Action",
    ),
    (
        [("Episode/raw_action_xy_norm_mean", "raw_xy_norm"), ("Episode/action_xy_norm_mean", "executed_xy_norm")],
        "26_raw_vs_executed_action_xy_norm.png",
        "Raw vs Executed Action XY Norm",
        "Action Norm",
    ),
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


def plot_combined(series_defs, filename, title, ylabel, ea, out_dir: Path, smooth: int):
    tags = ea.Tags().get("scalars", [])
    available = [(tag, label) for tag, label in series_defs if tag in tags]
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
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / filename, dpi=150)
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

    for series_defs, filename, title, ylabel in COMBINED_PLOTS:
        if plot_combined(series_defs, filename, title, ylabel, ea, out_dir, args.smooth):
            created.append(filename)

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
