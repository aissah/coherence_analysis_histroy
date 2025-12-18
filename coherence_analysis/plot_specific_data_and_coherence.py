"""Script to plot raw data.

This script reads raw data from a specified directory, selects a time range and
channel range, and generates a plot of the raw data. The plot is saved to a
designated results directory.

Usage:
    python plot_raw_data.py <data_path> <time_range> [-ch <channel_range>]
                            [-ds <channel_offset>] [-r <result_path>]
Arguments:
    data_path (str): Path to the directory containing the data files.
    time_range (str): Range of time (in Python list format,
                    e.g., "[start_time, end_time]").
    -ch, --channel_range (str, optional): Range of channels to plot.
                    (in Python list format, e.g., "[start_channel,
                    end_channel]"). Default is "(0, ...)".
    -ds, --channel_offset (int, optional): Channels to skip in between.
                    Default is 1.
    -r, --result_path (str, optional): Directory to save results. Default is
                    "../data/results".
Example:
    python plot_raw_data.py ./data "[2023-01-01 00:00:00, 2023-01-02 00:00:00]"
    -ch "[0, 100]" -ds 2 -r ./results
"""

import argparse
import os
from datetime import datetime, timedelta

import dascore as dc
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import utils.utils as func
from matplotlib.lines import Line2D


def parse_args():
    """Parse command line arguments.

    Raises
    ------
    ValueError
        Raise error if the method selected is not available.
    """
    # Initialize the parser
    parser = argparse.ArgumentParser(description="Plot Specific Data")

    # Add arguments
    parser.add_argument(
        "data_path",
        type=str,
        help="Path to the directory containing the data files",
    )
    parser.add_argument(
        "-ch",
        "--channel_range",
        type=str,
        help="Range of channels to use for coherence analysis "
        " (in Python list format)",
        default="(0, ...)",
    )
    parser.add_argument(
        "-ds",
        "--channel_offset",
        type=int,
        help="Channels to skip in between",
        default=1,
    )
    parser.add_argument(
        "-r",
        "--result_path",
        type=str,
        help="Directory to save results",
        default=os.path.join(
            os.path.dirname(__file__), os.pardir, "data/results"
        ),
    )

    return parser.parse_args()


if __name__ == "__main__":
    # record start time
    start_time = datetime.now()

    # Parse arguments
    args = parse_args()
    print(f"Arguments read from command line: {args}", flush=True)

    # Define time ranges and channel ranges
    time_length = timedelta(seconds=120)
    big_signal_start_time = datetime.strptime(
        "03/14/16 08:35:18", "%m/%d/%y %H:%M:%S"
    )
    # small_signal_start_time = datetime.strptime(
    #     "03/14/16 08:30:18", "%m/%d/%y %H:%M:%S"
    # )
    small_signal_start_time = datetime.strptime(
        "03/14/16 04:02:18", "%m/%d/%y %H:%M:%S"
    )  # 2 different events
    big_time_range = [
        big_signal_start_time,
        big_signal_start_time + time_length,
    ]
    small_time_range = [
        small_signal_start_time,
        small_signal_start_time + time_length,
    ]
    channel_range = [5000, 6500]

    frequencies = [7, 9, 17]  # frequencies to plot traces for
    big_event_times = [
        big_signal_start_time + timedelta(seconds=et) for et in [58, 69, 113]
    ]
    # small_event_times = [
    #     small_signal_start_time + timedelta(seconds=et) for et in [106]
    # ]
    small_event_times = [
        small_signal_start_time + timedelta(seconds=et) for et in [32, 43]
    ]  # 2 different events

    # Read the data
    print("Reading data...", flush=True)
    spool = dc.spool(args.data_path)
    big_signal_spool = spool.select(time=big_time_range, samples=True)
    small_signal_spool = spool.select(time=small_time_range, samples=True)

    # Get channel dimension and coordinates
    dims = spool[0].dims
    print(f"The data has the following dimensions: {dims}")
    print(
        f"""Channels will be grouped based on the '{dims[1]}'
            dimension. If another dimension is desired, use the
            method, '_set_channel_dim()' to set it.""",
        flush=True,
    )
    channel_dim = dims[1]
    channels = np.arange(
        channel_range[0],
        channel_range[1],
        args.channel_offset,
        dtype=int,
    )
    distance_coords = spool[0].coords.get_array(channel_dim)
    distance_array = distance_coords[channels]

    # Set plot parameters
    dpi = 600
    label_size = 16
    tick_size = 14
    legend_size = 12
    sns.set_theme(style="ticks", context="paper", font_scale=1.2)

    # ---- figure grid ----
    fig = plt.figure(figsize=(12, 7), dpi=dpi)

    gs = fig.add_gridspec(
        nrows=2,
        ncols=3,
        width_ratios=[20, 20, 1],  # last column = shared colorbar
        height_ratios=[3, 1],
        hspace=0.05,
        wspace=0.12,
    )

    # ---- axes ----
    ax_img_a = fig.add_subplot(gs[0, 0])
    ax_line_a = fig.add_subplot(gs[1, 0], sharex=ax_img_a)

    ax_img_b = fig.add_subplot(gs[0, 1])
    ax_line_b = fig.add_subplot(gs[1, 1], sharex=ax_img_b)

    cax = fig.add_subplot(gs[0, 2])

    # ---- imshow (use SAME vmin/vmax) ----
    vmax = 0.002
    vmin = -0.002
    data_array = np.concatenate(
        [
            d.select(**{channel_dim: distance_array}).data
            for d in big_signal_spool
        ],
        axis=0,
    )
    ima = ax_img_a.imshow(
        data_array.T,
        extent=[
            big_time_range[0],
            big_time_range[1],
            distance_array[-1],
            distance_array[0],
        ],
        origin="lower",
        aspect="auto",
        cmap="RdBu_r",
        vmin=vmin,
        vmax=vmax,
    )
    detection_significance_big, eig_estimates_big, _ = func.coherence(
        data_array.T,
        1,
        0,
        sample_interval=1 / 1000,
        method="qr",
    )

    # ---- second image plot ----
    data_array = np.concatenate(
        [
            d.select(**{channel_dim: distance_array}).data
            for d in small_signal_spool
        ],
        axis=0,
    )
    imb = ax_img_b.imshow(
        data_array.T,
        extent=[
            small_time_range[0],
            small_time_range[1],
            distance_array[-1],
            distance_array[0],
        ],
        origin="lower",
        aspect="auto",
        cmap="RdBu_r",
        vmin=vmin,
        vmax=vmax,
    )
    detection_significance_small, eig_estimates_small, _ = func.coherence(
        data_array.T,
        1,
        0,
        sample_interval=1 / 1000,
        method="qr",
    )

    # ---- axis formatting ----
    ax_img_a.set_ylabel("Channel")
    ax_img_a.tick_params(labelbottom=False)
    ax_img_b.tick_params(labelbottom=False)
    ax_img_b.set_yticklabels([])

    ax_line_a.set_xlabel("Time")
    ax_line_a.set_ylabel("Normalized Eigenvalue")
    ax_line_b.set_xlabel("Time")
    ax_line_b.set_yticklabels([])

    # ---- line plots ----
    time_ax = [
        timedelta(seconds=a) + big_signal_start_time
        for a in range(eig_estimates_big.shape[1])
    ]
    for a in frequencies:
        ax_line_a.plot(time_ax, eig_estimates_big[a, :] / 500, linewidth=2.5)

    time_ax = [
        timedelta(seconds=a) + small_signal_start_time
        for a in range(eig_estimates_small.shape[1])
    ]
    for a in frequencies:
        ax_line_b.plot(time_ax, eig_estimates_small[a, :] / 500, linewidth=2.5)

    # ---- vertical event lines (left axes) ----
    for et in big_event_times:
        for ax in (ax_img_a, ax_line_a):
            ax.axvline(et, color="k", linestyle="--", linewidth=1.3, alpha=0.7)

    # ---- vertical event lines (right axes) ----
    for et in small_event_times:
        for ax in (ax_img_b, ax_line_b):
            ax.axvline(et, color="k", linestyle="--", linewidth=1.3, alpha=0.7)

    # ---- shared colorbar ----
    cbar = fig.colorbar(ima, cax=cax, pad=0.02)
    cbar.set_label("Strain Rate")

    # -----------------------
    # Combined legend (below colorbar)
    # -----------------------
    legend_handles = [
        Line2D([0], [0], color="C0", lw=1.8, label=f"{frequencies[0]} Hz"),
        Line2D([0], [0], color="C1", lw=1.8, label=f"{frequencies[1]} Hz"),
        Line2D([0], [0], color="C2", lw=1.8, label=f"{frequencies[2]} Hz"),
        Line2D(
            [0], [0], color="k", lw=1.5, linestyle="--", label="Catalog Event"
        ),
    ]

    cax.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(-2.0, -0.06),
        frameon=False,
        # fontsize=9,
        handlelength=2.5,
        ncol=1,
    )

    # ---- panel labels ----
    for ax, label in zip([ax_img_a, ax_img_b], ["(a)", "(b)"]):
        ax.text(
            0.01,
            0.98,
            label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            # fontsize=12,
            fontweight="bold",
        )

    # ---- x-axis date formatting ----
    for ax in [ax_line_a, ax_line_b]:
        locator = mdates.AutoDateLocator()
        ax.xaxis.set_major_locator(locator)
        formatter = mdates.ConciseDateFormatter(locator)
        ax.xaxis.set_major_formatter(formatter)

    # fig.tight_layout()
    print("Saving plot...", flush=True)
    os.makedirs(args.result_path, exist_ok=True)
    plt.savefig(os.path.join(args.result_path, "combined_data_plot.png"))
