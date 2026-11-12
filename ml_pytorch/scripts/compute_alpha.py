#!/usr/bin/env python
"""Compute the alpha factor (N_ttbar / N_data) per region from PocketCoffea
outputs, for the multiclass QCD-morphing weight in convert_to_onnx.py.

alpha_R is needed in the morphing weight
    w = relu(r_0 - alpha * r_2) / relu(r_1 - alpha_den * r_3)
because the classifier is trained with all classes balanced to equal total
weight, so each ttbar output node is normalised as if ttbar were 100% of the
data. alpha_R scales it back to its physical size:
    alpha_R = (ttbar MC yield in R) / (data yield in R).
Pass the numerator-region row to convert_to_onnx.py as --alpha and the
denominator-region row as --alpha-den. For the morphing along TXbb (default),
the numerator is boosted_signal_sideband_region_C and the denominator
boosted_control_sideband_region_A; for the morphing along the mass, the
numerator is boosted_control_region_B.

Usage:
    python compute_alpha.py <coffea_output_dir>
    ml_alpha -c <training config> [--json alpha.json] [--print]
With -c, the coffea output directory is the first data_dirs of the config,
the numerator region is the region of the class with lbl 0 (data,
numerator) and the denominator region that of the class with lbl 1 (data,
denominator).
"""
import argparse
import glob
import json
import math
import os
import sys

from coffea.util import load

REGIONS = [
    "boosted_signal_region_D",
    "boosted_signal_sideband_region_C",
    "boosted_control_region_B",
    "boosted_control_sideband_region_A",
]


def walk(node):
    """Sum the innermost non-dict 'nominal' leaves of a region sub-tree."""
    if isinstance(node, dict):
        if "nominal" in node and not isinstance(node["nominal"], dict):
            return node["nominal"]
        return sum(walk(v) for v in node.values())
    return 0.0


def compute_alpha(in_dir, regions=REGIONS):
    """{region: (N_data, N_ttbar, alpha, MC-stat error of alpha)}."""
    data_files = sorted(glob.glob(os.path.join(in_dir, "output_DATA_*.coffea")))
    ttbar_files = sorted(glob.glob(os.path.join(in_dir, "output_TT*.coffea")))
    if not data_files or not ttbar_files:
        sys.exit(f"no DATA/TT coffea files found in {in_dir}")

    # data: event weight == 1, and the sumw accumulator is not filled for data,
    # so the raw cutflow count is the yield.
    # regions not in the outputs are skipped
    available = set(load(data_files[0])["cutflow"].keys())
    missing = [r for r in regions if r not in available]
    if missing:
        print(f"Regions not in the outputs, skipped: {missing}", file=sys.stderr)
    regions = [r for r in regions if r in available]

    data_n = {r: 0.0 for r in regions}
    for f in data_files:
        o = load(f)
        for r in regions:
            data_n[r] += walk(o["cutflow"][r])

    # ttbar: use the lumi-normalised sumw (and sumw2 for the MC-stat error).
    tt_sumw = {r: 0.0 for r in regions}
    tt_sumw2 = {r: 0.0 for r in regions}
    for f in ttbar_files:
        o = load(f)
        for r in regions:
            tt_sumw[r] += walk(o["sumw"][r])
            tt_sumw2[r] += walk(o["sumw2"][r])

    return {
        r: (
            data_n[r],
            tt_sumw[r],
            tt_sumw[r] / data_n[r],
            math.sqrt(tt_sumw2[r]) / data_n[r],
        )
        for r in regions
    }


def print_table(alphas):
    print(f"{'region':38s} {'N_data':>10s} {'N_ttbar':>12s} {'alpha':>9s} {'MCstat':>9s}")
    for r, (n_data, n_ttbar, a, da) in alphas.items():
        print(f"{r:38s} {n_data:10.0f} {n_ttbar:12.1f} {a:9.4f} {da:9.4f}")


def morphing_regions(cfg):
    """(numerator, denominator) regions of the data classes (lbl 0 and 1) of
    a multiclass morphing config."""
    classes = cfg.get("classes", None)
    if not classes:
        raise ValueError("The config has no classes: alpha is for the multiclass morphing")
    by_lbl = {int(c["lbl"]): c for c in classes.values()}
    if 0 not in by_lbl or 1 not in by_lbl:
        raise ValueError("The config needs the classes with lbl 0 and 1 (data)")
    return list(by_lbl[0]["region"])[0], list(by_lbl[1]["region"])[0]


def main():
    parser = argparse.ArgumentParser(
        description="Compute alpha = N_ttbar / N_data per region from PocketCoffea outputs"
    )
    parser.add_argument(
        "input",
        nargs="?",
        default=None,
        help="Directory with the PocketCoffea outputs (default: first data_dirs of the config, or .)",
    )
    parser.add_argument(
        "-c",
        "--config",
        default=None,
        help="Training config: data directory and numerator/denominator regions of the morphing",
    )
    parser.add_argument(
        "--json",
        default=None,
        help="Save the table and the alpha of the numerator and denominator regions to this file",
    )
    parser.add_argument(
        "--print",
        dest="print_values",
        action="store_true",
        help="Print only '<alpha numerator> <alpha denominator>' (needs --config)",
    )
    args = parser.parse_args()

    regions = list(REGIONS)
    num_region = den_region = None
    in_dir = args.input
    if args.config:
        from omegaconf import OmegaConf

        cfg = OmegaConf.load(args.config)
        num_region, den_region = morphing_regions(cfg)
        regions = list(dict.fromkeys(regions + [num_region, den_region]))
        if in_dir is None:
            in_dir = list(cfg.data_dirs)[0]
    elif args.print_values:
        parser.error("--print needs --config")
    in_dir = in_dir or "."

    alphas = compute_alpha(in_dir, regions)
    for r in [num_region, den_region]:
        if r is not None and r not in alphas:
            sys.exit(f"Region {r} of the morphing not found in the outputs in {in_dir}")
    if args.print_values:
        print(f"{alphas[num_region][2]} {alphas[den_region][2]}")
    else:
        print_table(alphas)
        if num_region:
            print(f"\nalpha numerator ({num_region}): {alphas[num_region][2]:.6g}")
            print(f"alpha denominator ({den_region}): {alphas[den_region][2]:.6g}")

    if args.json:
        out = {
            "input": os.path.abspath(in_dir),
            "regions": {
                r: dict(zip(["n_data", "n_ttbar", "alpha", "alpha_mc_stat"], v))
                for r, v in alphas.items()
            },
        }
        if num_region:
            out.update(
                numerator_region=num_region,
                denominator_region=den_region,
                alpha=alphas[num_region][2],
                alpha_den=alphas[den_region][2],
            )
        with open(args.json, "w") as f:
            json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
