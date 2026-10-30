"""Merge the many small parquet chunks saved by PocketCoffea into a single
parquet file per leaf directory (e.g. <dataset>/<region>/nominal).

Reading thousands of small files from dCache (/pnfs) is dominated by the
per-file latency, so merging them once (ideally into /work or a local disk)
makes the loading of the training dataset much faster. The directory
structure is kept, so the output directory can be used directly as
``parquet_dir`` in the training config.

The files are merged in the same order used to load them in training
(list_parquet_files), so the events order is unchanged.

With --copy-to, each merged file is then copied to another directory (e.g.
on /pnfs, where the files cannot be written directly), the copy is checked
(size, adler32 checksum and number of events) and the file in the output
directory is removed. The directory given with --copy-to is then the one to
use as ``parquet_dir``.

Usage:
    ml_merge_parquet -i <data_dir or parquet_files dir> -o <output_dir> [-j 32]
                     [--copy-to <final_dir>]
"""

import argparse
import json
import logging
import os
import shutil
import zlib
from concurrent.futures import ThreadPoolExecutor

import pyarrow as pa
import pyarrow.parquet as pq

from ml_pytorch.utils.dataset import list_parquet_files, root_to_local

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def find_leaf_dirs(path):
    """Directories directly containing parquet files (sorted)."""
    leaves = []
    for root, dirs, filenames in os.walk(path, followlinks=True):
        dirs[:] = sorted(d for d in dirs if not d.startswith((".", "_")))
        if any(
            f.endswith(".parquet") and not f.startswith((".", "_")) for f in filenames
        ):
            leaves.append(root)
    return leaves


def merge_dir(in_dir, out_file, n_threads, files_per_step):
    files = [f for f in list_parquet_files(in_dir) if f.endswith(".parquet")]
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    tmp_file = out_file + ".tmp"
    writer = None
    n_rows = 0
    with ThreadPoolExecutor(n_threads) as ex:
        # read the files in parallel in steps to limit the memory usage,
        # map keeps the order of the files
        for i in range(0, len(files), files_per_step):
            tables = list(ex.map(pq.read_table, files[i : i + files_per_step]))
            if writer is None:
                schema = tables[0].schema
                writer = pq.ParquetWriter(tmp_file, schema)
            tables = [t.cast(schema) for t in tables if t.num_rows > 0]
            if tables:
                # one large row group per step instead of one per small chunk
                table = pa.concat_tables(tables)
                writer.write_table(table, row_group_size=table.num_rows)
                n_rows += table.num_rows
    if writer is not None:
        writer.close()
        os.replace(tmp_file, out_file)
    logger.info(f"Merged {len(files)} files ({n_rows} events) into {out_file}")


def adler32(path, block_size=16 * 1024 * 1024):
    value = 1
    with open(path, "rb") as f:
        while block := f.read(block_size):
            value = zlib.adler32(block, value)
    return value


def copy_and_remove(src_file, dst_file, stop_dir):
    """Copy src_file to dst_file, check the copy and remove src_file and its
    parent directories left empty (up to stop_dir excluded).

    If the copy is not correct, dst_file is removed, src_file is kept and an
    error is raised."""
    os.makedirs(os.path.dirname(dst_file), exist_ok=True)
    # files on dCache cannot be overwritten, they have to be removed first
    if os.path.exists(dst_file):
        os.remove(dst_file)
    logger.info(f"Copying {src_file} to {dst_file}")
    shutil.copyfile(src_file, dst_file)

    errors = []
    src_size, dst_size = os.path.getsize(src_file), os.path.getsize(dst_file)
    if src_size != dst_size:
        errors.append(f"size {dst_size} != {src_size}")
    else:
        src_sum, dst_sum = adler32(src_file), adler32(dst_file)
        if src_sum != dst_sum:
            errors.append(f"adler32 {dst_sum:08x} != {src_sum:08x}")
        src_rows = pq.ParquetFile(src_file).metadata.num_rows
        dst_rows = pq.ParquetFile(dst_file).metadata.num_rows
        if src_rows != dst_rows:
            errors.append(f"events {dst_rows} != {src_rows}")
    if errors:
        os.remove(dst_file)
        raise RuntimeError(
            f"Copy of {src_file} to {dst_file} failed ({', '.join(errors)}): "
            f"the copy was removed and the original file kept"
        )
    logger.info(f"Copy checked (size, adler32, events), removing {src_file}")

    os.remove(src_file)
    parent = os.path.dirname(os.path.abspath(src_file))
    stop_dir = os.path.abspath(stop_dir)
    while parent != stop_dir and parent.startswith(stop_dir) and not os.listdir(parent):
        os.rmdir(parent)
        parent = os.path.dirname(parent)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="PocketCoffea output directory (with config.json, the parquet "
        "location is read from save_chunk) or the parquet files directory",
    )
    parser.add_argument(
        "-o", "--output", required=True, help="Output directory of the merged files"
    )
    parser.add_argument(
        "-j", "--threads", type=int, default=32, help="Number of reading threads"
    )
    parser.add_argument(
        "--files-per-step",
        type=int,
        default=256,
        help="Number of files kept in memory at the same time",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing merged files"
    )
    parser.add_argument(
        "--copy-to",
        default=None,
        help="Copy each merged file to this directory (e.g. on /pnfs), keeping "
        "the directory structure, check the copy (size, adler32, events) and "
        "remove the file from the output directory",
    )
    args = parser.parse_args()

    in_path = args.input
    config_file = os.path.join(in_path, "config.json")
    if os.path.isfile(config_file):
        with open(config_file) as f:
            in_path = root_to_local(
                json.load(f)["workflow"]["workflow_options"]["save_chunk"]
            )
    logger.info(f"Merging the parquet files in {in_path}")

    for leaf in find_leaf_dirs(in_path):
        rel_file = os.path.join(os.path.relpath(leaf, in_path), "merged.parquet")
        out_file = os.path.join(args.output, rel_file)
        if args.copy_to is None:
            if os.path.exists(out_file) and not args.overwrite:
                logger.info(f"Skipping existing {out_file}")
                continue
            merge_dir(leaf, out_file, args.threads, args.files_per_step)
        else:
            final_file = os.path.join(args.copy_to, rel_file)
            if os.path.exists(final_file) and not args.overwrite:
                logger.info(f"Skipping existing {final_file}")
                continue
            # a merged file left by a previous run whose copy failed is reused
            if not os.path.exists(out_file) or args.overwrite:
                merge_dir(leaf, out_file, args.threads, args.files_per_step)
            else:
                logger.info(f"Reusing existing {out_file}")
            # done after each merge, to not keep all the files in the output dir
            copy_and_remove(out_file, final_file, args.output)


if __name__ == "__main__":
    main()
