#!/usr/bin/env python3
"""
Resource monitor of the jobs of run_training.sh (started automatically by it,
see --no-monitor / --monitor-interval there).

Every --interval seconds, until the process --pid ends or the monitor gets
SIGTERM/SIGINT, it samples:
  * the CPU (cores used) and the memory of the whole job, from the Slurm cgroup
    of the job if there is one, otherwise from the processes of --pid;
  * the CPU, memory, GPU memory and GPU SM utilisation of every training (the
    process tree of each ml_train, named after its output directory, e.g.
    run00 or fold_3) and of the other steps (kfold_prepare, ml_kfold_merge,
    ml_onnx, ml_alpha);
  * the utilisation, memory and power of the GPUs allocated to the job
    (CUDA_VISIBLE_DEVICES), with nvidia-smi.

Outputs in --outdir:
  <tag>_job.csv        job CPU cores / memory vs time
  <tag>_gpu.csv        utilisation / memory / trainings of every GPU vs time
  <tag>_trainings.csv  CPU / memory / GPU of every training vs time
  <tag>_summary.txt    time occupancy, usage vs allocation, per-training and
  <tag>_summary.json   per-GPU numbers and suggested resources; rewritten every
                       --summary-every seconds, so it exists also if the job
                       is killed

Usage:
  python resource_monitor.py --pid PID --outdir DIR --tag TAG [--interval 10]
      [--phase-file FILE] [--cpus N] [--mem-per-cpu MB] [--summary-every 300]

The phase file has lines "<unix time> <phase> [exit code]", written by
run_training.sh at the start of every step ("end <exit code>" at the end), used
for the time spent in every phase.
"""

import argparse
import csv
import json
import math
import os
import shutil
import signal
import socket
import subprocess
import time
from collections import defaultdict

import psutil

MB = 1024 * 1024
STEP_NAMES = ("ml_kfold_merge", "ml_onnx", "ml_alpha")
NON_TRAININGS = ("other", "kfold_prepare", "ml_train") + STEP_NAMES


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--pid", type=int, required=True, help="Root process to monitor")
    p.add_argument("--outdir", required=True, help="Output directory")
    p.add_argument("--tag", required=True, help="Prefix of the output files")
    p.add_argument("--interval", type=float, default=10.0, help="Seconds between samples")
    p.add_argument("--summary-every", type=float, default=300.0,
                   help="Seconds between rewrites of the summary")
    p.add_argument("--phase-file", default=None, help="File with the phase markers")
    p.add_argument("--cpus", type=int, default=None,
                   help="Allocated CPUs (default: SLURM_CPUS_PER_TASK)")
    p.add_argument("--mem-per-cpu", type=float, default=None,
                   help="Allocated memory per CPU in MB (default: SLURM_MEM_PER_CPU)")
    return p.parse_args()


# ─── Helpers ─────────────────────────────────────────────────────────────────
def run(cmd, timeout=20):
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout if out.returncode == 0 else None


def to_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def slurm_time_to_s(s):
    """Slurm time (MM, MM:SS, HH:MM:SS, D-HH, D-HH:MM, D-HH:MM:SS) -> seconds."""
    if not s or s.strip().upper() in ("UNLIMITED", "INFINITE", "NOT_SET", "N/A"):
        return None
    try:
        s = s.strip()
        days = 0
        if "-" in s:
            d, s = s.split("-", 1)
            days = int(d)
        parts = [int(x) for x in s.split(":")]
        if days:
            parts += [0] * (3 - len(parts))
        elif len(parts) == 1:
            parts = [0, parts[0], 0]
        elif len(parts) == 2:
            parts = [0] + parts
        h, m, sec = parts
        return days * 86400 + h * 3600 + m * 60 + sec
    except ValueError:
        return None


def fmt_dur(s):
    if s is None:
        return "n/a"
    s = int(round(s))
    return f"{s // 3600}h{(s % 3600) // 60:02d}m{s % 60:02d}s"


def fmt_slurm_time(s):
    s = int(s)
    d, s = divmod(s, 86400)
    t = f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"
    return f"{d}-{t}" if d else t


def fmt(x, digits=1, unit=""):
    return "n/a" if x is None else f"{x:.{digits}f}{unit}"


def wmean(pairs):
    """Weighted mean of (weight, value) pairs."""
    pairs = [(w, v) for w, v in pairs if v is not None and w > 0]
    tot = sum(w for w, _ in pairs)
    return sum(w * v for w, v in pairs) / tot if tot > 0 else None


def percentile(vals, q):
    v = sorted(x for x in vals if x is not None)
    if not v:
        return None
    return v[min(len(v) - 1, max(0, int(round(q / 100 * (len(v) - 1)))))]


def vmax(vals):
    v = [x for x in vals if x is not None]
    return max(v) if v else None


# ─── Slurm cgroup of the job ─────────────────────────────────────────────────
def job_cgroup(pid):
    """Directory of the job_<id> cgroup (v2) of pid, None outside Slurm."""
    try:
        with open(f"/proc/{pid}/cgroup") as f:
            lines = [l.strip() for l in f if l.startswith("0::")]
    except OSError:
        return None
    if not lines:
        return None
    path = "/sys/fs/cgroup" + lines[0][3:]
    while path not in ("/sys/fs/cgroup", "/"):
        if os.path.basename(path).startswith("job_") and os.path.isfile(
            os.path.join(path, "cpu.stat")
        ):
            return path
        path = os.path.dirname(path)
    return None


def read_flat(path):
    out = {}
    try:
        with open(path) as f:
            for line in f:
                k, _, v = line.partition(" ")
                out[k] = int(v)
    except (OSError, ValueError):
        pass
    return out


def read_int(path):
    try:
        with open(path) as f:
            v = f.read().strip()
        return None if v == "max" else int(v)
    except (OSError, ValueError):
        return None


# ─── GPUs ────────────────────────────────────────────────────────────────────
class GpuReader:
    def __init__(self):
        self.ok = shutil.which("nvidia-smi") is not None
        self.pmon_ok = self.ok
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
        self.visible = None if cvd is None else [g.strip() for g in cvd.split(",") if g.strip()]

    def gpus(self):
        """Allocated GPUs: index (physical), uuid, name, util %, memory MB, power W."""
        if not self.ok or self.visible == []:
            return []
        out = run([
            "nvidia-smi",
            "--query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total,power.draw",
            "--format=csv,noheader,nounits",
        ])
        if out is None:
            return []
        rows = []
        for line in out.strip().splitlines():
            f = [x.strip() for x in line.split(",")]
            if len(f) < 7:
                continue
            rows.append(dict(index=f[0], uuid=f[1], name=f[2], util=to_float(f[3]),
                             mem=to_float(f[4]), mem_total=to_float(f[5]),
                             power=to_float(f[6])))
        if self.visible is None:
            return rows
        sel = [g for g in rows if g["index"] in self.visible or g["uuid"] in self.visible]
        # nvidia-smi restricted to the job GPUs and renumbered
        if not sel and len(rows) == len(self.visible):
            sel = rows
        return sel

    def procs(self):
        """pid -> [(gpu uuid, used memory MB)] of all the compute processes."""
        res = defaultdict(list)
        if not self.ok:
            return res
        out = run([
            "nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory",
            "--format=csv,noheader,nounits",
        ])
        for line in (out or "").strip().splitlines():
            f = [x.strip() for x in line.split(",")]
            if len(f) >= 3 and f[0].isdigit():
                res[int(f[0])].append((f[1], to_float(f[2])))
        return res

    def pmon(self):
        """pid -> SM utilisation % (nvidia-smi pmon, ~1 s)."""
        if not self.pmon_ok:
            return {}
        out = run(["nvidia-smi", "pmon", "-c", "1", "-s", "u"], timeout=15)
        if out is None:
            self.pmon_ok = False
            return {}
        header, res = None, defaultdict(float)
        for line in out.splitlines():
            if line.startswith("#"):
                if header is None:
                    header = line.lstrip("#").split()
                continue
            f = line.split()
            if not header or "pid" not in header or "sm" not in header or len(f) < len(header) - 1:
                continue
            pid, sm = f[header.index("pid")], to_float(f[header.index("sm")])
            if pid.isdigit() and sm is not None:
                res[int(pid)] += sm
        return res


# ─── Processes ───────────────────────────────────────────────────────────────
def step_key(cmdline):
    """Training / step a process belongs to, from its command line."""
    for i, a in enumerate(cmdline[:3]):
        name = os.path.basename(a)
        if name == "ml_train":
            args = cmdline[i + 1:]
            if "-o" in args and args.index("-o") + 1 < len(args):
                return os.path.basename(os.path.normpath(args[args.index("-o") + 1]))
            return "kfold_prepare" if "--kfold-prepare" in args else "ml_train"
        if name in STEP_NAMES:
            return name
    return None


# ─── Monitor ─────────────────────────────────────────────────────────────────
class Monitor:
    def __init__(self, args):
        self.args = args
        self.root = psutil.Process(args.pid)
        self.me = os.getpid()
        self.cg = job_cgroup(args.pid)
        self.gpu = GpuReader()
        self.t0 = time.time()
        start = to_float(os.environ.get("SLURM_JOB_START_TIME"))
        self.job_start = start if start and start <= self.t0 else self.t0
        self.time_limit = self._time_limit()
        self.alloc_cpus = args.cpus or int(os.environ.get("SLURM_CPUS_PER_TASK", 0)) or None
        self.alloc_mem = self._alloc_mem()
        self.prev_t = None
        self.prev_cg_usage = None
        self.cpu_last = {}  # (pid, create time) -> (key, cpu seconds)
        self.job = []  # job samples
        self.gpu_s = defaultdict(list)  # gpu index -> samples
        self.gpu_info = {}  # gpu index -> (name, memory MB)
        self.train = {}  # key -> statistics
        self.phases = []  # (time, phase, extra)
        self.stop_reason = None
        os.makedirs(args.outdir, exist_ok=True)
        base = os.path.join(args.outdir, args.tag)
        self.f_job, self.w_job = self._csv(base + "_job.csv", [
            "time", "elapsed_s", "phase", "cpu_cores", "mem_needed_MB",
            "mem_cgroup_MB", "rss_sum_MB", "n_trainings"])
        self.f_gpu, self.w_gpu = self._csv(base + "_gpu.csv", [
            "time", "elapsed_s", "gpu", "util_pct", "mem_used_MB", "mem_total_MB",
            "power_W", "n_trainings", "trainings", "job_mem_MB"])
        self.f_tr, self.w_tr = self._csv(base + "_trainings.csv", [
            "time", "elapsed_s", "training", "n_procs", "cpu_cores", "rss_MB",
            "gpus", "gpu_mem_MB", "sm_pct"])
        self.summary_base = base + "_summary"

    @staticmethod
    def _csv(path, header):
        f = open(path, "w", newline="")
        w = csv.writer(f)
        w.writerow(header)
        f.flush()
        return f, w

    def _time_limit(self):
        job = os.environ.get("SLURM_JOB_ID")
        if job:
            out = run(["squeue", "-h", "-j", job, "-o", "%l"], timeout=30)
            lim = slurm_time_to_s(out.strip().splitlines()[0]) if out and out.strip() else None
            if lim:
                return lim
        end, start = to_float(os.environ.get("SLURM_JOB_END_TIME")), to_float(
            os.environ.get("SLURM_JOB_START_TIME"))
        return end - start if end and start else None

    def _alloc_mem(self):
        if self.cg:
            limit = read_int(os.path.join(self.cg, "memory.max"))
            if limit:
                return limit / MB
        mpc = self.args.mem_per_cpu or to_float(os.environ.get("SLURM_MEM_PER_CPU"))
        if mpc and self.alloc_cpus:
            return mpc * self.alloc_cpus
        return to_float(os.environ.get("SLURM_MEM_PER_NODE"))

    def _read_phases(self):
        if not self.args.phase_file:
            return
        phases = []
        try:
            with open(self.args.phase_file) as f:
                for line in f:
                    p = line.split()
                    if len(p) >= 2 and to_float(p[0]) is not None:
                        phases.append((float(p[0]), p[1], " ".join(p[2:])))
        except OSError:
            return
        self.phases = phases

    def _phase_at(self, t):
        cur = "start"
        for pt, name, _ in self.phases:
            if pt <= t:
                cur = name
        return cur

    # ── one sample ──
    def sample(self):
        now = time.time()
        dt = now - self.prev_t if self.prev_t else 0.0
        self.prev_t = now
        elapsed = now - self.job_start
        self._read_phases()
        phase = self._phase_at(now)

        info = {}
        try:
            children = self.root.children(recursive=True)
        except psutil.Error:
            children = []
        for p in children:
            try:
                with p.oneshot():
                    ppid = p.ppid()
                    if p.pid == self.me or ppid == self.me:
                        continue
                    c = p.cpu_times()
                    info[p.pid] = dict(ppid=ppid, key=step_key(p.cmdline()),
                                       ct=p.create_time(), cpu=c.user + c.system,
                                       rss=p.memory_info().rss)
            except psutil.Error:
                continue

        def resolve(pid):
            for _ in range(100):
                if pid not in info:
                    return "other"
                if info[pid]["key"]:
                    return info[pid]["key"]
                pid = info[pid]["ppid"]
            return "other"

        cur = defaultdict(lambda: dict(cpu=0.0, rss=0, n=0, gpus=set(), gmem=0.0, sm=0.0))
        cpu_sum = 0.0
        for pid, d in info.items():
            key = d["key"] or resolve(pid)
            d["key"] = key
            prev = self.cpu_last.get((pid, d["ct"]))
            delta = d["cpu"] - prev[1] if prev else d["cpu"]
            self.cpu_last[(pid, d["ct"])] = (key, d["cpu"])
            cpu_sum += max(delta, 0.0)
            c = cur[key]
            c["cpu"] += max(delta, 0.0)
            c["rss"] += d["rss"]
            c["n"] += 1

        # job CPU and memory
        cores, mem_need, mem_cg = None, None, None
        rss_sum = sum(d["rss"] for d in info.values()) / MB
        if self.cg:
            usage = read_flat(os.path.join(self.cg, "cpu.stat")).get("usage_usec")
            if usage is not None:
                if self.prev_cg_usage is not None and dt > 0:
                    cores = (usage - self.prev_cg_usage) / 1e6 / dt
                self.prev_cg_usage = usage
            mem_cg = (read_int(os.path.join(self.cg, "memory.current")) or 0) / MB
            st = read_flat(os.path.join(self.cg, "memory.stat"))
            if st:
                mem_need = (st.get("anon", 0) + st.get("shmem", 0) + st.get("file_mapped", 0)) / MB
        if cores is None and dt > 0:
            cores = cpu_sum / dt
        if mem_need is None:
            mem_need = rss_sum

        # GPUs
        gpus = self.gpu.gpus()
        gprocs = self.gpu.procs() if gpus else {}
        sm = self.gpu.pmon() if gpus else {}
        uuid2idx = {g["uuid"]: g["index"] for g in gpus}
        gpu_keys = defaultdict(set)
        gpu_jobmem = defaultdict(float)
        for pid, d in info.items():
            for uuid, mem in gprocs.get(pid, []):
                if uuid not in uuid2idx:
                    continue
                idx = uuid2idx[uuid]
                c = cur[d["key"]]
                c["gpus"].add(idx)
                c["gmem"] += mem or 0.0
                c["sm"] += sm.get(pid, 0.0)
                gpu_keys[idx].add(d["key"])
                gpu_jobmem[idx] += mem or 0.0

        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now))
        n_tr = sum(1 for k in cur if k not in NON_TRAININGS)
        self.job.append(dict(t=now, dt=dt, phase=phase, cores=cores, need=mem_need,
                             cg=mem_cg, rss=rss_sum, n_tr=n_tr))
        self.w_job.writerow([ts, f"{elapsed:.0f}", phase, fmt(cores, 2, ""),
                             fmt(mem_need, 0), fmt(mem_cg, 0), fmt(rss_sum, 0), n_tr])
        for g in gpus:
            idx = g["index"]
            self.gpu_info[idx] = (g["name"], g["mem_total"])
            keys = sorted(gpu_keys[idx])
            self.gpu_s[idx].append(dict(dt=dt, util=g["util"], mem=g["mem"],
                                        keys=keys, jobmem=gpu_jobmem[idx], power=g["power"]))
            self.w_gpu.writerow([ts, f"{elapsed:.0f}", idx, fmt(g["util"], 0), fmt(g["mem"], 0),
                                 fmt(g["mem_total"], 0), fmt(g["power"], 1), len(keys),
                                 " ".join(keys), fmt(gpu_jobmem[idx], 0)])
        for key, c in sorted(cur.items()):
            ccores = c["cpu"] / dt if dt > 0 else None
            on_gpu = bool(c["gpus"])
            self.w_tr.writerow([ts, f"{elapsed:.0f}", key, c["n"], fmt(ccores, 2),
                                fmt(c["rss"] / MB, 0), " ".join(sorted(c["gpus"])),
                                fmt(c["gmem"], 0) if on_gpu else "",
                                fmt(c["sm"], 0) if on_gpu else ""])
            s = self.train.setdefault(key, dict(first=now, last=now, rss=0.0, gpus=set(),
                                                gmem=0.0, sm=[], cores=[]))
            s["last"] = now
            s["rss"] = max(s["rss"], c["rss"] / MB)
            s["gpus"] |= c["gpus"]
            s["gmem"] = max(s["gmem"], c["gmem"])
            if on_gpu and sm:
                s["sm"].append((dt, c["sm"]))
            if ccores is not None:
                s["cores"].append(ccores)
        for f in (self.f_job, self.f_gpu, self.f_tr):
            f.flush()

    # ── summary ──
    def summary(self, final):
        now = time.time()
        self._read_phases()
        elapsed = now - self.job_start
        end = [p for p in self.phases if p[1] == "end"]
        exit_code = end[-1][2] if end else None
        if not final:
            status = "running (partial summary)"
        elif exit_code is not None:
            status = f"finished, exit code {exit_code}" + (
                " (killed by SIGTERM: time limit or scancel)" if exit_code == "143" else "")
        else:
            status = f"monitor stopped ({self.stop_reason or 'root process ended'})"
        hit_limit = bool(self.time_limit and elapsed >= self.time_limit - 2 * self.args.interval - 60)

        L = []
        R = dict(tag=self.args.tag, host=socket.gethostname(), status=status,
                 job_id=os.environ.get("SLURM_JOB_ID"),
                 array_task=os.environ.get("SLURM_ARRAY_TASK_ID"))
        jid = R["job_id"] or "local"
        L.append(f"Resource usage of {self.args.tag} (Slurm job {jid}"
                 + (f", array task {R['array_task']}" if R["array_task"] else "")
                 + f", node {R['host']})")
        L.append(f"Status: {status}")
        L.append(f"Start {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.job_start))}, "
                 f"last update {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(now))}, "
                 f"sampled every {self.args.interval:g} s "
                 f"({'cgroup of the job' if self.cg else 'process tree'} for CPU/memory)")

        # time occupancy
        L.append("\n== Time ==")
        occ = elapsed / self.time_limit if self.time_limit else None
        L.append(f"Elapsed {fmt_dur(elapsed)} of a time limit of {fmt_dur(self.time_limit)}"
                 + (f" ({100 * occ:.0f}% of the limit)" if occ is not None else ""))
        if hit_limit:
            L.append("WARNING: the job reached its time limit")
        R["time"] = dict(elapsed_s=elapsed, limit_s=self.time_limit, fraction_of_limit=occ,
                         hit_limit=hit_limit)
        ph = [p for p in self.phases if p[1] != "end"]
        ph_end = end[-1][0] if end else now
        R["time"]["phases"] = {}
        if ph:
            L.append("Phases:")
            for i, (t, name, _) in enumerate(ph):
                t1 = ph[i + 1][0] if i + 1 < len(ph) else ph_end
                R["time"]["phases"][f"{i}_{name}"] = t1 - t
                L.append(f"  {name:<14} {fmt_dur(t1 - t)} ({100 * (t1 - t) / max(elapsed, 1):.0f}%)"
                         f"  from +{fmt_dur(t - self.job_start)}")

        trainings = {k: v for k, v in self.train.items() if k != "other"}
        if trainings:
            L.append("Trainings / steps (wall time, from the samples):")
            for k, s in sorted(trainings.items()):
                L.append(f"  {k:<16} {fmt_dur(s['last'] - s['first'])}  "
                         f"(+{fmt_dur(s['first'] - self.job_start)} -> "
                         f"+{fmt_dur(s['last'] - self.job_start)})")

        # CPU
        L.append("\n== CPU ==")
        cores = [s["cores"] for s in self.job]
        c_mean = wmean([(s["dt"], s["cores"]) for s in self.job])
        c_p95, c_max = percentile(cores, 95), vmax(cores)
        cpu_time = sum(s["dt"] * s["cores"] for s in self.job if s["cores"] is not None)
        a = self.alloc_cpus
        L.append(f"Allocated {a if a else 'n/a'} CPU(s)")
        L.append(f"Cores used: mean {fmt(c_mean, 2)}, 95th percentile {fmt(c_p95, 2)}, "
                 f"max {fmt(c_max, 2)}; CPU time {fmt_dur(cpu_time)}"
                 + (f"; efficiency (mean/allocated) {100 * c_mean / a:.0f}%" if a and c_mean else ""))
        R["cpu"] = dict(allocated=a, mean=c_mean, p95=c_p95, max=c_max, cpu_time_s=cpu_time)
        if ph:
            for name in dict.fromkeys(p[1] for p in ph):
                m = wmean([(s["dt"], s["cores"]) for s in self.job if s["phase"] == name])
                if m is not None:
                    L.append(f"  mean cores in phase {name}: {m:.2f}")

        # memory
        L.append("\n== Memory ==")
        need_peak = vmax(s["need"] for s in self.job)
        cg_peak = None
        if self.cg:
            cg_peak = read_int(os.path.join(self.cg, "memory.peak"))
            cg_peak = cg_peak / MB if cg_peak else vmax(s["cg"] for s in self.job)
        rss_peak = vmax(s["rss"] for s in self.job)
        am = self.alloc_mem
        L.append(f"Allocated {fmt(am, 0, ' MB')}")
        L.append(f"Peak used (non-reclaimable: anon + shmem + mapped files): {fmt(need_peak, 0, ' MB')}"
                 + (f" ({100 * need_peak / am:.0f}% of the allocation)" if am and need_peak else ""))
        if cg_peak is not None:
            L.append(f"Peak of the job cgroup (incl. page cache, reclaimable): {fmt(cg_peak, 0, ' MB')}")
        L.append(f"Peak sum of the RSS of the processes (shared pages counted several times): "
                 f"{fmt(rss_peak, 0, ' MB')}")
        R["memory"] = dict(allocated_MB=am, peak_needed_MB=need_peak, peak_cgroup_MB=cg_peak,
                           peak_rss_sum_MB=rss_peak)

        # GPUs
        L.append("\n== GPU ==")
        R["gpus"] = {}
        gpu_tr_mem = vmax(s["gmem"] for k, s in trainings.items() if k not in NON_TRAININGS
                          and s["gpus"])
        if not self.gpu_s:
            L.append("No GPU allocated" if self.gpu.visible == [] or self.gpu.visible is None
                     else "GPUs allocated but nvidia-smi not available")
        for idx, samples in sorted(self.gpu_s.items(), key=lambda x: int(x[0]) if x[0].isdigit() else 0):
            name, mtot = self.gpu_info[idx]
            T = sum(s["dt"] for s in samples)
            active = [s for s in samples if s["keys"]]
            Ta = sum(s["dt"] for s in active)
            busy = sum(s["dt"] for s in samples if (s["util"] or 0) > 0)
            u_all = wmean([(s["dt"], s["util"]) for s in samples])
            u_act = wmean([(s["dt"], s["util"]) for s in active])
            m_peak = vmax(s["mem"] for s in samples)
            conc = max((len(s["keys"]) for s in samples), default=0)
            keys = sorted({k for s in samples for k in s["keys"]})
            p_mean = wmean([(s["dt"], s["power"]) for s in samples])
            R["gpus"][idx] = dict(name=name, mem_total_MB=mtot, sampled_s=T, with_job_process_s=Ta,
                                  busy_s=busy, util_mean=u_all, util_mean_while_used=u_act,
                                  mem_peak_MB=m_peak, max_trainings_at_once=conc,
                                  trainings=keys, power_mean_W=p_mean)
            L.append(f"GPU {idx} ({name}, {fmt(mtot, 0, ' MB')}):")
            if T > 0:
                L.append(f"  used by the job (>= 1 process on it) {fmt_dur(Ta)} = {100 * Ta / T:.0f}% "
                         f"of the time; allocated but unused {fmt_dur(T - Ta)}")
                L.append(f"  busy (utilisation > 0) {100 * busy / T:.0f}% of the time; mean utilisation "
                         f"{fmt(u_all, 0, '%')} overall, {fmt(u_act, 0, '%')} while used by the job")
            L.append(f"  peak memory {fmt(m_peak, 0, ' MB')}"
                     + (f" ({100 * m_peak / mtot:.0f}%)" if m_peak and mtot else "")
                     + f"; mean power {fmt(p_mean, 0, ' W')}")
            L.append(f"  trainings on it: {', '.join(keys) if keys else 'none'} "
                     f"(at most {conc} at the same time)")
        if self.gpu_s:
            n = len(self.gpu_s)
            gh = sum(sum(s["dt"] for s in v) for v in self.gpu_s.values()) / 3600
            gh_used = sum(sum(s["dt"] for s in v if s["keys"]) for v in self.gpu_s.values()) / 3600
            if gh > 0:
                L.append(f"GPU-hours: {gh:.2f} allocated ({n} GPU(s)), {gh_used:.2f} with a process "
                         f"of the job ({100 * gh_used / gh:.0f}%)")

        # per training
        if trainings:
            L.append("\n== Per training / step ==")
            L.append(f"  {'name':<16} {'wall':>10} {'mean cores':>10} {'peak RSS MB':>11} "
                     f"{'GPU':>5} {'peak GPU MB':>11} {'mean SM%':>8}")
            R["trainings"] = {}
            cpu_by_key = defaultdict(float)
            for key, c in self.cpu_last.values():
                cpu_by_key[key] += c
            for k, s in sorted(trainings.items()):
                wall = s["last"] - s["first"]
                mc = cpu_by_key[k] / wall if wall > 0 else None
                smm = wmean(s["sm"])
                gmem = f"{s['gmem']:.0f}" if s["gpus"] else "-"
                R["trainings"][k] = dict(wall_s=wall, cpu_time_s=cpu_by_key[k], mean_cores=mc,
                                         peak_rss_MB=s["rss"], gpus=sorted(s["gpus"]),
                                         peak_gpu_mem_MB=s["gmem"] if s["gpus"] else None,
                                         mean_sm_pct=smm)
                L.append(f"  {k:<16} {fmt_dur(wall):>10} {fmt(mc, 2):>10} {s['rss']:>11.0f} "
                         f"{','.join(sorted(s['gpus'])) or '-':>5} "
                         f"{gmem:>11} {fmt(smm, 0):>8}")

        # suggestions
        L.append("\n== Suggestions (heuristic, check them against the numbers above) ==")
        sug = {}
        n_par = max((s["n_tr"] for s in self.job), default=0)
        if a and c_p95 is not None:
            if c_p95 >= 0.9 * a:
                sug["cpus"] = a
                L.append(f"--cpus {a}: the job used ~all its {a} CPU(s) (CPU-bound; run_training.sh "
                         "splits the CPUs among the parallel trainings, more CPUs may speed it up)")
            else:
                sug["cpus"] = max(1, n_par, math.ceil(c_p95 * 1.2))
                L.append(f"--cpus {sug['cpus']}: 95th percentile {c_p95:.1f} cores of {a} allocated"
                         f"{f', at least 1 per parallel training ({n_par})' if n_par else ''}")
        if need_peak:
            tot = need_peak * 1.2 + 1024
            ncpu = sug.get("cpus") or a or 1
            sug["mem_per_cpu_MB"] = int(math.ceil(tot / ncpu / 100) * 100)
            L.append(f"--mem-per-cpu {sug['mem_per_cpu_MB']}: peak {need_peak:.0f} MB +20% +1 GB "
                     f"= {tot:.0f} MB over {ncpu} CPU(s)")
        if hit_limit or exit_code == "143":
            L.append("--time: the job was stopped at the time limit (or cancelled): increase --time "
                     "(and resubmit with --load-last / --missing-folds to resume)")
        elif final and elapsed > 0:
            t = math.ceil(max(elapsed * 1.3, 900) / 900) * 900
            sug["time"] = fmt_slurm_time(t)
            L.append(f"--time {sug['time']}: elapsed {fmt_dur(elapsed)} +30% (rounded up to 15 min); "
                     "the trainings with early stopping can take longer with other seeds")
        for idx, g in R["gpus"].items():
            mtot, conc, u = g["mem_total_MB"], g["max_trainings_at_once"], g["util_mean_while_used"]
            if not g["trainings"]:
                L.append(f"GPU {idx}: never used by the job" + (
                    " (CPU-only step on the GPU partition?)" if not trainings or set(trainings) <= set(NON_TRAININGS)
                    else ": request fewer GPUs (-g)"))
                continue
            unused = 1 - g["with_job_process_s"] / g["sampled_s"] if g["sampled_s"] else 0
            if unused > 0.2:
                L.append(f"GPU {idx}: allocated but unused {100 * unused:.0f}% of the time (load/merge "
                         "phases or uneven trainings): consider --separate-merge, --slots-per-gpu or "
                         "one job per training/fold")
        if gpu_tr_mem and R["gpus"]:
            mtot = min(g["mem_total_MB"] for g in R["gpus"].values() if g["mem_total_MB"])
            fit = int(0.9 * mtot // gpu_tr_mem) if gpu_tr_mem > 0 else None
            conc = max(g["max_trainings_at_once"] for g in R["gpus"].values())
            used = [g["util_mean_while_used"] for g in R["gpus"].values() if g["util_mean_while_used"] is not None]
            u = sum(used) / len(used) if used else None
            sug["max_trainings_per_gpu_by_memory"] = fit
            L.append(f"GPU memory: peak {gpu_tr_mem:.0f} MB per training -> up to {fit} trainings fit "
                     f"in {mtot:.0f} MB (90%); this job ran at most {conc} per GPU")
            if u is not None:
                if u > 90:
                    L.append(f"GPU utilisation {u:.0f}% while used: the GPUs are saturated, more trainings "
                             "per GPU would slow each of them down")
                elif fit and fit > conc:
                    per_gpu = min(fit, max(conc + 1, math.ceil(conc * 80 / max(u, 1))))
                    sug["trainings_per_gpu"] = per_gpu
                    n_train = len([k for k in trainings if k not in NON_TRAININGS])
                    L.append(f"GPU utilisation {u:.0f}% while used with {conc} training(s) per GPU: about "
                             f"{per_gpu} per GPU should still fit (memory) and keep it busy, i.e. "
                             f"-g {math.ceil(n_train / per_gpu)} for the {n_train} training(s) of this job "
                             f"(or --slots-per-gpu {per_gpu}); watch the CPUs too")
                else:
                    L.append(f"GPU utilisation {u:.0f}% while used: the memory limits the trainings per GPU")
        R["suggestions"] = sug

        with open(self.summary_base + ".txt", "w") as f:
            f.write("\n".join(l for l in L if l is not None) + "\n")
        with open(self.summary_base + ".json", "w") as f:
            json.dump(R, f, indent=2, default=list)

    def loop(self):
        def stop(signum, _frame):
            self.stop_reason = signal.Signals(signum).name
        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        def root_alive():
            try:
                return self.root.is_running() and self.root.status() != psutil.STATUS_ZOMBIE
            except psutil.Error:
                return False

        last_summary = time.time()
        while self.stop_reason is None and root_alive():
            try:
                self.sample()
            except Exception as e:  # never kill the monitor for one bad sample
                print(f"resource_monitor: sample failed: {e!r}", flush=True)
            if time.time() - last_summary >= self.args.summary_every:
                try:
                    self.summary(final=False)
                except Exception as e:
                    print(f"resource_monitor: summary failed: {e!r}", flush=True)
                last_summary = time.time()
            t_next = time.time() + self.args.interval
            while self.stop_reason is None and time.time() < t_next:
                time.sleep(min(0.5, max(t_next - time.time(), 0)))
                if not root_alive():
                    break
        try:
            self.sample()
        except Exception as e:
            print(f"resource_monitor: sample failed: {e!r}", flush=True)
        self.summary(final=True)
        for f in (self.f_job, self.f_gpu, self.f_tr):
            f.close()


def main():
    args = parse_args()
    try:
        mon = Monitor(args)
    except psutil.Error as e:
        print(f"resource_monitor: cannot monitor pid {args.pid}: {e}")
        return
    mon.loop()


if __name__ == "__main__":
    main()
