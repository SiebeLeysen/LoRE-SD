"""LoRE-SD package initialisation.

Cap the number of threads used by the BLAS backend (OpenBLAS/MKL/etc.) to one
per process *before* NumPy is imported. LoRE-SD parallelises over voxels with a
multiprocessing pool (see ``lore_sd.optimisation.optimise.get_signal_decomposition``),
and the single-process tools are commonly launched many-at-once via SLURM array
jobs or GNU parallel. In both cases, letting each process *also* spin up one BLAS
thread per core causes massive thread oversubscription (processes x cores), which
collapses throughput to effectively a single active core.

These variables only take effect if read before the BLAS library is loaded
(i.e. before the first ``import numpy``), so this module must be imported ahead
of NumPy in every entry point.

We *force* the value to ``'1'`` rather than using ``setdefault``. HPC schedulers
and module systems (SLURM, etc.) routinely export e.g. ``OPENBLAS_NUM_THREADS``
equal to the allocated core count, so ``setdefault`` would silently defer to that
high value and the cap would never engage -- exactly the oversubscription we are
trying to prevent. Every LoRE-SD tool gets its parallelism from a per-voxel
process pool, so single-threaded BLAS per process is always the right choice here.
To run a single BLAS-heavy command with multiple BLAS threads, invoke it in its
own process with the variables exported in the shell beforehand.
"""
import os

for _var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'BLIS_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_var] = '1'
