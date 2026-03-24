"""
Chapter 20: Multiprocessing and Vectorization
===============================================
Advances in Financial Machine Learning – Marcos López de Prado

Multiprocessing is essential to ML: algorithms are computationally intensive
and require efficient use of all CPUs.  This chapter presents:
  - Vectorization vs explicit loops
  - Single-thread vs multiprocessing
  - Atoms/molecules partitioning
  - The mpPandasObj multiprocessing engine
  - Output reduction on-the-fly

Snippets:
  20.1 – Un-vectorized Cartesian product
  20.2 – Vectorized Cartesian product
  20.3 – Single-thread barrier touch
  20.4 – Multiprocessing barrier touch
  20.5 – linParts()    – linear partitions
  20.6 – nestedParts() – two-nested-loops partitions
  20.7 – mpPandasObj   – multiprocessing engine
  20.8 – processJobs_  – single-thread execution
  20.9 – processJobs   – parallel execution (imap_unordered)
  20.10 – expandCall   – unwrap callback arguments
  20.11 – pickle/unpickle bound methods
  20.12 – processJobsRedux – on-the-fly output reduction
  20.13 – mpJobList    – enhanced mpPandasObj with reduction
  20.14 – getPCs       – principal components example
"""

import numpy as np
import pandas as pd
import multiprocessing as mp
import time
import sys
import copy
import datetime as dt
from itertools import product as iproduct

sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))
from sample_data import generate_returns


# ============================================================================
# Snippet 20.1 – Un-Vectorized Cartesian Product
# ============================================================================
def cartesian_unvectorized():
    """
    Snippet 20.1: Un-vectorized Cartesian product using nested for loops.
    """
    dict0 = {'a': ['1', '2'], 'b': ['+', '*'], 'c': ['!', '@']}
    results = []
    for a in dict0['a']:
        for b in dict0['b']:
            for c in dict0['c']:
                results.append({'a': a, 'b': b, 'c': c})
    return results


# ============================================================================
# Snippet 20.2 – Vectorized Cartesian Product
# ============================================================================
def cartesian_vectorized():
    """
    Snippet 20.2: Vectorized Cartesian product using itertools.
    Replaces nested loops with fast iterators.
    """
    dict0 = {'a': ['1', '2'], 'b': ['+', '*'], 'c': ['!', '@']}
    jobs = (dict(zip(dict0, i)) for i in iproduct(*dict0.values()))
    return list(jobs)


# ============================================================================
# Snippet 20.3 – Single-Thread Barrier Touch
# ============================================================================
def barrierTouch(r, width=0.5):
    """
    Find the index of the earliest barrier touch for each path.

    Parameters
    ----------
    r : np.ndarray
        Random returns, shape (T, num_paths).
    width : float
        Barrier half-width.

    Returns
    -------
    dict
        {path_index: first_barrier_touch_time}
    """
    t = {}
    p = np.log((1 + r).cumprod(axis=0))
    for j in range(r.shape[1]):
        for i in range(r.shape[0]):
            if p[i, j] >= width or p[i, j] <= -width:
                t[j] = i
                break
    return t


def main0():
    """Snippet 20.3: Single-thread implementation of one-touch double barrier."""
    r = np.random.normal(0, 0.01, size=(1000, 10000))
    t = barrierTouch(r)
    return t


# ============================================================================
# Snippet 20.4 – Multiprocessing Barrier Touch
# ============================================================================
def main1(numThreads=None):
    """
    Snippet 20.4: Multiprocessing implementation of one-touch double barrier.
    Splits the problem into chunks and processes them in parallel.
    """
    if numThreads is None:
        numThreads = min(mp.cpu_count(), 8)
    r = np.random.normal(0, 0.01, size=(1000, 10000))
    parts = np.linspace(0, r.shape[1], min(numThreads, r.shape[1]) + 1)
    parts = np.ceil(parts).astype(int)
    jobs = []
    for i in range(1, len(parts)):
        jobs.append(r[:, parts[i - 1]:parts[i]])

    pool = mp.Pool(processes=numThreads)
    outputs = pool.imap_unordered(barrierTouch, jobs)
    out = []
    for out_ in outputs:
        out.append(out_)
    pool.close()
    pool.join()
    return out


# ============================================================================
# Snippet 20.5 – linParts: Linear Partitions
# ============================================================================
def linParts(numAtoms, numThreads):
    """
    Partition *numAtoms* atoms linearly into *numThreads* molecules.

    Parameters
    ----------
    numAtoms : int
        Number of indivisible tasks.
    numThreads : int
        Number of threads / partitions.

    Returns
    -------
    np.ndarray
        Array of partition boundaries (length = numThreads + 1).
    """
    parts = np.linspace(0, numAtoms, min(numThreads, numAtoms) + 1)
    parts = np.ceil(parts).astype(int)
    return parts


# ============================================================================
# Snippet 20.6 – nestedParts: Two-Nested Loops Partitions
# ============================================================================
def nestedParts(numAtoms, numThreads, upperTriang=False):
    """
    Partition atoms for double-nested loops so that each molecule
    contains approximately equal workload.

    Parameters
    ----------
    numAtoms : int
        Total number of rows (outer loop).
    numThreads : int
        Number of processors.
    upperTriang : bool
        If True, the first rows are the heaviest (upper triangular).

    Returns
    -------
    np.ndarray
        Partition boundaries.
    """
    parts = [0]
    numThreads_ = min(numThreads, numAtoms)
    for num in range(numThreads_):
        part = 1 + 4 * (parts[-1] ** 2 + parts[-1] +
                         numAtoms * (numAtoms + 1.0) / numThreads_)
        part = (-1 + part ** 0.5) / 2.0
        parts.append(part)
    parts = np.round(parts).astype(int)
    if upperTriang:  # first rows are heaviest
        parts = np.cumsum(np.diff(parts)[::-1])
        parts = np.append(np.array([0]), parts)
    return parts


# ============================================================================
# Snippet 20.10 – expandCall: Unwrap Callback Arguments
# ============================================================================
def expandCall(kargs):
    """
    Expand the arguments of a callback function.

    This transforms a dictionary (a job/molecule) into a function call.
    The dictionary must contain a 'func' key.

    Parameters
    ----------
    kargs : dict
        Must include 'func' (the callback) plus other kwargs.

    Returns
    -------
    object
        Result of func(**remaining_kargs).
    """
    func = kargs['func']
    del kargs['func']
    out = func(**kargs)
    return out


# ============================================================================
# Snippet 20.8 – processJobs_: Single-Thread Execution (for debugging)
# ============================================================================
def processJobs_(jobs):
    """
    Run jobs sequentially (single thread). Useful for debugging.

    Parameters
    ----------
    jobs : list of dict
        Each dict has 'func' plus other kwargs.

    Returns
    -------
    list
        List of results.
    """
    out = []
    for job in jobs:
        out_ = expandCall(job.copy())
        out.append(out_)
    return out


# ============================================================================
# Snippet 20.9 – processJobs: Asynchronous Parallel Execution
# ============================================================================
def reportProgress(jobNum, numJobs, time0, task):
    """Report progress as async jobs complete."""
    msg = [float(jobNum) / numJobs, (time.time() - time0) / 60.0]
    msg.append(msg[1] * (1 / msg[0] - 1))
    timeStamp = str(dt.datetime.fromtimestamp(time.time()))
    msg = (timeStamp + ' ' + str(round(msg[0] * 100, 2)) + '% ' + task +
           ' done after ' + str(round(msg[1], 2)) + ' minutes. Remaining ' +
           str(round(msg[2], 2)) + ' minutes.')
    if jobNum < numJobs:
        sys.stderr.write(msg + '\r')
    else:
        sys.stderr.write(msg + '\n')


def processJobs(jobs, task=None, numThreads=None):
    """
    Snippet 20.9: Run jobs in parallel using multiprocessing.

    Parameters
    ----------
    jobs : list of dict
        Each dict has 'func' plus other kwargs.
    task : str, optional
        Task name for progress reporting.
    numThreads : int, optional
        Number of parallel processes.

    Returns
    -------
    list
        Collected results.
    """
    if numThreads is None:
        numThreads = min(mp.cpu_count(), 8)
    if task is None:
        task = jobs[0]['func'].__name__
    pool = mp.Pool(processes=numThreads)
    outputs = pool.imap_unordered(expandCall, [j.copy() for j in jobs])
    out = []
    time0 = time.time()
    for i, out_ in enumerate(outputs, 1):
        out.append(out_)
        reportProgress(i, len(jobs), time0, task)
    pool.close()
    pool.join()
    return out


# ============================================================================
# Snippet 20.7 – mpPandasObj: Multiprocessing Engine
# ============================================================================
def mpPandasObj(func, pdObj, numThreads=None, mpBatches=1, linMols=True, **kargs):
    """
    Parallelize jobs, return a DataFrame or Series.

    Parameters
    ----------
    func : callable
        Callback function returning a DataFrame.
    pdObj : tuple
        (name_of_molecule_arg, list_of_atoms).
    numThreads : int
        Number of parallel threads.
    mpBatches : int
        Number of batches per core (>1 for load balancing).
    linMols : bool
        Whether to use linear partitions (True) or nested (False).
    **kargs :
        Additional keyword arguments for func.

    Returns
    -------
    pd.DataFrame or pd.Series or list
    """
    if numThreads is None:
        numThreads = min(mp.cpu_count(), 8)
    if linMols:
        parts = linParts(len(pdObj[1]), numThreads * mpBatches)
    else:
        parts = nestedParts(len(pdObj[1]), numThreads * mpBatches)
    jobs = []
    for i in range(1, len(parts)):
        job = {pdObj[0]: pdObj[1][parts[i - 1]:parts[i]], 'func': func}
        job.update(kargs)
        jobs.append(job)
    if numThreads == 1:
        out = processJobs_(jobs)
    else:
        out = processJobs(jobs, numThreads=numThreads)
    if isinstance(out[0], pd.DataFrame):
        return pd.concat([i for i in out if len(i) > 0]).sort_index()
    elif isinstance(out[0], pd.Series):
        return pd.concat([i for i in out if len(i) > 0]).sort_index()
    else:
        return out


# ============================================================================
# Snippet 20.12 – processJobsRedux: On-The-Fly Output Reduction
# ============================================================================
def processJobsRedux(jobs, task=None, numThreads=None, redux=None,
                     reduxArgs=None, reduxInPlace=False):
    """
    Run in parallel with on-the-fly output reduction to save memory.

    Parameters
    ----------
    jobs : list of dict
    task : str, optional
    numThreads : int, optional
    redux : callable, optional
        Reduction function (e.g. pd.DataFrame.add).
    reduxArgs : dict, optional
    reduxInPlace : bool
    """
    if reduxArgs is None:
        reduxArgs = {}
    if numThreads is None:
        numThreads = min(mp.cpu_count(), 8)
    if task is None:
        task = jobs[0]['func'].__name__
    pool = mp.Pool(processes=numThreads)
    imap = pool.imap_unordered(expandCall, [j.copy() for j in jobs])
    out = None
    time0 = time.time()
    for i, out_ in enumerate(imap, 1):
        if out is None:
            if redux is None:
                out, redux, reduxInPlace = [out_], list.append, True
            else:
                out = copy.deepcopy(out_)
        else:
            if reduxInPlace:
                redux(out, out_, **reduxArgs)
            else:
                out = redux(out, out_, **reduxArgs)
        reportProgress(i, len(jobs), time0, task)
    pool.close()
    pool.join()
    if isinstance(out, (pd.Series, pd.DataFrame)):
        out = out.sort_index()
    return out


# ============================================================================
# Snippet 20.13 – mpJobList: Enhanced mpPandasObj with On-The-Fly Reduction
# ============================================================================
def mpJobList(func, argList, numThreads=None, mpBatches=1, linMols=True,
              redux=None, reduxArgs=None, reduxInPlace=False, **kargs):
    """
    Enhanced version of mpPandasObj that supports on-the-fly output reduction.
    """
    if reduxArgs is None:
        reduxArgs = {}
    if numThreads is None:
        numThreads = min(mp.cpu_count(), 8)
    if linMols:
        parts = linParts(len(argList[1]), numThreads * mpBatches)
    else:
        parts = nestedParts(len(argList[1]), numThreads * mpBatches)
    jobs = []
    for i in range(1, len(parts)):
        job = {argList[0]: argList[1][parts[i - 1]:parts[i]], 'func': func}
        job.update(kargs)
        jobs.append(job)
    out = processJobsRedux(jobs, redux=redux, reduxArgs=reduxArgs,
                           reduxInPlace=reduxInPlace, numThreads=numThreads)
    return out


# ============================================================================
# DEMO
# ============================================================================
def _demo_barrier(molecule, width=0.5):
    """Callback for mpPandasObj demo: barrier touch on a molecule of paths."""
    results = {}
    for idx in molecule:
        rng = np.random.default_rng(idx)
        r = rng.normal(0, 0.01, 1000)
        p = np.log((1 + r).cumprod())
        for i in range(len(p)):
            if p[i] >= width or p[i] <= -width:
                results[idx] = i
                break
    return pd.Series(results, name='barrier_touch')


def main():
    """Demonstrate multiprocessing and vectorization techniques."""
    print("=" * 60)
    print("Chapter 20 – Multiprocessing and Vectorization")
    print("=" * 60)

    # --- 1) Cartesian product comparison ----------------------------------
    print("\n[1] Cartesian product:")
    r1 = cartesian_unvectorized()
    r2 = cartesian_vectorized()
    print(f"    Un-vectorized: {len(r1)} combinations")
    print(f"    Vectorized:    {len(r2)} combinations")
    assert len(r1) == len(r2)
    print("    ✓ Both produce identical number of results")

    # --- 2) linParts demo -------------------------------------------------
    print("\n[2] linParts (20 atoms, 6 threads):")
    parts = linParts(20, 6)
    print(f"    Boundaries: {parts}")
    sizes = np.diff(parts)
    print(f"    Molecule sizes: {sizes}")

    # --- 3) nestedParts demo ----------------------------------------------
    print("\n[3] nestedParts (20 atoms, 6 threads):")
    nparts = nestedParts(20, 6)
    print(f"    Boundaries: {nparts}")
    nsizes = np.diff(nparts)
    print(f"    Molecule sizes: {nsizes}")

    print("\n    nestedParts (20 atoms, 6 threads, upperTriang=True):")
    nparts_ut = nestedParts(20, 6, upperTriang=True)
    print(f"    Boundaries: {nparts_ut}")

    # --- 4) Single-thread barrier touch (small scale) ---------------------
    print("\n[4] Single-thread barrier touch (100x500 paths):")
    r = np.random.normal(0, 0.01, size=(100, 500))
    t0 = time.time()
    result = barrierTouch(r)
    elapsed = time.time() - t0
    print(f"    Paths that touched barrier: {len(result)}/500")
    print(f"    Elapsed: {elapsed:.4f}s")

    # --- 5) Multiprocessing with mpPandasObj ------------------------------
    print("\n[5] mpPandasObj demo (single-thread for safety in __main__):")
    atoms = list(range(100))
    out = mpPandasObj(_demo_barrier, ('molecule', atoms),
                      numThreads=1, width=0.5)
    if isinstance(out, list):
        total = sum(len(o) for o in out)
    else:
        total = len(out)
    print(f"    Paths touching barrier: {total}/100")

    # --- 6) expandCall demo -----------------------------------------------
    print("\n[6] expandCall demo:")
    job = {'func': barrierTouch, 'r': r[:, :10], 'width': 0.5}
    result = expandCall(job.copy())
    print(f"    Expanded call result: {len(result)} barrier touches")

    # --- 7) processJobs_ (sequential) demo --------------------------------
    print("\n[7] processJobs_ (sequential debugging mode):")
    jobs = [
        {'func': barrierTouch, 'r': r[:, :50], 'width': 0.5},
        {'func': barrierTouch, 'r': r[:, 50:100], 'width': 0.5},
    ]
    results = processJobs_(jobs)
    total = sum(len(res) for res in results)
    print(f"    Total barrier touches: {total}")

    print("\n✓ Chapter 20 complete")


if __name__ == "__main__":
    main()
