"""Isolation Forest vs RMS threshold on simulated bearing-fault data.
Runs 20 simulated experiments and reports average detection delay,
false alarms and missed detections for each method.
Both methods use the same rule: alarm after 4 abnormal windows in a row."""
import numpy as np
from sklearn.ensemble import IsolationForest
from scipy.stats import kurtosis

FS, WIN, BPFO = 2000, 200, 87
ONSET, STEPS, RUNS = 12.0, 400, 20

def make_window(rng, sev):
    t = np.arange(WIN)/FS
    x = rng.normal(0, .05, WIN) + .03*np.sin(2*np.pi*50*t + rng.uniform(0, 6.28))
    if sev > 0:
        for t0 in np.arange(rng.uniform(0, 1/BPFO), WIN/FS, 1/BPFO):
            m = t >= t0
            x[m] += sev*1.5*np.exp(-(t[m]-t0)*400)*np.sin(2*np.pi*800*(t[m]-t0))
    return x

def feats(x):
    r = np.sqrt(np.mean(x**2)); pk = np.max(np.abs(x))
    return [r, kurtosis(x), pk, pk/r]

def first_alarm(flags):
    c = 0
    for k, fl in enumerate(flags):
        c = c + 1 if fl else 0
        if c >= 4:
            return k*0.1
    return None

def run(seed):
    rng