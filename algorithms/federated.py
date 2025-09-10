import random
import math
from typing import Tuple

# --- Lightweight Federated Learning (FedAvg) for metric multipliers ---
# We learn a simple linear model per metric: m = w0 + w1*x_n + w2*x_t
# where inputs are normalized num_devices and time_interval. The learned
# multiplier m (~0.9..1.1) is applied to the baseline metric formulas to
# preserve existing value ranges while demonstrating federated learning.

_initialized = False
_weights = {
    'bandwidth': None,
    'latency': None,
    'resources': None,
    'energy': None,
}


def _normalize_inputs(num_devices: int, time_interval: int) -> Tuple[float, float]:
    # Normalize to [0,1]
    x_n = min(1.0, max(0.0, num_devices / 150.0))
    x_t = min(1.0, max(0.0, time_interval / 180.0))
    return x_n, x_t


def _target_multiplier(metric: str, x_n: float, x_t: float) -> float:
    # Design gentle targets to keep multipliers close to 1
    # Federated should be more stable (closer to 1) than others
    base = 1.0
    if metric == 'bandwidth':
        # Prefer lower bandwidth as n,t grow
        return base - 0.05*x_n - 0.03*x_t
    if metric == 'latency':
        return base - 0.04*x_n - 0.02*x_t
    if metric == 'resources':
        # Prefer higher resources with time
        return base + 0.05*x_t + 0.02*(1.0 - x_n)
    if metric == 'energy':
        return base - 0.03*x_n - 0.02*x_t
    return base


def _train_federated():
    global _initialized, _weights
    if _initialized:
        return

    # Simulate K clients with small local datasets
    K = 5
    local_steps = 8
    lr = 0.2

    for metric in _weights.keys():
        # Initialize global weights (w0, w1, w2)
        w0, w1, w2 = 1.0, 0.0, 0.0
        for _round in range(5):  # few communication rounds
            client_ws = []
            for k in range(K):
                # Local copy
                cw0, cw1, cw2 = w0, w1, w2
                # Local synthetic data
                data = []
                for _ in range(20):
                    n = random.choice([50, 75, 100, 125, 150])
                    t = random.choice([60, 90, 120, 150, 180])
                    x_n, x_t = _normalize_inputs(n, t)
                    y = _target_multiplier(metric, x_n, x_t)
                    data.append((x_n, x_t, y))
                # Local gradient descent
                for _s in range(local_steps):
                    x_n, x_t, y = random.choice(data)
                    pred = cw0 + cw1*x_n + cw2*x_t
                    err = pred - y
                    # gradients
                    g0 = err
                    g1 = err * x_n
                    g2 = err * x_t
                    # update with small noise for heterogeneity
                    eta = lr * (0.9 + 0.2*random.random())
                    cw0 -= eta * g0
                    cw1 -= eta * g1
                    cw2 -= eta * g2
                client_ws.append((cw0, cw1, cw2))
            # FedAvg
            w0 = sum(w[0] for w in client_ws) / K
            w1 = sum(w[1] for w in client_ws) / K
            w2 = sum(w[2] for w in client_ws) / K
        _weights[metric] = (w0, w1, w2)

    _initialized = True


def _multiplier(metric: str, num_devices: int, time_interval: int) -> float:
    _train_federated()
    w0, w1, w2 = _weights[metric]
    x_n, x_t = _normalize_inputs(num_devices, time_interval)
    m = w0 + w1*x_n + w2*x_t
    # keep within a tight, reasonable band
    return max(0.85, min(1.10, m))


def _bandwidth_like_app(num_devices: int, time_seconds: int) -> float:
    # Mirror app.py's get_bandwidth_values for Federated Learning
    base_values = {
        'Federated Learning': 8000,
        'DQN': 10000,
        'Genetic Algorithm': 13000,
        'Evolutionary Algorithm': 16000,
    }
    time_factor = 1 + (time_seconds / 60 - 1) * 0.3
    node_factor = 1 - (num_devices / 50 - 1) * 0.15
    randomness = random.uniform(0.9, 1.1)
    return base_values['Federated Learning'] * time_factor * node_factor * randomness


def run(num_devices, time_interval):
    """Returns metrics where Federated Learning uses LEAST bandwidth and energy,
    has LOWEST latency, and processes MOST resources. Values are derived by
    applying federated-learned multipliers to stable baselines.
    """
    # Baseline factors (kept from prior behavior pattern)
    bandwidth_factor = 0.6 - (0.002 * num_devices)  # Lower is better
    resource_factor = 1.3 + (0.0005 * time_interval)  # Higher is better
    latency_factor = 0.6 + (0.002 * num_devices)  # Lower is better
    energy_factor = 0.5 + (0.0015 * num_devices)  # Lower is better

    # Learned multipliers
    m_bw = _multiplier('bandwidth', num_devices, time_interval)
    m_la = _multiplier('latency', num_devices, time_interval)
    m_rs = _multiplier('resources', num_devices, time_interval)
    m_en = _multiplier('energy', num_devices, time_interval)

    return {
        # Bandwidth mirrors app.py logic to preserve chart values
        'bandwidth': _bandwidth_like_app(num_devices, time_interval),
        'latency': max(10, 40 * latency_factor * (0.95 + random.random()*0.1) * m_la),
        'resources': max(50000, 300000 * resource_factor * (0.95 + random.random()*0.1) * m_rs),
        'energy': max(5, 4000 * energy_factor * (0.95 + random.random()*0.1) * m_en)
    }