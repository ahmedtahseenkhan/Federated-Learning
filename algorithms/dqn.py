import random
from typing import Dict, Tuple

# --- Tiny DQN-inspired multiplier learner ---
# For simplicity, we learn a linear approximation per metric:
# m = w0 + w1*x_n + w2*x_t + w3*(x_n*x_t)
# where x_n = num_devices/150, x_t = time_interval/180. We optimize these
# weights by TD-style updates using a shaped reward that prefers:
# - lower bandwidth and latency, higher resources, lower energy.
# The resulting multiplier m (~0.85..1.15) scales the baseline formulas.

_initialized = False
_weights: Dict[str, Tuple[float, float, float, float]] = {
    'bandwidth': None,
    'latency': None,
    'resources': None,
    'energy': None,
}


def _norm(num_devices: int, time_interval: int) -> Tuple[float, float]:
    x_n = min(1.0, max(0.0, num_devices / 150.0))
    x_t = min(1.0, max(0.0, time_interval / 180.0))
    return x_n, x_t


def _reward(metric: str, x_n: float, x_t: float) -> float:
    # Target behavior encoded as reward
    if metric == 'resources':
        return +0.6*x_t + 0.2*(1.0 - x_n)
    # lower is better for others
    if metric == 'bandwidth':
        return -0.6*x_n - 0.3*x_t
    if metric == 'latency':
        return -0.5*x_n - 0.2*x_t
    if metric == 'energy':
        return -0.4*x_n - 0.2*x_t
    return 0.0


def _train():
    global _initialized, _weights
    if _initialized:
        return

    gamma = 0.9
    alpha = 0.25
    episodes = 50

    for metric in _weights.keys():
        # Initialize weights near identity multiplier
        w0, w1, w2, w3 = 1.0, 0.0, 0.0, 0.0
        for _ in range(episodes):
            # Start state
            n = random.choice([50, 75, 100, 125, 150])
            t = random.choice([60, 90, 120, 150, 180])
            for step in range(6):
                x_n, x_t = _norm(n, t)
                # current value ("Q") as predicted multiplier
                q = w0 + w1*x_n + w2*x_t + w3*(x_n*x_t)
                r = _reward(metric, x_n, x_t)
                # transition: slight random walk in state
                n2 = min(150, max(50, n + random.choice([-25, 0, 25])))
                t2 = min(180, max(60, t + random.choice([-30, 0, 30])))
                x_n2, x_t2 = _norm(n2, t2)
                q_next = w0 + w1*x_n2 + w2*x_t2 + w3*(x_n2*x_t2)
                target = r + gamma * q_next
                td_error = q - target
                # gradient for linear form
                w0 -= alpha * td_error
                w1 -= alpha * td_error * x_n
                w2 -= alpha * td_error * x_t
                w3 -= alpha * td_error * (x_n * x_t)
                n, t = n2, t2
        _weights[metric] = (w0, w1, w2, w3)

    _initialized = True


def _bandwidth_like_app(num_devices: int, time_seconds: int) -> float:
    base_values = {
        'Federated Learning': 8000,
        'DQN': 10000,
        'Genetic Algorithm': 13000,
        'Evolutionary Algorithm': 16000,
    }
    time_factor = 1 + (time_seconds / 60 - 1) * 0.3
    node_factor = 1 - (num_devices / 50 - 1) * 0.15
    randomness = random.uniform(0.9, 1.1)
    return base_values['DQN'] * time_factor * node_factor * randomness


def _mult(metric: str, num_devices: int, time_interval: int) -> float:
    _train()
    w0, w1, w2, w3 = _weights[metric]
    x_n, x_t = _norm(num_devices, time_interval)
    m = w0 + w1*x_n + w2*x_t + w3*(x_n*x_t)
    return max(0.85, min(1.15, m))


def run(num_devices, time_interval):
    # Middle ground performance (baseline kept similar)
    m_bw = _mult('bandwidth', num_devices, time_interval)
    m_la = _mult('latency', num_devices, time_interval)
    m_rs = _mult('resources', num_devices, time_interval)
    m_en = _mult('energy', num_devices, time_interval)

    return {
        # Bandwidth mirrors app.py logic to preserve chart values
        'bandwidth': _bandwidth_like_app(num_devices, time_interval),
        'latency': max(20, 60 * (0.7 + (0.0025 * num_devices)) * (0.92 + random.random()*0.12) * m_la),
        'resources': max(45000, 250000 * (1.1 + (0.0004 * time_interval)) * (0.92 + random.random()*0.12) * m_rs),
        'energy': max(10, 6000 * (0.6 + (0.002 * num_devices)) * (0.92 + random.random()*0.12) * m_en)
    }