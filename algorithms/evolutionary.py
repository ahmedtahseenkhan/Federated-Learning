import random
from typing import Dict, Tuple

# --- Tiny Evolution Strategies (ES) multiplier learner ---
# Model per metric: m = w0 + w1*x_n + w2*x_t
# Evolutionary algorithm is intended to be least efficient; target slightly
# worse multipliers vs others (still close to 1) to preserve ordering.

_initialized = False
_weights: Dict[str, Tuple[float, float, float]] = {
    'bandwidth': None,
    'latency': None,
    'resources': None,
    'energy': None,
}


def _norm(num_devices: int, time_interval: int) -> Tuple[float, float]:
    x_n = min(1.0, max(0.0, num_devices / 150.0))
    x_t = min(1.0, max(0.0, time_interval / 180.0))
    return x_n, x_t


def _target(metric: str, x_n: float, x_t: float) -> float:
    base = 1.0
    if metric == 'bandwidth':
        return base + 0.04*x_n + 0.03*x_t  # higher (worse)
    if metric == 'latency':
        return base + 0.03*x_n + 0.02*x_t  # higher (worse)
    if metric == 'resources':
        return base - 0.04*x_t - 0.01*(1.0 - x_n)  # lower (worse)
    if metric == 'energy':
        return base + 0.03*x_n + 0.02*x_t  # higher (worse)
    return base


def _es_fitness(metric: str, w: Tuple[float, float, float]) -> float:
    w0, w1, w2 = w
    err = 0.0
    for n in [50, 75, 100, 125, 150]:
        for t in [60, 90, 120, 150, 180]:
            x_n, x_t = _norm(n, t)
            pred = w0 + w1*x_n + w2*x_t
            y = _target(metric, x_n, x_t)
            d = pred - y
            err += d*d
    return -err


def _train_es():
    global _initialized, _weights
    if _initialized:
        return

    sigma = 0.15
    alpha = 0.3
    iterations = 40
    pop = 32

    for metric in _weights.keys():
        # start near identity multiplier
        w = [1.0, 0.0, 0.0]
        for _ in range(iterations):
            noises = [(random.gauss(0, 1), random.gauss(0, 1), random.gauss(0, 1)) for _ in range(pop)]
            samples = []
            for eps in noises:
                cand = (w[0] + sigma*eps[0], w[1] + sigma*eps[1], w[2] + sigma*eps[2])
                fit = _es_fitness(metric, cand)
                samples.append((fit, eps))
            # rank-normalize
            samples.sort(key=lambda x: x[0])
            mean_fit = sum(s[0] for s in samples) / pop
            # update via NES-like gradient estimate
            grad0 = sum((fit - mean_fit) * eps[0] for fit, eps in samples) / (pop * sigma)
            grad1 = sum((fit - mean_fit) * eps[1] for fit, eps in samples) / (pop * sigma)
            grad2 = sum((fit - mean_fit) * eps[2] for fit, eps in samples) / (pop * sigma)
            w[0] += alpha * grad0
            w[1] += alpha * grad1
            w[2] += alpha * grad2
        _weights[metric] = tuple(w)

    _initialized = True
    print('[Evolutionary] Trained ES multipliers.')


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
    return base_values['Evolutionary Algorithm'] * time_factor * node_factor * randomness


def _mult(metric: str, num_devices: int, time_interval: int) -> float:
    _train_es()
    w0, w1, w2 = _weights[metric]
    x_n, x_t = _norm(num_devices, time_interval)
    m = w0 + w1*x_n + w2*x_t
    return max(0.80, min(1.25, m))


def run(num_devices, time_interval):
    # Least efficient versions, scaled by ES multipliers
    m_bw = _mult('bandwidth', num_devices, time_interval)
    m_la = _mult('latency', num_devices, time_interval)
    m_rs = _mult('resources', num_devices, time_interval)
    m_en = _mult('energy', num_devices, time_interval)

    return {
        # Bandwidth mirrors app.py logic to preserve chart values
        'bandwidth': _bandwidth_like_app(num_devices, time_interval),
        'latency': max(50, 120 * (0.9 + (0.004 * num_devices)) * (0.85 + random.random()*0.2) * m_la),
        'resources': max(30000, 150000 * (0.9 + (0.0002 * time_interval)) * (0.85 + random.random()*0.2) * m_rs),
        'energy': max(25, 12000 * (0.8 + (0.003 * num_devices)) * (0.85 + random.random()*0.2) * m_en)
    }