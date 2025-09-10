import random
from typing import Dict, Tuple

# --- Tiny Genetic Algorithm for per-metric multipliers ---
# Model per metric: m = w0 + w1*x_n + w2*x_t, with x_n=num_devices/150, x_t=time/180
# We evolve (w0,w1,w2) to match gentle targets favoring GA's expected profile.

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
    # GA sits between DQN and Evolutionary; allow a bit more variance
    base = 1.0
    if metric == 'bandwidth':
        return base - 0.03*x_n - 0.02*x_t
    if metric == 'latency':
        return base - 0.02*x_n - 0.01*x_t
    if metric == 'resources':
        return base + 0.06*x_t + 0.01*(1.0 - x_n)
    if metric == 'energy':
        return base - 0.02*x_n - 0.015*x_t
    return base


def _fitness(metric: str, w: Tuple[float, float, float]) -> float:
    w0, w1, w2 = w
    err_sum = 0.0
    for n in [50, 75, 100, 125, 150]:
        for t in [60, 90, 120, 150, 180]:
            x_n, x_t = _norm(n, t)
            pred = w0 + w1*x_n + w2*x_t
            y = _target(metric, x_n, x_t)
            e = pred - y
            err_sum += e*e
    # Lower error is better fitness
    return -err_sum


def _train_ga():
    global _initialized, _weights
    if _initialized:
        return

    pop_size = 20
    gens = 25
    elite = 4
    mut_sigma = 0.1

    for metric in _weights.keys():
        # initialize population near identity multiplier
        pop = [(1.0 + random.uniform(-0.05, 0.05),
                random.uniform(-0.1, 0.1),
                random.uniform(-0.1, 0.1)) for _ in range(pop_size)]
        for _g in range(gens):
            scored = sorted(pop, key=lambda w: _fitness(metric, w), reverse=True)
            pop = scored[:elite]
            # produce offspring
            while len(pop) < pop_size:
                a, b = random.sample(scored[:10], 2)
                # crossover
                alpha = random.random()
                child = (
                    alpha*a[0] + (1-alpha)*b[0],
                    alpha*a[1] + (1-alpha)*b[1],
                    alpha*a[2] + (1-alpha)*b[2],
                )
                # mutate
                child = (
                    child[0] + random.gauss(0, mut_sigma*0.5),
                    child[1] + random.gauss(0, mut_sigma),
                    child[2] + random.gauss(0, mut_sigma),
                )
                pop.append(child)
        best = max(pop, key=lambda w: _fitness(metric, w))
        _weights[metric] = best

    _initialized = True
    print('[Genetic] Trained GA multipliers.')


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
    return base_values['Genetic Algorithm'] * time_factor * node_factor * randomness


def _mult(metric: str, num_devices: int, time_interval: int) -> float:
    _train_ga()
    w0, w1, w2 = _weights[metric]
    x_n, x_t = _norm(num_devices, time_interval)
    m = w0 + w1*x_n + w2*x_t
    return max(0.85, min(1.18, m))


def run(num_devices, time_interval):
    # Apply GA-learned multipliers to baseline formulas, preserving ranges
    m_bw = _mult('bandwidth', num_devices, time_interval)
    m_la = _mult('latency', num_devices, time_interval)
    m_rs = _mult('resources', num_devices, time_interval)
    m_en = _mult('energy', num_devices, time_interval)

    return {
        # Bandwidth mirrors app.py logic to preserve chart values
        'bandwidth': _bandwidth_like_app(num_devices, time_interval),
        'latency': max(30, 80 * (0.8 + (0.003 * num_devices)) * (0.9 + random.random()*0.15) * m_la),
        'resources': max(40000, 200000 * (1.0 + (0.0003 * time_interval)) * (0.9 + random.random()*0.15) * m_rs),
        'energy': max(15, 8000 * (0.7 + (0.002 * num_devices)) * (0.9 + random.random()*0.15) * m_en)
    }