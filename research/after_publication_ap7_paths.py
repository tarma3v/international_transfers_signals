"""Causal path libraries, chronological distribution splits, scenario utilities."""
import datetime as dt

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor

from ml.targets import HORIZONS
from research.after_publication_ap1 import SEED
from research.after_publication_ap5_learning import available_mask

FAMILIES = ('global64', 'global128', 'wide128', 'local32', 'shrink', 'recent128',
            'unconditional', 'forest', 'ridge_empirical', 'ridge_gaussian', 'ridge_local')


def normalized_paths(series, panel, scale):
    paths = np.full((len(panel), 20), np.nan)
    for row, (c, i) in enumerate(zip(panel.currency, panel.announced_index)):
        i = int(i)
        prices = series[c].values
        if i + 20 < len(prices):
            paths[row] = 1e4 * np.log(prices[i + 1:i + 21] / prices[i]) / scale[row]
    return paths


def scenario_summary(paths, weights, scale, known_past):
    if not np.isfinite(paths).all() or not np.isfinite(weights).all():
        raise ValueError('Only finite mature scenarios can be evaluated')
    if len(paths) != len(weights) or (weights < 0).any() or not np.isclose(weights.sum(), 1.):
        raise ValueError('Scenario weights must be nonnegative and sum to one')
    log_ratios = paths * scale / 1e4
    clipped = int((np.abs(log_ratios) > 20).sum())
    ratios = np.exp(np.clip(log_ratios, -20, 20))
    survival, forward, symmetric = [], [], []
    for j, h in enumerate(HORIZONS):
        survival.append(weights @ (paths[:, :h].min(axis=1) >= 0))
        future_sum = ratios[:, :h].sum(axis=1)
        forward.append(weights @ (1e4 * (1 - h / future_sum)))
        reference_sum = known_past[j] + 1 + future_sum
        symmetric.append(weights @ (1e4 * (1 - (2 * h + 1) / reference_sum)))
    return np.array(survival), np.array(forward), np.array(symmetric), clipped


class NeighborLibrary:
    def __init__(self, X, ids, k):
        if not len(ids):
            raise ValueError('Empty analog library')
        self.ids, self.k = np.asarray(ids), min(k, len(ids))
        self.mean = X[ids].mean(axis=0)
        self.scale = np.maximum(X[ids].std(axis=0), 1e-6)
        self.Z = (X[ids] - self.mean) / self.scale

    def query(self, x):
        distance = np.sum((self.Z - (x - self.mean) / self.scale) ** 2, axis=1)
        chosen = np.argsort(distance, kind='stable')[:self.k]
        bandwidth = max(float(np.median(distance[chosen])), 1e-8)
        weight = np.exp(-distance[chosen] / bandwidth)
        weight /= weight.sum()
        return self.ids[chosen], weight


def chronological_split(ids, dates, maturity):
    days = np.unique(dates[ids])
    if len(days) < 3:
        return np.array([], int), np.array([], int), None
    boundary = days[min(int(.6 * len(days)), len(days) - 1)]
    structure = ids[(dates[ids] < boundary) & (maturity[ids] < boundary - dt.timedelta(days=2))]
    estimation = ids[dates[ids] >= boundary]
    return structure, estimation, boundary


class SplitPathModel:
    """Structure/mean learned earlier than every held-out residual/leaf outcome."""
    def __init__(self, X, paths, structure, estimation, kind):
        self.estimation, self.kind = estimation, kind
        self.mean, self.scale = X[structure].mean(axis=0), np.maximum(X[structure].std(axis=0), 1e-6)
        train = (X[structure] - self.mean) / self.scale
        calibration = (X[estimation] - self.mean) / self.scale
        self.paths = paths[estimation]
        if kind == 'forest':
            self.trees = []
            target = paths[structure][:, np.array(HORIZONS) - 1] / np.sqrt(HORIZONS)
            for k in range(32):
                tree = DecisionTreeRegressor(max_leaf_nodes=12, min_samples_leaf=30,
                    max_features=.8, random_state=SEED + k)
                tree.fit(train, target)
                self.trees.append((tree, tree.apply(calibration)))
        else:
            self.model = Ridge(alpha=100.).fit(train, paths[structure])
            self.residual = paths[estimation] - self.model.predict(calibration)
            if kind == 'gaussian':
                covariance = np.cov(self.residual, rowvar=False)
                covariance = .5 * covariance + .5 * np.diag(np.diag(covariance)) + 1e-8 * np.eye(20)
                noise = np.random.default_rng(SEED).normal(size=(128, 20))
                self.gaussian_residual = (np.vstack((noise, -noise)) @ np.linalg.cholesky(covariance).T
                                          + self.residual.mean(axis=0))

    def query(self, x):
        xx = ((x - self.mean) / self.scale)[None, :]
        if self.kind == 'forest':
            weights = np.zeros(len(self.estimation))
            for tree, leaves in self.trees:
                same = leaves == tree.apply(xx)[0]
                if not same.any():
                    same[:] = True
                weights += same / same.sum() / len(self.trees)
            return self.paths, weights
        residual = self.gaussian_residual if self.kind == 'gaussian' else self.residual
        return self.model.predict(xx) + residual, np.full(len(residual), 1. / len(residual))


def forecast_paths(panel, core, wide, paths, scale, past_sums, maturity):
    dates, currencies = panel.date.to_numpy(), panel.currency.to_numpy()
    complete = np.isfinite(paths).all(axis=1)
    source = np.isfinite(core).all(axis=1) & np.isfinite(wide).all(axis=1)
    values = {name: {k: np.full((len(panel), 5), np.nan) for k in ('probability', 'forward', 'symmetric')}
              for name in FAMILIES}
    neighbors = {k: np.full((len(panel), n), -1, dtype=np.int32) for k, n in (('global128', 128), ('local32', 32))}
    neighbor_weights = {k: np.zeros_like(v, dtype=float) for k, v in neighbors.items()}
    train_logs, scenario_logs = [], []
    months = pd.period_range('2022-07', pd.Timestamp(max(dates)).to_period('M'), freq='M')
    for month in months:
        origin, end = month.start_time.date(), (month + 1).start_time.date()
        query_ids = np.flatnonzero((dates >= origin) & (dates < end) & source)
        if not len(query_ids):
            continue
        ids = np.flatnonzero(available_mask(dates, maturity, origin, complete) & source)
        if not len(ids):
            raise ValueError('Queries precede available mature path library')
        recent = ids[dates[ids] >= origin - dt.timedelta(days=730)]
        libraries = {'global64': NeighborLibrary(core, ids, 64), 'global128': NeighborLibrary(core, ids, 128),
            'wide128': NeighborLibrary(wide, ids, 128), 'recent128': NeighborLibrary(core, recent, 128)}
        local_ids, local_libraries, split_models = {}, {}, {}
        for c in sorted(set(currencies[query_ids])):
            own = ids[currencies[ids] == c]
            local_ids[c] = own
            local_libraries[c] = NeighborLibrary(core, own, 32) if len(own) >= 20 else libraries['global128']
        for key, library in [('ALL', ids), *list(local_ids.items())]:
            structure, estimation, boundary = chronological_split(library, dates, maturity)
            minimum_train, minimum_est = (100, 30) if key == 'ALL' else (30, 12)
            fitted = len(structure) >= minimum_train and len(estimation) >= minimum_est
            kinds = ('forest', 'empirical', 'gaussian') if key == 'ALL' else ('empirical',)
            for kind in kinds:
                split_models[key, kind] = SplitPathModel(core, paths, structure, estimation, kind) if fitted else None
            train_logs.append({'origin': str(origin), 'currency': key, 'n_library': len(library),
                'n_structure': len(structure), 'n_estimation': len(estimation), 'fitted': fitted,
                'split_date': str(boundary) if boundary else None,
                'last_structure_maturity': str(max(maturity[structure])) if len(structure) else None,
                'last_library_maturity': str(max(maturity[library])) if len(library) else None,
                'last_library_date': str(max(dates[library])) if len(library) else None})
        for i in query_ids:
            c = currencies[i]
            scenarios = {}
            for key, library in libraries.items():
                ni, w = library.query(wide[i] if key == 'wide128' else core[i])
                scenarios[key] = paths[ni], w
                if key == 'global128':
                    neighbors[key][i, :len(ni)], neighbor_weights[key][i, :len(ni)] = ni, w
            ni, w = local_libraries[c].query(core[i])
            scenarios['local32'] = paths[ni], w
            if len(ni) <= 32:
                neighbors['local32'][i, :len(ni)], neighbor_weights['local32'][i, :len(ni)] = ni, w
            # In current data all local libraries have >=20 rows; tests can exercise fallback.
            fraction = len(local_ids[c]) / (len(local_ids[c]) + 200.)
            global_p, global_w = scenarios['global128']
            scenarios['shrink'] = np.vstack((paths[ni], global_p)), np.r_[fraction * w, (1 - fraction) * global_w]
            scenarios['unconditional'] = paths[ids], np.full(len(ids), 1. / len(ids))
            for name, key, kind in (('forest', 'ALL', 'forest'), ('ridge_empirical', 'ALL', 'empirical'),
                                   ('ridge_gaussian', 'ALL', 'gaussian'), ('ridge_local', c, 'empirical')):
                model = split_models[key, kind]
                scenarios[name] = model.query(core[i]) if model is not None else scenarios['global128' if key == 'ALL' else 'local32']
            for name, (scenario, weight) in scenarios.items():
                probability, forward, symmetric, clips = scenario_summary(scenario, weight, scale[i], past_sums[i])
                for key, vector in zip(('probability', 'forward', 'symmetric'), (probability, forward, symmetric)):
                    values[name][key][i] = vector
                scenario_logs.append({'row': int(i), 'date': str(dates[i]), 'currency': c, 'origin': str(origin),
                    'model': name, 'n_scenarios': len(weight), 'effective_scenarios': float(1. / np.sum(weight ** 2)),
                    'weight_sum': float(weight.sum()), 'min_weight': float(weight.min()), 'clipped_log_ratios': clips})
        print(f'AP7 {origin}: {len(ids)} mature paths, {len(query_ids)} queries', flush=True)
    return values, train_logs, scenario_logs, neighbors, neighbor_weights
