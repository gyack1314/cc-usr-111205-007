"""Hawkes分析:递推核对、约束生效、空窗、违例报告、截短不泄漏与诊断口径。"""
import math
import unittest

from response_data.io import load
from response_data import hawkes


def _analyze(name, fracs=(0.5, 1.0)):
    v, digest = load(f'data/{name}')
    return v, hawkes.analyze(v, digest, list(fracs))


class KernelTests(unittest.TestCase):
    def test_recursion_matches_direct_sum(self):
        v, _ = load('data/normal.json')
        cfg = hawkes.parse_config(v)
        windows, _ = hawkes.valid_windows(v)
        for w in windows:
            for b in cfg['beta_candidates']:
                rec = hawkes.recursion_A(w['times'], b)
                dire = hawkes.recursion_A_direct(w['times'], b)
                for x, y in zip(rec, dire):
                    self.assertAlmostEqual(x, y, places=12)

    def test_rescaled_identity_and_tail(self):
        # 完整间隔tau之和+右删失尾=闭式积分;尾不进经验分布
        _, rep = _analyze('normal.json')
        sel = rep['fit']['selected']
        mu, alpha, beta = sel['mu'], sel['alpha'], sel['beta']
        v, _ = load('data/normal.json')
        windows, _ = hawkes.valid_windows(v)
        n_events = 0
        for w in windows:
            rows = hawkes.rescaled_intervals(w, mu, alpha, beta)
            self.assertEqual(rows[-1]['kind'], 'censored_tail')
            closed = mu * w['T'] + alpha / beta * hawkes.tail_sum(w['times'], w['T'], beta)
            self.assertAlmostEqual(sum(r['tau'] for r in rows), closed, places=9)
            n_events += sum(1 for r in rows if r['kind'] == 'complete')
            n_events += 0  # 尾不算完整间隔
        dev_n = sum(len(w['times']) for w in windows if w['split'] == 'development')
        self.assertEqual(rep['diagnostics']['development']['complete_intervals'], dev_n)
        self.assertEqual(len(rep['diagnostics']['development']['censored_tails']),
                         len(rep['splits']['development']))

    def test_decomposition_sums_to_one_and_first_event_background(self):
        _, rep = _analyze('normal.json')
        sel = rep['fit']['selected']
        v, _ = load('data/normal.json')
        windows, _ = hawkes.valid_windows(v)
        for w in windows:
            sums = {}
            for row in hawkes.decomposition_rows(w, sel['mu'], sel['alpha'], sel['beta']):
                sums[row[2]] = sums.get(row[2], 0.0) + row[7]
            for s in sums.values():
                self.assertAlmostEqual(s, 1.0, places=12)
            if w['times']:
                first = [r for r in hawkes.decomposition_rows(w, sel['mu'], sel['alpha'], sel['beta'])
                         if r[2] == 1]
                self.assertEqual(len(first), 1)
                self.assertEqual(first[0][5], 'background')
                self.assertAlmostEqual(first[0][7], 1.0, places=12)  # 窗前历史为空


class FitTests(unittest.TestCase):
    def test_solver_certificate_and_stability(self):
        _, rep = _analyze('normal.json')
        cfg = rep['config']
        for c in rep['fit']['candidates']:
            self.assertEqual(c['status'], 'converged')
            self.assertLessEqual(c['kkt_residual'], 1e-6)
            self.assertLessEqual(c['dual_gap'], 1e-6)
            self.assertLessEqual(c['alpha'], cfg['r_max'] * c['beta'] + 1e-12)
            self.assertLessEqual(c['alpha'], cfg['alpha_bounds'][1] + 1e-12)
            self.assertGreaterEqual(c['mu'], cfg['mu_bounds'][0])

    def test_stability_constraint_binds_during_solve(self):
        # 强聚集窗+小r_max:无约束解会越界,约束须在求解中生效而非事后回裁
        times = []
        for k in range(20):
            times += [10 + 40 * k + 0.1 * j for j in range(5)]
        v = {'windows': [{'id': 'c1', 'split': 'development', 'end': 1000, 'times': times}],
             'beta_candidates': [0.5, 1, 2], 'mu_bounds': [0.01, 3],
             'alpha_bounds': [0, 2], 'r_max': 0.3,
             'solver': {'tolerance': 1e-09, 'max_iterations': 1000},
             'diagnostic_grid': [0, 1, 2]}
        rep = hawkes.analyze(v, 'test', [1.0])
        active = 0
        for c in rep['fit']['candidates']:
            self.assertEqual(c['status'], 'converged')
            self.assertLessEqual(c['alpha'], 0.3 * c['beta'] + 1e-12)
            self.assertLessEqual(c['kkt_residual'], 1e-6)
            self.assertLessEqual(c['dual_gap'], 1e-6)
            self.assertLessEqual(c['branching_ratio'], 0.3 + 1e-12)
            if c['stability_constraint_active']:
                active += 1
                self.assertAlmostEqual(c['alpha'], 0.3 * c['beta'], places=9)
        self.assertGreater(active, 0)

    def test_grid_selection_is_finite_best_and_earliest_tie(self):
        _, rep = _analyze('normal.json')
        cands = rep['fit']['candidates']
        best_ll = max(c['loglik'] for c in cands)
        sel = rep['fit']['selected']
        # 所选与网格最大者之差在求解证书噪声内(有限网格最佳,不声称连续全局最优)
        self.assertLessEqual(best_ll - sel['loglik'], 1e-8 * max(1.0, abs(best_ll)))
        self.assertEqual(sel['beta'], rep['config']['beta_candidates'][0])  # 同目标取先列者
        c1 = {'loglik': 1.0, 'beta': 0.5}
        c2 = {'loglik': 1.0, 'beta': 1.0}
        idx, tie = hawkes.select_candidate([dict(c1), dict(c2), {'loglik': 0.5, 'beta': 2.0}])
        self.assertEqual((idx, tie), (0, True))

    def test_hawkes_never_worse_than_poisson(self):
        # Poisson是alpha=0的可行特例,约束最优不得更差;normal.json上自激不可识别,如实给alpha=0
        _, rep = _analyze('normal.json')
        delta = rep['clustering']['poisson_reference']['hawkes_minus_poisson_loglik']
        self.assertGreaterEqual(delta, -1e-9)
        self.assertEqual(rep['fit']['selected']['alpha'], 0.0)
        self.assertEqual(rep['fit']['selected']['alpha_bound'], 'lower')
        self.assertTrue(rep['fit']['tie'])  # alpha=0时beta不可识别,同目标取先列者
        self.assertEqual(rep['fit']['selected']['beta'], rep['config']['beta_candidates'][0])

    def test_empty_development_window(self):
        _, rep = _analyze('empty-development-window.json')
        sel = rep['fit']['selected']
        self.assertEqual(sel['status'], 'converged')
        dev_a = next(r for r in rep['fit']['dev_windows'] if r['window'] == 'dev-a')
        self.assertEqual(dev_a['events'], 0)
        self.assertAlmostEqual(dev_a['loglik'], -sel['mu'] * dev_a['T'], places=12)  # 空窗仍贡献-mu*T
        tail = next(t for t in rep['diagnostics']['development']['censored_tails']
                    if t['window'] == 'dev-a')
        self.assertAlmostEqual(tail['tau'], sel['mu'] * dev_a['T'], places=9)

    def test_long_tail_residual_kept_in_integral(self):
        _, rep = _analyze('long-unobserved-event-tail.json')
        sel = rep['fit']['selected']
        dev_a = next(r for r in rep['fit']['dev_windows'] if r['window'] == 'dev-a')
        # T=100,最后事件9.5:积分必须覆盖到T,残尾保留
        closed = sel['mu'] * 100 + sel['alpha'] / sel['beta'] * sum(
            1 - math.exp(-sel['beta'] * (100 - t)) for t in (1, 1.2, 2.3, 5.1, 5.4, 9.5))
        self.assertAlmostEqual(dev_a['integral'], closed, places=9)
        tail = next(t for t in rep['diagnostics']['development']['censored_tails']
                    if t['window'] == 'dev-a')
        self.assertAlmostEqual(tail['end'] - tail['start'], 90.5, places=9)


class DomainTests(unittest.TestCase):
    def test_invalid_window_reported_not_jittered(self):
        v, rep = _analyze('invalid-domain.json')
        self.assertEqual(len(rep['violations']), 1)
        self.assertEqual(rep['violations'][0]['window'], 'dev-a')
        self.assertIn('dev-a', rep['splits']['excluded'])
        self.assertEqual(rep['splits']['development'], ['dev-b'])
        # 时刻保持原值,未抖动
        self.assertEqual(v['windows'][0]['times'], [1, 1])
        # 其余窗照常拟合与外期评分
        self.assertEqual(rep['fit']['selected']['status'], 'converged')
        self.assertEqual(len(rep['external']['windows']), 1)

    def test_truncation_uses_only_past_and_scores_full_external(self):
        _, rep = _analyze('normal.json', fracs=(0.5, 1.0))
        rows = {r['fraction']: r for r in rep['truncation']}
        self.assertEqual(rows[0.5]['kept_events'], 6)  # dev-a[1,1.2,2.3]+dev-b[0.5,3,3.1]
        self.assertEqual(rows[1.0]['kept_events'], 10)
        # f=1的截短拟合与主拟合一致
        sel = rep['fit']['selected']
        self.assertAlmostEqual(rows[1.0]['selected']['mu'], sel['mu'], places=9)
        self.assertAlmostEqual(rows[1.0]['selected']['beta'], sel['beta'], places=12)
        # 截短模型的外期评分用完整external窗(4事件,冻结参数)
        s = rows[0.5]['selected']
        v, _ = load('data/normal.json')
        ext = [w for w in hawkes.valid_windows(v)[0] if w['split'] == 'external']
        manual = sum(hawkes.window_loglik(w, s['mu'], s['alpha'], s['beta'])['loglik']
                     for w in ext)
        self.assertAlmostEqual(rows[0.5]['external_total_loglik'], manual, places=9)
        self.assertEqual(sum(len(w['times']) for w in ext), 4)

    def test_verification_errors_small(self):
        _, rep = _analyze('normal.json')
        ver = rep['verification']
        self.assertLess(ver['recursion_max_abs_err'], 1e-12)
        self.assertLess(ver['rescaled_identity_max_abs_err'], 1e-9)
        self.assertLess(ver['quadrature_max_abs_err'], 1e-4)
        self.assertLess(ver['decomposition_sum_max_abs_err'], 1e-12)


if __name__ == '__main__':
    unittest.main()
