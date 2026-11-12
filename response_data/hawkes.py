"""指数核单变量Hawkes分析:固定beta网格约束似然、冻结外期评分、强度分解与时间变换诊断。

口径(见docs/data-contract.md):
- 各窗独立起窗,窗前历史为空,窗间不传播激励;强度在事件自身发生前计算,t_i不含自身。
- development窗共享(mu,alpha,beta),似然为各窗之和;external窗用冻结参数按自身事件更新条件强度,不重拟合。
- 每个固定beta内,在公开有限边界上最大化点过程log似然;稳定约束alpha<=r_max*beta与alpha_bounds共同生效,不在拟合后回裁。
- 似然积分覆盖到T,保留每个事件的激励残尾alpha/beta*(1-exp(-beta*(T-t_i)));空开发窗仍贡献-mu*T。
- 不符合简单点过程假设(同刻、越窗、非数值)的窗只报告并剔除,不抖动时刻。
- 固定beta下log似然对(mu,alpha)为凹函数,投影Newton求解,并以投影梯度KKT残差与凹函数线性化对偶间隙作证。
- 背景/核份额是事件时刻的条件强度分解,分母与似然相同,不是因果传播链。
- 时间变换诊断只用事件间完整积分;窗末无事件尾为右删失,单列且不进经验分布;同资料拟合的诊断不套独立同分布检验p值。
"""
import csv
import json
import math
from pathlib import Path

from .io import number, positive, unique


# ---------- 输入解析与窗校验 ----------

def _bounds(pair, name, lo_positive):
    if not isinstance(pair, list) or len(pair) != 2:
        raise ValueError(f'{name}必须为[下界,上界]')
    lo, hi = number(pair[0]), number(pair[1])
    if not lo < hi:
        raise ValueError(f'{name}下界须小于上界')
    if lo_positive and lo <= 0:
        raise ValueError(f'{name}下界须为正')
    if not lo_positive and lo < 0:
        raise ValueError(f'{name}下界须非负')
    return lo, hi


def parse_config(v):
    """读取外给网格、公开边界、稳定上限与求解设置;mu正有限边界不是自动估出的下限。"""
    mu_b = _bounds(v['mu_bounds'], 'mu_bounds', True)
    a_b = _bounds(v['alpha_bounds'], 'alpha_bounds', False)
    r_max = number(v['r_max'])
    if not 0 < r_max < 1:
        raise ValueError('稳定上限无效')
    betas = [positive(b) for b in v['beta_candidates']]
    if not betas:
        raise ValueError('beta候选为空')
    solver = v.get('solver', {})
    tol = positive(solver.get('tolerance', 1e-9))
    max_iter = solver.get('max_iterations', 1000)
    if type(max_iter) is not int or max_iter < 1:
        raise ValueError('max_iterations须为正整数')
    grid = [number(x) for x in v.get('diagnostic_grid', [])]
    if any(x < 0 for x in grid) or any(grid[i] >= grid[i + 1] for i in range(len(grid) - 1)):
        raise ValueError('diagnostic_grid须非负且严格递增')
    return {'mu_bounds': mu_b, 'alpha_bounds': a_b, 'r_max': r_max,
            'beta_candidates': betas, 'tolerance': tol, 'max_iterations': max_iter,
            'diagnostic_grid': grid}


def valid_windows(v):
    """逐窗校验(0,T]内严格递增;违例窗报告并剔除,不抖动时刻。返回(有效窗,违例)。"""
    unique(v['windows'], 'id')
    valid, violations = [], []
    for w in v['windows']:
        wid = w.get('id')
        try:
            T = positive(w['end'])
            split = w['split']
            if split not in ('development', 'external'):
                raise ValueError(f'未知split:{split!r}')
            times = [number(x) for x in w['times']]
            last = 0.0
            for x in times:
                if not last < x <= T:
                    raise ValueError(f'时刻{x}不严格递增或越窗(0,{T}];同刻事件无先后顺序,简单点过程假设不成立')
                last = x
        except (KeyError, TypeError, ValueError) as e:
            violations.append({'window': wid, 'reason': str(e), 'action': '该窗剔除,其余窗继续;时刻保持原值未抖动'})
            continue
        valid.append({'id': wid, 'split': split, 'T': T, 'times': times})
    return valid, violations


# ---------- 核量:O(n)递推,不为每个候选构造事件对矩阵 ----------

def recursion_A(times, beta):
    """A_i=sum_{j<i}exp(-beta*(t_i-t_j)),不含自身,O(n)递推。"""
    out, a, prev = [], 0.0, None
    for t in times:
        if prev is not None:
            a = math.exp(-beta * (t - prev)) * (1.0 + a)
        out.append(a)
        prev = t
    return out


def recursion_A_direct(times, beta):
    """直接双重求和,仅用于小资料核对递推。"""
    return [sum(math.exp(-beta * (t - s)) for s in times[:i]) for i, t in enumerate(times)]


def tail_sum(times, T, beta):
    """B=sum_i(1-exp(-beta*(T-t_i))):每个事件到T的激励残尾,积分覆盖到T。"""
    return sum(1.0 - math.exp(-beta * (T - t)) for t in times)


def _objective(windows, beta, mu, alpha):
    """固定beta的总log似然、梯度与Hessian;对(mu,alpha)为凹。"""
    ll = g_mu = g_a = h_mm = h_ma = h_aa = 0.0
    for w in windows:
        A = recursion_A(w['times'], beta)
        B = tail_sum(w['times'], w['T'], beta)
        for a in A:
            lam = mu + alpha * a
            ll += math.log(lam)
            inv = 1.0 / lam
            g_mu += inv
            g_a += a * inv
            h_mm -= inv * inv
            h_ma -= a * inv * inv
            h_aa -= a * a * inv * inv
        ll -= mu * w['T'] + alpha / beta * B
        g_mu -= w['T']
        g_a -= B / beta
    return ll, (g_mu, g_a), (h_mm, h_ma, h_aa)


# ---------- 固定beta的约束最大化 ----------

def _kkt_residual(x, g, lo0, hi0, lo1, hi1):
    """盒约束最大化的一阶必要条件:||x-Proj(x+g)||_inf。"""
    p0 = min(max(x[0] + g[0], lo0), hi0)
    p1 = min(max(x[1] + g[1], lo1), hi1)
    return max(abs(x[0] - p0), abs(x[1] - p1))


def _dual_gap(x, g, lo0, hi0, lo1, hi1):
    """凹目标线性化上界:max_{y in box} g.(y-x);loglik* <= loglik(x)+gap。"""
    gap = g[0] * ((hi0 - x[0]) if g[0] > 0 else (lo0 - x[0]))
    gap += g[1] * ((hi1 - x[1]) if g[1] > 0 else (lo1 - x[1]))
    return max(gap, 0.0)


def _bound_flag(x, lo, hi):
    tol = 1e-9 * max(1.0, abs(lo), abs(hi))
    if x - lo <= tol:
        return 'lower'
    if hi - x <= tol:
        return 'upper'
    return None


def fit_fixed_beta(windows, beta, cfg):
    """固定beta,在mu_bounds与[alpha_lo,min(alpha_hi,r_max*beta)]上最大化总log似然。

    稳定约束在求解中生效,不是拟合后回裁。返回目标、梯度、KKT残差、对偶间隙、
    贴边、Hessian特征值、平坦标记与求解状态;未收敛如实报告,不跳过。
    """
    mu_lo, mu_hi = cfg['mu_bounds']
    a_lo = cfg['alpha_bounds'][0]
    a_hi = min(cfg['alpha_bounds'][1], cfg['r_max'] * beta)
    tol, max_iter = cfg['tolerance'], cfg['max_iterations']
    n_tot = sum(len(w['times']) for w in windows)
    T_tot = sum(w['T'] for w in windows)
    base = {'beta': beta, 'alpha_upper_effective': a_hi,
            'stability_bound': cfg['r_max'] * beta, 'alpha_bound_given': cfg['alpha_bounds'][1]}
    if a_lo > a_hi:
        return {**base, 'status': 'infeasible-bounds', 'iterations': 0, 'loglik': None,
                'mu': None, 'alpha': None, 'branching_ratio': None, 'grad': None,
                'kkt_residual': None, 'dual_gap': None, 'mu_bound': None, 'alpha_bound': None,
                'stability_constraint_active': None, 'neg_hessian_eigs': None, 'flat_direction': None}
    if n_tot == 0:
        # 空开发集:ll=-mu*T_tot,对alpha平坦,最优在mu下界
        ll = -mu_lo * T_tot
        return {**base, 'status': 'closed-form-empty', 'iterations': 0, 'loglik': ll,
                'mu': mu_lo, 'alpha': a_lo, 'branching_ratio': a_lo / beta,
                'grad': [-T_tot, 0.0], 'kkt_residual': 0.0, 'dual_gap': 0.0,
                'mu_bound': 'lower', 'alpha_bound': 'lower' if a_lo == a_hi else None,
                'stability_constraint_active': False, 'neg_hessian_eigs': None,
                'flat_direction': True}
    x = [min(max(n_tot / T_tot, mu_lo), mu_hi), 0.5 * (a_lo + a_hi)]
    ll, g, H = _objective(windows, beta, x[0], x[1])
    status, it = 'max-iterations', 0
    while True:
        kkt = _kkt_residual(x, g, mu_lo, mu_hi, a_lo, a_hi)
        gap = _dual_gap(x, g, mu_lo, mu_hi, a_lo, a_hi)
        if kkt <= tol and gap <= tol * max(1.0, abs(ll)):
            status = 'converged'
            break
        if it >= max_iter:
            break
        h11, h12, h22 = H
        # 活跃集:贴边且梯度指向界外的坐标固定,只对自由坐标做Newton
        free_mu = not ((x[0] <= mu_lo and g[0] < 0) or (x[0] >= mu_hi and g[0] > 0))
        free_a = not ((x[1] <= a_lo and g[1] < 0) or (x[1] >= a_hi and g[1] > 0))
        if free_mu and free_a:
            det = h11 * h22 - h12 * h12
            if det > 1e-13 * (abs(h11 * h22) + 1e-13):
                d = [(-h22 * g[0] + h12 * g[1]) / det, (h12 * g[0] - h11 * g[1]) / det]
            else:
                d = list(g)
            if d[0] * g[0] + d[1] * g[1] <= 0:
                d = list(g)
        elif free_mu:
            d = [-g[0] / h11 if h11 < 0 else g[0], 0.0]
        elif free_a:
            d = [0.0, -g[1] / h22 if h22 < 0 else g[1]]
        else:
            status = 'converged'  # 两坐标均被约束钉住:KKT点
            break
        step, ok = 1.0, False
        for _ in range(60):
            nx = [min(max(x[0] + step * d[0], mu_lo), mu_hi),
                  min(max(x[1] + step * d[1], a_lo), a_hi)]
            nll, ng, nh = _objective(windows, beta, nx[0], nx[1])
            if nll >= ll + 1e-4 * (g[0] * (nx[0] - x[0]) + g[1] * (nx[1] - x[1])):
                ok = True
                break
            step *= 0.5
        if not ok:
            status = 'line-search-failed'
            break
        x, ll, g, H = nx, nll, ng, nh
        it += 1
    kkt = _kkt_residual(x, g, mu_lo, mu_hi, a_lo, a_hi)
    gap = _dual_gap(x, g, mu_lo, mu_hi, a_lo, a_hi)
    h11, h12, h22 = H
    tr, det = -(h11 + h22), h11 * h22 - h12 * h12
    disc = math.sqrt(max(tr * tr - 4.0 * det, 0.0))
    e_max, e_min = 0.5 * (tr + disc), 0.5 * (tr - disc)
    flat = e_max <= 0.0 or e_min <= 1e-10 * e_max
    mu_b = _bound_flag(x[0], mu_lo, mu_hi)
    a_b = _bound_flag(x[1], a_lo, a_hi)
    stab_active = (a_b == 'upper') and cfg['r_max'] * beta < cfg['alpha_bounds'][1]
    return {**base, 'status': status, 'iterations': it, 'loglik': ll,
            'mu': x[0], 'alpha': x[1], 'branching_ratio': x[1] / beta,
            'grad': list(g), 'kkt_residual': kkt, 'dual_gap': gap,
            'mu_bound': mu_b, 'alpha_bound': a_b,
            'stability_constraint_active': stab_active,
            'neg_hessian_eigs': [e_min, e_max], 'flat_direction': flat}


def select_candidate(candidates):
    """总似然最大者;同目标(1e-8相对容差内,小于求解证书噪声)取先列者。返回(下标,是否同目标)。"""
    def tol_of(ll):
        return 1e-8 * max(1.0, abs(ll))
    best = None
    for i, c in enumerate(candidates):
        if c['loglik'] is None:
            continue
        if best is None or c['loglik'] > candidates[best]['loglik'] + tol_of(candidates[best]['loglik']):
            best = i
    tie = False
    if best is not None:
        bl = candidates[best]['loglik']
        tie = any(i != best and c['loglik'] is not None and abs(c['loglik'] - bl) <= tol_of(bl)
                  for i, c in enumerate(candidates))
    return best, tie


def fit_grid(windows, cfg):
    cands = [fit_fixed_beta(windows, b, cfg) for b in cfg['beta_candidates']]
    best, tie = select_candidate(cands)
    if best is not None:
        cands[best]['selected'] = True
    return cands, best, tie


# ---------- 冻结参数评分、分解与时间变换 ----------

def window_loglik(w, mu, alpha, beta):
    """单窗条件log似然:强度在事件前计算,积分覆盖到T并保留残尾。"""
    A = recursion_A(w['times'], beta)
    B = tail_sum(w['times'], w['T'], beta)
    sll = sum(math.log(mu + alpha * a) for a in A)
    integral = mu * w['T'] + alpha / beta * B
    return {'window': w['id'], 'split': w['split'], 'events': len(w['times']), 'T': w['T'],
            'sum_log_lambda': sll, 'integral': integral, 'loglik': sll - integral,
            'mean_log_lambda': sll / len(A) if A else None}


def decomposition_rows(w, mu, alpha, beta):
    """逐事件产出(背景,各先前事件)份额行,分母与似然相同;可流式写出。"""
    A = recursion_A(w['times'], beta)
    for i, t in enumerate(w['times']):
        lam = mu + alpha * A[i]
        yield [w['id'], w['split'], i + 1, t, lam, 'background', '', mu / lam]
        for j in range(i):
            share = alpha * math.exp(-beta * (t - w['times'][j])) / lam
            yield [w['id'], w['split'], i + 1, t, lam, 'event', w['times'][j], share]


def rescaled_intervals(w, mu, alpha, beta):
    """相邻事件间强度积分tau_i;末段到T的无事件尾为右删失,单列不进经验分布。"""
    A = recursion_A(w['times'], beta)
    rows, prev = [], 0.0
    for i, t in enumerate(w['times']):
        s_prev = 1.0 + A[i - 1] if i else 0.0
        tau = mu * (t - prev) + alpha / beta * (s_prev - A[i])
        rows.append({'window': w['id'], 'split': w['split'], 'index': i + 1,
                     'start': prev, 'end': t, 'tau': tau, 'kind': 'complete'})
        prev = t
    if w['times']:
        s_end = 1.0 + A[-1]
        tail = mu * (w['T'] - prev) + alpha / beta * s_end * (1.0 - math.exp(-beta * (w['T'] - prev)))
    else:
        tail = mu * w['T']
    rows.append({'window': w['id'], 'split': w['split'], 'index': len(w['times']) + 1,
                 'start': prev, 'end': w['T'], 'tau': tail, 'kind': 'censored_tail'})
    return rows


def _cdf_rows(taus, grid):
    n = len(taus)
    rows = []
    for x in grid:
        emp = sum(1 for t in taus if t <= x) / n if n else None
        ref = 1.0 - math.exp(-x)
        rows.append({'grid': x, 'empirical_cdf': emp, 'exp1_cdf': ref,
                     'diff': emp - ref if emp is not None else None})
    return rows


def diagnostics(windows, mu, alpha, beta, grid):
    """时间变换诊断:只用完整间隔;末段右删失尾单列;不套iid检验p值。"""
    taus, tails = [], []
    for w in windows:
        for r in rescaled_intervals(w, mu, alpha, beta):
            if r['kind'] == 'complete':
                taus.append(r['tau'])
            else:
                tails.append({'window': r['window'], 'start': r['start'], 'end': r['end'], 'tau': r['tau']})
    n = len(taus)
    mean = sum(taus) / n if n else None
    var = sum((t - mean) ** 2 for t in taus) / n if n else None
    return {'complete_intervals': n, 'mean': mean, 'variance': var,
            'min': min(taus) if taus else None, 'max': max(taus) if taus else None,
            'censored_tails': tails, 'cdf': _cdf_rows(taus, grid)}


def excitation_share(windows, mu, alpha, beta):
    """事件时刻强度中自激占比的均值:条件强度分解,不是因果链。"""
    vals = []
    for w in windows:
        for a in recursion_A(w['times'], beta):
            vals.append(alpha * a / (mu + alpha * a))
    return sum(vals) / len(vals) if vals else None


# ---------- 小资料核对:直接求和、恒等式、数值积分 ----------

def _quadrature(w, mu, alpha, beta, steps=4000):
    """中点法数值积分int_0^T lambda dt,事件时刻并入节点;仅小资料核对用。"""
    n = max(steps, 8 * len(w['times']) + 8)
    h = w['T'] / n
    pts = sorted({i * h for i in range(n + 1)} | set(w['times']))
    total = 0.0
    for a, b in zip(pts, pts[1:]):
        m = 0.5 * (a + b)
        lam = mu + alpha * sum(math.exp(-beta * (m - s)) for s in w['times'] if s < m)
        total += (b - a) * lam
    return total


def verification(windows, cfg, mu, alpha, beta):
    rec = 0.0
    for w in windows:
        for b in cfg['beta_candidates']:
            for x, y in zip(recursion_A(w['times'], b), recursion_A_direct(w['times'], b)):
                rec = max(rec, abs(x - y))
    ident = quad = 0.0
    for w in windows:
        closed = mu * w['T'] + alpha / beta * tail_sum(w['times'], w['T'], beta)
        s = sum(r['tau'] for r in rescaled_intervals(w, mu, alpha, beta))
        ident = max(ident, abs(s - closed))
        if len(w['times']) <= 2000:
            quad = max(quad, abs(_quadrature(w, mu, alpha, beta) - closed))
    decomp = 0.0
    for w in windows:
        A = recursion_A(w['times'], beta)
        for i, t in enumerate(w['times']):
            lam = mu + alpha * A[i]
            s = mu / lam + sum(alpha * math.exp(-beta * (t - u)) / lam for u in w['times'][:i])
            decomp = max(decomp, abs(s - 1.0))
    return {'recursion_max_abs_err': rec,
            'rescaled_identity_max_abs_err': ident,
            'quadrature_max_abs_err': quad,
            'decomposition_sum_max_abs_err': decomp,
            'note': '递推对全部beta候选做直接双重求和核对;数值积分为中点法,仅n<=2000的窗;分解每事件份额和应为1'}


# ---------- 截短敏感性 ----------

def truncate_windows(windows, f):
    """观测窗截短到(0,f*T]:只用不超过f*T的事件,积分到f*T,不引入未来事件。"""
    out = []
    for w in windows:
        T2 = f * w['T']
        out.append({'id': w['id'], 'split': w['split'], 'T': T2,
                    'times': [t for t in w['times'] if t <= T2]})
    return out


# ---------- 主流程 ----------

def analyze(v, digest, truncations, outdir=None):
    cfg = parse_config(v)
    windows, violations = valid_windows(v)
    dev = [w for w in windows if w['split'] == 'development']
    ext = [w for w in windows if w['split'] == 'external']
    fracs = []
    for f in truncations:
        f = number(f)
        if not 0 < f <= 1:
            raise ValueError('截短比例须在(0,1]')
        if f not in fracs:
            fracs.append(f)

    candidates, best, tie = fit_grid(dev, cfg)
    sel = candidates[best] if best is not None else None
    n_dev = sum(len(w['times']) for w in dev)
    T_dev = sum(w['T'] for w in dev)
    mu_lo, mu_hi = cfg['mu_bounds']
    if n_dev:
        mu_p = min(max(n_dev / T_dev, mu_lo), mu_hi)
        ll_p = n_dev * math.log(mu_p) - mu_p * T_dev
    else:
        mu_p, ll_p = mu_lo, -mu_lo * T_dev

    report = {'source_sha256': digest, 'config': {
        'mu_bounds': list(cfg['mu_bounds']), 'alpha_bounds': list(cfg['alpha_bounds']),
        'r_max': cfg['r_max'], 'beta_candidates': cfg['beta_candidates'],
        'solver': {'tolerance': cfg['tolerance'], 'max_iterations': cfg['max_iterations']},
        'diagnostic_grid': cfg['diagnostic_grid']},
        'violations': violations,
        'splits': {'development': [w['id'] for w in dev], 'external': [w['id'] for w in ext],
                   'excluded': [x['window'] for x in violations]},
        'notes': [
            '网格上的总似然最大者只是有限候选最佳,不声称连续beta全局最优',
            '背景/核份额为事件时刻条件强度分解,分母与似然相同,不是实际因果传播链',
            '时间变换诊断基于同资料拟合参数,不套独立同分布检验p值;末段无事件尾为右删失,不进经验分布',
            '截短拟合只用不超过f*T的事件且积分到f*T;外期评分始终用完整external窗与冻结参数',
            '空开发窗仍贡献-mu*T;未完成候选如实标注状态,不跳过后报已求完']}
    if tie:
        report['notes'].append('多个beta候选总似然在1e-8相对容差内相同(alpha=0时beta不可识别是常见情形),按约定取先列者')

    if sel is None:
        report['fit'] = {'candidates': candidates, 'selected': None,
                         'error': '无有效候选,拟合未完成'}
        report['headline'] = {'source_sha256': digest, 'violations': violations,
                              'error': '无有效候选'}
        return report

    mu, alpha, beta = sel['mu'], sel['alpha'], sel['beta']
    provisional = sel['status'] not in ('converged', 'closed-form-empty')
    dev_rows = [window_loglik(w, mu, alpha, beta) for w in dev]
    ext_rows = [window_loglik(w, mu, alpha, beta) for w in ext]
    clustering = {
        'branching_ratio': alpha / beta,
        'dev_event_excitation_share_mean': excitation_share(dev, mu, alpha, beta),
        'external_event_excitation_share_mean': excitation_share(ext, mu, alpha, beta),
        'poisson_reference': {'mu': mu_p, 'loglik': ll_p,
                              'hawkes_minus_poisson_loglik': sel['loglik'] - ll_p},
        'note': 'branching_ratio=alpha/beta为稳态下自激份额;事件级份额为条件强度分解均值,均非因果链'}
    report['fit'] = {
        'candidates': candidates, 'selected_index': best, 'selected': sel, 'tie': tie,
        'provisional': provisional,
        'selection_note': '有限网格总似然最大' + (';所选候选未收敛,结论暂定' if provisional else ''),
        'dev_windows': dev_rows, 'dev_total_loglik': sum(r['loglik'] for r in dev_rows)}
    report['clustering'] = clustering
    report['external'] = {'windows': ext_rows,
                          'total_loglik': sum(r['loglik'] for r in ext_rows),
                          'note': '冻结参数,各external窗独立起窗按自身事件更新条件强度,不重拟合'}
    report['diagnostics'] = {
        'development': diagnostics(dev, mu, alpha, beta, cfg['diagnostic_grid']),
        'external': diagnostics(ext, mu, alpha, beta, cfg['diagnostic_grid'])}

    trunc_rows = []
    for f in fracs:
        dev_f = truncate_windows(dev, f)
        cands_f, best_f, tie_f = fit_grid(dev_f, cfg)
        sel_f = cands_f[best_f] if best_f is not None else None
        row = {'fraction': f,
               'kept_events': sum(len(w['times']) for w in dev_f),
               'full_events': n_dev, 'tie': tie_f,
               'candidate_statuses': {str(c['beta']): c['status'] for c in cands_f}}
        if sel_f is not None:
            ext_f = [window_loglik(w, sel_f['mu'], sel_f['alpha'], sel_f['beta']) for w in ext]
            row.update({'selected': {'beta': sel_f['beta'], 'mu': sel_f['mu'],
                                     'alpha': sel_f['alpha'],
                                     'branching_ratio': sel_f['branching_ratio'],
                                     'status': sel_f['status']},
                        'dev_total_loglik': sel_f['loglik'],
                        'external_total_loglik': sum(r['loglik'] for r in ext_f)})
        else:
            row.update({'selected': None, 'dev_total_loglik': None,
                        'external_total_loglik': None})
        trunc_rows.append(row)
    report['truncation'] = trunc_rows
    report['verification'] = verification(dev + ext, cfg, mu, alpha, beta)

    report['headline'] = {
        'source_sha256': digest, 'violations': violations,
        'selected': {'beta': beta, 'mu': mu, 'alpha': alpha,
                     'branching_ratio': alpha / beta, 'status': sel['status']},
        'tie': tie, 'provisional': provisional,
        'candidate_statuses': {str(c['beta']): c['status'] for c in candidates},
        'dev_total_loglik': report['fit']['dev_total_loglik'],
        'poisson_delta_loglik': clustering['poisson_reference']['hawkes_minus_poisson_loglik'],
        'clustering': clustering,
        'external_total_loglik': report['external']['total_loglik'],
        'truncation': [{'fraction': r['fraction'], 'kept_events': r['kept_events'],
                        'selected': r['selected'],
                        'external_total_loglik': r['external_total_loglik']} for r in trunc_rows],
        'verification': report['verification']}

    if outdir is not None:
        write_outputs(report, dev, ext, mu, alpha, beta, outdir)
    return report


# ---------- 输出 ----------

def write_outputs(report, dev, ext, mu, alpha, beta, outdir):
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    cands = report['fit']['candidates']
    with open(out / 'fit_by_beta.csv', 'w', newline='', encoding='utf8') as fh:
        w = csv.writer(fh)
        w.writerow(['beta', 'status', 'iterations', 'loglik', 'mu', 'alpha',
                    'branching_ratio', 'grad_mu', 'grad_alpha', 'kkt_residual',
                    'dual_gap', 'mu_bound', 'alpha_bound', 'stability_constraint_active',
                    'neg_hessian_eig_min', 'neg_hessian_eig_max', 'flat_direction', 'selected'])
        for c in cands:
            eigs = c['neg_hessian_eigs'] or [None, None]
            grad = c['grad'] or [None, None]
            w.writerow([c['beta'], c['status'], c['iterations'], c['loglik'], c['mu'],
                        c['alpha'], c['branching_ratio'], grad[0], grad[1],
                        c['kkt_residual'], c['dual_gap'], c['mu_bound'], c['alpha_bound'],
                        c['stability_constraint_active'], eigs[0], eigs[1],
                        c['flat_direction'], bool(c.get('selected'))])
    with open(out / 'truncation.csv', 'w', newline='', encoding='utf8') as fh:
        w = csv.writer(fh)
        w.writerow(['fraction', 'kept_events', 'full_events', 'beta', 'mu', 'alpha',
                    'branching_ratio', 'status', 'dev_total_loglik', 'external_total_loglik'])
        for r in report['truncation']:
            s = r['selected'] or {}
            w.writerow([r['fraction'], r['kept_events'], r['full_events'],
                        s.get('beta'), s.get('mu'), s.get('alpha'),
                        s.get('branching_ratio'), s.get('status'),
                        r['dev_total_loglik'], r['external_total_loglik']])
    with open(out / 'external_scores.csv', 'w', newline='', encoding='utf8') as fh:
        w = csv.writer(fh)
        w.writerow(['window', 'events', 'T', 'sum_log_lambda', 'integral', 'loglik',
                    'mean_log_lambda'])
        for r in report['external']['windows']:
            w.writerow([r['window'], r['events'], r['T'], r['sum_log_lambda'],
                        r['integral'], r['loglik'], r['mean_log_lambda']])
    with open(out / 'decomposition.csv', 'w', newline='', encoding='utf8') as fh:
        w = csv.writer(fh)
        w.writerow(['window', 'split', 'event_index', 'event_time', 'lambda',
                    'parent', 'parent_time', 'share'])
        for win in dev + ext:
            for row in decomposition_rows(win, mu, alpha, beta):
                w.writerow(row)
    with open(out / 'rescaled_intervals.csv', 'w', newline='', encoding='utf8') as fh:
        w = csv.writer(fh)
        w.writerow(['window', 'split', 'index', 'start', 'end', 'tau', 'kind'])
        for win in dev + ext:
            for r in rescaled_intervals(win, mu, alpha, beta):
                w.writerow([r['window'], r['split'], r['index'], r['start'], r['end'],
                            r['tau'], r['kind']])
    with open(out / 'diagnostic_cdf.csv', 'w', newline='', encoding='utf8') as fh:
        w = csv.writer(fh)
        w.writerow(['split', 'grid', 'empirical_cdf', 'exp1_cdf', 'diff', 'complete_intervals'])
        for split in ('development', 'external'):
            d = report['diagnostics'][split]
            for r in d['cdf']:
                w.writerow([split, r['grid'], r['empirical_cdf'], r['exp1_cdf'], r['diff'],
                            d['complete_intervals']])
    (out / 'report.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8')
