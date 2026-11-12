"""检查各独立观测窗的原始事件顺序。"""
from .io import unique,positive,number

def summarize(v):
    unique(v['windows'],'id');events=0;duration=0
    for w in v['windows']:
        T=positive(w['end']);last=0
        for x in w['times']:
            if not last<number(x)<=T:raise ValueError('事件时刻不严格递增或越窗')
            last=x;events+=1
        duration+=T
    for beta in v['beta_candidates']:positive(beta)
    if not 0<number(v['r_max'])<1:raise ValueError('稳定上限无效')
    return {'windows':len(v['windows']),'events':events,'total_observation_time':duration}
