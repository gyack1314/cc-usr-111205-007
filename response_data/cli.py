"""显示已有原始资料摘要,或对到达资料做指数核Hawkes自激分析。"""
import argparse
import json
from pathlib import Path

from .io import load
from .domain import summarize
from . import hawkes

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['inspect','hawkes']);p.add_argument('file')
    p.add_argument('--out',default=None,help='hawkes输出目录,默认out/<资料名>')
    p.add_argument('--truncations',default='0.5,0.6,0.7,0.8,0.9,1.0',help='截短比例,逗号分隔')
    a=p.parse_args()
    v,digest=load(a.file)
    if a.command=='inspect':
        print(json.dumps({'source_sha256':digest,'summary':summarize(v)},ensure_ascii=False,indent=2,allow_nan=False))
        return
    fracs=[float(x) for x in a.truncations.split(',') if x.strip()]
    out=Path(a.out) if a.out else Path('out')/Path(a.file).stem
    report=hawkes.analyze(v,digest,fracs,out)
    print(json.dumps(report['headline'],ensure_ascii=False,indent=2,allow_nan=False))
if __name__=='__main__':main()
