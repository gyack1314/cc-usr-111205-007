"""显示已有原始资料摘要。"""
import argparse
import json
from .io import load
from .domain import summarize

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['inspect']);p.add_argument('file');a=p.parse_args()
    v,digest=load(a.file)
    print(json.dumps({'source_sha256':digest,'summary':summarize(v)},ensure_ascii=False,indent=2,allow_nan=False))
if __name__=='__main__':main()
