"""演练资料的原字节身份与严格数值读取。"""
import hashlib
import json
import math
from fractions import Fraction
from pathlib import Path

def load(path):
    raw=Path(path).read_bytes()
    v=json.loads(raw.decode('utf8'),parse_constant=lambda x:(_ for _ in ()).throw(ValueError('JSON不得含非有限常量')))
    if not isinstance(v,dict):raise ValueError('顶层资料必须为对象')
    return v,hashlib.sha256(raw).hexdigest()

def number(x):
    if isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x):raise ValueError('需要有限数值')
    return x

def positive(x):
    if number(x)<=0:raise ValueError('需要正数')
    return x

def integer(x,minimum=0):
    if type(x) is not int or x<minimum:raise ValueError('整数越界')
    return x

def fraction(x):
    if isinstance(x,bool) or not isinstance(x,(str,int)):raise ValueError('精确量使用整数或分数字符串')
    try:return Fraction(x)
    except (ValueError,ZeroDivisionError):raise ValueError('有理数无效') from None

def unique(rows,key):
    seen=set()
    for row in rows:
        x=row[key]
        if not isinstance(x,str) or not x or x in seen:raise ValueError('身份必须非空且唯一')
        seen.add(x)
    return seen
