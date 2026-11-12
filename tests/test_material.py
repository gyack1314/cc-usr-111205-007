"""检查原始资料摘要、数值和输入边界。"""
import json
import tempfile
import unittest
from pathlib import Path
from response_data.io import load,number,integer,fraction,unique
from response_data.domain import summarize

class MaterialTests(unittest.TestCase):
    def test_normal(self):
        source,digest=load('data/normal.json');value=summarize(source)
        self.assertEqual(value,summarize(json.loads(json.dumps(source))))
        self.assertEqual(len(digest),64)
        self.assertEqual(value['windows'],3);self.assertEqual(value['events'],14)
    def test_domain_error(self):
        with self.assertRaises((ValueError,KeyError,TypeError)):summarize(load('data/invalid-domain.json')[0])
    def test_boundary_material(self):
        for p in Path('data').glob('*.json'):
            if p.name.startswith('invalid-'):continue
            with self.subTest(file=p.name):self.assertIsNotNone(summarize(load(p)[0]))
    def test_strict_numbers(self):
        for x in (True,float('inf'),float('nan'),'3'):
            with self.assertRaises(ValueError):number(x)
        for x in (-1,True,1.5):
            with self.assertRaises(ValueError):integer(x)
        self.assertEqual(str(fraction('2/4')),'1/2')
        for x in ('1/0',0.5,False):
            with self.assertRaises(ValueError):fraction(x)
    def test_unique_identity(self):
        with self.assertRaises(ValueError):unique([{'id':'same'},{'id':'same'}],'id')
    def test_json_identity(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'资料.json'
            for raw in ('[]','null','{"a":Infinity}'):
                p.write_text(raw,encoding='utf8')
                with self.assertRaises(ValueError):load(p)
            p.write_text('{"人数":3}',encoding='utf8');first=load(p)
            self.assertEqual(first[0],{'人数':3});self.assertEqual(first[1],load(p)[1])
