"""Contract and rejection tests for the symptom editorial build."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('symptom_builder',ROOT/'scripts/build_symptomes_carte.py')
builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)

class SymptomBuild(unittest.TestCase):
    def setUp(self):
        self.original=builder.read
    def build_changed(self,filename,change):
        def read(path):
            value=self.original(path)
            if path.name==filename: change(value)
            return value
        with patch.object(builder,'read',read):return builder.build()
    def test_deterministic_complete_partition(self):
        before=(ROOT/'data/cartographie.json').read_bytes()
        a=builder.build();b=builder.build()
        self.assertEqual(a,b)
        self.assertEqual(before,(ROOT/'data/cartographie.json').read_bytes())
        self.assertEqual(a['coverage']['raw_symptoms']-a['coverage']['merged'],len(a['factors']))
        self.assertLessEqual(max(a['coverage']['questions_by_theme'].values()),30)
        self.assertEqual(len({x['id'] for x in a['factors']}),len(a['factors']))
    def test_classification_gated_by_explicit_product(self):
        a=builder.build()
        self.assertEqual(a['requires']['icd_6c40_2'],['usage_alcool'])
        self.assertEqual(a['requires']['ten_usage_cannabis'],['usage_cannabis'])
        self.assertEqual(a['requires']['sym_6c45'],['sup_conso_cocaine'])
    def test_no_unknown_requirement(self):
        with self.assertRaisesRegex(ValueError,'Requires invalide'):
            self.build_changed('requires.json',lambda d:d['requires'].update(sym_6c45=['nonexistent']))
    def test_rejects_missing_clinical_source(self):
        def change(d):d['symptoms'][0]['links'][0]['sources']=['unknown_source']
        with self.assertRaisesRegex(ValueError,'Source non résolue'):
            self.build_changed('liens-supplementaires.json',change)
    def test_rejects_unjustified_merger(self):
        with self.assertRaisesRegex(ValueError,'Fusion non justifiée'):
            self.build_changed('fusion.json',lambda d:d['merge'][0].update(why=''))
    def test_safety_cannot_disappear(self):
        def change(d):
            next(s for s in d['symptoms'] if s['id']=='idees_de_mort')['safety']=None
        with self.assertRaisesRegex(ValueError,'Signes de sécurité absents'):
            self.build_changed('canoniques.json',change)
    def test_long_help_rejected(self):
        with self.assertRaisesRegex(ValueError,'Aide trop longue'):
            self.build_changed('canoniques.json',lambda d:d['symptoms'][0].update(description='x'*121))
    def test_editorial_metadata_materialized_from_childhood(self):
        a=builder.build();child={o['id']:o for o in self.original(ROOT/'data/cartographie.json')['outcomes']}
        for o in a['outcomes']:
            if o['id'] in child:
                for k in ['label','description','url']:self.assertEqual(o.get(k),child[o['id']].get(k))

if __name__=='__main__':unittest.main()
