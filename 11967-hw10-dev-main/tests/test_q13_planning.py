"""CPU/stdlib-only checks: no model imports, optimizer steps or GPU requests."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

SCRIPTS=Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0,str(SCRIPTS))
from q13_plan import plan_run, search_batch_limit

CODE=SCRIPTS.parent


class BatchSearchTests(unittest.TestCase):
    def test_exact_maximum_and_failing_neighbor(self):
        seen=[]
        def attempt(b):
            seen.append(b)
            return 'ok' if b<=5 else 'oom'
        r=search_batch_limit(attempt,64)
        self.assertEqual(r['maximum_batch_size'],5)
        self.assertEqual(r['first_failing_batch_size'],6)
        self.assertTrue(r['maximum_confirmed'])
        self.assertIn(5,seen); self.assertIn(6,seen)

    def test_cap_is_not_claimed_as_true_maximum(self):
        r=search_batch_limit(lambda b:'ok',8)
        self.assertEqual(r['largest_successful_batch_size'],8)
        self.assertIsNone(r['maximum_batch_size'])
        self.assertFalse(r['maximum_confirmed'])

    def test_no_feasible_batch(self):
        r=search_batch_limit(lambda b:'oom',64)
        self.assertEqual(r['maximum_batch_size'],0)
        self.assertEqual(r['first_failing_batch_size'],1)

    def test_unrelated_errors_abort_search(self):
        with self.assertRaisesRegex(RuntimeError,'other than CUDA OOM'):
            search_batch_limit(lambda b:'error',64)

    def test_invalid_cap(self):
        with self.assertRaises(ValueError): search_batch_limit(lambda b:'ok',0)


class PlanningTests(unittest.TestCase):
    def test_paired_configs_differ_only_in_initialization_and_names(self):
        a=plan_run(CODE,'retrofit'); b=plan_run(CODE,'scratch')
        changes={k for k in a['config'].keys()|b['config'].keys() if a['config'].get(k)!=b['config'].get(k)}
        self.assertEqual(changes,{'model_to_train','run_name','output_dir'})
        self.assertNotEqual(a['output'],b['output'])
        for plan in (a,b):
            c=plan['config']
            self.assertEqual(plan['tokens_per_update'],16384)
            self.assertEqual(c['gradient_checkpointing_kwargs'],{'use_reentrant':False})
            self.assertEqual(c['num_train_epochs'],1)
            self.assertEqual(c['per_device_eval_batch_size'],2)
            self.assertTrue(c['bf16'])

    def test_reject_overwriting_input(self):
        with tempfile.TemporaryDirectory() as temp:
            code=Path(temp)/'code';(code/'configs').mkdir(parents=True)
            path=CODE/'configs/2048_flash_gc_retrofit_wikitext.json'
            c=json.loads(path.read_text());c['output_dir']=c['model_to_train']
            (code/'configs'/path.name).write_text(json.dumps(c))
            with self.assertRaises(ValueError):plan_run(code,'retrofit')

    def test_dry_runs_do_not_import_ml_or_create_results(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);code=root/'code'
            shutil.copytree(SCRIPTS,code/'scripts',ignore=shutil.ignore_patterns('__pycache__'))
            shutil.copytree(CODE/'configs',code/'configs')
            packages=root/'fake';packages.mkdir()
            for module in ('torch','transformers','datasets'):
                (packages/(module+'.py')).write_text("raise RuntimeError('ML import forbidden in dry-run')\n")
            env=os.environ.copy();env['PYTHONPATH']=str(packages)
            for script,args in [('run_q13.py',['retrofit']),('run_q13.py',['scratch']),('probe_q13_batch_size.py',[])]:
                output=subprocess.check_output([sys.executable,str(code/'scripts'/script),*args,'--dry-run'],env=env,text=True)
                self.assertIsInstance(json.loads(output),dict)
            self.assertFalse((root/'q13-results').exists())
            self.assertFalse((code/'models').exists())


if __name__=='__main__':unittest.main()
