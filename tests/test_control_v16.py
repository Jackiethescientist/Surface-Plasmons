"""Exercise real child-process scheduling and protection of saved controls."""
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, main
from unittest.mock import patch
import importlib.util
import json
import shutil
import sys

SCRIPT = Path(__file__).parents[1] / "scripts" / "run_control_v16.py"
SPEC = importlib.util.spec_from_file_location("control_v16_under_test", SCRIPT)
control = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = control
SPEC.loader.exec_module(control)

FAKE_RUNNER = '''from pathlib import Path
import argparse, json, os, time
p=argparse.ArgumentParser()
p.add_argument('--config');p.add_argument('--name');p.add_argument('--results-root')
p.add_argument('--field-maps');p.add_argument('--resume',action='store_true')
a=p.parse_args();c=json.loads(Path(a.config).read_text())
t=Path(a.results_root)/a.name;t.mkdir(parents=True,exist_ok=True)
started=time.monotonic()
time.sleep(.5)
(t/'execution.json').write_text(json.dumps({'started':started,'finished':time.monotonic(),
 'pol':c['source']['polarisation'],'python':__import__('sys').executable,
 'threads':{k:os.environ[k] for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']}}))
print('worker '+c['source']['polarisation'],flush=True)
raise SystemExit(7 if os.environ.get('CONTROL_TEST_FAIL')==c['source']['polarisation'] else 0)
'''


class ControlTests(TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / "configs").mkdir()
        for pol in ("ex", "ez"):
            name = f"validated_sphere_{pol}_v15.json"
            shutil.copyfile(SCRIPT.parents[1] / "configs" / name, self.root / "configs" / name)
        self.results = self.root / "results"

    def jobs(self, resume=False):
        return control.plan_jobs(self.root, "test_control", self.results, resume)

    def save_manifest(self, job, *, status="completed", digest=None):
        job.target.mkdir(parents=True, exist_ok=True)
        (job.target / "manifest.json").write_text(json.dumps({
            "configuration_hash": digest or control.config_hash(job.config), "status": status
        }))

    def prepare_runner(self):
        (self.root / "scripts").mkdir()
        (self.root / "scripts" / "run_single.py").write_text(FAKE_RUNNER)
        batch = self.root / "batch"
        batch.mkdir()
        return batch

    def test_preflight_has_separate_reference_keys_and_no_writes(self):
        jobs = self.jobs()
        self.assertNotEqual(jobs[0].reference_key, jobs[1].reference_key)
        self.assertFalse(self.results.exists())
        self.assertTrue(all(not j.config["numerics"]["record_database"] for j in jobs))

    def test_completed_solver_results_are_reused_without_overwrite(self):
        for j in self.jobs():
            self.save_manifest(j)
        self.assertTrue(all(j.reuse for j in self.jobs()))
        self.assertEqual(control.memory_budget_bytes(self.jobs(), 2), 0)

    def test_configuration_collision_rejects_entire_plan(self):
        self.save_manifest(self.jobs()[1], digest="different")
        with self.assertRaisesRegex(ValueError, "different settings"):
            self.jobs()
        self.assertFalse(self.results.joinpath("test_control_ex").exists())

    def test_partial_result_needs_explicit_resume(self):
        self.save_manifest(self.jobs()[0], status="interrupted")
        with self.assertRaisesRegex(ValueError, "--resume"):
            self.jobs()
        self.assertTrue(self.jobs(resume=True)[0].resume)

    def test_pair_template_is_rejected_before_any_output(self):
        p = self.root / "configs" / "validated_sphere_ex_v15.json"
        c = json.loads(p.read_text());c["geometry"]["particle_count"] = 2
        p.write_text(json.dumps(c))
        with self.assertRaisesRegex(ValueError, "requires one|particle_count"):
            self.jobs()
        self.assertFalse(self.results.exists())

    def test_dry_run_does_not_create_batch_or_results(self):
        with patch.object(control, "PROJECT_ROOT", self.root), redirect_stdout(StringIO()):
            self.assertEqual(control.main(["--dry-run", "--results-root", str(self.results)]), 0)
        self.assertFalse(self.results.exists())
        self.assertFalse((self.root / "data").exists())

    def test_low_memory_refuses_before_writing_outputs(self):
        with patch.object(control, "PROJECT_ROOT", self.root), \
             patch.object(control, "available_memory_bytes", return_value=1), redirect_stdout(StringIO()):
            with self.assertRaisesRegex(ValueError, "Insufficient available RAM"):
                control.main(["--results-root", str(self.results)])
        self.assertFalse(self.results.exists())
        self.assertFalse((self.root / "data").exists())

    def test_two_workers_overlap_and_keep_environment_and_logs_separate(self):
        batch = self.prepare_runner()
        jobs = self.jobs()
        with redirect_stdout(StringIO()):
            code = control.execute_jobs(jobs, root=self.root, batch=batch, workers=2)
        self.assertEqual(code, 0)
        records = [json.loads((j.target / "execution.json").read_text()) for j in jobs]
        self.assertLess(max(r["started"] for r in records), min(r["finished"] for r in records))
        for r in records:
            self.assertEqual(r["python"], sys.executable)
            self.assertTrue(all(v == "1" for v in r["threads"].values()))
        self.assertEqual((batch / "ex.log").read_text().strip(), "worker Ex")
        self.assertEqual((batch / "ez.log").read_text().strip(), "worker Ez")

    def test_one_worker_is_serial_and_failure_stops_queued_job(self):
        batch = self.prepare_runner()
        with patch.dict("os.environ", {"CONTROL_TEST_FAIL": "Ex"}), redirect_stdout(StringIO()):
            code = control.execute_jobs(self.jobs(), root=self.root, batch=batch, workers=1)
        self.assertEqual(code, 1)
        self.assertFalse((self.results / "test_control_ez").exists())


if __name__ == "__main__":
    main()
