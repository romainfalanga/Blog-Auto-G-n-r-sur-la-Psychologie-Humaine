import contextlib
import io
import pathlib
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(pathlib.Path(__file__).parents[2] / 'scripts'))
import generate_personal_articles as personal
import run_journals as runner


def job(kind='portrait'):
    return {'schema_version':2,'id':'job-test','kind':kind,'source_version':7,'lease_token':'private-lease'}


class PipelineTests(unittest.TestCase):







    def test_runner_reports_only_safe_stage_and_sends_supported_failure_code(self):
        calls=[]
        def internal(path,payload):
            calls.append((path,payload))
            return {'jobs':[job()]} if len(calls)==1 else {'jobs':[]}
        out=io.StringIO()
        with patch.object(runner,'internal',side_effect=internal),patch.object(runner,'generate',side_effect=personal.GenerationFailure('support','provider_error')),contextlib.redirect_stdout(out):
            self.assertEqual(runner.main(),1)
        self.assertIn('stage=support, code=provider_error',out.getvalue())
        self.assertNotIn('private-lease',out.getvalue())
        self.assertEqual(calls[1][1]['reason'],'provider_error')

    def test_runner_preserves_completion_transport_codes_without_retrying_nonprovider_failures(self):
        for code in ('context_changed','lease_expired','generation_paused','quota_exceeded'):
            with self.subTest(code=code):
                calls=[];claimed=False
                def internal(path,payload):
                    nonlocal claimed
                    calls.append((path,payload))
                    if path=='/claim':
                        if claimed:return {'jobs':[]}
                        claimed=True;return {'jobs':[job()]}
                    if path.endswith('/complete'):raise runner.TransportFailure('PRIVATE TRANSPORT DETAILS',code)
                    return {}
                out=io.StringIO()
                with patch.object(runner,'internal',side_effect=internal),patch.object(runner,'generate',return_value={}),contextlib.redirect_stdout(out):
                    self.assertEqual(runner.main(),1)
                failures=[payload for path,payload in calls if path.endswith('/fail')]
                self.assertEqual(failures[0]['reason'],code)
                self.assertEqual(sum(path.endswith('/complete') for path,_ in calls),1)
                self.assertIn('stage=completion, code='+code,out.getvalue())
                self.assertNotIn('PRIVATE TRANSPORT DETAILS',out.getvalue());self.assertNotIn('private-lease',out.getvalue())


    def test_runner_logs_only_allowlisted_validation_codes(self):
        calls=[]
        def internal(path,payload):
            calls.append((path,payload))
            return {'jobs':[job()]} if len(calls)==1 else {'jobs':[]}
        failure=personal.GenerationFailure('support','validation_error')
        # Defense in depth: even a malformed diagnostic property cannot put
        # source text or a provider response in public Actions logs.
        failure.validation_codes=('support_context_budget','PRIVATE SOURCE MUST STAY PRIVATE',{},'support_context_budget')
        out=io.StringIO()
        with patch.object(runner,'internal',side_effect=internal),patch.object(runner,'generate',side_effect=failure),contextlib.redirect_stdout(out):
            self.assertEqual(runner.main(),1)
        self.assertIn('stage=support, code=validation_error, checks=support_context_budget',out.getvalue())
        self.assertNotIn('PRIVATE SOURCE',out.getvalue())
        self.assertNotIn('private-lease',out.getvalue())
        self.assertEqual(calls[1][1]['reason'],'validation_error')


    def test_runner_ignores_retired_leases_without_provider_or_publication_calls(self):
        for kind in ('characters','story'):
            with self.subTest(kind=kind), patch.object(runner,'internal',return_value={'jobs':[job(kind)]}) as internal, patch.object(runner,'generate') as generate, contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(runner.main(),0)
                internal.assert_called_once_with('/claim',{})
                generate.assert_not_called()
                with self.assertRaises(personal.GenerationFailure):
                    personal.generate(job(kind),generate)
                generate.assert_not_called()


if __name__=='__main__':
    unittest.main()
