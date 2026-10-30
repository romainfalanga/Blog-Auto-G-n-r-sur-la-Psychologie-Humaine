import importlib.util
import os
import pathlib
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('journals', pathlib.Path(__file__).parents[2] / 'scripts/generate_journals.py')
j = importlib.util.module_from_spec(spec)
spec.loader.exec_module(j)


class GenerationTests(unittest.TestCase):
    def test_openrouter_preferred_without_leaking_secret(self):
        with patch.dict(os.environ, {'OPENROUTER_API_KEY': 'test-or', 'MAMMOUTH_API_KEY': 'test-ma'}, clear=True):
            self.assertEqual(j.provider()[0], 'https://openrouter.ai/api/v1/chat/completions')

    def test_only_narrative_context_is_sent(self):
        c = j.context_for({'profile': {'display_name': 'Camille'}, 'email': 'private', 'lease_token': 'secret', 'billing': {}})
        self.assertNotIn('email', c)
        self.assertNotIn('lease_token', c)
        self.assertNotIn('billing', c)

    def test_missing_provider_fails_before_claim(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(j, 'internal') as internal:
            with self.assertRaises(RuntimeError):
                j.main()
            internal.assert_not_called()

    def test_malformed_short_or_truncated_story_is_rejected(self):
        with self.assertRaises(ValueError):
            j.validate_story({'title': 'Une histoire', 'body': 'trop court'})
        with self.assertRaises(ValueError):
            j.parse_object('[]')

    def test_review_must_approve(self):
        with patch.object(j, 'llm', side_effect=[{}, {}, {'approved': False}]):
            with self.assertRaisesRegex(ValueError, 'review_rejected'):
                j.generate({})

    def test_completion_keeps_version_and_lease(self):
        story = {'title': 'La seconde possibilité', 'body': 'Une situation fictive. ' * 100,
                 'summary': 'Une scène inventée pour explorer une autre réponse.',
                 'practice': 'Observer une situation ordinaire sans se juger.',
                 'reflection_questions': ['Quelles autres possibilités voyez-vous ?']}
        job = {'lease_token': 'lease', 'profile_version': 7, 'memory': {'text': 'Un fait', 'version': 3}}
        with patch.object(j, 'llm', side_effect=[{}, story, {'approved': True, 'story': story}, {'proposal': 'Piste à confirmer, selon le profil.'}]):
            out = j.generate(job)
        self.assertEqual(out['lease_token'], 'lease')
        self.assertEqual(out['profile_version'], 7)
        self.assertEqual(out['memory_version'], 3)
        self.assertEqual(out['memory_proposal'], 'Piste à confirmer, selon le profil.')


if __name__ == '__main__':
    unittest.main()
