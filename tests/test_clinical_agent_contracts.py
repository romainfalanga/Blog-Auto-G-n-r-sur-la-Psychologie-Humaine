"""Verify actual agent contracts against an isolated existing twelve-part arc.

The synthetic design is a test fixture; no editorial corpus is changed. Clinical
accuracy still requires reading the complete arc and its sources independently.
"""
from contextlib import redirect_stdout
from copy import deepcopy
from datetime import date
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import clinical_agent_contracts as contracts
import clinical_editorial as clinical

CHECKOUT = Path(__file__).resolve().parents[1]
TODAY = date.today().isoformat()


class ClinicalAgentContractTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.editorial = self.root / "editorial/clinical"
        self.editorial.mkdir(parents=True)
        self.character = deepcopy(clinical.sources_and_characters()[0]["mireille"])
        for relative in ("data/clinical_sources.json", "editorial/clinical/charter.md",
                         "editorial/automation/legacy-publications.json",
                         "editorial/clinical/mireille.json", "editorial/clinical/dossiers/mireille.json"):
            destination = self.root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(CHECKOUT / relative, destination)
        self.write("data/clinical_catalog.json", {"characters": [self.character]})
        self.write("editorial/clinical/additional_characters.json", [])
        self.write("editorial/clinical/releases.json", {"arcs": {}})
        self.write("data/parcours_topics.json", {})
        self.manuscript = self.read("editorial/clinical/mireille.json")
        self.design = {
            "schema_version": 1, "character_id": "mireille", "code": self.character["code"],
            "article_format": 2, "bio": self.manuscript["bio"],
            "profile_topics": self.manuscript["profile_topics"],
            "voice": "Une voix précise qui garde ses préférences et ses observations.",
            "series_goal": "Partager son carnet avec Alice tout en conservant sa propre écriture.",
            "continuity_rules": ["Mireille garde ses décisions et ses responsabilités choisies.",
                                 "Alice propose une aide que Mireille peut refuser.",
                                 "La consultation mémoire assure le suivi clinique."],
            "episodes": [{"number": episode["number"], "focus": episode["clinical_focus"],
                          "scene": episode["opening"], "development": episode["sections"][0]["text"],
                          "ending": episode["ending"], "next": episode["continuity"]["next"],
                          "source_refs": episode["sources"], "diagnostic_stage": "known"}
                         for episode in self.manuscript["episodes"]],
        }
        self.save_design()
        for module, field, value in ((contracts, "ROOT", self.root),
                                     (clinical, "ROOT", self.root),
                                     (clinical, "EDITORIAL", self.editorial)):
            patcher = patch.object(module, field, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.report = self.make_review()
        self.save_review()

    def write(self, relative, value):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def read(self, relative):
        return json.loads((self.root / relative).read_text(encoding="utf-8"))

    def sha(self, relative):
        return hashlib.sha256((self.root / relative).read_bytes()).hexdigest()

    def save_design(self):
        self.write("editorial/clinical/designs/mireille.json", self.design)

    def save_review(self):
        self.write("editorial/clinical/reviews/mireille.json", self.report)

    def make_review(self):
        return {"schema_version": 1, "character_id": "mireille", "status": "passed",
                "reviewer": "independent-reviewer", "reviewed_on": TODAY,
                "files": contracts.validate_story("mireille")["files"],
                "summary": "La relecture vérifie chaque épisode et les éléments précis du parcours.",
                "findings": [],
                "episode_checks": [{"number": episode["number"], "focus": episode["clinical_focus"],
                                    "opening_quote": episode["opening"][:80],
                                    "ending_quote": episode["ending"][:80],
                                    "continuity_note": episode["continuity"]["next"],
                                    "clinical_note": "La portée du trouble reste celle du dossier consulté.",
                                    "relatives_note": episode["mirror"][0]}
                                   for episode in self.manuscript["episodes"]]}

    def ask_for_revision(self):
        self.report["status"] = "revision_requested"
        self.report["findings"] = [{"episode": 4, "issue": "Le raccord ne reprend pas la décision précédente.",
                                    "requested_change": "Réécrire ce raccord en conservant la décision de Mireille."}]
        self.save_review()

    def test_existing_full_arc_validates_with_real_contract_and_cumulative_hashes(self):
        design = contracts.validate_design("mireille")
        story = contracts.validate_story("mireille")
        review = contracts.validate_review("mireille")
        self.assertEqual(design["episodes"], 12)
        self.assertEqual(set(design["files"]), {"editorial/clinical/designs/mireille.json",
                                               "editorial/clinical/dossiers/mireille.json"})
        self.assertEqual(set(story["files"]), set(design["files"]) | {"editorial/clinical/mireille.json"})
        self.assertEqual(set(review["files"]), set(story["files"]) | {"editorial/clinical/reviews/mireille.json"})
        self.assertEqual(review["status"], "passed")
        for relative, digest in review["files"].items():
            self.assertEqual(digest, self.sha(relative))

    def test_biography_preserves_canonical_name_age_diagnosis_and_length(self):
        for bio in ("Mireille présente un trouble neurocognitif léger.",
                    self.manuscript["bio"].replace("trouble neurocognitif léger", "difficulté encore mal comprise"),
                    self.manuscript["bio"].replace("Mireille", "Valérie"),
                    self.manuscript["bio"].replace("66 ans", "67 ans")):
            with self.subTest(bio=bio[:80]):
                self.design["bio"] = bio
                self.save_design()
                with self.assertRaises(ValueError):
                    contracts.validate_design("mireille")

    def test_established_diagnosis_cannot_be_under_assessment(self):
        self.design["episodes"][0]["diagnostic_stage"] = "under_assessment"
        self.save_design()
        with self.assertRaisesRegex(ValueError, "[Cc]hronologie|[Tt]emporalité"):
            contracts.validate_design("mireille")

    def test_early_assessment_has_exact_diagnosis_boundary(self):
        self.character["diagnosis"] = {"timing": "early_professional_assessment", "episode": 3}
        self.write("data/clinical_catalog.json", {"characters": [self.character]})
        for episode in self.design["episodes"]:
            episode["diagnostic_stage"] = ("under_assessment" if episode["number"] < 3 else
                                             "diagnosis_explained" if episode["number"] == 3 else "known")
        self.save_design()
        contracts.validate_design("mireille")
        self.design["episodes"][2]["diagnostic_stage"] = "under_assessment"
        self.save_design()
        with self.assertRaisesRegex(ValueError, "[Cc]hronologie|[Tt]emporalité"):
            contracts.validate_design("mireille")

    def test_duplicate_focus_is_rejected(self):
        self.design["episodes"][1]["focus"] = self.design["episodes"][0]["focus"]
        self.save_design()
        with self.assertRaisesRegex(ValueError, "[Aa]spect"):
            contracts.validate_design("mireille")

    def test_unknown_plan_source_is_rejected(self):
        self.design["episodes"][0]["source_refs"] = ["oms-cddr", "source-inventee"]
        self.save_design()
        with self.assertRaisesRegex(ValueError, "[Ss]ources"):
            contracts.validate_design("mireille")

    def test_story_cannot_silently_replace_the_approved_biography(self):
        manuscript = deepcopy(self.manuscript)
        manuscript["bio"] = manuscript["bio"].replace("66 ans", "67 ans")
        self.write("editorial/clinical/mireille.json", manuscript)
        with self.assertRaisesRegex(ValueError, "diverge|reprendre le personnage"):
            contracts.validate_story("mireille")

    def test_story_cannot_silently_replace_an_approved_focus(self):
        manuscript = deepcopy(self.manuscript)
        manuscript["episodes"][0]["clinical_focus"] = "Fatigue après les rencontres"
        self.write("editorial/clinical/mireille.json", manuscript)
        with self.assertRaisesRegex(ValueError, "[Aa]spect différent|douze aspects"):
            contracts.validate_story("mireille")

    def test_changed_valid_story_invalidates_existing_review(self):
        manuscript = deepcopy(self.manuscript)
        manuscript["episodes"][0]["description"] = "Une nouvelle description pour cette scène du carnet."
        self.write("editorial/clinical/mireille.json", manuscript)
        contracts.validate_story("mireille")
        with self.assertRaisesRegex(ValueError, "Relecture périmée"):
            contracts.validate_review("mireille")

    def test_invented_review_quote_is_rejected(self):
        self.report["episode_checks"][0]["opening_quote"] = "Une citation entièrement inventée qui ne figure pas dans la scène."
        self.save_review()
        with self.assertRaisesRegex(ValueError, "Citation"):
            contracts.validate_review("mireille")

    def test_revision_request_keeps_findings_and_cannot_be_released(self):
        self.ask_for_revision()
        validated = contracts.validate_review("mireille")
        self.assertEqual(validated["findings"], self.report["findings"])
        with self.assertRaises(ValueError):
            clinical.record_review("mireille", TODAY, "Contrôle du refus de publication.")
        self.assertEqual(self.read("editorial/clinical/releases.json"), {"arcs": {}})

    def test_passed_review_cannot_have_open_findings(self):
        self.ask_for_revision()
        self.report["status"] = "passed"
        self.save_review()
        with self.assertRaises(ValueError):
            contracts.validate_review("mireille")

    def test_review_covers_every_episode_and_rejects_null_checks(self):
        baseline = deepcopy(self.report["episode_checks"])
        for invalid in (baseline[:-1], [None] + baseline[1:]):
            with self.subTest(kind="missing" if len(invalid) == 11 else "null"):
                self.report["episode_checks"] = invalid
                self.save_review()
                with self.assertRaises(ValueError):
                    contracts.validate_review("mireille")

    def test_null_design_episode_is_a_validation_error(self):
        self.design["episodes"][0] = None
        self.save_design()
        with self.assertRaises(ValueError):
            contracts.validate_design("mireille")

    def test_release_binds_reviewed_files_and_integration_rechecks_hashes(self):
        clinical.record_review("mireille", TODAY, "Contrôle éditorial indépendant de cette version.")
        release = self.read("editorial/clinical/releases.json")["arcs"]["mireille"]
        for key, relative in (("sha256", "mireille.json"), ("dossier_sha256", "dossiers/mireille.json"),
                              ("design_sha256", "designs/mireille.json"),
                              ("independent_review_sha256", "reviews/mireille.json")):
            self.assertEqual(release[key], self.sha("editorial/clinical/" + relative))
        with redirect_stdout(io.StringIO()):
            clinical.integrate(check=False)
            clinical.integrate(check=True)
        self.assertEqual(len(list((self.root / "content/posts").rglob("*.md"))), 12)
        self.assertTrue((self.root / "content/personnages/mireille.md").is_file())
        self.design["voice"] += " Cette modification attend une nouvelle relecture."
        self.save_design()
        with self.assertRaises(ValueError):
            clinical.integrate(check=True)

    def test_removing_design_cannot_bypass_a_present_review_report(self):
        self.ask_for_revision()
        (self.editorial / "designs/mireille.json").unlink()
        with self.assertRaises(ValueError):
            clinical.record_review("mireille", TODAY, "Tentative de contournement du refus.")
        self.assertEqual(self.read("editorial/clinical/releases.json"), {"arcs": {}})

    def test_bound_release_cannot_be_downgraded_by_deleting_design_and_review(self):
        clinical.record_review("mireille", TODAY, "Version liée à la relecture indépendante.")
        before = self.read("editorial/clinical/releases.json")
        (self.editorial / "designs/mireille.json").unlink()
        (self.editorial / "reviews/mireille.json").unlink()
        with self.assertRaises(ValueError):
            clinical.record_review("mireille", TODAY, "Tentative de retrait des preuves de lecture.")
        self.assertEqual(self.read("editorial/clinical/releases.json"), before)

    def test_exact_legacy_manuscript_remains_compatible_without_agent_artifacts(self):
        (self.editorial / "designs/mireille.json").unlink()
        (self.editorial / "reviews/mireille.json").unlink()
        clinical.record_review("mireille", TODAY, "Relecture du manuscrit historique inchangé.")
        release = self.read("editorial/clinical/releases.json")["arcs"]["mireille"]
        self.assertEqual(release["sha256"], self.sha("editorial/clinical/mireille.json"))
        self.assertNotIn("design_sha256", release)
        self.assertNotIn("independent_review_sha256", release)

    def test_modified_legacy_manuscript_requires_agent_artifacts_even_if_both_deleted(self):
        (self.editorial / "designs/mireille.json").unlink()
        (self.editorial / "reviews/mireille.json").unlink()
        manuscript = deepcopy(self.manuscript)
        manuscript["episodes"][0]["description"] = "Une nouvelle description qui modifie la version historiquement relue."
        self.write("editorial/clinical/mireille.json", manuscript)
        clinical.validate_manuscript("mireille", TODAY)
        with self.assertRaises(ValueError):
            clinical.record_review("mireille", TODAY, "Tentative de réutilisation de l'exemption historique.")
        self.assertEqual(self.read("editorial/clinical/releases.json"), {"arcs": {}})


if __name__ == "__main__":
    unittest.main()
