"""Exercise real queue locking, leases, draft preservation and evidence gates."""
from datetime import datetime, timedelta, timezone
import hashlib
import json
import multiprocessing
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from editorial_pipeline import Pipeline, PipelineError


def write(root, relative, data):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data) if not isinstance(data, str) else data, encoding="utf-8")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class EditorialFixture:
    def __init__(self, root):
        self.root = Path(root)

    def sources_and_characters(self):
        return {ident: {"id": ident, "name": ident.title(), "code": str(number)}
                for number, ident in enumerate(("louise", "constance", "mila"))}, {}

    def brief(self, ident):
        return {"character": self.sources_and_characters()[0][ident], "total_episodes": 12}

    def character_evidence(self, character, shared):
        path = self.root / "editorial/clinical/dossiers" / (character["id"] + ".json")
        return {}, {"sha256": sha(path)}

    def agent_review_required(self, ident, release):
        migration = json.loads((self.root / "editorial/automation/legacy-publications.json").read_text())
        original = migration["manuscripts"].get(ident)
        return original is None or release.get("sha256") != original

    def arc_outputs(self, character, manuscript, publication, order, evidence, article_format):
        ident = character["id"]
        return {self.root / "content/personnages" / (ident + ".md"): manuscript["public_text"]}, {ident: ["repère"]}


class ContractsFixture:
    """Independent fixture contract: approval and twelve actual episode records."""
    def __init__(self, root):
        self.root = Path(root)

    def check(self, ident, roles):
        files = {"design": f"editorial/clinical/designs/{ident}.json",
                 "dossier": f"editorial/clinical/dossiers/{ident}.json",
                 "story": f"editorial/clinical/{ident}.json",
                 "review": f"editorial/clinical/reviews/{ident}.json"}
        paths = [files[role] for role in roles]
        for path in paths:
            data = json.loads((self.root / path).read_text())
            if not data.get("approved"):
                raise ValueError("Editorial approval is absent")
        if "story" in roles:
            manuscript = json.loads((self.root / files["story"]).read_text())
            if [episode["number"] for episode in manuscript["episodes"]] != list(range(1, 13)):
                raise ValueError("The complete twelve-episode arc is required")
        return {"character_id": ident, "files": {path: sha(self.root / path) for path in paths}}

    def validate_design(self, ident):
        return self.check(ident, ("design", "dossier"))

    def validate_story(self, ident):
        return self.check(ident, ("design", "dossier", "story"))

    def validate_review(self, ident):
        result = self.check(ident, ("design", "dossier", "story", "review"))
        report = json.loads((self.root / f"editorial/clinical/reviews/{ident}.json").read_text())
        result.update({key: report[key] for key in ("status", "findings", "reviewer")})
        return result


def concurrent_claim(root, barrier, queue, owner):
    pipeline = Pipeline(root, EditorialFixture(root), ContractsFixture(root))
    barrier.wait(timeout=10)
    try:
        job = pipeline.claim("character", owner)
        queue.put((job["job_id"], job["character_id"]))
    except Exception as error:
        queue.put(("error", str(error)))


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        policy = {"schema_version": 1, "lease_seconds": 60, "max_revision_rounds": 2,
                  "max_active_jobs": 6, "priority_characters": ["louise", "constance", "mila"]}
        write(self.root, "editorial/automation/policy.json", policy)
        write(self.root, "editorial/automation/legacy-publications.json", {"schema_version": 1, "manuscripts": {}})
        write(self.root, "editorial/clinical/releases.json", {"arcs": {}})
        for name in ("characters", "stories", "review"):
            write(self.root, f"editorial/agents/{name}.md", "Real editorial instructions: " + name)
        for path in ("data/clinical_catalog.json", "data/clinical_sources.json",
                     "editorial/clinical/additional_characters.json", "editorial/clinical/charter.md",
                     "scripts/clinical_editorial.py", "scripts/clinical_agent_contracts.py"):
            write(self.root, path, {})
        self.current = datetime(2026, 11, 7, 12, 0, tzinfo=timezone.utc)
        self.pipeline = Pipeline(self.root, EditorialFixture(self.root), ContractsFixture(self.root), lambda: self.current)
        self.pipeline.init()

    def error(self, code, operation):
        with self.assertRaises(PipelineError) as raised:
            operation()
        self.assertEqual(raised.exception.code, code)

    def state(self):
        return json.loads((self.root / "editorial/automation/state.json").read_text())

    def finish(self, job):
        return self.pipeline.finish(job["job_id"], job["token"])

    def make_design(self, ident="louise"):
        write(self.root, f"editorial/clinical/designs/{ident}.json", {"approved": True, "voice": "une voix propre"})
        write(self.root, f"editorial/clinical/dossiers/{ident}.json", {"approved": True, "sources": ["une source"]})

    def complete_design(self, ident="louise"):
        job = self.pipeline.claim("character", "designer", ident)
        self.make_design(ident)
        self.finish(job)
        return job

    def complete_story(self, ident="louise", owner="author"):
        job = self.pipeline.claim("story", owner, ident)
        write(self.root, f"editorial/clinical/{ident}.json", {"approved": True, "public_text": "Published prose\n",
              "episodes": [{"number": number, "text": f"Scène {number}"} for number in range(1, 13)]})
        self.finish(job)
        return job

    def complete_review(self, ident="louise", owner="reviewer", status="passed"):
        job = self.pipeline.claim("review", owner, ident)
        write(self.root, f"editorial/clinical/reviews/{ident}.json", {"approved": True, "reviewer": owner,
              "status": status, "findings": [] if status == "passed" else [{"episode": 2, "issue": "Raccord à revoir", "requested_change": "Réécrire le raccord"}]})
        result = self.finish(job)
        return job, result

    def integrated_release(self):
        release = {"date": "2026-11-07", "status": "editorially_reviewed", "article_format": 2,
                   "sha256": sha(self.root / "editorial/clinical/louise.json"),
                   "dossier_sha256": sha(self.root / "editorial/clinical/dossiers/louise.json"),
                   "design_sha256": sha(self.root / "editorial/clinical/designs/louise.json"),
                   "independent_review_sha256": sha(self.root / "editorial/clinical/reviews/louise.json"),
                   "review": "Relecture indépendante effectuée"}
        write(self.root, "editorial/clinical/releases.json", {"arcs": {"louise": release}})
        write(self.root, "data/parcours_topics.json", {"louise": ["repère"]})
        write(self.root, "content/personnages/louise.md", "Published prose\n")
        return release

    def test_two_processes_cannot_claim_the_same_character(self):
        # Both processes start together against the same physical state file.
        context = multiprocessing.get_context("fork")
        barrier, queue = context.Barrier(2), context.Queue()
        processes = [context.Process(target=concurrent_claim, args=(str(self.root), barrier, queue, owner))
                     for owner in ("parallel-a", "parallel-b")]
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=15)
            self.assertEqual(process.exitcode, 0)
        claims = [queue.get(timeout=2), queue.get(timeout=2)]
        self.assertNotIn("error", [job[0] for job in claims])
        self.assertEqual(len({job[0] for job in claims}), 2)
        self.assertEqual({job[1] for job in claims}, {"louise", "constance"})
        self.assertEqual(len(self.state()["jobs"]), 2)

    def test_dependency_gates_and_independent_people(self):
        self.error("no_job", lambda: self.pipeline.claim("story", "author", "louise"))
        self.complete_design()
        self.error("role_conflict", lambda: self.pipeline.claim("story", "designer", "louise"))
        self.complete_story()
        self.error("role_conflict", lambda: self.pipeline.claim("review", "author", "louise"))
        _, result = self.complete_review()
        self.assertEqual(result["stage"], "ready_to_publish")
        self.error("not_integrated", lambda: self.pipeline.publish("louise"))
        self.assertEqual(self.state()["cases"]["louise"]["stage"], "ready_to_publish")

    def test_invalid_token_never_finishes_or_extends_a_job(self):
        job = self.pipeline.claim("character", "designer", "louise")
        self.make_design()
        before = self.state()
        self.error("invalid_token", lambda: self.pipeline.heartbeat(job["job_id"], "bad-token"))
        self.error("invalid_token", lambda: self.pipeline.finish(job["job_id"], "bad-token"))
        self.assertEqual(self.state(), before)
        self.assertNotIn(job["token"], json.dumps(before))

    def test_changed_inputs_create_a_new_snapshot_only_on_owner_resume(self):
        job = self.pipeline.claim("character", "designer", "louise")
        self.make_design()
        write(self.root, "editorial/agents/characters.md", "Updated instruction")
        self.error("inputs_changed", lambda: self.finish(job))
        self.assertEqual(self.state()["jobs"][job["job_id"]]["status"], "stale")
        self.error("no_job", lambda: self.pipeline.claim("character", "someone-else", "louise", resume=True))
        resumed = self.pipeline.claim("character", "designer", "louise", resume=True)
        self.assertNotEqual(resumed["job_id"], job["job_id"])
        self.assertEqual(self.pipeline.brief(job["job_id"])["prompt"], "Real editorial instructions: characters")
        self.assertEqual(self.pipeline.brief(resumed["job_id"])["prompt"], "Updated instruction")
        self.finish(resumed)
        self.assertEqual(self.state()["cases"]["louise"]["stage"], "story")

    def test_incomplete_contract_cannot_advance_and_can_be_corrected(self):
        job = self.pipeline.claim("character", "designer", "louise")
        self.make_design()
        write(self.root, "editorial/clinical/designs/louise.json", {"approved": False})
        self.error("contract_invalid", lambda: self.finish(job))
        self.assertEqual(self.state()["cases"]["louise"]["stage"], "character")
        self.assertEqual(self.state()["jobs"][job["job_id"]]["status"], "active")
        self.make_design()
        self.assertEqual(self.finish(job)["stage"], "story")

    def test_expiration_and_resume_preserve_partial_manuscript_bytes(self):
        manuscript = self.root / "editorial/clinical/louise.json"
        write(self.root, "editorial/clinical/louise.json", {"episodes": [{"number": 1, "text": "Original déjà rédigé"}, {"number": 2}]})
        original = manuscript.read_bytes()
        self.pipeline.init()
        self.assertEqual(self.state()["cases"]["louise"]["stage"], "character")
        job = self.pipeline.claim("character", "designer", "louise")
        self.pipeline.heartbeat(job["job_id"], job["token"], checkpoint=True)
        self.current += timedelta(seconds=61)
        self.assertEqual(self.pipeline.status()["jobs"][0]["status"], "expired")
        self.error("inactive_job", lambda: self.finish(job))
        self.error("no_job", lambda: self.pipeline.claim("character", "other", "louise"))
        self.error("no_job", lambda: self.pipeline.claim("character", "other", "louise", resume=True))
        resumed = self.pipeline.claim("character", "designer", "louise", resume=True)
        self.assertEqual(resumed["job_id"], job["job_id"])
        self.assertNotEqual(resumed["token"], job["token"])
        self.error("invalid_token", lambda: self.pipeline.heartbeat(job["job_id"], job["token"]))
        self.assertEqual(manuscript.read_bytes(), original)
        self.assertEqual(self.state()["jobs"][job["job_id"]]["checkpoints"][0]["observed"]["episodes_present"], 2)

    def test_a_different_valid_design_is_still_a_stale_dependency(self):
        self.complete_design()
        write(self.root, "editorial/clinical/designs/louise.json", {"approved": True, "voice": "modified elsewhere"})
        self.error("inputs_changed", lambda: self.pipeline.claim("story", "author", "louise"))

    def test_two_revision_loops_then_block_with_findings_preserved(self):
        self.complete_design()
        for number in range(3):
            self.complete_story()
            _, result = self.complete_review(status="revision_requested")
            self.assertEqual(result["stage"], "story" if number < 2 else "blocked")
        item = self.state()["cases"]["louise"]
        self.assertEqual(item["revision_rounds"], 3)
        self.assertEqual(item["last_error"]["code"], "revision_limit")
        self.assertEqual(item["findings"][0]["episode"], 2)

    def test_publication_requires_exact_integrated_public_bytes(self):
        self.complete_design()
        self.complete_story()
        self.complete_review()
        manuscript = "editorial/clinical/louise.json"
        release = {"date": "2026-11-07", "status": "editorially_reviewed", "article_format": 2,
                   "sha256": sha(self.root / manuscript), "dossier_sha256": sha(self.root / "editorial/clinical/dossiers/louise.json"),
                   "design_sha256": sha(self.root / "editorial/clinical/designs/louise.json"),
                   "independent_review_sha256": sha(self.root / "editorial/clinical/reviews/louise.json"),
                   "review": "Relecture indépendante effectuée"}
        write(self.root, "editorial/clinical/releases.json", {"arcs": {"louise": release}})
        write(self.root, "data/parcours_topics.json", {"louise": ["repère"]})
        self.error("not_integrated", lambda: self.pipeline.publish("louise"))
        write(self.root, "content/personnages/louise.md", "Published prose\n")
        self.assertEqual(self.pipeline.publish("louise")["stage"], "published")
        write(self.root, "content/personnages/louise.md", "Changed public prose\n")
        self.error("not_integrated", lambda: self.pipeline.publish("louise"))
        self.assertFalse(next(case for case in self.pipeline.status()["cases"]
                              if case["character_id"] == "louise")["publication_files_current"])

    def test_status_checks_all_publication_evidence_not_only_public_prose(self):
        self.complete_design()
        self.complete_story()
        self.complete_review()
        self.integrated_release()
        self.pipeline.publish("louise")

        def current():
            return next(case for case in self.pipeline.status()["cases"]
                        if case["character_id"] == "louise")["publication_files_current"]

        self.assertTrue(current())
        changes = {
            "editorial/clinical/dossiers/louise.json": lambda data: {**data, "extra": "changed dossier"},
            "editorial/clinical/designs/louise.json": lambda data: {**data, "voice": "changed design"},
            "editorial/clinical/reviews/louise.json": lambda data: {**data, "extra": "changed review"},
            "editorial/clinical/louise.json": lambda data: {**data, "extra": "changed manuscript"},
            "editorial/clinical/releases.json": lambda data: {"arcs": {"louise": {**data["arcs"]["louise"], "review": "changed release"}}},
            "data/parcours_topics.json": lambda data: {"louise": ["wrong public topic"]},
        }
        for relative, mutation in changes.items():
            with self.subTest(evidence=relative):
                path = self.root / relative
                original = path.read_bytes()
                try:
                    write(self.root, relative, mutation(json.loads(original)))
                    self.assertFalse(current())
                finally:
                    path.write_bytes(original)
                self.assertTrue(current())

    def test_new_publication_cannot_drop_its_independent_review_evidence(self):
        self.complete_design()
        self.complete_story()
        self.complete_review()
        release = self.integrated_release()
        self.pipeline.publish("louise")
        (self.root / "editorial/clinical/designs/louise.json").unlink()
        (self.root / "editorial/clinical/reviews/louise.json").unlink()
        release.pop("design_sha256")
        release.pop("independent_review_sha256")
        write(self.root, "editorial/clinical/releases.json", {"arcs": {"louise": release}})
        self.assertFalse(next(case for case in self.pipeline.status()["cases"]
                              if case["character_id"] == "louise")["publication_files_current"])

    def test_migration_registry_is_an_immutable_job_input(self):
        job = self.pipeline.claim("character", "designer", "louise")
        self.make_design()
        write(self.root, "editorial/automation/legacy-publications.json", {"schema_version": 1, "manuscripts": {"louise": "invented"}})
        self.error("inputs_changed", lambda: self.finish(job))

    def test_all_stages_are_reported_and_paths_stay_inside_checkout(self):
        self.assertEqual(set(self.pipeline.status()["counts"]),
                         {"character", "story", "review", "ready_to_publish", "published", "blocked"})
        self.error("unknown_character", lambda: self.pipeline.claim("character", "designer", "../outside"))
        outside = Path(self.temporary.name).parent / "outside-pipeline-test"
        (self.root / "outside-link").symlink_to(outside)
        self.error("unsafe_path", lambda: self.pipeline.path("outside-link/private.json"))


if __name__ == "__main__":
    unittest.main()
