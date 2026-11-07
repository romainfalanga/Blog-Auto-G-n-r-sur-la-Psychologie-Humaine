"""Durable work queue for real ChatGPT editorial agents; never runs an AI model.

The CLI assigns and checks work performed in the current ChatGPT session.  It
does not schedule future sessions, call an API, generate prose, or publish files.
All paths are anchored to this checkout. Drafts are only read, never rewritten.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import fcntl
import hashlib
import hmac
import importlib
import json
import os
from pathlib import Path
import re
import secrets
import tempfile
import uuid


ROOT = Path(__file__).resolve().parents[1]
ROLES = ("character", "story", "review")
STAGES = (*ROLES, "ready_to_publish", "published", "blocked")
PROMPTS = {"character": "characters.md", "story": "stories.md", "review": "review.md"}
IDENT = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*\Z")
JOB_ID = re.compile(r"[a-f0-9]{32}\Z")


class PipelineError(ValueError):
    def __init__(self, code, message):
        self.code = code
        super().__init__(message)


def require(condition, code, message):
    if not condition:
        raise PipelineError(code, message)


def now():
    return datetime.now(timezone.utc)


def timestamp(value):
    return value.isoformat(timespec="seconds").replace("+00:00", "Z")


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def json_bytes(data):
    return (json.dumps(data, ensure_ascii=False, indent=2) + "\n").encode()


def safe_error(error):
    message = re.sub(r"sk-[A-Za-z0-9_-]{12,}", "[redacted]", str(error))
    return re.sub(r"[\x00-\x1f\x7f]", " ", message)[:600]


class Pipeline:
    """Injectable clock and adapters keep concurrency and contracts testable."""

    def __init__(self, root=ROOT, editorial=None, contracts=None, clock=now):
        self.root = Path(root).resolve()
        self.editorial = editorial or importlib.import_module("clinical_editorial")
        self.contracts = contracts
        self.clock = clock

    def path(self, relative):
        require(isinstance(relative, str) and not Path(relative).is_absolute()
                and ".." not in Path(relative).parts, "unsafe_path", "Chemin hors du projet interdit.")
        path = self.root / relative
        require(path.resolve().is_relative_to(self.root), "unsafe_path", "Lien hors du projet interdit.")
        return path

    def read(self, relative):
        return json.loads(self.path(relative).read_text(encoding="utf-8"))

    def hash_file(self, relative):
        path = self.path(relative)
        return digest(path.read_bytes()) if path.is_file() else None

    def policy(self):
        policy = self.read("editorial/automation/policy.json")
        require(policy.get("schema_version") == 1, "invalid_policy", "Version de politique inconnue.")
        require(isinstance(policy.get("lease_seconds"), int) and 60 <= policy["lease_seconds"] <= 86400,
                "invalid_policy", "Durée de bail invalide.")
        require(policy.get("max_revision_rounds") == 2, "invalid_policy", "Deux boucles de correction au maximum.")
        require(isinstance(policy.get("max_active_jobs"), int) and policy["max_active_jobs"] >= 1,
                "invalid_policy", "Limite de travaux invalide.")
        return policy

    def atomic_write(self, relative, data):
        path = self.path(relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix=".pipeline-", dir=path.parent)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(json_bytes(data))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
            directory = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    @contextmanager
    def locked(self, initialize=False):
        path = self.path("editorial/automation/.pipeline.lock")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                state_path = self.path("editorial/automation/state.json")
                if state_path.exists():
                    state = self.read("editorial/automation/state.json")
                    require(state.get("schema_version") == 1, "invalid_state", "Version d'état inconnue.")
                else:
                    require(initialize, "not_initialized", "Initialiser la file avec init.")
                    state = {"schema_version": 1, "revision": 0, "created_at": timestamp(self.clock()),
                             "cases": {}, "jobs": {}}
                self.expire(state)
                yield state
                state["revision"] += 1
                state["updated_at"] = timestamp(self.clock())
                self.atomic_write("editorial/automation/state.json", state)
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    def expire(self, state):
        current = timestamp(self.clock())
        for job in state["jobs"].values():
            if job["status"] == "active" and job["expires_at"] <= current:
                job["status"] = "expired"
                job["expired_at"] = current

    def characters(self):
        characters, shared = self.editorial.sources_and_characters()
        require(all(IDENT.fullmatch(ident) for ident in characters), "invalid_catalogue", "Identifiant de personnage invalide.")
        return characters, shared

    def outputs(self, role, ident):
        paths = {
            "character": [f"editorial/clinical/designs/{ident}.json", f"editorial/clinical/dossiers/{ident}.json"],
            "story": [f"editorial/clinical/{ident}.json"],
            "review": [f"editorial/clinical/reviews/{ident}.json"],
        }
        return paths[role]

    def observe(self, ident):
        manuscript = f"editorial/clinical/{ident}.json"
        path = self.path(manuscript)
        episodes, readable = 0, True
        if path.is_file():
            try:
                data = self.read(manuscript)
                episodes = len(data.get("episodes", []))
            except (ValueError, TypeError, AttributeError):
                readable = False
        return {"manuscript_sha256": self.hash_file(manuscript), "episodes_present": episodes,
                "manuscript_readable": readable,
                "design_sha256": self.hash_file(f"editorial/clinical/designs/{ident}.json"),
                "dossier_sha256": self.hash_file(f"editorial/clinical/dossiers/{ident}.json"),
                "review_sha256": self.hash_file(f"editorial/clinical/reviews/{ident}.json")}

    def publication_proof(self, ident):
        """Require the release AND every generated public file to match exactly."""
        characters, shared = self.characters()
        require(ident in characters, "unknown_character", "Personnage inconnu.")
        releases = self.read("editorial/clinical/releases.json")["arcs"]
        require(ident in releases, "not_integrated", "Parcours absent du registre de publication.")
        release = releases[ident]
        require(release.get("status") == "editorially_reviewed" and bool(release.get("review")),
                "not_integrated", "Relecture enregistrée requise.")
        manuscript = f"editorial/clinical/{ident}.json"
        require(self.hash_file(manuscript) == release.get("sha256"), "not_integrated", "Hash du manuscrit publié différent.")
        for field, directory in (("design_sha256", "designs"), ("independent_review_sha256", "reviews")):
            relative = f"editorial/clinical/{directory}/{ident}.json"
            current_hash = self.hash_file(relative)
            if current_hash is not None or field in release:
                require(current_hash is not None and current_hash == release.get(field),
                        "not_integrated", "Hash du contrat publié différent : " + directory)
        if self.editorial.agent_review_required(ident, release):
            approval = self.validator("review", ident)
            require(approval.get("status") == "passed", "not_integrated", "Relecture indépendante acceptée requise.")
            for field, directory in (("design_sha256", "designs"), ("independent_review_sha256", "reviews")):
                relative = f"editorial/clinical/{directory}/{ident}.json"
                require(self.hash_file(relative) is not None and self.hash_file(relative) == release.get(field),
                        "not_integrated", "Preuve indépendante non enregistrée : " + directory)
        evidence, dossier_review = self.editorial.character_evidence(characters[ident], shared)
        dossier_sha = dossier_review["sha256"] if dossier_review else None
        require(dossier_sha == release.get("dossier_sha256"), "not_integrated", "Hash du dossier publié différent.")
        outputs, topics = self.editorial.arc_outputs(
            characters[ident], self.read(manuscript), release["date"], list(releases).index(ident),
            evidence, release.get("article_format", 1))
        proof = {}
        for path, content in outputs.items():
            relative = str(Path(path).relative_to(self.root))
            require(self.hash_file(relative) == digest(content.encode()), "not_integrated",
                    "Fichier public absent ou différent : " + relative)
            proof[relative] = self.hash_file(relative)
        published_topics = self.read("data/parcours_topics.json")
        require(all(published_topics.get(key) == value for key, value in topics.items()),
                "not_integrated", "Repères publics absents ou différents.")
        return {"verified_at": timestamp(self.clock()), "release": release, "files": proof,
                "topics_sha256": digest(json_bytes(topics))}

    def init(self):
        characters, _ = self.characters()
        releases = self.read("editorial/clinical/releases.json")["arcs"]
        proofs, failures = {}, {}
        for ident in releases:
            if ident in characters:
                try:
                    proofs[ident] = self.publication_proof(ident)
                except (ValueError, OSError, KeyError) as error:
                    failures[ident] = safe_error(error)
        with self.locked(initialize=True) as state:
            for ident, character in characters.items():
                if ident in state["cases"]:
                    continue
                item = {"character_id": ident, "code": character["code"], "stage": "character",
                        "created_at": timestamp(self.clock()), "active_job": None,
                        "owners": {role: [] for role in ROLES}, "revision_rounds": 0,
                        "accepted": {}, "observed_at_init": self.observe(ident)}
                if ident in proofs:
                    item.update(stage="published", publication=proofs[ident])
                elif ident in failures:
                    item.update(stage="blocked", last_error={"code": "publication_mismatch", "message": failures[ident]})
                state["cases"][ident] = item
        return self.status()

    def ordered(self, cases):
        priority = self.policy().get("priority_characters", [])
        rank = {ident: number for number, ident in enumerate(priority)}
        return sorted(cases, key=lambda ident: (rank.get(ident, len(rank)), ident))

    def status(self):
        with self.locked() as state:
            # Full publication validation may inspect hundreds of public files.
            # Take a consistent queue snapshot, then release the short state lock.
            snapshot = json.loads(json.dumps(state))
        counts = {stage: 0 for stage in STAGES}
        cases = []
        for ident in self.ordered(snapshot["cases"]):
            item = snapshot["cases"][ident]
            counts[item["stage"]] += 1
            row = {key: item[key] for key in ("character_id", "code", "stage", "active_job", "revision_rounds", "owners")}
            row["observed"] = self.observe(ident)
            row["validated_evidence_current"] = {
                role: all(self.hash_file(path) == sha for path, sha in accepted["validation"]["files"].items())
                for role, accepted in item.get("accepted", {}).items()}
            if item.get("publication"):
                try:
                    current = self.publication_proof(ident)
                    row["publication_files_current"] = all(
                        current[key] == item["publication"].get(key)
                        for key in ("release", "files", "topics_sha256"))
                except (ValueError, OSError, KeyError, TypeError) as error:
                    row["publication_files_current"] = False
                    row["publication_error"] = safe_error(error)
            if item.get("last_error"):
                row["last_error"] = item["last_error"]
            if item.get("findings"):
                row["findings"] = item["findings"]
            cases.append(row)
        jobs = [{key: job.get(key) for key in ("id", "character_id", "role", "owner", "status", "created_at", "expires_at", "last_error")}
                for job in snapshot["jobs"].values() if job["status"] in ("active", "expired", "stale")]
        return {"ok": True, "execution": "ChatGPT session agents; no background scheduler or model API",
                "state_revision": snapshot["revision"], "counts": counts, "cases": cases, "jobs": jobs}

    def snapshot(self, role, ident, case):
        """Capture actual agent inputs once; finish checks their exact bytes."""
        prompt_file = "editorial/agents/" + PROMPTS[role]
        require(self.path(prompt_file).is_file(), "missing_prompt", "Carte de rôle absente : " + prompt_file)
        input_paths = ["data/clinical_catalog.json", "data/clinical_sources.json",
                       "editorial/clinical/additional_characters.json", "editorial/clinical/charter.md",
                       "editorial/automation/policy.json", "scripts/clinical_editorial.py",
                       "editorial/automation/legacy-publications.json", "scripts/clinical_agent_contracts.py", prompt_file]
        if role in ("story", "review"):
            input_paths += self.outputs("character", ident)
        if role == "review":
            input_paths += self.outputs("story", ident)
        # The previous review is an input to a revision, never a generated story.
        if role == "story" and case.get("revision_rounds"):
            input_paths += self.outputs("review", ident)
        before = {path: self.hash_file(path) for path in input_paths}
        brief = self.editorial.brief(ident)
        data = {"clinical_brief": brief, "findings": case.get("findings", []),
                "existing_manuscript": self.read(f"editorial/clinical/{ident}.json")
                if self.path(f"editorial/clinical/{ident}.json").is_file() else None}
        if role in ("story", "review"):
            data["design"] = self.read(f"editorial/clinical/designs/{ident}.json")
        snapshot = {"prompt_file": prompt_file, "prompt": self.path(prompt_file).read_text(encoding="utf-8"),
                    "inputs": before, "data": data, "allowed_outputs": self.outputs(role, ident),
                    "output_baseline": {path: self.hash_file(path) for path in self.outputs(role, ident)}}
        require(before == {path: self.hash_file(path) for path in input_paths},
                "inputs_changed", "Les entrées ont changé pendant la préparation ; réessayer.")
        return snapshot

    def unchanged(self, snapshot):
        return all(self.hash_file(path) == expected for path, expected in snapshot["inputs"].items())

    def ownership(self, case, role, owner):
        if role == "story":
            require(owner not in case["owners"]["character"], "role_conflict", "Le rédacteur doit être différent du concepteur.")
        if role == "review":
            require(owner not in case["owners"]["story"], "role_conflict", "Un auteur ne peut pas relire son propre parcours.")

    def prerequisites(self, case, role):
        required = ("character", "story") if role == "review" else ("character",) if role == "story" else ()
        for prior in required:
            accepted = case["accepted"].get(prior)
            require(accepted is not None, "dependencies", "Validation préalable absente : " + prior)
            require(all(self.hash_file(path) == sha for path, sha in accepted["validation"]["files"].items()),
                    "inputs_changed", "Une entrée validée a été modifiée : " + prior)

    def claim(self, role, owner, ident=None, resume=False):
        require(role in ROLES, "unknown_role", "Rôle inconnu.")
        require(isinstance(owner, str) and 1 <= len(owner) <= 100 and owner == owner.strip()
                and not re.search(r"[\x00-\x1f\x7f]", owner), "invalid_owner", "Nom de propriétaire invalide.")
        if ident is not None:
            require(IDENT.fullmatch(ident), "unknown_character", "Identifiant de personnage invalide.")
        policy = self.policy()
        with self.locked() as state:
            if ident is not None:
                require(ident in state["cases"], "unknown_character", "Personnage inconnu.")
                candidates = [ident]
            else:
                candidates = self.ordered(state["cases"])
            eligible = []
            for candidate in candidates:
                case = state["cases"][candidate]
                if case["stage"] != role:
                    continue
                previous = state["jobs"].get(case.get("active_job"))
                if previous:
                    if not resume or previous["owner"] != owner or previous["role"] != role:
                        continue
                elif resume:
                    continue
                try:
                    self.ownership(case, role, owner)
                except PipelineError:
                    if ident:
                        raise
                    continue
                eligible.append((candidate, case, previous))
                break
            require(eligible, "no_job", "Aucun travail disponible pour ce rôle et ce propriétaire (vérifier les dépendances et --resume).")
            ident, case, previous = eligible[0]
            self.prerequisites(case, role)
            active = [job for job in state["jobs"].values() if job["status"] == "active" and (not previous or job["id"] != previous["id"])]
            require(len(active) < policy["max_active_jobs"], "capacity", "Limite de travaux actifs atteinte.")
            role_limit = policy.get("role_limits", {}).get(role, policy["max_active_jobs"])
            require(sum(job["role"] == role for job in active) < role_limit, "capacity", "Limite de ce rôle atteinte.")
            token = secrets.token_urlsafe(32)
            stamp = timestamp(self.clock())
            if previous and self.unchanged(previous["snapshot"]):
                job = previous
                job["resumed_at"] = stamp
                job["resume_count"] = job.get("resume_count", 0) + 1
            else:
                snapshot = self.snapshot(role, ident, case)
                job_id = uuid.uuid4().hex
                job = {"id": job_id, "character_id": ident, "role": role, "owner": owner,
                       "created_at": stamp, "snapshot": snapshot, "snapshot_sha256": digest(json_bytes(snapshot))}
                state["jobs"][job_id] = job
                if previous:
                    previous.update(status="superseded", superseded_by=job_id)
                    job["resumes_job"] = previous["id"]
            job.update(status="active", token_sha256=digest(token.encode()), heartbeat_at=stamp,
                       expires_at=timestamp(self.clock() + timedelta(seconds=policy["lease_seconds"])))
            case["active_job"] = job["id"]
            if owner not in case["owners"][role]:
                case["owners"][role].append(owner)
            return {"ok": True, "job_id": job["id"], "token": token, "character_id": ident,
                    "role": role, "owner": owner, "expires_at": job["expires_at"],
                    "snapshot_sha256": job["snapshot_sha256"], "prompt_file": job["snapshot"]["prompt_file"],
                    "data": job["snapshot"]["data"], "allowed_outputs": job["snapshot"]["allowed_outputs"],
                    "resumed": bool(previous)}

    def get_job(self, state, job_id, token=None):
        require(isinstance(job_id, str) and JOB_ID.fullmatch(job_id) and job_id in state["jobs"],
                "unknown_job", "Travail inconnu.")
        job = state["jobs"][job_id]
        require(digest(json_bytes(job["snapshot"])) == job["snapshot_sha256"],
                "invalid_snapshot", "Le snapshot du travail a été modifié.")
        if token is not None:
            require(isinstance(token, str) and hmac.compare_digest(digest(token.encode()), job["token_sha256"]),
                    "invalid_token", "Jeton de travail invalide.")
            require(job["status"] == "active", "inactive_job", "Travail inactif ; le propriétaire doit utiliser claim --resume.")
            require(state["cases"][job["character_id"]]["active_job"] == job_id,
                    "inactive_job", "Ce travail n'est plus attribué.")
        return job

    def brief(self, job_id):
        with self.locked() as state:
            job = self.get_job(state, job_id)
            return {"ok": True, "job_id": job_id, "role": job["role"], "owner": job["owner"],
                    "status": job["status"], "snapshot_sha256": job["snapshot_sha256"], **job["snapshot"]}

    def heartbeat(self, job_id, token, checkpoint=False):
        with self.locked() as state:
            job = self.get_job(state, job_id, token)
            job["heartbeat_at"] = timestamp(self.clock())
            job["expires_at"] = timestamp(self.clock() + timedelta(seconds=self.policy()["lease_seconds"]))
            if checkpoint:
                job.setdefault("checkpoints", []).append({"at": timestamp(self.clock()),
                    "outputs": {path: self.hash_file(path) for path in job["snapshot"]["allowed_outputs"]},
                    "observed": self.observe(job["character_id"])})
            return {"ok": True, "job_id": job_id, "expires_at": job["expires_at"], "checkpoint": checkpoint}

    def validator(self, role, ident):
        contracts = self.contracts or importlib.import_module("clinical_agent_contracts")
        return getattr(contracts, {"character": "validate_design", "story": "validate_story", "review": "validate_review"}[role])(ident)

    def finish(self, job_id, token):
        # Validation can read a full twelve-episode arc; it runs outside the lock.
        with self.locked() as state:
            job = self.get_job(state, job_id, token)
            snapshot_sha = job["snapshot_sha256"]
            ident, role = job["character_id"], job["role"]
            initial_outputs = {path: self.hash_file(path) for path in job["snapshot"]["allowed_outputs"]}
            stale = not self.unchanged(job["snapshot"])
            if stale:
                job.update(status="stale", last_error={"code": "inputs_changed", "message": "Entrées modifiées ; reprendre avec --resume."})
        require(not stale, "inputs_changed", "Les entrées ont changé ; reprendre avec --resume.")
        validation, failure = None, None
        try:
            validation = self.validator(role, ident)
            require(isinstance(validation, dict) and validation.get("character_id") == ident,
                    "contract_invalid", "Identifiant absent du résultat de validation.")
            files = validation.get("files")
            expected_files = set(self.outputs("character", ident))
            if role in ("story", "review"):
                expected_files.update(self.outputs("story", ident))
            if role == "review":
                expected_files.update(self.outputs("review", ident))
            require(isinstance(files, dict) and set(files) == expected_files
                    and all(sha is not None and self.hash_file(path) == sha for path, sha in files.items())
                    and all(initial_outputs[path] is not None and files[path] == initial_outputs[path] for path in initial_outputs),
                    "contract_invalid", "Sorties du contrat différentes des fichiers du travail.")
            if role == "review":
                require(validation.get("status") in ("passed", "revision_requested"), "contract_invalid", "Décision de relecture invalide.")
                require(validation["status"] != "revision_requested" or bool(validation.get("findings")),
                        "contract_invalid", "Les corrections demandées doivent être documentées.")
                require(validation.get("reviewer") == job["owner"], "contract_invalid", "Le nom du relecteur doit correspondre au propriétaire du travail.")
        except (ValueError, OSError, KeyError, TypeError, ImportError) as error:
            failure = {"code": "contract_invalid", "message": safe_error(error)}
        with self.locked() as state:
            job = self.get_job(state, job_id, token)
            require(job["snapshot_sha256"] == snapshot_sha, "inputs_changed", "Le travail a été remplacé.")
            if not self.unchanged(job["snapshot"]):
                failure = {"code": "inputs_changed", "message": "Les entrées ont changé pendant la validation."}
                job["status"] = "stale"
            elif initial_outputs != {path: self.hash_file(path) for path in initial_outputs}:
                failure = {"code": "outputs_changed", "message": "Les sorties ont changé pendant la validation ; réessayer."}
            if failure:
                job["last_error"] = failure
            else:
                case = state["cases"][ident]
                self.ownership(case, role, job["owner"])
                case["accepted"][role] = {"job_id": job_id, "owner": job["owner"],
                    "validated_at": timestamp(self.clock()), "validation": validation}
                case["active_job"] = None
                case.pop("last_error", None)
                job.update(status="finished", finished_at=timestamp(self.clock()), validation=validation)
                job.pop("last_error", None)
                if role == "character":
                    case["stage"] = "story"
                elif role == "story":
                    case["stage"] = "review"
                elif validation["status"] == "passed":
                    case["stage"] = "ready_to_publish"
                    case.pop("findings", None)
                else:
                    case["revision_rounds"] += 1
                    case["findings"] = validation["findings"]
                    if case["revision_rounds"] <= self.policy()["max_revision_rounds"]:
                        case["stage"] = "story"
                    else:
                        case["stage"] = "blocked"
                        case["last_error"] = {"code": "revision_limit", "message": "Deux boucles de correction épuisées ; arbitrage humain requis."}
                result = {"ok": True, "job_id": job_id, "character_id": ident, "stage": case["stage"], "validation": validation}
        if failure:
            raise PipelineError(failure["code"], failure["message"])
        return result

    def publish(self, ident):
        require(isinstance(ident, str) and IDENT.fullmatch(ident), "unknown_character", "Identifiant de personnage invalide.")
        with self.locked() as state:
            require(ident in state["cases"], "unknown_character", "Personnage inconnu.")
            case = state["cases"][ident]
            require(case["stage"] in ("ready_to_publish", "published"), "dependencies", "Une relecture indépendante réussie est requise.")
            accepted = case.get("accepted", {}).get("review")
            accepted_story = case.get("accepted", {}).get("story")
            if case["stage"] != "published":
                require(accepted and accepted_story, "dependencies", "Validations du manuscrit et de la relecture absentes.")
                files = {**accepted_story["validation"]["files"], **accepted["validation"]["files"]}
                require(all(self.hash_file(path) == sha for path, sha in files.items()),
                        "inputs_changed", "Le manuscrit ou la relecture a changé après validation.")
        proof = self.publication_proof(ident)
        with self.locked() as state:
            case = state["cases"][ident]
            require(case["stage"] in ("ready_to_publish", "published") and case.get("accepted", {}).get("review") == accepted,
                    "inputs_changed", "L'état éditorial a changé pendant la vérification.")
            # Check the public bytes again before accepting the observed publication.
            require(all(self.hash_file(path) == sha for path, sha in proof["files"].items()),
                    "inputs_changed", "Les fichiers publics ont changé pendant la vérification.")
            require(self.read("editorial/clinical/releases.json")["arcs"].get(ident) == proof["release"],
                    "inputs_changed", "Le registre a changé pendant la vérification.")
            if accepted:
                require(all(self.hash_file(path) == sha for path, sha in {**accepted_story["validation"]["files"], **accepted["validation"]["files"]}.items()),
                        "inputs_changed", "Les éléments relus ont changé pendant la vérification.")
            case.update(stage="published", publication=proof)
        return {"ok": True, "character_id": ident, "stage": "published", "publication": proof}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("init")
    sub.add_parser("status")
    claim = sub.add_parser("claim")
    claim.add_argument("--role", choices=ROLES, required=True)
    claim.add_argument("--owner", required=True)
    claim.add_argument("--character")
    claim.add_argument("--resume", action="store_true")
    brief = sub.add_parser("brief")
    brief.add_argument("--job", required=True)
    for action in ("heartbeat", "checkpoint", "finish"):
        command = sub.add_parser(action)
        command.add_argument("--job", required=True)
        command.add_argument("--token", required=True)
    publish = sub.add_parser("publish")
    publish.add_argument("--character", required=True)
    args = parser.parse_args()
    pipeline = Pipeline()
    try:
        if args.action in ("init", "status"):
            result = getattr(pipeline, args.action)()
        elif args.action == "claim":
            result = pipeline.claim(args.role, args.owner, args.character, args.resume)
        elif args.action == "brief":
            result = pipeline.brief(args.job)
        elif args.action in ("heartbeat", "checkpoint"):
            result = pipeline.heartbeat(args.job, args.token, args.action == "checkpoint")
        elif args.action == "finish":
            result = pipeline.finish(args.job, args.token)
        else:
            result = pipeline.publish(args.character)
    except (ValueError, OSError, KeyError, ImportError) as error:
        result = {"ok": False, "error": {"code": getattr(error, "code", "pipeline_error"), "message": safe_error(error)}}
    print(json.dumps(result, ensure_ascii=False, separators=(",", ":")))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
