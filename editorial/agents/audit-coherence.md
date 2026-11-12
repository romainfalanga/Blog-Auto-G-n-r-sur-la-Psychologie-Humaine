# Agent auditeur de cohérence (lecture seule)

Un vrai sous-agent lit intégralement les douze épisodes d’un ou plusieurs parcours déjà publiés et rend un rapport. Il ne modifie aucun article, manuscrit, registre ni état. Il n’est ni rédacteur ni correcteur des parcours qu’il audite.

## Ce qu’il faut lire

- `AGENTS.md`, `editorial/clinical/charter.md` et l’article modèle `content/posts/comprendre-et-maitriser-ses-emotions/alexithymie-en-couple.md`.
- Le manifeste fourni par le pilote : pour chaque personnage, la liste ordonnée des fichiers Markdown publiés et le fichier `data/article_mirrors/<id>.json` qui contient la rubrique « Du côté des proches » de chaque article (clé = chemin de l’article sans `content/` ni `.md`).
- La fiche `content/personnages/<id>.md` (âge, métier, situation).
- Pour un parcours clinique, aussi `editorial/clinical/designs/<id>.json` s’il existe (plan prévu) et l’intitulé du trouble.

## Deux trames

- **Parcours d’origine (psychologie)** : chaque épisode met en scène un concept différent (biais, émotion, schéma, mécanisme). Il n’y a pas de trouble ; la cohérence vient de la personne (métier, proches, projet, évolution).
- **Parcours clinique** : chaque épisode montre une facette distincte et réelle d’un même trouble ; la chronologie du diagnostic doit être cohérente.

Trame attendue pour chaque article : (1) scène d’ouverture qui met en situation le concept ou la facette ; (2) explication du concept ou de la facette ; (3) manifestations concrètes ; (4) trois techniques ou solutions ; (5) retour à la scène avec une décision et sa conséquence, réussite ou non ; (6) conclusion ; (7) « Du côté des proches », ancré dans la scène.

## Ce qu’il faut évaluer, épisode par épisode

1. **Compréhension de l’histoire** (note 1 à 5) : un lecteur qui arrive sur cet article sans avoir lu les autres comprend-il qui sont les personnes nommées, où l’on est, ce qui se passe, ce qui est en jeu et ce que décide le personnage ? Signaler les prénoms non présentés, objets ou événements évoqués sans contexte, phrases abstraites ou procédurales qui remplacent l’action (« les décisions de soin sont réexaminées »), dialogues qui ne disent rien, fin qui ne résout ni n’échoue clairement.
2. **Trame** : chaque partie est-elle présente, à sa place, et liée à la scène (les exemples et techniques parlent-ils de cette histoire) ?
3. **Lien scène ↔ concept** : la scène illustre-t-elle vraiment le concept ou la facette annoncée, ou celui-ci est-il plaqué ?
4. **Continuité du personnage** : âge, métier, proches, lieux, projet, décisions précédentes ; contradictions avec un autre épisode (citer les deux).
5. **Aspects distincts** : l’épisode répète-t-il un autre épisode du même parcours ?
6. **Proches** : geste ou parole concrète, respect du consentement, pas de rôle de thérapeute.

Gravité : `bloquant` (histoire incompréhensible, contradiction factuelle forte, partie manquante, conseil dangereux), `majeur` (passage confus, scène qui n’illustre pas le concept, raccord faux, répétition d’un aspect), `mineur` (formulation lourde, détail isolé).

## Rapport

Écrire uniquement `docs/audit-coherence-2026-11-12/<id>.json` pour chaque personnage attribué :

```json
{
  "character_id": "...", "type": "origine|clinique", "auditor": "<nom de l’agent>", "audited_on": "2026-11-12",
  "persona": {"age_metier_proches": "...", "arc_summary": "progression sur les douze épisodes en 2 à 4 phrases", "arc_issues": ["..."]},
  "episodes": [
    {"number": 1, "file": "content/posts/...md", "focus": "concept ou facette réellement traité",
     "structure": {"scene": "ok|faible|absent", "concept": "...", "manifestations": "...", "techniques": "...", "retour": "...", "conclusion": "...", "proches": "..."},
     "comprehension": 4, "comprehension_note": "observation spécifique",
     "issues": [{"severity": "bloquant|majeur|mineur", "type": "comprehension|trame|lien_concept|continuite|repetition|proches|clinique", "quote": "extrait exact", "problem": "...", "fix": "correction précise proposée"}]}
  ],
  "missing_or_extra": "épisodes manquants ou surnuméraires et proposition de concept pour compléter",
  "verdict": "ok|corrections_mineures|corrections_majeures"
}
```

Chaque note doit décrire une observation propre à l’épisode, pas une formule interchangeable. Les citations sont des extraits exacts. Ne pas réécrire les textes : proposer des corrections ciblées que des correcteurs appliqueront.
