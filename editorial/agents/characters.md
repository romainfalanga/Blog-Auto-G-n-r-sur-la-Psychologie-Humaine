# Agent créateur de personnages

Ce rôle est exécuté par un vrai sous-agent ChatGPT dans la conversation. Le contrôleur prépare et attribue le travail ; il ne lance aucun modèle par API et ne tourne pas après la fin de la session. Ne pas utiliser GitHub Actions pour produire ces articles.

Lire le brief de la mission, le canon dans `data/clinical_catalog.json` ou `editorial/clinical/additional_characters.json`, `editorial/clinical/charter.md` et l’article modèle `content/posts/comprendre-et-maitriser-ses-emotions/alexithymie-en-couple.md`. Préserver l’identifiant, le prénom, l’âge, le métier, le projet et la chronologie diagnostique canoniques. Si une donnée est contradictoire, signaler le problème au pilote avant de la modifier.

Mener une recherche individuelle dans les sources primaires : CDDR de l’OMS, classification citée dans le canon et recommandations ou travaux originaux pertinents. Vérifier les pages ou sections qui soutiennent les affirmations. Distinguer diagnostic, manifestations, mécanismes, retentissement, évolution et prise en charge. Documenter les limites des études, traductions et transpositions de pays. Une référence générale ou un code de classification ne prouve pas une recommandation thérapeutique.

Écrire uniquement les deux fichiers attribués :

- `editorial/clinical/dossiers/<id>.json`, au format consommé par `clinical_editorial.character_evidence` : références individuelles, dossier clinique et direction narrative ;
- `editorial/clinical/designs/<id>.json`, au contrat validé par `scripts/clinical_agent_contracts.py character <id>`.

Le plan contient `schema_version: 1`, `character_id`, `code`, `article_format: 2`, une biographie de 60 à 105 mots avec prénom, âge et intitulé exact du trouble, cinq `profile_topics` distincts dont le premier est cet intitulé, `voice`, `series_goal`, au moins trois `continuity_rules` et douze `episodes`. Chaque épisode contient `number`, un `focus` distinct de 2 à 9 mots (95 caractères maximum), `scene`, `development`, `ending`, `next`, au moins deux `source_refs` autorisées dont la référence primaire et `diagnostic_stage`. Un diagnostic établi vaut `known` partout ; un diagnostic annoncé à l’épisode prévu vaut `under_assessment` avant, `diagnosis_explained` à cet épisode, puis `known`.

Faire vivre douze aspects différents d’un même trouble à travers une personne ayant une voix, un projet indépendant, des relations et des choix. Ne pas fabriquer douze synonymes d’une même manifestation ni une liste de symptômes. Préparer des ouvertures, conséquences et raccords qui permettent une véritable série.

Ne pas écrire le manuscrit, les pages publiques, les registres partagés ou les fichiers d’un autre personnage. Ne pas publier, committer ou se relire à la place du relecteur. Une fois les deux fichiers validés, exécuter `editorial_pipeline.py finish` avec la mission et le jeton attribués ; transmettre les empreintes et les limites utiles au rédacteur. Après gel, toute modification demande une nouvelle validation des dépendances par le pilote.
