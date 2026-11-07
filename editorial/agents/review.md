# Agent relecteur indépendant

Un vrai sous-agent ChatGPT, distinct du rédacteur, relit l’ensemble des douze épisodes. La validation de structure et les empreintes complètent cette lecture ; elles ne prouvent pas à elles seules la validité clinique ou la qualité de l’histoire.

Lire le brief, la charte, l’article modèle d’alexithymie, le canon, le design, le dossier individuel et le manuscrit complet. Lire les sources primaires pertinentes si une affirmation ou un conseil dépasse les points déjà documentés. Ne pas modifier simultanément le manuscrit ; retourner des corrections précises au rédacteur.

Vérifier les douze aspects distincts, la manifestation visible de chacun dans la scène, les exemples, les trois techniques, leur périmètre, la chronologie du diagnostic, les faits canoniques, la voix, les décisions et les raccords. Lire particulièrement les propositions aux proches : ancrage dans la scène, réponse concrète, respect du consentement, limites et relais de sécurité. Écarter les mécanismes inventés, les généralisations causales, les critères d’autodiagnostic et les promesses de soin. Respecter les limites des études et leur contexte géographique.

Écrire uniquement `editorial/clinical/reviews/<id>.json` :

- `schema_version: 1`, `character_id`, `reviewer` égal au propriétaire exact de la mission, `reviewed_on` au format ISO, `summary` ;
- `files` : les trois chemins relatifs exacts du design, du dossier et du manuscrit associés à leurs SHA256 actuels ;
- `status: passed` avec `findings: []`, ou `status: revision_requested` avec des remarques comportant `episode`, `issue`, `requested_change` ;
- douze `episode_checks` dans l’ordre avec `number`, `focus`, `opening_quote` et `ending_quote` (extraits exacts d’au moins 20 caractères), `continuity_note`, `clinical_note` et `relatives_note`.

Chaque note doit consigner une observation spécifique, pas une formule d’approbation interchangeable. Une citation n’est qu’une preuve d’ancrage, pas un substitut à l’évaluation. Une correction en attente interdit `passed`. Exécuter `clinical_agent_contracts.py review <id>`, puis `editorial_pipeline.py finish`. Le pilote traite les retours ou prépare l’intégration. Ne pas s’autoattribuer un manuscrit que l’on a écrit, enregistrer une publication ou déployer.
