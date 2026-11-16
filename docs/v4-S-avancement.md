# Lot S (sûreté clinique et qualité d'Ambre) — avancement

Base : plan v4.1 (`docs/plan-ambre-v4-2026-11-18.md`), recherche v3 (§1.2, §3, §5, §6, §7), audit de robustesse (F5–F7, F17–F20).
Suite de départ : `npm --prefix journal test` 594/594.

## Fait
- Corpus de crise indépendant : `journal/tests/fixtures/v4-crise-corpus.json` (65 à risque, 65 neutres trompeurs), écrit par une session Claude distincte sans accès au code, avant toute modification des règles. Mesure avant règles : rappel lexical 9/65 (13,8 %), faux positifs 10/65.
- Nouveau module de citations `journal/src/quote-guard.mjs` (normalisation typographique, négation dans la proposition, longueur minimale 12 caractères ou phrase entière, appui des valeurs sur la citation, correspondance approchée pour le contrôle de risque).
- F20 : notes de cartographie, propositions de profil et de plan passent par ce module.
- Filet lexical à trois niveaux (`chat-safety.mjs` : explicite, possible, indice), F17 (ressources sur toutes les erreurs d'envoi).

- F18/F19 dans le moteur (vérification de risque dédiée en cas de panne d'analyse, voie prudente `cautiousReply`, ancrage approché, critère indépendant du relecteur).
- F5–F7 (`mergeMemory` : `safety_history` et `key_facts` épinglés et datés, dédoublonnage approché, éléments invalides écartés un à un, lot « non résumé » après 3 échecs, phrase du signal), R6, R16.
- Signaux cliniques (`clinical-signals.mjs`) branchés dans l'analyse, la rédaction et la relecture ; R1/R24/R27/R33 dans le rédacteur.
- Tests : `v4-S-corpus`, `v4-S-safety`, `v4-S-memory`.

## En cours
- Tests des signaux (R2–R8, R10, R13, R14, R17, R20, R21, #12, #34), budget, batterie R11.

## Reste
- Troisième corpus de contrôle (demandé à la session indépendante), rapport final.
