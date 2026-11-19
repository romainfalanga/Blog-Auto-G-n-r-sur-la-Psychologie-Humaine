# « Dans le passé » sans précision obligatoire (19 novembre 2026)

Demande de Romain : dans Mes symptômes, « Dans le passé » s'enregistre comme Non, Oui ou Je ne sais pas.

## Changements
- `static/js/cartographie-ui.js` : un clic sur « Dans le passé » enregistre aussitôt la réponse `"past"`, sans fenêtre ; la précision n'est plus obligatoire et reste accessible par « Préciser » / « Modifier ma précision » ; vider la précision la retire sans changer la réponse ; les précisions existantes sont conservées lors des changements de réponse.
- `journal/src/cartographie.mjs` : le serveur accepte `"past"` sans note.
- `journal/src/companion-readiness.mjs` : `"past"` compte parmi les 516 réponses sans note, dans le comptage JavaScript et dans la garde SQL (parité testée) ; 515 réponses restent insuffisantes.
- `journal/src/clinical-assessment.mjs` : une réponse passée sans note est acceptée et reste rangée dans `historical` (jamais dans la fenêtre actuelle), sans référence de note.
- Inchangés : valeur `"past"` et son sens (avant les 12 derniers mois), exclusion du classement actuel et de la sécurité actuelle, contexte historique d'Ambre (`symptom-map-context.mjs` gère déjà l'absence de note), Mon enfance, contrôles de propriétaire, de session et de révision, archives des bilans.

## Vérifications
- `npm --prefix journal test` : 951/951 ; `node --test tests/*.test.cjs` avec Hugo 0.140 : 62/62.
- `tests/symptom-map-browser.cjs` à 390 tactile, 1280 et 1440 : clic sans fenêtre, enregistrement, rechargement, ajout puis retrait manuel de la précision avec réponse conservée (attente du rendu paresseux rendue robuste).
- Mon enfance : `tests/cartographie-browser.cjs`, `cartographie-choices-browser.cjs`, `v4-C-cartographie-browser.cjs`, `v5-C-cartographie-browser.cjs` passent.
- Aucune génération IA payante lancée.

## Toujours ouverts (non traités ici)
`context_changed` quand l'analyse du portrait se termine pendant la préparation d'un bilan ; `review_not_approved` sur un profil stabilisé ; aucun bilan réel complet validé (voir `docs/recette-openrouter-recharge-2026-11-19.md`).

## Retrait de la mention « ces 12 derniers mois » (19/11, soir)
Demande de Romain : ne plus afficher la période. « Oui » = la personne l'a ; « Dans le passé » = plus maintenant.
- Retirés : la ligne « Pour ces 12 derniers mois. » sous le titre (`layouts/_default/cartographie.html`) et le suffixe « — Ces 12 derniers mois. » des thèmes (`static/js/cartographie-ui.js`) ; descriptions des 15 thèmes concernés reformulées sans la période (`editorial/symptomes/v2/themes.json`, données régénérées).
- Inchangés : le sens interne de `"past"` (avant les 12 derniers mois) et ce qu'Ambre en reçoit ; libellés de questions ; empreinte du jeu de données mise à jour dans `clinical-assessment.mjs`.
