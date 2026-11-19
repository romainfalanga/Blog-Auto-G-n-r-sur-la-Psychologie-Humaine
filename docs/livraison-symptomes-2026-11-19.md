# Livraison du 19 novembre 2026 : arrivée sur Ambre, moments marquants, Mes symptômes

Publié par [PR 102](https://github.com/romainfalanga/Blog-Auto-G-n-r-sur-la-Psychologie-Humaine/pull/102) (fusion 82bccad).

## Contenu
- `/` → `/mon-accompagnement/` (302, netlify.toml) ; page Ambre avant lancement (carte « Vos informations ») ; Mon enfance allégée sur mobile ; Mes proches en grille ; retour tactile ; « Ajouter un moment » depuis la page d'un proche ; pied de page = nom du site seul (la page Personnages n'a plus de lien, adresse `/personnages/`).
- Moments « Anecdotique / Marquant » (`life_events.significance`, migration 0027), distinct de `current_importance`.
- Page `/mes-symptomes/` (migration 0028 : `symptom_consents`, `symptom_checks`, `symptom_drafts`, `symptom_safety_alerts`). Plan : `docs/plan-mes-symptomes-2026-11-19.md` (v1.1 fait foi) ; contrat : `docs/symptomes-contrat-2026-11-19.md` ; source : `editorial/symptomes/symptomes.json` → `scripts/build_symptomes.py` ; relectures : `docs/relecture-clinique-symptomes-2026-11-19.md`.

## Production
- Migrations 0027 et 0028 appliquées via MCP et inscrites dans `d1_migrations`.
- Worker `decode-journal` redéployé (54 modules, 7 liaisons, cron `* * * * *`, `/api/health` ok).
- Recette réelle avec compte synthétique (supprimé, zéro ligne restante vérifiée dans D1) : tous points OK.

## Restant
- Lien TSPT complexe → sujet `icd_6b40` dans le croisement (règle `itq_tspt_c`) : modification du JSON bloquée par la protection automatique ; en attente de l'accord du propriétaire.
- Points incertains : codes CIM-11 à reconfirmer dans le navigateur officiel ; seuil d'entrée GAD-2 ≥ 2 (choix du site) ; cotation de l'item règles du PHQ-15 ; confirmation du Cn2r ; relecture par un clinicien humain souhaitable.
- Mon compte accepte une date de naissance de mineur sans avertissement (Mes symptômes filtre).
