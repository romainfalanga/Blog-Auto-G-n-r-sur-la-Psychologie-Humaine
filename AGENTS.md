# Décode ton esprit — consignes de reprise

## Le produit aujourd’hui

Un espace personnel, sans catalogue public. Le menu (`data/navigation.json`) comprend : Ambre (`/ambre/`), Ma ligne de vie, Mes proches, Mon enfance (`/cartographie/`), Mes symptômes (`/mes-symptomes/`), Mes sujets, Mes bilans, Mes diagnostics et Mon compte. Netlify redirige `/` vers `/ambre/`.

- **Hugo 0.140 + Netlify** (racine du dépôt) : Mon enfance et Mes symptômes (`layouts/_default/cartographie.html`, `static/js/cartographie*.js`), la page 404, les redirections et les relais vers le Worker (`netlify.toml`, `netlify/edge-functions/`).
- **Cloudflare Worker + D1** (`journal/`) : Ambre, les comptes et toutes les autres pages. Les migrations sont dans `journal/migrations/`.
- Les données `data/cartographie.json`, `data/symptomes_*.json` et les modules `journal/src/*.generated.mjs` sont générés par `scripts/build_*` à partir de `editorial/cartographie/` et `editorial/symptomes/` : modifier les sources, relancer le script, ne jamais éditer la sortie à la main.

Le catalogue public (personnages, articles, pipeline éditorial) a été retiré ; ses anciennes adresses redirigent vers `/ambre/`. Ne pas le recréer.

## Avant de travailler

Lire `docs/reprise-2026-11-19-codex.md` (note de reprise), puis examiner l’état du dépôt et les derniers commits. Les fichiers du dépôt font foi ; un ancien résumé de conversation peut être périmé.

## Tests

```sh
npm --prefix journal test
HUGO_BIN=<chemin de hugo 0.140> node --test tests/*.test.cjs
```

Les autres contrôles (scripts `--check`, tests Python) sont listés dans `.github/workflows/verify.yml` ; les faire tous passer avant de livrer.

## Règles

- Ne jamais utiliser les données d’une personne réelle dans les tests : uniquement des comptes et des textes fictifs.
- Aucun appel IA payant sans l’accord explicite du propriétaire.
- Aucun secret dans le dépôt (clés, jetons, mots de passe) : ils restent dans les secrets Cloudflare, Netlify ou GitHub.
- Les migrations D1 sont additives : ajouter des tables ou des colonnes, ne pas supprimer ni réécrire de données sans demande explicite ; ne pas rejouer une migration déjà appliquée.
- Une mise en ligne n’est acquise qu’après vérification du déploiement distant.
