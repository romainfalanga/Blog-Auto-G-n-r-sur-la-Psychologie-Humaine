# Décode ton esprit

Site existant : https://decodetonesprit.fr ; journal privé : https://decodetonesprit.fr/journal/.

Les 231 articles et 20 personnages d’origine sont conservés. Des mots-clés tirés de leurs caractéristiques et thèmes facilitent la découverte sur les personnages, l’accueil, les catégories, les suggestions et les articles.

## Parcours

Le lecteur aime les articles qui lui parlent. Il peut expliquer pourquoi, mais ce commentaire reste facultatif. Sa collection n’a plus de limite de trois articles. À partir de dix articles aimés et de son consentement, il peut prendre l’abonnement à 10 EUR par mois. Aucun questionnaire biographique, prénom réel, âge ou métier n’est demandé.

L’IA construit un portrait de lecture hypothétique avec axes, sources et incertitudes. Elle choisit un personnage fictif dont l’identité reste stable. Chez les abonnés, chaque nouveau like, retrait, explication ou retour sur une histoire personnelle invalide le portrait antérieur. La prochaine génération attend la réanalyse du contexte courant. Une réponse « Non » est aussi prise en compte. Les récits fictifs ne deviennent jamais une biographie réelle.

## Automatisation historique réutilisée

Le workflow original `.github/workflows/generate-articles.yml` est restauré. Il traite les révisions de portraits et les récits privés toutes les quinze minutes (avec déclenchement manuel et sur modification du pipeline). La base limite chaque utilisateur à une histoire par journée Europe/Paris.

`scripts/generate_personal_articles.py` importe le générateur original `scripts/generate_articles.py`. Il réutilise ses fonctions de cadrage, prompts, parsing, validation, relecture, cohérence, structure et extraction du résumé. Aucun texte privé n’est écrit dans les articles publics, les commits, les logs ou les artefacts GitHub.

La clé **OPENROUTER_API_KEY reste un secret Cloudflare**, dans le Worker `decode-journal`. GitHub Actions s’authentifie avec OIDC et utilise une passerelle OpenRouter limitée aux tâches payées et autorisées. Aucune clé IA n’est à ajouter dans GitHub ni Netlify. L’ancien Cron Cloudflare et le workflow de remplacement `journal-daily.yml` sont retirés pour éviter deux orchestrations concurrentes.

## Hébergement et vérification

Netlify conserve Hugo, les articles et le domaine. Les routes `/journal/*` et `/api/*` sont transmises au backend Cloudflare sur le même domaine. D1 conserve comptes, likes, portraits, versions et récits privés.

```sh
hugo --minify
node --test journal/tests/backend.test.mjs
python -m unittest discover -s journal/tests -p 'test_personal_pipeline.py'
```

La compilation Hugo, 19 tests backend et 7 tests Python passent. Le test de volume couvre les 231 articles avec de longues explications, sans perte des sources. Voir `journal/README.md` et `docs/evolution-par-articles-aimes.md` pour la mise en service et les validations réelles.
