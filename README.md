# Décode ton esprit

Le site existant https://decodetonesprit.fr conserve ses 231 articles, ses 20 personnages, ses catégories et ses adresses. Il évolue avec un journal personnel accessible sur `/journal/`, intégré à la même navigation et au même domaine.

## Parcours

Le lecteur choisit jusqu’à trois articles qui lui parlent, explique pourquoi et précise les différences avec le personnage. Il renseigne son propre profil. Les abonnés disposent d’une histoire fictive quotidienne adaptée à leurs déclarations et à leurs retours. Le personnage public est une inspiration, jamais une biographie attribuée au lecteur.

## Hébergement

- Netlify : site Hugo existant, articles, personnages, ressources et domaine.
- Cloudflare Worker `decode-journal` : comptes, API privée, génération OpenRouter et facturation.
- D1 européenne `decode-journal` : données privées et catalogue public de référence séparé.
- Netlify transmet `/journal/*` et `/api/*` au Worker par réécriture 200 : l’utilisateur reste sur `decodetonesprit.fr`.

## Clé IA : un seul emplacement

Cloudflare → Workers & Pages → `decode-journal` → Settings → Variables and Secrets → Add → **Secret**.

Nom exact : `OPENROUTER_API_KEY`.

Enregistrer puis déployer. Ne pas la placer dans Netlify, le navigateur, GitHub ou un fichier du dépôt. Le journal utilise uniquement OpenRouter, sans fournisseur de secours. La variable non secrète `OPENROUTER_MODEL` vaut `openai/gpt-5-mini` et peut être modifiée.

## Automatisation

Un Cron Cloudflare toutes les cinq minutes sélectionne au maximum trois abonnés éligibles. Chaque compte reçoit au maximum un récit par journée Europe/Paris. Il faut un personnage renseigné, le consentement IA, au moins une inspiration enregistrée et un abonnement actif. Sans clé IA, aucun travail n’est réservé.

La publication automatique historique de trois nouveaux articles publics est arrêtée : les articles existants constituent désormais le fonds d’exemples. Le générateur historique Python reste conservé et a été converti à OpenRouter pour un éventuel usage manuel par un développeur; il n’est plus exécuté par le site. Le workflow GitHub « Relance des journaux Cloudflare » est uniquement un déclencheur manuel authentifié, sans clé IA ni donnée personnelle dans ses logs.

## Développement

```sh
hugo --minify
cd journal
npm test
npm run check
```

Hugo 0.140.0 ; Node 24 pour les tests SQLite ; Python et PyYAML pour reconstruire le catalogue : `python scripts/build_examples.py --seed /tmp/decode-examples.json`.

Voir `journal/README.md` pour les migrations, la publication et la configuration Stripe.
