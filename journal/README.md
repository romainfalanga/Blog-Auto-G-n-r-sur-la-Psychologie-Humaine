# Journal fondé sur les articles aimés

URL : https://decodetonesprit.fr/journal/.

## Données et migrations

Appliquer 0001.sql, 0002_inspirations.sql et 0003_likes_portraits.sql. La troisième migration importe les choix antérieurs sans supprimer leurs explications ni les tables d’origine. Les likes n’ont plus de plafond de trois. Un like ou une modification réelle d’explication incrémente la version des sources; les écritures identiques ne déclenchent pas d’analyse inutile.

Le seuil de souscription est dix articles aimés et le consentement IA. Le motif reste facultatif. Un abonné qui retire ensuite un article continue avec sa collection courante dès qu’elle n’est pas vide. À zéro article, les nouvelles analyses et histoires sont suspendues.

Les portraits sont hypothétiques, sourcés par les articles aimés et les retours sur les histoires privées. Les versions protègent chaque appel fournisseur et chaque publication : un portrait ou un récit commencé avant une correction ne peut pas être publié comme actuel. Le personnage fictif conserve son prénom lors des réanalyses. Les nouveaux épisodes ne modifient pas automatiquement le portrait réel ; seul un retour de l’utilisateur apporte une nouvelle source.

## Génération historique GitHub Actions

Le workflow original generate-articles.yml appelle scripts/run_journals.py, qui traite d’abord les portraits obsolètes, puis les histoires des portraits courants. Le pipeline importe réellement generate_articles.py et ses fonctions build_system_prompt, build_article_prompt, parse_article_response, validate_article, gemini_comprehensive_review, validate_coherence, verify_article_structure, _strip_bold et extract_narrative_summary.

Le transport de ces fonctions est adapté par une passerelle OIDC dans le Worker. OPENROUTER_API_KEY reste un secret Cloudflare; aucun secret IA permanent dans GitHub. Seul le dépôt exact, main, le workflow exact et les événements attendus sont acceptés, avec signature RSA vérifiée. Chaque appel doit porter le bail du travail, rester dans les versions, le consentement et l’abonnement autorisés. Budget de vingt appels fournisseur par travail, délai de 180 secondes par appel. Le runner cesse de réserver de nouveaux travaux après vingt minutes. Chaque travail cesse de lancer des appels après vingt-cinq minutes, afin de rester dans son bail de trente minutes et la limite globale de cinquante minutes. Le transport interne des Actions utilise directement le Worker, avec un User-Agent explicite; les visiteurs restent sur le domaine existant.

Une histoire maximum par utilisateur et journée Europe/Paris. Les réanalyses interviennent au passage suivant du workflow, avec regroupement possible des changements rapprochés. Après épuisement des essais d’un portrait, reprise automatique au bout de 24 heures; retry_after est visible dans l’interface. GitHub peut retarder une exécution planifiée; aucune promesse de livraison à la minute.

## Publication

1. Appliquer la migration0003 à la D1 existante sans effacer les données.
2. Déployer worker.mjs, ui.mjs et examples.mjs en conservant les secrets existants.
3. Retirer le Cron Cloudflare; publier les fichiers du workflow original et supprimer journal-daily.yml.
4. Publier Hugo et ses nouveaux fichiers data/partials/static/js. Conserver la correction de réécriture /journal dans netlify.toml.
5. Contrôler un compte de recette, les likes, les portraits, le workflow et un récit réel.

Les textes canoniques publics sont déjà dans public_examples. scripts/build_examples.py reconstruit les métadonnées, mots-clés et l’export public si de nouveaux articles sont ajoutés.

## Paiement et limites

Stripe attend ses secrets STRIPE_SECRET_KEY et STRIPE_WEBHOOK_SECRET et le prix STRIPE_PRICE_ID (1000 centimes EUR mensuels). Sans cette configuration, les comptes et collections fonctionnent et le paiement reste honnêtement indisponible. Le webhook est /api/billing/webhook. La récupération du mot de passe par email reste à ajouter avant ouverture commerciale complète.

La base est européenne. Les données soumises au fournisseur IA ou temporairement traitées dans GitHub Actions ne sont pas garanties hébergées en Europe. Les prompts, textes, identifiants de connexion et clé IA ne sont jamais inscrits dans des fichiers publics ni des logs. Les frais des fournisseurs restent distincts du tarif client.

## Tests

19 tests backend et 7 tests Python passent : migration, collection, seuil, absence de questionnaire, isolation, versions concurrentes, preuve privée, consentement, souscription, quota et signature OIDC réelle. Le volume maximal des 231 articles avec raisons et différences longues est couvert. La compilation Hugo conserve les adresses et le texte source des articles et personnages.

## Recette réelle du 31 octobre 2026

Le workflow historique [41190861894](https://github.com/romainfalanga/Blog-Auto-G-n-r-sur-la-Psychologie-Humaine/actions/runs/41190861894) a produit un portrait puis un récit privé de 2520 mots sur un compte fictif : cadrage, rédaction, relecture, cohérence et publication ont abouti (cinq appels IA pour le récit). Un retour négatif a ensuite incrémenté la version des sources de 13 à 14; la même automatisation a réanalysé et publié le portrait à jour (version 14). Aucun achat réel : un droit temporaire de recette a été réservé à ce compte, supprimé après vérification.
