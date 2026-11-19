# Livraison — symptômes passés, archives des bilans et suivi

Publié le 19 novembre 2026 à 17 h 07 (Paris), PR [107](https://github.com/romainfalanga/Blog-Auto-G-n-r-sur-la-Psychologie-Humaine/pull/107). Le code et les migrations sont en production. **La génération réelle d’un bilan reste bloquée : OpenRouter a renvoyé HTTP 402 pendant la recette.** Vérifier les crédits et les limites de facturation de la clé fournisseur ; aucun paiement ni changement de clé n’a été effectué. Le statut `/api/health` (`ai: true`) indique une configuration présente, pas un appel fournisseur réussi.

## Fonctionnalités livrées

- Mes symptômes : tous les thèmes fermés à l’arrivée ; Non, Oui, Je ne sais pas, Dans le passé en grille de quatre boutons. Une précision est obligatoire pour le passé ; annuler conserve la réponse précédente. Le passé désigne une période antérieure aux douze derniers mois et ne compte pas comme symptôme actuel. Mon enfance conserve ses trois réponses.
- Nouveaux bilans : instantané daté des données personnelles utiles, avec libellés et version des règles, consultation par rubrique et export. Les changements ultérieurs ne réécrivent pas cet instantané. Les suppressions, retraits de consentement et invalidations prévus continuent de s’appliquer. Les anciens bilans ne reçoivent pas de faux instantané reconstitué.
- Analyse par blocs de toutes les données capturées, couverture traçable et reprise du travail interrompu. Une limite dépassée produit une erreur explicite, jamais une affirmation de traitement complet. L’analyse supplémentaire consomme du budget fournisseur ; les contrôles de budget et le décompte d’un bilan sont conservés.
- Mes diagnostics : hypothèses datées, indices, associations, informations manquantes et conclusions professionnelles attribuées, présentées séparément. L’algorithme ne transforme pas ces rapprochements en diagnostics médicaux établis.
- Mon compte : prochain rendez-vous avec le psychologue, modifiable et supprimable. Ambre reçoit ce contexte ainsi que les évaluations datées et conclusions disponibles.
- Dossier professionnel : informations et notes enrichies, archives filtrées par autorisations, mémoire et évaluations avec accords distincts. Une conclusion professionnelle exige les accès bilans et diagnostics. Révocation contrôlée côté serveur ; l’identité professionnelle demeure déclarée, non vérifiée.

## Méthode et limites

Lire [la recherche clinique](recherche-evaluation-clinique-2026-11-19.md), [sa relecture](relecture-evaluation-clinique-2026-11-19.md), [le contrat des instantanés](contrat-instantanes-bilans-2026-11-19.md) et [la relecture du partage](relecture-partage-archives-2026-11-19.md).

La recherche s’appuie sur les CDDR de l’OMS, la HAS et NICE. Certains accès directs NICE ont renvoyé 403 ; cette limite est documentée. Les 157 sujets ont une couverture explicite : 61 sujets cliniques partiels, 12 agrégats, 68 constructions, 6 qualificatifs et 10 sujets hors CDDR en attente. Ce n’est pas une procédure diagnostique exhaustive pour tous les troubles. Durée, simultanéité, retentissement, début, exclusions et diagnostic différentiel restent signalés comme inconnus lorsqu’ils ne sont pas établis. Ni le vécu d’enfance ni un texte produit par l’IA ne deviennent une preuve de critère clinique ; aucune probabilité diagnostique n’est inventée.

Limites techniques explicites : 512 Kio pour les sources et le contenu total de l’instantané, 10 000 lignes par section, synthèse finale de 40 000 caractères, export d’archives de 16 Mio. L’archive conserve les données personnelles et la mémoire utiles, pas une copie graphique du site, des secrets ou du catalogue public. Une synthèse de chaque bloc ne garantit pas une fidélité clinique parfaite.

## Vérification

- Local : 948 tests serveur et 62 tests site réussis ; construction Hugo de 7 933 pages réussie. Tests ciblés des symptômes passés, des instantanés, du partage et des rendez-vous réussis.
- CI GitHub `44561695110`, job `135692412244` : succès sur `932b247fafa74f0edff6a7e7716f3e36384b537a`, 948 tests serveur ; 61 tests site réussis, 1 ignoré faute de Hugo dans ce contexte CI.
- Navigateur local : parcours archives, évaluations et rendez-vous à 390 et 1 440 px ; tests des symptômes à 390, 1 280 et 1 440 px.
- Production : inscription, profil, réponses, proches, événements, rendez-vous et prérequis vérifiés. **L’essai réel de génération a échoué avec `provider_http_402` : aucune génération complète réelle n’est validée par cette livraison.**
- Une archive explicitement synthétique, réservée à un compte de recette, a ensuite permis de vérifier les API d’archive, l’immutabilité après modification, l’export, les autorisations professionnelles, la conclusion attribuée et la révocation. Ce test ne constitue pas une validation du fournisseur IA. En navigateur de production, thèmes fermés, page diagnostics, ouverture de la fenêtre d’archive et absence de débordement/erreur JavaScript vérifiés à 390 et 1 440 px. Les captures de la fenêtre d’archive montrent son état d’ouverture ; les données complètes ont été vérifiées par API et par le parcours local.
- Les quatre comptes temporaires ont été supprimés : aucune ligne de compte ni donnée liée dans les 47 tables à `user_id` contrôlées ; aucune violation de clé étrangère. Fichiers de récupération des accès temporaires supprimés.

## Publication et exploitation

- Source : `932b247fafa74f0edff6a7e7716f3e36384b537a` ; fusion : `1ed6271f7dca9e1f8bd89c687ac5ec64e0635b95`.
- Point de restauration D1 avant intervention : `000000b3-00000cfd-00005128-319c5cfe0e3795869204ab58bf580310`.
- Migrations 0031 et 0032 appliquées et inscrites une seule fois ; historique 0001 à 0032 et clés étrangères vérifiés. **Ne pas rejouer ces migrations.**
- Worker `decode-journal`, déploiement `a700fb7c36144cc19d5f5aecd22668bc`, 54 modules vérifiés par empreinte, placement smart et secrets conservés. Suspension temporaire des générations levée après déploiement.
- Netlify `6aff1eb9c0bc2aadadad1c80`, état ready, commit de fusion exact, publié le 19 novembre à 16:07:59 UTC ; fonction Edge présente.
- Accès gratuit conservé, limite quotidienne de dépense laissée désactivée, production éditoriale publique non relancée.

Les reçus techniques et journaux de recette locaux sont dans `work/bilans-historique-nov19/` et `work/reprise-nov19/` (répertoires ignorés par Git). Le blocage fournisseur doit être résolu puis un bilan réel complet doit être testé avant d’annoncer la validation intégrale de la génération.
