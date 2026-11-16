# Livraison Ambre et Cartographie — 17 novembre 2026

Chantier repris sur la PR 83 (`1b86169`), puis publié par la [PR 84](https://github.com/romainfalanga/Blog-Auto-G-n-r-sur-la-Psychologie-Humaine/pull/84). Le chantier éditorial des personnages reste inchangé. La publication et la recette décrites ici ont été constatées le 16 novembre 2026 (UTC) ; le nom du dossier de travail conserve la date de la demande de reprise.

## Résultat livré

Les défauts de l’audit Ambre §14 sont corrigés, à l’exception du code mort n°15 explicitement exclu : budget de réécriture, date, mémoire récente, fin des réponses dans l’historique, faux positifs de sécurité, bilans et observations utiles. Le moteur prévoit la réécriture et sa relecture dans sa réserve nominale de 30 000 jetons ; ses reprises techniques restent bornées et comptabilisées.

Le catalogue comporte 36 stratégies sourcées. Le plan V3 a été retenu après deux cycles de critique et de révision avant implémentation. Objectifs, étapes, essais, résultats, consignes rapportées et préparation personnelle sont lisibles et modifiables dans Mon suivi. Les propositions demandent un choix explicite ; les invitations de retour demandent un accord distinct. Paramètres conserve ses trois onglets et présente le contexte, les méthodes, les modèles et les appels effectivement utilisés. Le partage professionnel du plan reste facultatif et exclut les consignes privées, propositions et préparations personnelles.

Les neuf lots cartographiques sont intégrés : 59 questions, 109 cartes actives, 493 liens, 260 sources, 11 ressources et 3 identifiants historiques conservés sans réinterprétation. Les 161 correspondances ont fait l’objet d’une relecture sémantique indépendante. Le registre compte 1 567 décisions documentées, dont les références CIM et DSM examinées. Les simplifications de la PR 83 sont conservées : six volets initialement fermés, résultats en liste, portrait Mon portrait/Ambre. L’encadré 3114 et le bouton de suppression sont retirés de la page Cartographie selon la décision du propriétaire ; le dispositif de sécurité d’Ambre et l’API de suppression restent disponibles.

## Relectures et vérifications

Les relectures indépendantes sont favorables après correction des défauts reproduits : confusion entre repère contextuel et jeune enfant, budget des preuves, catégories santé, provenance avant transmission, export complet, révocations inutiles et courses entre suppression, mémoire et finalisation des traces.

- Journal : **594/594 tests réussis** sur la version gelée.
- Site : **32/32** ; Python : **43 + 13 + 21**.
- Toutes les commandes de `.github/workflows/verify.yml`, générateurs, audit, navigation, catalogues et contrôles éditoriaux réussis.
- [CI de la PR 84](https://github.com/romainfalanga/Blog-Auto-G-n-r-sur-la-Psychologie-Humaine/actions/runs/43916820144) réussie.
- Cartographie : 42 tests, 1 770 comparaisons public/Worker, 16 scénarios navigateur ; module généré de 292 325 octets.
- `tests/change-plan-browser.cjs` : vrai Worker avec SQLite local, aux largeurs 1440, 390 et 320 px ; édition, dates inconnues, conflit préservant la saisie, préparation privée sans professionnel, catalogue, arrêt et effacement de conversation conservant le plan.
- Hugo complet : 7 931 pages. Aucun débordement horizontal ni erreur JavaScript constaté dans les parcours vérifiés.

Les fixtures du moteur simulent le fournisseur pour provoquer aussi les reprises et les courses. Elles ne mesurent pas l’efficacité clinique. Sur le profil rempli avec deux méthodes, environ 11 742 jetons d’entrée sont estimés et 29 098 réservés avec réécriture. Un contexte maximal peut omettre une méthode entière et interdire l’exercice ; ses exclusions ne sont jamais coupées pour la faire tenir.

## Publication constatée

Une seule migration a été appliquée : `0021_change_plan.sql`. La structure distante confirmait auparavant 0020 et l’absence de 0021 ; cette base n’a pas de table `d1_migrations`. Après application, colonne, trois tables, déclencheurs et clés étrangères ont été vérifiés.

| Élément | Preuve |
|---|---|
| Repère D1 avant migration | `000000a1-00000022-00005125-7c60dfea939e3a3f622208ce26c985a9` |
| Commit source testé | `b933dc91316420fdd661bfd633db1cfc29c1ae74` |
| Worker complet | 38 modules, tous retéléchargés avec empreintes SHA256 identiques aux sources |
| Déploiement Worker | `67dc37c9f87441379a4405e6fac80e9f` |
| Etag Worker | `05f50a88252421afdae69e45dcc8f6e3bb5f0fae4ad442fb23d61a4ddc3d7208` |
| Fusion dans main | `ab663b603dfd7e438c7cb32eb708fe17642d0625` |
| Déploiement Netlify | `6afb3cd4325cabd54a5d213d`, état `ready`, commit de fusion correspondant |

Les sept liaisons, le secret OpenRouter et le cron `* * * * *` sont conservés. La pause temporaire de génération a été retirée. L’ordre migration → Worker complet → fusion/site a été respecté.

## Recette réelle et nettoyage

Trois comptes synthétiques distincts ont exercé les rôles membre, psychologue et membre non autorisé. En production : création et lecture du plan, résultat sans date inventée, préparation privée, isolation des comptes, interface aux trois largeurs, six volets fermés et suppressions visuelles demandées sont validés.

L’analyse initiale a abouti. Un échange réel demandant de l’écoute sans exercice a été publié après trois appels terminés : analyse `openai/gpt-5-nano`, rédaction `anthropic/claude-sonnet-5.5`, relecture `openai/gpt-5-nano`. Les entrées et modèles reçus sont visibles dans la trace ; la relecture est passée et les critères de changement sans objet restent explicitement sans objet. Le fournisseur rapporte 9 067 jetons d’entrée et 672 de sortie sur cet échange synthétique ; ce profil de recette ne remplace pas les fixtures remplies/maximales.

La commande d’arrêt suspend les invitations sans ajouter de tour modèle. Le point professionnel partagé a été généré à partir du seul périmètre plan, sans consigne ou préparation privée. Après révocation, les lectures du dossier et du point par le psychologue sont refusées. L’export du propriétaire contient le plan versionné. L’effacement du chat conserve le plan par défaut ; le choix explicite d’effacer aussi le plan le supprime.

Les trois comptes ont ensuite été supprimés par l’API. Un contrôle D1 sur **43 tables liées aux propriétaires et acteurs** confirme zéro ligne restante pour leurs identifiants, et `foreign_key_check` ne remonte aucune anomalie. Les compteurs temporaires identifiables par leurs identifiants/adresses de test ont également été retirés. Aucune donnée d’un compte réel n’a été utilisée pour cette recette.

## Limites documentaires

Le registre décrit ce qui a été examiné, exclu ou reste à examiner. Il ne signifie pas que tous les critères du manuel DSM ont été lus ni que chaque trouble possède un lien causal établi avec l’enfance. Les niveaux de lecture (résumé ou texte complet) restent conservés. Les associations ne constituent pas un diagnostic individuel, et la relecture logicielle ne certifie pas l’efficacité clinique d’Ambre.

Rapports détaillés : `relecture-finale-cartographie-2026-11-17.md`, `relecture-finale-moteur-changement-2026-11-17.md`, `relecture-suivi-plan-2026-11-17.md`, `integration-moteur-changement-2026-11-17.md` et les neuf rapports de recherche.
