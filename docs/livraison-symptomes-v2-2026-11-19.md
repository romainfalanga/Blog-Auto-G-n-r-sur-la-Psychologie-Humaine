# Livraison Mes symptômes v2 — 19 novembre 2026

Cet état remplace les consignes de chantier interrompu de la reprise du 19 novembre. La v1 des questionnaires est retirée. L’éditorial des personnages reste en pause.

## Résultat publié

Mes symptômes reprend l’interface de Mon enfance, avec réponses Non / Oui / Je ne sais pas, onglets de saisie et résultats, précisions modifiables, sauvegarde et brouillon distincts. La période est « ces 12 derniers mois ». Aucun bouton d’accord supplémentaire. Mon enfance conserve son classement et ses données.

La banque compte **516 questions, 157 sujets et 1 120 liens**, en 28 thèmes de 5 à 30 questions. Les 34 fusions sont explicites. Les variantes d’un signe ne sont pas comptées plusieurs fois. Le classement `essential_share_desc` impose les contextes nécessaires : produit déclaré, période périnatale, contexte relationnel, absence d’intention suicidaire explicitement indiquée pour le rapprochement correspondant, coût attendu de la procrastination. Une simple ressemblance n’est pas une probabilité diagnostique.

Les questions sont rendues à l’ouverture d’un thème. Banque publique et explications sont deux JSON à empreinte, chargés séparément ; les réponses personnelles restent privées. Le module Worker est inférieur à 500 Ko. Le générateur contrôle sources, identifiants, poids, familles, exceptions argumentées et conditions de contexte. Certaines catégories ont légitimement plus de douze caractéristiques : les critères documentés ne sont pas abaissés artificiellement pour respecter un plafond technique.

Ambre reçoit les deux cartes sous forme d’aperçus fixes et d’index de détails, avec réponses négatives/incertaines, notes et signes manquants. Les hypothèses doivent nommer leurs appuis et leurs limites. Les parcours historiques sans carte symptômes conservent leurs allocations et leurs sources. Avec les deux cartes, rédaction au plus +40 % et bilan au plus +20 % des références ; réécriture réservée. La sécurité conserve un drapeau serveur et un horodatage de la question effectivement publiée ; une alerte actuelle dans le message reste prioritaire.

Partage avec le psychologue uniquement sur choix du périmètre `symptoms`, export et effacement pris en charge. Contrôle de révision monotone pour empêcher une publication sur données remplacées, y compris effacement/recréation dans la même seconde. Les deux interfaces de carte vérifient aussi l’identité attendue : une ancienne fenêtre A avec la nouvelle session B reçoit `409 session_changed` et conserve son brouillon sans écrire sur B.

## Validation

- **900 tests serveur, 61 tests site**, contrôles complets de `verify.yml`, tests Python personnels et générateur ; CI PR 105 réussie ; CI finale PR 106 : `44548433687` réussie.
- **33 parcours navigateur Mon enfance** ; symptômes à 390/1280/1440 px, erreurs de détails et reprise, changement de session, notes, rechargement et classement. Parité sur 200 profils enfance et 60 profils symptômes.
- Hugo : 7 933 pages ; banque complète réellement rendue à 390/1440 px, sans débordement ni erreur JavaScript.
- Relectures indépendantes : [backend](relecture-backend-symptomes-v2-2026-11-19.md), [interface](relecture-interface-symptomes-v2-2026-11-19.md), [Ambre](relecture-ambre-symptomes-v2-2026-11-19.md), [CDDR](relecture-clinique-symptomes-v2-2026-11-19.md), [autres sources](relecture-noncim-symptomes-v2-2026-11-19.md), [vitesse](relecture-performance-2026-11-19.md). Les niveaux de lecture des sources sont précisés ; ce n’est pas une certification par un psychologue humain ni une validation psychométrique du score.
- Recette de production : sauvegarde sans accord supplémentaire, conflit de version, identité attendue, trois largeurs, édition/rechargement des notes, détails, export et conservation d’une ancienne réponse enfance `2`. Après 0030, seconde vérification authentifiée des sauvegardes et de l’export.
- Après PR 106 : vérification anonyme en production à 390/1440 px du texte des questions de contexte et des questions ordinaires ; aucun débordement, aucune erreur JavaScript, aucune requête de modification, captures inspectées.
- Trois comptes synthétiques supprimés ; aucun reliquat dans **46 tables par compte**, aucune violation de clé étrangère, fichiers temporaires d’identifiants supprimés. Le premier passage a été interrompu par une attente erronée du script de recette après modification d’une note ; son compte a aussi été supprimé. Aucun échange IA réel n’a été lancé par ces recettes. Les tests de génération utilisent des fournisseurs simulés.

## Vitesse mesurée

Les fichiers d’application à empreinte et le chargement différé du catalogue étaient déjà présents dans la PR 104 : ils ont été audités, pas réimplémentés. La comparaison locale PR 103/104 réduit le HTML gzip de 119 464 à 1 734 octets ; la première visite transfère encore environ 120 Ko avec les fichiers statiques. Le bénéfice porte surtout sur leur réutilisation en cache.

Le même script navigateur a mesuré le site avant/après, en session anonyme isolée et à 390 px. Mon compte : LCP observé 1 668→484 ms à froid, 208→96 ms à chaud ; Personnages : 684→740 ms à froid, 136→260 ms à chaud. Ces échantillons réseau ponctuels ne prouvent pas un gain universel. Mes symptômes passe d’un écran de connexion v1 à un questionnaire réellement utilisable : les contenus ne sont pas comparables ; après publication, LCP observé 1 832 ms à froid et 300 ms à chaud. Aucun résultat de latence authentifiée ou de modèle distant n’est déduit de ces mesures.

## Publication et restauration

- Source PR 105 : `db30aefd77328eef1848c3f68cf9e2bc3d6fcc3c`, arbre `972b54168b4fce7ae8895b9246d0ac0e98064094`, fusion `5905271789e769b625a725d5b192e1560edcf73b`. CI `44536868021` réussie.
- Avant 0029 : bookmark `000000b3-00000a88-00005128-b35c971275a3e7a34b138b23d2f8ff6f`. Migration appliquée et inscrite une seule fois. Pause temporaire de génération retirée lors de l’envoi complet du Worker.
- Worker `decode-journal` : déploiement `6077fcd928af40348688400725e58383`, 50 modules relus et SHA-256 identiques, sept liaisons et secret conservés, placement smart, cron minute. `/api/health` : 200, ok/ai vrais, billing faux. Accès gratuit maintenu.
- Netlify PR 105 : `6aff017c34c5cdbbef74ed8b`, ready, publié le 19 novembre à 15:03:11 Paris, fusion exacte et Edge présent.
- Après recette réussie : 0030 retire les quatre tables v1. Aucun résultat, brouillon ni alerte v1 présent avant suppression ; seul l’ancien accord existait. Bookmark préalable `000000b3-00000b36-00005128-5312ec598701105cdcb37f93e2a007fa`. **30 migrations inscrites**, tables v1 absentes, intégrité vérifiée. Ne rien rejouer.
- PR 106 : clarification des questions de contexte sans lien direct et correction de portabilité JSON SQLite des traces compactées. Le premier passage CI `44543944271` a révélé deux agrégats sérialisés comme chaînes ; `json()` rétablit explicitement leur type aux frontières des sous-requêtes. Le test force la perte de sous-type avec `substr(...,1)` ; un contrôle négatif indépendant échoue avec l’ancien SQL et réussit avec le correctif. La première tentative de test utilisait `CAST`, qui préservait encore le sous-type sur SQLite local ; elle a été corrigée avant sauvegarde. Requête synthétique en lecture seule vérifiée sur D1 : les deux types sont `array`, les octets d’entrée et le verdict sont conservés. Aucun seuil relevé ; aucune trace malformée présente en production lors du contrôle. Source `3a44e2f46ab5283b56795e226b5a9669c3f884ae`, CI `44548433687` réussie. Fusion `99fa1c562d243d96ba7c3005708afdcf4c9ded8a` ; Netlify `6aff0d89f9b7ead919046530`, publié le 19 novembre 2026 à 15:54:37 (Paris). Worker final `f33f48d7dc494d10a214afd5f36fb6ea`, 50 modules vérifiés.

Preuves locales : `work/reprise-nov19/`, dont reçus de publication, manifestes, tests, captures, mesures et rapports de recette. Les sources et rapports durables sont sauvegardés dans Git.

## Décisions restant ouvertes

Les catégories laissées de côté dans la banque (paraphilies, transe, démences/delirium et troubles principalement de l’enfance) n’ont pas été ajoutées. Restent également ouverts l’avertissement pour une date de naissance de mineur, le lien public vers Personnages retiré du pied de page et une relecture par un psychologue humain avant diffusion large. Aucun de ces choix n’a été modifié implicitement.
