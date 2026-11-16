# Plan d'amélioration v4 — Ambre et Cartographie (18 novembre 2026)

Sources : `docs/audit-robustesse-ambre-2026-11-18.md` (23 failles F1–F23, démontrées), `docs/recherche-ambre-psychologue-v3-2026-11-18.md` (36 améliorations cliniques, §7), `docs/revue-exhaustivite-cartographie-2026-11-18.md` (corrections §8, hors périmètre §9). Déjà publié le 18/11 : conversation sans menu d'options, Cartographie sans défilement, pied de page minimal (PR 85).

Principe : d'abord ce qui protège (données, crise, affirmations inventées, tenue en charge), ensuite ce qui améliore la relation et la compréhension, sans augmenter le coût nominal d'un tour (≈ 12 000 jetons d'entrée) de plus de 10 %. Chaque point a un test qui échoue avant et passe après.

## Lot R — robustesse et tenue en charge (Worker)

| Réf. | Correction | Critère d'acceptation |
|---|---|---|
| F1 | Rétention des traces : détail complet 30 jours puis résumé (étapes, tailles, jetons, verdicts) ; purge par lots bornés dans le cron ; plafond par personne (200 traces détaillées). Paramètres signale les traces résumées. | Base simulée de 3 000 tours : volume des traces ≤ 2 × la conversation ; export et Paramètres cohérents. |
| F2 | Tâches terminées supprimées après 7 jours ; index partiel sur les tâches dues ; plus aucun balayage complet. | `EXPLAIN` du cron sans `SCAN chat_engine_jobs`. |
| F3 | Sondage incrémental (`after` = dernier identifiant connu ; état seul tant que rien ne change). | Un sondage sans nouveauté lit un nombre borné de lignes, indépendant de la longueur. |
| F4 | Export des traces paginé avec couverture déclarée, comme le suivi. | Export de 3 000 traces complet ou couverture explicite. |
| F8/F9 | Cron : tâches traitées en parallèle bornée (4), budget de temps par passage, équité (plus ancienne échéance d'abord, mémoire comprise). | Simulation : 20 tours terminés en ≤ 2 passages ; mémoire servie au premier passage disponible. |
| F10 | 429/402 du fournisseur : respect de `Retry-After`, refroidissement commun alimenté par le moteur actif, tours mis en attente (pas échoués), budget non débité pour un appel refusé. | Simulation 429 : aucun tour échoué, aucun jeton débité pour les refus. |
| F11 | Limiteur de connexion par IP réelle (`CF-Connecting-IP` transmise par Netlify via en-tête signé ou `x-nf-client-connection-ip`) + par adresse ; jamais par IP du proxy. | 31 échecs d'une IP n'empêchent pas une autre IP de se connecter. |
| F12 | Plafond global quotidien de dépense configurable (`DAILY_PROVIDER_TOKEN_CAP`), file d'attente au-delà ; plafond de nouveaux comptes par IP et par jour. | Simulation : au-delà du plafond, tours en attente, message clair. |
| F13 | Compteurs agrégés sans contenu (tours, échecs par code, jetons, attentes) consultables par l'administrateur. | Aucun texte privé dans les compteurs. |
| F22 | Table `schema_migrations` créée par `0022` et renseignée pour 0001–0022 ; documentation de la procédure. | Rejeu documenté sans risque. |
| F23 | Contexte maximal : lecture des sources une seule fois par tour (mémoïsation). | CPU du profil maximal divisé par ≥ 3. |

## Lot S — sûreté clinique et qualité d'Ambre

| Réf. | Correction | Critère |
|---|---|---|
| F17 | Un message de danger reçoit **toujours** les ressources (3114, 15/112) immédiatement, même pendant un bilan, avant l'analyse initiale ou pendant un autre tour. | Les trois cas de l'audit reçoivent 3114. |
| F18 | Échec ou paraphrase du contrôle de risque → réponse prudente avec ressources, jamais un échec nu. | Test de panne du contrôle. |
| F19 | Filet lexical élargi (automutilation, surdosage, moyens, formulations indirectes de l'audit) ; analyse en panne → voie prudente avec ressources si un signal même faible existe. | 9/9 messages ambigus de l'audit traités ; faux positifs connus toujours non déclenchés. |
| F20 | Notes de cartographie, objectifs et propositions : citation exacte non niée et non vide (≥ 12 caractères), cohérente avec la réponse de la personne ; négation détectée → rejet ; note sur un vécu répondu « Non » interdite. | Les trois exemples de l'audit sont rejetés. |
| F5/F6/F7 | Mémoire : rubrique `safety` et faits majeurs datés jamais évincés par l'âge ; citation inexacte → élément écarté, pas le lot ; la citation de sécurité retenue est la phrase du signal. | 1 000 échanges : la tentative de 2019 reste ; une apostrophe n'arrête plus la mémoire. |
| R1 | Réintroduire dans le rédacteur les consignes de fond perdues (proche : vu / ressenti / imaginé ; emprise sans confrontation ; croyances de persécution ; explications alternatives ; exceptions), ≤ 700 caractères. | Présentes dans le prompt actif ; testées. |
| R2–R4 | Violences conjugales : signal `ipv_signal`, fiche (3919, 17/112, 114 SMS, traces numériques, ni confrontation ni injonction à partir) ; facteurs de gravité déterministes (étranglement, arme, menaces de mort) ; critère « aucune incitation à confronter » sur toute réponse. | Jeux d'essai IPV. |
| R5–R7 | Psychose/manie (`reality_concern` : pas de validation, pas d'activation, orientation médicale ; croyances rapportées hors `facts`, compteur longitudinal) ; troubles alimentaires (`eating_concern`, aucune donnée chiffrée de poids ou calories, critère déterministe). | Jeux d'essai. |
| R8 | Mineurs : `minor_signal` → réponse adaptée, 119, adulte de confiance. | Jeu d'essai. |
| R10, R13 | Pensées intrusives non désirées distinguées d'une intention ; grille unique signal → ressource vérifiée → rappel, contrôlée par la relecture. Numéros : seuls ceux vérifiés sur source officielle. | Tests. |
| R14, R17 | Réactance : après deux propositions refusées, retour à l'écoute ; après une réparation, aucune méthode au tour suivant. | Tests. |
| R16 | Rubrique mémoire `style` (préférences de forme déclarées) transmise à chaque tour (≤ 100 jetons). | Test. |
| R20 | Questions sur la relation à l'IA : réponse honnête ; critère contre toute déclaration d'attachement. | Test. |
| R21 | Complaisance : critère 11 décomposé (raison/tort, cadrage accepté, décision nuisible, validation de l'acte), avec contrôle déterministe des formules d'approbation inconditionnelle. | Tests ELEPHANT adaptés. |
| R24, R27, R33 | Pas de cause biographique sur un épisode isolé ; identité et orientation jamais un problème à corriger ; consigne culturelle. | Présents et testés. |
| R11 | Batterie multi-tours en français (crise directe/indirecte, IPV, psychose explicite/implicite, TCA, mineur, complaisance, longue conversation) rejouée sur le vrai moteur avec fournisseur simulé, script réutilisable à chaque changement de modèle. | `npm --prefix journal test` l'inclut. |

Différé, documenté : formulation de cas partagée (R23), retour rapide sur chaque réponse (R15), fiches situations (R26), nouvelles stratégies (R32, circuit de critique complet), bilan de clôture (R22), avis juridique sur l'intitulé (R9, décision du propriétaire).

## Lot C — Cartographie

1. Tri : le nombre affiché et le tri ne comptent que les vécus reliés par une **association empirique** (forte, modérée, limitée) ; liens théoriques, définitionnels et contextes de classification affichés à part et non comptés ; cartes contextuelles en fin de liste, sans étiquette « Preuves solides ». `interpretation.ranking` et Ambre alignés sur ce même ordre. (Respecte la demande « trié par nombre de vécus associés » en ne comptant que ce qui est réellement établi.)
2. Scinder `controle` (contrôle psychologique / autorité stricte), en ne réaffectant que les liens dont la source mesure la dimension.
3. Forces harmonisées ou abaissées (§8 points 4–6), formulations (point 8), exil → 6B40/6B41 en contexte.
4. Ajouts de liens de priorité 2 de la revue, après lecture des méthodes et résultats ; sinon `limitee` au plus ou différé.
5. Questions sans lien : mention « aucune association suffisamment documentée n'est retenue pour l'instant » ; région lecteur d'écran limitée au compteur.
6. Hors périmètre §9 visible (lien discret « Ce que la carte ne couvre pas »).
7. Encadré 3114 et bouton de suppression : **pas réintroduits** (décision du propriétaire) ; question posée au propriétaire.

## Ordre, tests, publication

Lots R, S, C en parallèle avec propriété de fichiers ; migration `0022` (R) appliquée avant le Worker ; suites complètes, `verify.yml`, scénarios navigateur ; relecture indépendante finale ; recette réelle avec comptes synthétiques supprimés ; publication.
