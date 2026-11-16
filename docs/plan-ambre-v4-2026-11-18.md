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

---

# Révision v4.1 après critique indépendante (fait foi en cas de contradiction avec ce qui précède)

## Lot R — corrections
- **F11** : pas de redirection signée dans cette livraison (secret Netlify = décision du propriétaire). Limiteur par adresse e-mail conservé ; limiteur par IP calculé sur `x-nf-client-connection-ip` si présent, sinon `CF-Connecting-IP` (IPv6 par préfixe /64). Une falsification ne permet que d'échapper au limiteur IP (le limiteur e-mail demeure), plus de blocage global de toutes les connexions. Pas de 403, `workers_dev` reste ouvert. Purge bornée de `rate_limits` et des `sessions` expirées dans le cron. Test : 31 échecs d'une IP n'empêchent pas une autre IP de se connecter.
- **F10** : garde fournisseur consultée **avant** toute réservation et toute trace ; un refus 429/402/503 ne compte ni dans `attempt` ni dans `attempts`, rembourse `provider_budgets` et `chat_engine_jobs.reserved_tokens` ; `Retry-After` respecté ; refroidissement commun alimenté par le moteur actif ; bilans et portrait passent par le même module commun d'accès au fournisseur. R livre ce correctif du bloc d'erreur de `work()` en premier.
- **F12** : plafond global quotidien `DAILY_PROVIDER_TOKEN_CAP` (variable, défaut 0 = désactivé, documenté), commun à tous les usages, conversation prioritaire ; au-delà : refus clair (avec ressources si un signal de danger est présent), pas de file longue. Plafond de création de comptes par IP et par jour.
- **F8/F9** : priorité d'étape (relecture et publication avant analyse) puis `created_at` ; `queued_at` pour la mémoire ; au plus 5 appels simultanés par invocation ; critère : 20 tours en ≤ 4 passages, mémoire servie dans les 3 premiers.
- **F3** : index `chat_turns(user_id, …)` dans 0022 ; comptages horaires et quotidiens bornés ; critère `EXPLAIN` sans `SCAN` ni `TEMP B-TREE` sur les requêtes de sondage.
- **F22** : script de ligne de base hors `migrations/` (`journal/ops/d1-baseline.sql`) qui crée `d1_migrations` au format Wrangler et y inscrit 0001–0021 ; appliqué une fois en production avant 0022 ; critère : `migrations list` n'indique que 0022.
- **F1** : purge limitée aux tâches terminées (`done`/`failed`) ; détail 30 jours puis résumé, 200 traces détaillées par personne ; critère mesuré par `sum(length(trace))` à J+31 simulé ; Paramètres et texte de confidentialité mis à jour.
- **F13** : compteurs agrégés lisibles par une route `/api/internal/metrics` protégée comme les autres routes internes ; retirer `throw portraits.reason` (worker.mjs).
- **F23** : critère « lectures de `life_events` ≤ 2 par tour ».
- **F16** (message d'erreur brut, CSP), **F21** (modèle de secours pour les petits appels : `OPENROUTER_ANALYSIS_FALLBACK_MODEL`), **F14** partiel (bouton « Déconnecter tous les appareils » dans Mon compte ; mention « identité professionnelle déclarée, non vérifiée » visible partout où le psychologue apparaît) ajoutés. F15 différé.

## Lot S — corrections
- **F17** : tous les codes d'erreur de `send` (409, 403, 429, 503…) portent les ressources quand le message contient un signal de danger ; affichage vérifié dans le navigateur.
- **F19** : corpus indépendant ≥ 50 messages à risque et ≥ 50 neutres (écrit par un agent distinct, avant la règle), rappel et faux positifs publiés ; en cas de panne de l'analyse, **vérification de risque dédiée** (petit appel) ; voie prudente avec ressources seulement si un signal lexical (liste documentée) est présent ou si la vérification échoue elle aussi sur un tel message.
- **Ordre des signaux** : un signal déterministe n'est jamais rétrogradé (ni par R10 ni par R5) ; `reality_concern` jamais sur une croyance religieuse ou culturelle seule ; R21 vise l'approbation d'actes nuisibles, jamais la validation d'émotions ; R8 seulement sur l'âge actuel déclaré ; R16 ne stocke aucun diagnostic.
- **F5** : séparer `safety_history` (épinglé, daté, dédoublonné, modifiable et supprimable par la personne) des appuis évincables ; format daté compatible avec `memoryPrompt`, `PUT /api/chat/memory`, l'export et Paramètres.
- **R14** : « proposition refusée » = proposition de plan refusée, ou refus explicite d'un exercice ou d'une piste dans le message (liste d'expressions testée).
- **Ajouts** : #12 (registre d'événements indésirables sans contenu, consultable par l'administrateur), #34 (consommations : EM d'abord, jamais d'arrêt brutal d'alcool ou de benzodiazépines sans médecin ; ressources vérifiées uniquement).
- **Budget** : test de budget sur les 3 étapes ; ajout ≤ +10 % sur le profil rempli, compensé en réduisant les `details` si besoin.
- **R11** : batterie simulée dans la CI ; passe réelle (≈ 40 scénarios × 6 tours) documentée comme procédure, non lancée automatiquement.

## Lot C — corrections
- **C1** : tri et nombre affiché = vécus reliés par une association empirique (forte, modérée, limitée) ; liens théoriques, définitionnels et de classification affichés à part, non comptés ; cartes contextuelles après toutes les autres ; Ambre reçoit le même ordre. Critères par simulation : top 5 affiché = top 5 transmis à Ambre ; aucune carte contextuelle avant une carte non contextuelle ; parité navigateur/Worker.
- **C2 (scinder `controle`) et « Je ne sais pas »** : différés, décision du propriétaire (changement de sens des réponses enregistrées).
- **C3–C6** maintenus ; tout changement de lien relance `audit_cartographie.py` et une relecture des liens modifiés, pas seulement l'empreinte.
- Saisie des ressources (PCE/BCE) : décision du propriétaire.

## Propriété des fichiers (implémentation parallèle)
- **R** : `worker.mjs`, `netlify.toml`, `wrangler.jsonc`, `journal/migrations/0022_*`, `journal/ops/`, `ambre-insight.mjs`, `personal-backend.mjs` (appel fournisseur), nouveau module commun d'accès au fournisseur ; dans `ambre-engine.mjs` : `generateText`, bloc d'erreur de `work()`, `startTrace`/`finishTrace`/`saveTrace`, `resume`, `snapshot`, planification du cron ; dans `chat-backend.mjs` : `state`, `exportData`, `providerCooldown`, `failureInfo`, appel des bilans, requêtes de sondage ; dans `chat-interface.mjs` : boucle de sondage ; `ui.mjs` (Mon compte : déconnexion globale).
- **S** : `chat-safety.mjs`, `psychology-framework.mjs`, `change-engine.mjs`, nouveau module de citations ; dans `ambre-engine.mjs` : consignes, analyse, matrice, relecture, `checkedRisk`, `updateMemory` ; dans `chat-backend.mjs` : `send`, `makeProposal`, filtrage des notes ; rendu des erreurs dans `chat-interface.mjs` ; migration `0023_*` si indispensable.
- **C** : `data/`, `editorial/cartographie/`, `scripts/build_cartographie*`, `scripts/audit_cartographie*`, `static/js/cartographie*.js`, `layouts/_default/cartographie.html`, `journal/src/cartographie.mjs`, `journal/src/cartographie.generated.mjs`, `tests/cartographie*`.
- Chacun écrit ses tests dans de nouveaux fichiers ; Edit uniquement, relire avant chaque modification d'un fichier partagé.

## Décisions laissées au propriétaire (rien n'est fait sans accord)
Secret de signature Netlify ; montant du plafond quotidien ; vérification d'adresse e-mail ; accès des mineurs ; scission de `controle` et « Je ne sais pas » ; saisie des ressources ; revue humaine mensuelle ; intitulé « psychologue IA » (avis juridique) ; encadré 3114 dans la Cartographie.
