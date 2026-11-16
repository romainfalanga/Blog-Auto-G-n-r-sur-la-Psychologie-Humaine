# Correctifs du moteur Ambre — 17 novembre 2026

Périmètre : défauts 1 à 14, 17 et partie moteur de 18 de `audit-ambre-2026-11-17.md`. Le nouveau plan de changement et le suivi (16) sont intégrés séparément. Le code mort (15) reste volontairement hors périmètre. Aucune migration, aucun déploiement ni appel de génération distant réalisé pour cette recette.

## Résultats et méthode

Recette sur le **vrai moteur activé**, ses routes de conversation et une **base SQLite en mémoire contenant toutes les migrations du dépôt**. Le fournisseur est simulé pour imposer précisément succès, réécriture, erreurs HTTP et sorties tronquées. Il ne s’agit donc pas d’une mesure facturée par OpenRouter ni d’une validation clinique des réponses générées.

La mesure d’entrée est celle utilisée par les réservations : `ceil((système + JSON d’entrée + schéma JSON).length / 3)`. La réserve ajoute le plafond de sortie de chaque appel. Les sorties nominales sont plafonnées à 1 500 pour analyse/relecture et 1 800 pour rédaction ; les reprises de relecture tronquée montent à 2 250 puis 3 000.

Le profil rempli utilise des données réellement enregistrées : 6 proches, 20 moments, présentation détaillée, cartographie renseignée, bilan, 6 tours longs et mémoire de continuité. Le maximal contient 50 proches, 500 moments, champs de présentation aux limites, message de 6 000 caractères, cartographie, 6 tours, les huit rubriques de mémoire remplies dont 30 sujets fermés. Aucun remplacement du fournisseur de contexte personnel pour ces fixtures.

Dernière exécution complète : **472 tests journal réussis, aucun échec** (`npm --prefix journal test`). Les tests nouveaux portent le préfixe `audit17` dans `journal/tests/ambre-engine.test.mjs`. Les protections préexistantes de bail, version, consentement, suppression, export et isolation des comptes restent couvertes. Le contrôle `git diff --check` des fichiers moteur passe.

| Profil / parcours | Analyse | Rédaction | Relecture | Réécriture | Relecture finale | Total entrée estimée | Réservé |
|---|---:|---:|---:|---:|---:|---:|---:|
| Rempli, normal | 2 201 | 6 624 | 2 465 | — | — | 11 290 | 16 090 |
| Rempli, réécriture | 2 201 | 6 624 | 2 465 | 6 727 | 2 465 | 20 482 | 28 582 |
| Maximal, normal | 4 833 | 5 558 | 2 372 | — | — | 12 763 | 17 563 |
| Maximal, réécriture | 4 833 | 5 558 | 2 372 | 5 595 | 2 372 | 20 730 | 28 830 |

Le maximal coûte moins en rédaction car son long message laisse moins de place aux extraits facultatifs. Les omissions sont explicites dans `context_omissions` et les entrées exactes des étapes. Quelques unités peuvent varier avec la date ou la longueur des libellés de cartographie.

## Budget complet, reprises et bornes

Le **plafond nominal reste 30 000 jetons réservés**. Après analyse et éventuelle vérification de risque, le moteur calcule ce qui reste pour **deux rédactions et deux relectures**, sorties comprises. Le budget d’entrée de la rédaction découle de ce calcul ; la relecture réserve au plus 8 000 caractères d’entrée, schéma compris. Les entrées effectivement construites sont vérifiées avant l’appel. Le contrat de rédaction est propre au moteur et plus court ; l’ancien contrat est conservé uniquement dans le chemin historique non activé.

Les reprises techniques sont une enveloppe distincte, fondée sur les réservations des tentatives déjà tracées : au plus trois tentatives pour chacune des phases `analysis`, `risk_check`, `writing`, `review`, `rewrite`, `rewrite_review` (l’analyse initiale n’est pas rejouée). Le cumul réel reste intégralement dans `chat_engine_jobs.reserved_tokens` et `provider_budgets.reserved_tokens`. Aucun appel échoué n’est rendu artificiellement gratuit. Une limite finale de 94 500 borne aussi le cumul ; ce n’est pas un nouveau budget nominal ouvert à chaque message. Une réservation refusée par le vrai budget hebdomadaire ne reste pas ajoutée au compteur du tour.

Preuves extrêmes avec profil maximal :

- **13 appels**, analyse unique, deux rédactions chacune avec deux erreurs HTTP avant succès, deux relectures chacune tronquée deux fois : succès relu, **78 324 réservés**. Réservations successives : `6333, 7358, 7358, 7358, 3872, 4622, 5372, 7395, 7395, 7395, 3872, 4622, 5372`.
- **16 appels**, mêmes reprises plus trois tentatives de vérification du risque : succès relu avec ressources de sécurité, **79 448 réservés**, analyse unique. La voie sécurité utilise un contrat et un schéma dédiés sans proposition ni note.
- Le plafond du tour produit `turn_budget`, avec un message propre à la limite de traitement. `generation_budget` reste réservé au budget hebdomadaire. L’analyse, les baux et les compteurs ne sont pas réinitialisés pour contourner une limite.

Les bornes sont conservatrices, pas une promesse de succès lors d’une panne permanente : trois échecs de relecture publient le repli prudent, trois échecs de vérification du risque arrêtent le tour sans inventer une crise, et le vrai budget hebdomadaire peut toujours refuser des appels.

## Traçabilité par défaut de l’audit

| N° | Correction et preuve |
|---|---|
| 1 | Réécriture et relecture finale planifiées avant la première rédaction ; profils rempli/maximal aboutissent sous 30 000 en nominal. Erreur de tour distincte de la semaine, test de son message. |
| 2 | Contrat compact, présélection d’index, budget du JSON complet des détails, compression explicite des extraits. Maximal avec message de 6 000 caractères **et toutes les rubriques mémoire pleines** : réponse relue et réécriture possibles. |
| 3 | `memoryPrompt` parcourt chaque rubrique du plus récent au plus ancien. La conservation complète reste chronologique ; les fermetures omises imposent les contraintes prudentes. Test du dernier objectif conservé. |
| 4 | Extraits gardant début et fin des réponses, donc question finale. Six tours datés dans la voie ordinaire ; pour un contexte de sécurité saturé, l’échange immédiatement précédent est prioritaire et l’omission des anciens est signalée. Test avec question au-delà de 700 caractères. |
| 5 | `current_time`, `timezone=Europe/Paris`, dates des échanges et `last_session_local`/`next_session_local` lisibles. Tests hiver/été et passage de minuit. Les secondes Unix originales du suivi sont conservées. |
| 6 | Contrat moteur séparé, zéro ou une question ; schéma à une question, préparation limitée à une. Plus de références aux champs absents de l’ancien moteur. Le contrat historique à deux questions demeure dans le code mort, explicitement hors périmètre 15. |
| 7 | « J’ai noté que… », « C’est noté » et délai de 48 heures ne bloquent plus. Les promesses de sauvegarde et faux numéros directs restent détectés. Le relecteur reçoit aussi cette distinction. |
| 8 | Expressions de disparition qualifiées par un sujet personnel et danger réellement décrit ; « faire disparaître cette habitude » et « travail… fardeau » ne déclenchent plus le 3114. « pas moi » isolé et honte ordinaire ne forcent plus une stabilisation sans question. `avis` exige une frontière de mot. Les signaux directs, moyens, tiers, fardeau personnel et allusions actuelles restent protégés, plus vérification sémantique indépendante conditionnelle. |
| 9 | URL de personnage retirée des pistes transmises ; rédaction interdit URLs et suggestions de lecture/personnage. Les URL des **preuves scientifiques** sont conservées avec leurs limites. Test d’absence de `/personnages/`. |
| 10 | `cartography_targets` fournit les cibles exactes sélectionnées ; `observation_candidates` les citations autorisées et l’ID du proche. Contrat explicite pour `factor/outcome/general`, citations actuelles et absence d’annonce avant écriture. Les contrôles déterministes et les écritures existantes restent en place. |
| 11 | `draft.questions` contient les vraies questions ajoutées à la réponse ; relecteur informé de ne pas les compter deux fois. Bloc `care` et date transmis à la relecture. Test du contenu exact. |
| 12 | Une panne ou un résultat de mémoire mal formé n’ajoute aucun début de message aux faits, n’avance pas `through_turn` et conserve le lot. Reprise avec attente croissante (jusqu’à une heure), état `retry_pending` exposé. Test panne puis réussite et couverture inchangée pendant la panne. |
| 13 | Aucun complément d’extraits de score nul. Recherche lexicale bornée avec titres pondérés ; seules sélection du modèle autorisée et correspondances pertinentes remplissent les détails. Test : « Bonjour » ne remplit aucune fiche sans rapport. |
| 14 | Libellés d’index jusqu’à 100 caractères, types explicites, espaces normalisés. Test d’un intitulé négatif long conservé. |
| 15 | Code mort non supprimé, conformément au périmètre demandé. |
| 16 | Suivi et préparation de séance : travaux séparés à intégrer par le coordinateur. Aucun changement ici dans `care-backend.mjs` ou ses tables. |
| 17 | Présélection à 48 entrées/2 600 caractères ; total et omissions communiqués. Test de plus de 1 100 entrées. Les IDs choisis doivent appartenir à la présélection réellement envoyée. Pas de recours silencieux aux règles pour un index géant. |
| 18 | Commentaire des traces V2 corrigé ; mémoire permanente libellée en français ; `safety_seen` conservé lors d’une correction sans modification de sécurité, effacé lors d’un effacement complet. Politique de contexte, instantanés et descriptions des bilans/observations actualisés. README et libellés statiques Paramètres relèvent de l’intégration du coordinateur. |
| 19 | Élément documentaire hors moteur, sans correctif logiciel. |

Bilans et observations, également relevés dans les constats généraux : les quatre derniers bilans prêts sont indexés comme **synthèses de l’assistant**, jamais comme preuves nouvelles. Les observations ne sont rappelées que par le sélecteur existant, pour le proche identifié et sous accord actif. La révision des observations entre dans l’empreinte de contexte et dans la garde SQL de publication. Une révocation pendant la relecture invalide la réponse. Tests de citation candidate, note réellement écrite, rappel au tour suivant et révocation tardive.

## Interfaces pour la greffe du plan de changement

- `createAmbreEngine` garde la signature de `generateText` utilisée par le suivi. Les paramètres additionnels de trace/budget restent internes et optionnels.
- `snapshot` est le point de chargement : ajouter le plan et sa révision au chargement **et au hash**. `publishEngine` doit aussi vérifier la révision au commit.
- Après l’analyse, `data` contient `current_time`, `timezone`, `care`, `memory`, `memory_revision`, `history`, `identity`, `self_context`, `details`, `index_total`, `observation_revision`, `reported_relationship_observations`, `observation_candidates`, `cartography_targets` et `budget`.
- `fitInput` retire d’abord les détails et sources facultatives. Toute nouvelle donnée constante doit entrer dans ce budget et ses tests, sans augmenter implicitement la réserve. La version effectivement envoyée est persistée avant la relecture ; les contraintes renforcées lors d’une omission de sujets fermés sont aussi conservées.
- La relecture reçoit explicitement `care`. Ajouter les stratégies et le plan aux seules entrées nécessaires et à `traceDetail`/`USAGE_POLICY`. Le nombre de critères reste 15 dans ce lot.
- Une proposition de plan ne doit pas contourner le lot atomique de publication, les baux, le consentement, les révisions, l’export ni les suppressions.

## Limites restantes

Le classement lexical est une présélection, pas une garantie de compréhension : une source pertinente peut être omise quand le vocabulaire diffère. La mémoire demeure une organisation de citations de la personne, pas encore un plan d’action structuré. Aucun de ces tests ne valide la sensibilité clinique d’un juge IA ou le tarif réel d’un fournisseur. Une sortie exceptionnellement longue ou un futur bloc constant trop grand peut dépasser l’enveloppe prévue : le contrôle l’arrête avant appel ; les tests de budget doivent être relancés lors de toute greffe.
