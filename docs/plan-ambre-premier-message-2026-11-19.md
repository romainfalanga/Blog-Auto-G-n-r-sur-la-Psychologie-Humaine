# Plan : Ambre commence par un premier message et un bilan lancé depuis la conversation (19/11)

## Demande du propriétaire
1. La page Ambre est toujours la conversation, même avant lancement : un premier message d'Ambre (stylisé) explique son fonctionnement et qu'elle a besoin de toutes les informations pour un premier bilan.
2. Ce message liste les pages à compléter (lien + compteur « fait / total » par page) ; la saisie reste fermée tant que le bilan n'est pas lancé ; un bouton en fin de message lance le bilan.
3. Au clic : message d'Ambre « je prépare votre bilan », avec une durée estimée, une progression, et quand revenir.
4. Bilan prêt : il est dans Mes bilans, Mes diagnostics est à jour, et Ambre publie un message qui résume ce qui ressort et pose des questions (ce que la personne veut creuser, ce qu'elle en pense). La conversation s'ouvre.
5. Le bilan garde tout ce qui existe : instantané complet (enfance, symptômes, proches, ligne de vie, notes, observations, bilans précédents, conversation, mémoire, portrait, suivi, conclusions professionnelles), évaluation clinique déterministe (méthode OMS/CDDR, HAS, NICE), analyse de tous les blocs, relecture. Rien n'est retiré.

## Lot 1 — Fiabiliser le bilan (préalable, sans quoi le bouton mène à un échec)
- `context_changed` : après la capture, toutes les étapes travaillent uniquement sur l'instantané figé (déjà daté et complet) ; les contrôles d'empreinte des sources vivantes après capture sont retirés (le bilan reflète l'état au moment de la capture, comme un document daté). Les contrôles d'accord, d'accès, de prérequis et de quota restent faits à la création et avant publication. Le cron du portrait ne change plus le résultat. Une réponse modifiée pendant le bilan sera prise en compte au bilan suivant.
- `review_not_approved` : le contrôle déterministe des citations ne considère plus comme citations les libellés de l'interface présents dans l'instantané (questions, réponses « Non/Oui/Je ne sais pas/Dans le passé », titres de thèmes et de sujets) ; le contrat de relecture est mis en cohérence avec la règle d'Ambre (« hypothèse : X, appuyée par A et B ; il manque C » autorisé, jamais « vous avez ») pour supprimer les verdicts contradictoires. Aucun critère n'est désactivé.
- Tests : rejeu d'un bilan complet avec un portrait qui change pendant le traitement (doit aboutir) ; bilan contenant « Dans le passé » (pas de citation inventée).

## Lot 2 — Sortie du bilan : document complet + message de conversation
- Le schéma de sortie du bilan gagne un champ `chat_message` (≤ 2 500 caractères) : ce qui ressort en quelques points, puis 2 à 3 questions (ce que la personne veut creuser, ce qu'elle en pense). Le document complet va dans Mes bilans ; le message publié dans la conversation est `chat_message` avec les liens « Lire le bilan complet » et « Voir Mes diagnostics ». Un seul appel de rédaction (pas de génération supplémentaire), relu avec les mêmes critères.
- Pour un point mensuel ultérieur, même principe.

## Lot 3 — Page Ambre : premier message, bouton, attente, échec
- Le fil et le formulaire sont toujours affichés ; la carte « Avant de commencer » disparaît. Un premier message d'Ambre rendu par le code (pas d'IA, pas de stockage), stylisé : présentation courte, puis une liste « Ce dont j'ai besoin » (Mon compte n/4, Mes proches n/3, Ma ligne de vie n/3, Mon enfance n/N, Mes symptômes n/516) avec liens et coche quand c'est complet, puis l'accord fournisseur (case) et le bouton « Lancer mon bilan » (actif seulement quand tout est complet et l'accord coché ; sinon la raison est écrite sous le bouton).
- Saisie visible mais désactivée, avec la mention « Vous pourrez écrire à Ambre après votre premier bilan ».
- Après le clic : message d'Ambre « Je prépare votre bilan », avec progression (lecture de vos informations n/N, rédaction, relecture), durée estimée affichée en fourchette, et « Vous pouvez fermer cette page : votre bilan apparaîtra ici et dans Mes bilans ». `GET /api/chat` expose `journey.review` : `stage`, `blocks_done`, `blocks_total`, `started_at`, `estimate_seconds`.
- Estimation : `estimate = 60 s + 35 s × blocs + 60 s` (marge de réécriture), arrondie en minutes, recalibrée sur la médiane réelle des bilans `ready` quand il y en a (≥ 3), et recalculée pendant le traitement à partir du rythme observé. Valeur de départ validée par une mesure réelle sur un compte de test complet (lot 4).
- Échec : message d'Ambre clair (« Le bilan n'a pas pu aboutir : … Rien n'a été perdu ») avec un bouton « Relancer » quand c'est possible (le crédit mensuel est restitué en cas d'échec, limites quotidiennes inchangées).
- Après le bilan : le message `chat_message` d'Ambre apparaît, la saisie s'ouvre (accord de conversation déjà donné au lancement).
- Visiteur non connecté : même premier message, avec « Se connecter / Créer un compte » à la place du bouton.

## Lot 4 — Vérifications et mise en ligne
- Tests unitaires et navigateur (390 tactile, 1280, 1440) : premier message, compteurs, bouton, attente, échec, message post-bilan.
- Relecture de code indépendante.
- Recette réelle payante limitée : un compte de test complet (516 symptômes, enfance, 3 proches, 3 moments) → bilan réel jusqu'au bout ; mesurer la durée réelle et le nombre de blocs ; vérifier Mes bilans, Mes diagnostics, le message de conversation ; supprimer le compte.
- Hors périmètre (à proposer) : e-mail « votre bilan est prêt ».

---

# Version 1.1 après relecture critique (fait foi)

## Lot 1 (bilan fiable)
1. Retirer seulement `snapshotGuard` (l.~851 et dans `cas` l.~925, 931, 972) ; garder `changeGuard`, bail, accord, accès, préparation, réservation mensuelle, budgets.
2. Garde de suppression avant chaque appel et à la publication : les éléments saisis par la personne et présents dans l'instantané doivent toujours exister (modifications et ajouts ignorés) ; sinon échec `sources_changed` qui efface l'instantané (droit à l'effacement).
3. Garde de sécurité vivante à la publication : un nouveau signal de sécurité apparu après la capture impose les ressources (3114, 15/112) dans le message, ou rejette le bilan.
4. Capture tolérante : relire et réessayer jusqu'à 2 fois avant `context_changed`.
5. Avant la capture, attendre au plus ~3 min qu'un travail de portrait en attente se termine ; sinon capturer sans portrait et le signaler dans le bilan. Le cron n'est pas suspendu.
6. Citations en trois classes : mots de la personne (vérifiés dans `corpus`, pas dans le texte produit par l'IA) ; libellés de l'interface (`question_labels`) admis seulement présentés comme libellés (« à la question … vous avez répondu Oui »), jamais après « vous écrivez / dites » ; le reste = invention.
7. Une seule règle partagée par rédaction et relecture : hypothèses nommées « hypothèse : X, appuyée par A et B ; il manque C », jamais « vous avez [trouble] » (unifier `no_diagnosis`, `REVIEW_COMMON`, `MAP_HYPOTHESIS_RULES`, `REVIEW_INITIAL`) ; tests des deux verdicts contradictoires observés.

## Lot 2 (sortie)
8. `chat_message` (≤ 2 500 caractères, ce qui ressort puis 2–3 questions) dans la même sortie ; `REVIEW_WRITE_TOKENS` et la réservation relevés ; contrôles déterministes et relecture du modèle sur `chat_message` ; repli déterministe construit depuis le résumé si seule la longueur pose problème ; stockage dans l'enveloppe `chat_turns` (`chat_message`, `review_id`) ; les anciennes enveloppes affichent `content` ; aucun faux message pour les anciens bilans ; Mes bilans garde `content`.

## Lot 3 (page) — contrat d'API figé
- `GET /api/chat` : `journey.review` = `{running_id, stage, blocks_done, blocks_total, started_at, estimate:{p50_seconds,p90_seconds,basis}, last_failure:{id,error_code,retryable,retry_at}|null}` ; `blocks_total` stocké à la capture ; `readiness` par page avec `complete` tenant compte de l'enregistrement (`consent`/`locked_at`) en plus des réponses.
- Message de conversation après bilan : enveloppe `{review_id, chat_message}` ; liens vers `/mes-bilans/` et `/mes-diagnostics/`.
9. Le premier message remplace `data.welcome` et la case `chat-gate` (un seul message, un seul accord) ; l'accord n'est enregistré qu'à la création réussie du bilan (ou annulé si la création échoue).
10. Compteurs avec « enregistrez » si la page n'est pas enregistrée ; visiteur non connecté : même message sans compteurs, avec connexion.
11. Échec affiché depuis `last_failure` (survit au rechargement) ; table de messages par code ; « Relancer » seulement si `retryable`, avec les limites quotidiennes affichées.
12. Estimation en fourchette p50–p90 (données réelles quand ≥ 3 bilans, sinon valeurs initiales mesurées en recette, cron et réécriture inclus), mention « plus long que prévu » au-delà ; sondage toutes les 15 s, suspendu quand la page est masquée.
13. Instantanés des bilans échoués supprimés (ou limités au dernier).
14. Accessibilité : barre de progression avec valeurs, un seul `aria-live` poli, saisie `aria-disabled` reliée à son explication, mouvement réduit, focus.

## Lot 4
15. Tests : portrait modifié pendant le bilan (aboutit), suppression d'un proche (échoue), retrait d'accord, suppression du compte, signal de sécurité tardif, « Dans le passé » et libellés, coupure de longueur, ancienne enveloppe, rechargement pendant l'attente et après un échec, relance et quotas.
16. Répétition locale avec fournisseur simulé, puis une seule recette payante (au plus une relance), coût mesuré par `provider_budgets.observed_cost` ; un agent pour les lots 1 puis 2 (même fichier), un agent pour l'interface en parallèle sur le contrat figé, puis relecture indépendante.
