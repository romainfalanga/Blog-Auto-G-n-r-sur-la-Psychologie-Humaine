# Intégration du plan de changement au moteur Ambre — 17 novembre 2026

Le plan v3 et ses deux cycles de relecture ont été lus avant l’intégration. Cette livraison concerne le moteur, les propositions de conversation, la continuité et la transparence. Les routes propriétaires du plan, la migration 0021, le partage professionnel et l’interface sont intégrés dans leurs chantiers séparés. Aucun déploiement ni appel à un fournisseur réel n’a été effectué ici.

## Contrats livrés

`journal/src/change-engine.mjs` assemble le plan compact, les consignes privées, la sélection et la relecture de `change-strategies.mjs`. L’analyse reçoit zéro à six candidats ; seuls zéro à deux IDs de cette liste peuvent charger des fiches. Les restrictions particulières restent entières ; le contrat actif contient P0/P1/P2/P3 et les exclusions communes C0. Une correction ponctuelle autorisée par l’auteur du module normalise les traits d’union des demandes françaises (« Pouvez-vous », « Aide-moi »).

La révision du plan, son epoch, sa génération et la version du catalogue entrent dans l’empreinte des sources. `context_generation` est inscrit au moment SQL de création du tour et n’est jamais remplacé à sa reprise. Historique et lots de mémoire ne lisent que cette génération. Les bilans écrits suivent la même règle et contrôlent le CAS du plan avant publication.

Une proposition reste `proposed`, avec un ID, `source_turn_id` et une citation exacte du message courant. Le modèle ne peut adopter un objectif, choisir un résultat, fixer une date ou ouvrir un partage. Les résultats reprennent les mots exacts de la personne, avec `attempt:uncertain` et `outcome:null` tant que la carte n’a pas été confirmée. La réponse du chat contient seulement `change_proposal_ids` ; l’interface recharge la carte dans l’API du plan.

La publication compose le CAS du plan, les écritures de propositions/relance et la réponse dans le même `DB.batch`, sous le bail du moteur. Après la mutation du plan, les écritures restantes utilisent son jeton `last_mutation`, pour ne pas invalider leur propre CAS. Une interruption après publication ne répète ni carte ni relance. Une révision, un epoch ou une génération obsolètes empêchent la publication.

Les huit critères C-R1…C-R8 gardent leur applicabilité et leur verdict : `pass`, `fail`, `not_applicable` ou `not_checked`. Un critère applicable manquant n’est pas validé. Le relecteur peut signaler un exercice implicite non détecté initialement ; un échec active alors le critère correspondant. Les contrôles déterministes prévalent sur les réponses du modèle. Une méthode omise pour manque de budget, une instruction privée trop longue, un refus ou une voie de sécurité interdisent tout nouvel exercice, y compris non structuré.

## Relances, suppressions et provenance

Les relances exigent le seul essai admissible, le choix explicite de la personne, moins de deux relances, sept jours pleins entre elles et un tour intermédiaire. L’invitation doit être une citation exacte de la réponse finale. Une réécriture qui la retire ne débite rien. La commande entière d’arrêt suspend immédiatement les retours sans appel modèle, crédit ou tour artificiel ; les objectifs restent conservés.

Les références `change_artifact_refs` sont enregistrées avant l’appel fournisseur à partir des entrées réellement envoyées, sans dépendre d’une déclaration d’utilisation du modèle. Elles comprennent plan, consignes et dépendances héritées des tours et bilans effectivement fournis. Lorsqu’un extrait de mémoire est envoyé, les références connues jusqu’à son watermark sont héritées conservativement. Un `INSERT SELECT` groupé évite une requête par ancienne référence. Les bilans héritent eux aussi des dépendances des échanges réellement retenus après réduction.

`DELETE /api/chat` utilise le helper CAS du plan, en première écriture du batch. Il purge les copies de conversation, citations, propositions non adoptées, mémoire, traces, jobs, observations et bilans. Les choix adoptés restent conservés par défaut ; `include_plan:true` les efface aussi. La réponse explique les éléments conservés et la nécessité d’un nouvel accord et d’une nouvelle analyse initiale. Un ancien bilan supprimé n’est pas présenté comme prêt. Le test ancien de reprise après purge a été adapté à ce comportement validé par l’agent principal.

L’effacement explicite de toute la mémoire utilise son helper distinct et la révision affichée dans l’interface. Il avance la génération et efface les indicateurs de mémoire conformément au contrat, sans réactiver les propositions ou les relances. Les purges ordinaires conservent seulement les booléens de refus et de vigilance. Une demande actuelle explicite de rouvrir un sujet autorise ce tour sans supprimer le refus durable ; une vigilance passée ne devient jamais un danger actuel.

## Mesures de coût

Les tests appellent le vrai moteur et un SQLite réel avec toutes les migrations ; seul le fournisseur est simulé. L’estimation d’entrée est `ceil((système + schéma + données sérialisées)/3)`, par appel. Les réservations incluent les maxima de sortie et toutes les reprises. Ces mesures ne sont ni une facture fournisseur ni une mesure de qualité clinique. Une variation de quelques jetons peut venir des dates et IDs.

| Fixture | Analyse | Rédaction | Relecture | Réécriture | Seconde relecture | Entrée nominale, trois appels | Réservation totale, cinq appels |
|---|---:|---:|---:|---:|---:|---:|---:|
| Rempli, plan et consigne | 2 686 | 6 609 | 2 534 | 6 291 | 2 534 | 11 829 | 28 754 |
| Rempli, deux méthodes actives | 2 629 | 6 572 | 2 541 | 6 715 | 2 541 | 11 742 | 29 098 |
| Maximal, plan, mémoire complète et message de 6 000 caractères | 5 256 | 5 466 | 2 433 | 5 360 | 2 270 | 13 155 | 28 885 |
| Maximal, deux méthodes demandées, puis omises faute de place | 4 835 | 5 546 | 2 265 | 5 575 | 2 265 | 12 646 | 28 586 |
| Maximal et agrégat aux plafonds de nombre d’objets | 5 260 | 5 466 | 2 433 | 5 360 | 2 270 | 13 159 | 28 889 |

Le profil rempli comporte six proches, vingt moments, cartographie, six tours et mémoire. Le maximal comporte cinquante proches, cinq cents moments, mémoire saturée et texte courant maximal. L’agrégat plein contient vingt objectifs, soixante étapes, deux cents essais, quarante propositions, vingt préparations et dix consignes, tout en restant sous 256 Kio. Une consigne trop longue est omise entière et impose une vérification en séance.

Le contexte de plan remplace une partie des autres détails ; il ne s’ajoute pas simplement à leur ancien plafond. Le moteur réserve les deux rédactions et deux relectures avant la première rédaction. Le plafond nominal reste 30 000, distinct du budget hebdomadaire. Si le contexte maximal ne peut contenir une méthode avec toutes ses restrictions, il omet ce bloc entier, inscrit `disabled_reason:context_budget` et interdit l’exercice. Cette limitation est intentionnelle et visible dans les traces.

Les reprises restent bornées à trois tentatives par phase, comptées séparément et imputées au vrai budget hebdomadaire. Le stress maximal de réécriture avec toutes les reprises utilise treize appels et 78 365 réservés ; avec vérification de sécurité, seize appels et 78 389 réservés. L’analyse n’est pas rejouée après sa réussite. Ces réservations cumulées restent sous la borne de récupération 94 500, et ne sont pas présentées comme un tour nominal de 30 000.

## Transparence et interfaces stabilisées

- `GET /api/chat/insight` expose `change={plan,strategies,private_care_instructions,due_trial,instructions_omitted}` et `current.change`.
- `methods={version,common_rules,methods,review_criteria}` contient le catalogue réellement chargé.
- Un tour expose `snapshot.change`, `snapshot.analysis.strategy_ids`, `review.change.criteria`, et les entrées exactes de chaque appel dans `stages[].input.change`.
- `usage_policy` décrit la substitution du contexte et les bornes de plan, candidats, fiches et consignes.
- `followups_paused:true` et `notice` signalent l’arrêt immédiat ; la suppression du chat possède sa propre notice.

La réduction des anciennes voies conversation/bilan a aussi été complétée pour les nouveaux blocs cartographiques `contextual` et `exploratory`. Leurs éléments partent entiers, avec `truncated:true`, avant les dernières notes de la personne ; le test de contexte surdimensionné passe sur le nouveau graphe.

## Vérification et limites

`journal/tests/change-engine.test.mjs` : 18 tests SQLite réussis, couvrant proposition/adoption, résultat exact, refus/crise, revue manquante, exercice implicite, relances 1/2/jamais 3, réécriture sans débit, reprise après publication, CAS en vol, génération, purge chat, héritage des références, bilan en vol et coûts rempli/maximal/agrégat plein.

La passe finale dédiée réussit **194/194 tests**, sans échec, avec :

```sh
node --test journal/tests/ambre-engine.test.mjs journal/tests/change-engine.test.mjs journal/tests/chat.test.mjs journal/tests/chat-safety.test.mjs journal/tests/change-strategies.test.mjs
```

Le test cartographique de contexte surdimensionné réussit aussi séparément. La suite globale du Worker reste coordonnée par l’agent principal après les autres intégrations simultanées. Sa première passe avait révélé trois harness nécessitant de réécrire le nouvel import `change-backend.mjs` avant leur import `data:` ; ils ont été signalés à leur propriétaire.

Le test d’une ancienne association empirique religion/dépression retirée par la relecture a été remplacé par la véritable liaison clinique `emprise_religieuse → sch_assujettissement`, avec statut théorique et source Young. La non-troncature d’une longue nuance foi/coercition reste couverte par une fixture technique séparée, sans la présenter comme une association du catalogue. Un test distingue les libellés contextuels adultes de la restriction `young_child_only`.

L’efficacité clinique, la qualité des sorties d’un fournisseur réel, l’affichage final sur navigateur et le déploiement ne sont pas établis par ces tests. Le plafond de contexte peut volontairement empêcher une nouvelle méthode sur un cas maximal ; aucune restriction clinique n’est coupée pour la faire tenir.


## Correctif issu de la relecture indépendante — course de finalisation de trace

Le relecteur indépendant a reproduit une suppression de plan intercalée entre la lecture de la trace finale et sa réinsertion. L’ancienne finalisation pouvait recréer les entrées privées supprimées. Le correctif remplace cette reconstruction JavaScript par une fusion SQL atomique des seuls champs de résumé ; les entrées privées ne sont jamais renvoyées depuis l’ancienne lecture. Un CAS sur la trace relue, le bail actif, le consentement et sa révision, la génération de création du tour et les contrôles courants du plan protègent l’écriture. Les résumés remplacent leurs anciennes valeurs, sans conserver un ancien verdict d’échec par fusion récursive.

Les mêmes gardes couvrent le début et la fin d’une étape de trace. L’enregistrement des références exige la trace qui vient réellement d’être écrite. Le job mémoire planifié après publication est gardé lui aussi, afin qu’une purge qui gagne cette course ne soit pas suivie par sa recréation.

Le test indépendant `change-engine-review.test.mjs` nommé « a late trace finalizer cannot restore plan inputs after owner deletion » passe sans modification du test. Deux nouvelles régressions couvrent la suppression pendant la finalisation d’un échec et le remplacement du bail avant le retour du fournisseur. Les deux modules de tests moteur passent ensemble : 83/83 ; la dernière vérification ciblée de suppression, bail et reprise après publication passe également. Les autres courses de cutoff relevées dans les mutations du plan relèvent de `change-backend.mjs` et sont corrigées par son propriétaire.
