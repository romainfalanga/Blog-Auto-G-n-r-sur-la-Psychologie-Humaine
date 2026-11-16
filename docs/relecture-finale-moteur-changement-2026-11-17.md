# Relecture indépendante finale du moteur de changement — 17 novembre 2026

Relecteur : `final_engine_review17`, distinct des auteurs du moteur et de la persistance. Branche examinée : `codex/ambre-changement-couverture-20261117`.

**Avis favorable sur le périmètre technique relu, après correction et reproduction des trois défauts ci-dessous. Aucun bloquant identifié ne reste ouvert dans ce périmètre.** La recette navigateur, les tests globaux, la transmission/export du suivi professionnel et le déploiement restent coordonnés séparément par le pilote.

Le contrat v3, ses deux cycles de critique et le rapport d’intégration du moteur ont été lus. La relecture cible `change-engine`, `change-plan`, `change-strategies`, `ambre-engine`, `chat-backend` et les chemins de suppression associés de `change-backend`. Aucun fichier source ni test existant n’a été modifié par le relecteur ; seuls ce rapport et le nouveau fichier `journal/tests/change-engine-review.test.mjs` lui appartiennent. Aucun commit, appel à un fournisseur réel ni déploiement effectué.

## Défauts reproduits puis clos

Les reproductions emploient le vrai moteur, les routes propriétaires et SQLite avec toutes les migrations. Le fournisseur seul est simulé. Les points de suspension sont des retours de lecture SQL : ils imposent un entrelacement autorisé entre deux requêtes, sans reproduire la logique métier dans le test.

| Défaut initial | Reproduction et conséquence observée avant correction | Correction relue et résultat |
| --- | --- | --- |
| F1 — Finalisation tardive restaurant une trace purgée | Dans l’ancien `ambre-engine.mjs:200–203`, `saveTrace` relisait l’ancienne trace complète, puis la réinsérait sous le seul contrôle `purged=0`. Après publication du tour, une vraie suppression d’objectif placée entre les deux opérations retirait la trace, mais le finaliseur la recréait avec le texte de l’objectif supprimé. | `traceWriteGuard` et `saveTrace` (`ambre-engine.mjs:182`, `:204`) contrôlent le bail, le consentement, la génération et le plan ; `saveTrace` applique seulement un patch de métadonnées à la trace encore présente, avec comparaison de la trace lue. Le test indépendant ne retrouve plus de trace. Les auteurs ont aussi couvert la finalisation d’un échec et le remplacement du bail avant `finishTrace`. |
| F2 — Transmission manquée par une suppression concurrente | Dans l’ancien `change-backend.mjs:78–89`, `cutoff` dépendait d’une lecture de références hors transaction. Un vrai tour transmis après cette lecture, avant le commit de suppression, ajoutait ses dépendances sans changer la révision du plan. La suppression réussissait sans avancer la génération ; le message source revenait dans `writing.history` au tour suivant. | Le premier CAS du batch recontrôle maintenant l’existence effective des références concernées (`change-backend.mjs:89–98`). En cas de changement de cette source avec plan inchangé, le batch est reconstruit ; un conflit réel de plan reste un 409. La suppression reproduite avance désormais la génération et le prochain prompt exclut l’ancien message. |
| F3 — Refus explicite perdu pendant une purge ordinaire | Une suppression lisait l’ancienne mémoire, puis un tour « Je ne veux plus parler de ma famille. » publiait sa limite dans `memory.closed`. La suppression vidait ensuite la mémoire avec `exploration_reconsent_required=0`, calculé sur la lecture périmée. | Le CAS initial compare aussi la valeur du résumé effectivement lu pour les indicateurs (`change-backend.mjs:90–98`). Le batch reconstruit conserve le refus sous forme non textuelle. Le test vérifie `exploration_reconsent_required=1` après suppression. |

Les trois tests indépendants ont échoué sur le code initial avec les conséquences décrites ; ils réussissent sans modifier leur scénario après les corrections des auteurs. Leurs assertions n’affichent aucun contenu privé réel.

## Contrats examinés

- **Consentement, arrêt et adoption** : commande d’arrêt entière traitée avant génération, arrêt durable, reprise uniquement par action explicite, refus et crise neutralisant les propositions. Le modèle ne crée que des cartes `proposed` ; l’adoption et les résultats acceptés passent par l’API propriétaire. Les résultats proposés exigent des mots exacts et restent `attempt:uncertain`, `outcome:null` avant confirmation.
- **Méthodes et relecture** : présélection bornée, liste blanche des IDs, détails entiers avec permissions/exclusions, C09 limitée à la préparation de séance et exclusion de l’assertivité à tester en présence de violence. Un critère applicable absent devient `not_checked` et bloque la publication de l’exercice. Le relecteur sémantique peut activer un critère pour un exercice implicite. Les contrôles déterministes gardent priorité.
- **Relances** : opt-in séparé, un seul essai admissible, deux invitations au maximum, sept jours pleins et un tour intermédiaire, citation exacte de la réponse finale. Une réécriture retirant l’invitation ne débite rien. La publication et le compteur partagent le batch gardé ; la reprise après publication ne les répète pas.
- **Provenance et suppression** : références enregistrées avant l’appel à partir des entrées fournies ; héritage des tours, bilans et mémoire envoyés ; références privées séparées du sérialiseur professionnel. Les corrections/suppressions transmises coupent la continuité, les modifications administratives la préservent. Les tours gardent leur génération de création ; historique et lots de mémoire respectent la génération courante. Les trois fenêtres concurrentes découvertes sont couvertes par les reproductions ci-dessus.
- **Traces** : instantanés détaillés réservés au propriétaire, verdicts applicables distingués des sans objet, méthodes omises pour budget signalées explicitement. La finalisation ne peut plus reconstruire les anciennes entrées après une purge.

## Vérifications exécutées

Dernière commande, après les corrections :

```sh
node --test journal/tests/change-engine.test.mjs journal/tests/change-plan.test.mjs journal/tests/change-strategies.test.mjs journal/tests/change-engine-review.test.mjs
```

**76 tests réussis, aucun échec**, dont les trois nouvelles reproductions indépendantes et les deux compléments des auteurs sur les finalisations. `git diff --check` réussi.

Mesures reproduites par les fixtures du moteur, estimation `ceil((système + schéma + données)/3)` :

| Fixture | Entrées nominales, trois appels | Réservation avec réécriture et seconde relecture |
| --- | ---: | ---: |
| Profil rempli, plan et consigne | 11 830 | 28 755 |
| Profil rempli, deux méthodes | 11 742 | 29 098 |
| Profil maximal, plan chargé | 13 155 | 28 885 |
| Profil maximal, méthodes omises faute de place | 12 646 | 28 586 |
| Profil maximal, agrégat aux plafonds | 13 159 | 28 889 |

Les fixtures habituelles restent sous la tolérance de 13 800 entrées ; toutes ces voies avec réécriture restent sous la réserve nominale de 30 000. Le cas maximal peut omettre entièrement les méthodes et interdire le nouvel exercice. Les reprises fournisseur ont une enveloppe distincte : ce rapport ne les assimile pas à un tour nominal de 30 000. Ces estimations ne sont ni un relevé de facturation ni une mesure de qualité clinique.

## Limites et versions relues

Cette relecture ne certifie pas la qualité de chaque sortie d’un modèle réel, l’efficacité clinique de l’accompagnement, une absence universelle de défaut, ni le résultat du déploiement. Les gardes lexicales sont conservatrices et la relecture sémantique reste nécessaire pour les formulations implicites. Le partage/export `care` et la cartographie font l’objet de relectures distinctes ; ils n’ont pas été réaudités intégralement ici.

Empreintes SHA-256 des sources principales à la fin de cette passe :

```text
4edbcc4a818a8eaa2745c8d8bf65a79fa018cbd76d983115b681be7df2e915f7  journal/src/change-engine.mjs
520ceb2fbadd590be1b7b0763eedec03e0eebac495fcffe1944e3f79b6dcedba  journal/src/change-plan.mjs
cde8af5ff6a18925d6808e2b7670ba3dd850cf2d3f80ec02f25e0d1e6ecf8d35  journal/src/change-strategies.mjs
24ae32bf22bf1f60126d867339b01e24b4a3d1afb03a5523aac7b02bee123e10  journal/src/ambre-engine.mjs
9c9fdb27b06e2557f93f274b4f0d88ff9b3e687c8b52874884e6c334106edf2e  journal/src/chat-backend.mjs
2d9bc70c2aa93eb07e873513abd567e0c57e372b13ad2d69d280e5a8267c2ff4  journal/src/change-backend.mjs
c869cad5c66f8928e19f88987a0403da90db1c4370707fba93e148df1cd932aa  journal/tests/change-engine-review.test.mjs
```
