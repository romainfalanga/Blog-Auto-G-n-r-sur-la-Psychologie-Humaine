# Relecture indépendante de l’intégration suivi / plan — 17 novembre 2026

Relecteur : `plan_tests17`, distinct de l’auteur `care17`. Périmètre : `care-backend.mjs`, `change-plan.mjs`, `change-backend.mjs`, leur migration et leurs contrats de partage, de préparation, de suppression et d’export. Aucun fichier de production n’a été modifié par ce relecteur. Les seuls nouveaux livrables sont ce rapport et `journal/tests/care-plan-review.test.mjs`.

## Défauts reproduits et corrections retestées

### 1. Transmission du premier lot non enregistrée pendant l’appel fournisseur

Références initiales : [care-backend.mjs](../journal/src/care-backend.mjs), fonction `processBrief`, appel fournisseur puis enregistrement des dépendances ; [change-backend.mjs](../journal/src/change-backend.mjs), décision de coupure dans `save`. Test : `the first provider call already counts as transmission when the member removes its source during generation`.

Le point reçoit bien les données choisies du plan, mais les références d’artefacts sont enregistrées seulement après le retour fournisseur et la publication du lot. Une suppression du même objectif pendant le premier appel ne trouve donc aucune trace de la transmission : `context_generation` reste à zéro et `continuity_reset` est absent. Le test utilise une suppression par la véritable route du plan, après avoir vérifié que l’objectif figure dans les entrées du fournisseur.

**Limite précise :** la publication du point est correctement bloquée et son contenu est vide. Aucune restitution du texte supprimé n’a été observée. L’écart concerne le contrat v3 selon lequel une suppression de contenu déjà transmis est reconnue comme telle, y compris pendant un appel, avec progression de génération. Les transmissions doivent être enregistrées sous garde de source, consentement et bail avant l’appel ; un état déjà périmé ne doit pas provoquer de transmission ou de référence fictive.

**Correction validée :** `care17` enregistre désormais les identifiants et révisions des entrées avant `generateText`, dans un lot gardé par bail, source, partage et séance, sans publier de texte. Le cas de suppression pendant le premier appel passe avec progression de génération et point vide. Les tests complémentaires de l’auteur, relancés indépendamment, prouvent qu’une garde perdue avant transmission ne produit ni appel ni références ; un échec fournisseur après transmission conserve seulement les métadonnées effectivement nécessaires.

### 2. Une correction strictement privée détruit un point partagé inchangé

Références : [change-plan.mjs](../journal/src/change-plan.mjs), révocation large des points liés au scope `plan`, ligne 110 ; [migration 0021](../journal/migrations/0021_change_plan.sql), trigger `care_revision_change_plans_update`, ligne 50. Test : `editing a private instruction does not invalidate an unchanged shared plan point`.

Une consigne privée est créée, puis un point `plan` est préparé à partir des seuls objectifs, étapes et essais autorisés. La consigne ne figure ni dans les entrées du fournisseur, ni dans les références. Corriger ensuite son texte révoque le point et fait répondre `404` à sa lecture, alors que les trois collections du sérialiseur partagé sont strictement identiques avant et après la correction. Le test correspond à `cutoff:false` : aucun autre moteur n’a reçu la consigne.

Deux causes se cumulent : l’invalidation révoque tous les points associés à une liaison comprenant `plan`, même sans dépendance touchée ; le trigger de révision partagée réagit à toute modification de l’agrégat, y compris les consignes et points privés. L’effet concret est la perte d’un point encore valable et sa régénération inutile, avec consommation éventuelle du quota de préparation. Aucune fuite du texte privé n’a été observée.

**Correction validée :** l’intégrateur a limité le trigger aux trois collections `goals`, `steps` et `trials`, et la révocation directe aux références réellement touchées. La correction d’une consigne jamais transmise laisse le point partagé prêt et inchangé. Une coupure justifiée par un contenu effectivement transmis conserve son invalidation globale prudente.

### 3. Export incomplet de l’historique conservé

Référence initiale : [care-backend.mjs](../journal/src/care-backend.mjs), fonction `exportData`, plafonds `LIMIT 1000` pour points/liens et `LIMIT 10000` pour accès. Test : `owner care export includes all retained link, point and access histories beyond the display limits`.

Le jeu comporte des historiques valides conservés, indépendants des plafonds d’affichage et de liens actifs : 1 002 liaisons, 1 001 points révoqués et 10 003 accès. L’export restitue respectivement **1 000, 1 000 et 10 000** enregistrements, sans curseur ni indicateur d’omission. Les contraintes courantes ne plafonnent pas le nombre total de liaisons révoquées, de points historiques ou d’accès. La limitation à 500 séances correspond, elle, au plafond de création des séances ; elle n’est pas à l’origine du cas démontré.

L’export doit parcourir les pages conservées ou exposer une suite explicite, tout en gardant le filtrage des textes révoqués/périmés et l’exclusion des secrets. Un export courant inférieur aux seuils ne démontre pas cette complétude.

**Correction validée :** l’export parcourt les historiques par pages SQL de 200 avec curseur et bornes de départ. Les 1 002 liaisons, 1 001 points et 10 003 accès du test sont tous présents. Les compteurs de couverture et la cohérence non transactionnelle sont explicités. Les contenus de points sont contrôlés par page puis revérifiés avant remise ; une invalidation d’une page antérieure fait échouer l’export avec `409 export_sources_changed`. Les tests complémentaires couvrent la suppression d’une ligne située derrière le curseur et la restitution des 500 séances autorisées.

### 4. Une attente après la dernière garde réouvrait la course de suppression

Référence : [care-backend.mjs](../journal/src/care-backend.mjs), fin de `exportData`. Test ajouté après la première correction d’export : `a deletion during the export metadata reads cannot return point text checked earlier in the same export`.

La première version corrigée vérifiait les points, puis attendait encore les lectures de `care_settings` et `professional_profile`. Le nouveau test supprime l’objectif par l’API pendant l’une de ces lectures : l’ancien texte apparaissait encore dans le JSON d’export, bien que le point ait été révoqué en base. Ce cas a été reproduit avant correction, avec le marqueur `OBJECTIF_CHOISI` effectivement présent dans la sortie.

**Correction validée :** toutes les métadonnées sont chargées avant l’ultime garde ; la fin du module ne contient plus d’attente après celle-ci. Le neuvième test passe sans modification de son assertion. L’appelant doit conserver cette propriété jusqu’à la réponse : l’assemblage global `/api/export` du Worker a été signalé à l’intégrateur, car une lecture de `change.exportData` effectuée après `care.exportData` réouvrirait la même fenêtre.

## Vérifications utiles sans défaut observé

Les tests existants `care.test.mjs` et `care-change.test.mjs` couvrent les liaisons sans scope `plan`, les listes blanches imbriquées, les identifiants étrangers, les lectures concurrentes avec révocation, les mutations de source à la publication, les exports de points périmés avant lecture et les cascades de suppression. Le code de migration n’ajoute pas `plan` aux scopes des liaisons existantes.

Les nouveaux cas passent aussi sur les points suivants :

- vingt-cinq objets du plan réellement transmis en deux lots donnent exactement vingt-cinq références, même si le fournisseur ne retient aucune citation ; les consignes, la mémoire et les préparations personnelles sont absentes des entrées ;
- une suppression d’un essai situé dans la page suivante, jamais envoyée, ne déclenche pas de coupure de continuité ; le point partiellement préparé ne peut ensuite publier ses anciennes entrées ;
- corriger par l’API un résultat transmis mais non cité retire immédiatement le point déjà prêt, son contenu et ses notes, y compris dans l’export ;
- une préparation personnelle peut être prévisualisée puis sauvegardée sans professionnel ; l’ajout volontaire d’une note de mémoire conserve sa provenance privée et une liaison `plan` créée ensuite ne la partage pas ; le propriétaire la retrouve dans son propre export du plan ;
- supprimer le compte propriétaire pendant l’appel de préparation retire les lignes de plan, reçus, références, liaisons, points, séances, accès et budgets concernés ; aucun point ne réapparaît et l’autre compte reste intact.

## Méthode et état

Les tests réutilisent le harnais SQLite du suivi, qui exécute toutes les migrations et de vraies transactions, et ajoutent des appels directs à `createChangeBackend`. Le fournisseur est simulé uniquement pour déclencher les courses et observer les entrées. Le Worker, l’interface et le moteur conversationnel en cours d’intégration ne sont pas validés par cette revue. Le test d’interface déjà présent dans la suite `care-change` n’est pas une recette supplémentaire d’interface.

Commande finale exécutée : `cd journal && node --test tests/care.test.mjs tests/care-change.test.mjs tests/care-plan-review.test.mjs tests/change-plan.test.mjs`.

État initial : 46 tests, 43 réussis et trois échecs ; la dernière course d’export a ensuite été démontrée par un neuvième cas. État final après les quatre corrections : **84 tests exécutés et réussis, aucun échec, aucun test ignoré**, dont les neuf nouveaux cas indépendants. Le lot comprend 52 cas de suivi/partage et les 32 cas de non-régression du backend du plan. Les tests ayant exposé les défauts sont conservés.

**Intégration des modules suivi / plan retenue après corrections et retest indépendant.** Aucun défaut bloquant ne reste ouvert dans les fonctions de ce périmètre. La branche globale d’export du Worker doit terminer par la garde du suivi ; sa recette et celles du moteur/interface restent du ressort de l’intégration. Les requêtes d’export sont bornées, mais le résultat est encore assemblé en mémoire : ce contrôle ne constitue pas une mesure de charge de très grands historiques de textes sur D1 distant. Aucun commit, déploiement ou appel fournisseur réel n’a été effectué par le relecteur.
