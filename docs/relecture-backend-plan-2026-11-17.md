# Relecture indépendante du backend du plan — 17 novembre 2026

Relecteur : agent `plan_tests17`, distinct de l’auteur des modules. Périmètre confié : `journal/src/change-plan.mjs`, `journal/src/change-backend.mjs` et la migration `0021_change_plan.sql`, comparés au plan v3 et à ses deux cycles de critique. Le relecteur a écrit uniquement `journal/tests/change-plan.test.mjs` et ce rapport. Les corrections de production appartiennent à l’intégrateur.

## Méthode et limites de la preuve

Les routes sont appelées directement via `createChangeBackend`, avec des requêtes HTTP et deux comptes membres ainsi qu’un compte professionnel. L’adaptateur `SQLiteD1` utilise `node:sqlite`, les clés étrangères et de vraies transactions SQL `BEGIN`/`COMMIT`/`ROLLBACK`. Chaque cas charge la chaîne complète des migrations disponibles, de `0001` à `0021`, y compris `0019` et `0020` déjà déployées. Les comparaisons de révision exécutent le SQL réel ; aucun faux stockage JavaScript ne simule le CAS.

Les courses contrôlées placent deux écritures au même instantané, ou font gagner une mutation concurrente juste avant la transaction. Une erreur volontaire dans l’insertion du reçu vérifie le rollback du document et des révisions de partage. Les suppressions sont testées avec des copies de réponse, des traces, des brouillons, un tour en cours, une mémoire, une tâche mémoire et un bilan existants.

Ce contrôle porte sur le backend local. Il ne prouve pas le routage HTTP du Worker, l’authentification de bout en bout, une transaction D1 distante, la publication conversationnelle complète, la sélection des stratégies, l’interface ou l’utilité clinique. Les intégrations moteur et partage font l’objet des missions séparées `engine17` et `care17`. Aucun appel fournisseur, commit, push ou déploiement n’a été effectué par ce relecteur.

## Couverture

La suite comporte 32 cas indépendants, couvrant notamment :

- création manuelle séparée d’une proposition ; adoption explicite des objectifs, étapes, essais et résultats ; édition au moment du choix ; cartes refusées, expirées et déjà résolues ; retrait du texte des cartes résolues ; suppression d’une carte sans supprimer l’objet adopté séparément ;
- rejeu après correction, suppression et purge ; clé utilisée avec un autre contenu ou une autre route ; même clé indépendante entre propriétaires ; deux clics simultanés ; réponse courante sans réapplication du texte initial ; registre sans payload ;
- CAS concurrent, rollback transactionnel, refus de publication avec un instantané ancien et impossibilité pour le perdant d’effacer les données du gagnant ;
- refus du rôle professionnel sur les routes privées ; isolation des identifiants directs, cibles de proposition, références de préparation, séances et relations entre objets ; cascades après suppression du compte ;
- trois objectifs actifs ; plafonds de 20 objectifs, 60 étapes, 200 essais, 40 propositions encore proposées, 20 préparations et 10 consignes ; longueurs de texte, absence de troncature silencieuse et plafond global de 256 Kio vérifié en octets UTF-8 ;
- pagination de 50 objets et parcours complet des collections ; export des 200 essais, des 60 étapes et de toutes les cartes conservées, y compris au-delà d’une première page ; maintien d’un export complet après refus pour stockage saturé ;
- résultats sans date, « non essayé », résultat pénible, dates civiles invalides ; maintien des observations précédentes et des instantanés d’action lorsqu’une étape est corrigée puis répétée ;
- arrêt depuis un écran périmé, reprise CAS après collision, réactivation volontaire sous révision, maintien des compteurs et absence de réautorisation d’un essai individuellement arrêté ;
- jour local Europe/Paris à la frontière du passage à l’heure d’été ; sept jours UTC complets ; prochain message après adoption ; absence de relance sur deux tours consécutifs, de relance hors sujet ou de troisième relance ; priorité de l’objectif à date égale ;
- suppression de contenu transmis pendant génération : copies et traces supprimées, bail annulé, crédit libéré, mémoire et brouillons retirés, génération augmentée et ancien CAS refusé ; messages originaux distingués dans la réponse ; refus et vigilance conservés sous forme de booléens sans ancien texte ; autre compte intact ;
- conservation de la continuité lors des modifications de priorité, statut, date prévue et date de retour, ainsi que pour une formulation jamais transmise ; coupure lors de la correction d’un texte effectivement référencé ;
- préparation personnelle sans lien professionnel, prévisualisation non enregistrée, résultats sans date présentés explicitement, omissions déclarées, notes mémoire séparées et limitées aux rubriques permises, document privé modifiable et invalidé avec ses sources ; provenance `memory_unverified` d’un ajout volontaire ;
- sérialisation partagée par liste blanche récursive : objets adoptés et relations autorisées uniquement ; exclusion des consignes, préparations, propositions, citations et objets malformés imbriqués dans des champs normalement scalaires ;
- helpers de suspension, d’effacement de conversation et d’effacement explicite de mémoire : garde composable, révision de mémoire propre, isolation, progression de l’epoch/génération et conservation des arrêts de retour.

## Constats et corrections

Trois points identifiés à la lecture ont été corrigés par l’intégrateur avant la validation des premiers tests :

1. **Liste blanche récursive du partage.** Copier une valeur autorisée sans vérifier son type permettait à un objet imbriqué de transporter des champs privés. Le sérialiseur refuse désormais ces objets et ne transmet que les types scalaires attendus. Le test adverse conserve le texte public légitime et vérifie l’absence des marqueurs privés.
2. **Ordre des relances.** À date égale, le choix utilisait seulement l’ancienneté. Il tient désormais compte de la priorité de l’objectif avant l’ancienneté. Le test place volontairement l’essai de faible priorité avant l’autre.
3. **Provenance mémoire du point personnel.** L’enregistrement volontaire conserve désormais `memory_origin: "memory_unverified"`, en plus de la révision de mémoire ; le point reste privé et exclu du sérialiseur partagé.

Deux défauts supplémentaires ont été reproduits, transmis à l’intégrateur puis corrigés :

4. **Perte d’une observation par changement administratif.** Un résultat créé via l’API avec texte et effet pénible acceptait `status: "cancelled"` puis effaçait ces champs. Le serveur refuse désormais la transition d’un résultat rapporté vers un autre statut ; les observations restent conservées. Leur retrait explicite reste possible par suppression de l’essai. Le test passe sans être assoupli.
5. **Coupure de mémoire sans changement de contenu.** L’adoption d’une proposition reprenant exactement un résultat déjà accepté et transmis augmentait la génération et effaçait la mémoire. La comparaison porte désormais sur les champs de contenu avant mise à jour des métadonnées. Le test vérifie l’adoption effective, la génération inchangée et la mémoire conservée.

L’écart documentaire concernant la version de schéma est également corrigé : `schema_version: 1` figure dans le document persistant, les vues paginées et l’export. Des assertions ajoutées aux cas existants vérifient cette présence et son maintien après effacement ; la révision de concurrence reste distincte.

## Compatibilité des helpers de suppression

La relecture finale de `clearChatChangeState` et `clearChangeMemoryState` ne relève pas de régression de suppression ou de garde dans le périmètre testé. Le premier conserve les objets adoptés lorsqu’ils ne sont pas inclus dans l’effacement, retire les propositions issues du chat et les citations, invalide les préparations issues de mémoire, supprime les références d’artefacts et fait progresser epoch et génération. Le second vérifie la révision de mémoire attendue dans la garde SQL, retire la mémoire et ses tâches, invalide les préparations qui l’utilisaient, augmente la génération et conserve les arrêts de retour décidés séparément. L’effacement explicite de mémoire peut retirer les indicateurs de refus/vigilance ; une suppression ordinaire de contenu transmis les conserve.

Ces helpers renvoient un lot à composer avec les autres suppressions du parcours : ils ne prétendent pas effacer seuls toute la conversation. Le test de garde ajoute une suppression du côté appelant au même lot et prouve qu’un CAS perdant ne touche ni celle-ci, ni les reçus, ni les dépendances. Les lectures et exports du partage restent soumis à la vérification de source du module `care`, couverte par sa revue distincte.

## État de la validation

Commande : `cd journal && node --test tests/change-plan.test.mjs`.

Passage complet final après les corrections et les assertions de version de schéma : **32 tests exécutés, 32 réussis, aucun échec, aucun test ignoré**, sous Node `v24.22.1`. Les deux tests ayant reproduit les défauts 4 et 5 sont conservés et passent. Le contrôle de format des deux livrables ne relève pas d’espace fautif.

**Périmètre backend local retenu après correction et retest indépendant.** Aucun défaut bloquant ne reste ouvert dans cette revue. Cette acceptation ne remplace pas les vérifications d’intégration, la recette distante et la relecture des réponses prévues par le plan.
