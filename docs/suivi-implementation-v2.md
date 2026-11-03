# Reprise de l'implémentation V2 — 3 novembre 2026

Autorisation : Romain a validé le [plan](plan-portrait-et-trois-personnages.md) par « go ». Base Git : `cc6ce8668e026157d744b93392d3d2140d6007f0`.

## Suivi

| Exigences | Travail | État |
|---|---|---|
| R1–R2 | Navigation des cartes et barre de recherche épurée | Implémenté ; recherche, Hugo et navigation réelle validés. |
| R3–R5 | Proches, ligne de vie, sources et portrait | Implémenté ; API, portrait IA réel et actualisation après retour validés. |
| R6–R8 | Trois personnages, continuités et sept crédits hebdomadaires | Implémenté ; récit publié, réservation/restitution/débit et idempotence vérifiés en réel. Réserves éditoriales ci-dessous. |
| R9 | Retours, classement, génération et provenance | Implémenté ; recommandations sans like, retours, sources et chaîne de publication vérifiés. Pas de gain général de pertinence démontré. |
| R10 | Contrats, migration, revue et preuves | Publié ; migration, revue indépendante, CI et contrôles de production validés. |

## Contrats et organisation

Le [contrat V2](contrats-implementation-v2.md) fixe les noms de champs partagés entre Worker, interface et runner. Backend/migrations, interface et génération/recommandations ont des responsables d'écriture distincts. L'intégration et le déploiement restent coordonnés centralement.

La recette utilise une base Cloudflare distincte avec des comptes et histoires synthétiques. Aucune donnée personnelle de production n'y est copiée. Les contrôles incluent une base contenant d’anciennes histoires et leurs retours avant migration.

L'accès Stripe est actuellement non configuré dans les bindings de production ; le chantier conserve les droits d'abonnement existants mais n'effectue aucun achat et n'affirme pas que l'encaissement est activé.

Les versions et limites de recette sont consignées ci-dessous.

## Vérifications avant publication

- 93 tests JS du journal, 16 tests de recherche, 35 tests Python : tous passent. Hugo compile 1 076 pages ; les sources des articles publics restent intactes.
- Dix contrôles API passent sur une D1 isolée avec uniquement des données synthétiques : authentification, migration des archives, crédits historiques, CRUD, conflits concurrents, dates partielles, liens privés, suppression d’un proche sans perte d’événement, export et recommandations sans like.
- Navigateur réel : recherche « rumination », fermeture par Échap sans effacer la requête ni déplacer la carte (1 062,09375 px avant/après), ouverture de Sophie ; recherche « peur de décevoir » puis clic direct avec suggestions ouvertes : page Thomas ouverte.
- Revue indépendante : échec tardif d’un ancien bail et facettes périmées corrigés ; reproductions concurrentes repassées.
- Production : point de restauration D1 enregistré avant opération ; migrations 0005/0006 appliquées sans erreur (50 instructions). Contrôle des clés étrangères vide et compte préexistant conservé. Génération temporairement suspendue pendant le changement de protocole.

Les formulaires privés sont vérifiés par tests d’interface et appels authentifiés au serveur ; leur parcours connecté n’a pas encore été exercé dans le navigateur de cette recette. L’évaluation du classement est synthétique et ne démontre pas de gain général sur la cohorte ayant des likes. Voir la note algorithmique.

## Publication et contrôles réels

- Implémentation Git : `095d15093b77d3b304a3ee66d13a55e8acf08ed5` ; dépendance de validation YAML ajoutée dans `72e10286582ae30c4ea93df26c4abf7da244870a` après le premier passage CI. L’environnement local possédait déjà YAML ; le runner GitHub neuf a révélé l’absence de déclaration de cette dépendance.
- Worker actif : déploiement `2feeb057aa614e7d8174380e76117543`, modules V2 et bindings existants conservés. Génération réactivée.
- Netlify : déploiement initial `6ae9e7ae797054c68ea33f48`, puis `6ae9e83e2f34a036b8ba4b80` sur le commit `72e1028`, tous deux `ready`, sur le domaine existant.
- [CI finale](https://github.com/romainfalanga/Blog-Auto-G-n-r-sur-la-Psychologie-Humaine/actions/runs/41648990943) : succès, 144 tests au total. La première CI a échoué sur l’import YAML ; la correction est vérifiée sur un runner neuf.
- [Génération V2 réelle](https://github.com/romainfalanga/Blog-Auto-G-n-r-sur-la-Psychologie-Humaine/actions/runs/41648627110) : succès. Un compte fictif avec une relation décrite, un événement positif et zéro like a obtenu un portrait `ready` puis trois personnages. Trois appels pour le portrait, un pour les fiches, sans erreur ni nouvelle tentative. Compteurs fournisseur du seul compte de recette : 8 762 tokens d’entrée, 4 167 de sortie ; coût enregistré `0.0105245` par la passerelle. La préparation conserve les sept crédits (zéro consommé/réservé).
- Navigateur en production : « rumination » → suggestion « rumination mentale » → « Découvrir ses histoires » ouvre `/personnages/sophie/`. Le menu et la page `/mes-personnages/` affichent le nouveau nom et l’offre hebdomadaire.
- Catalogue D1 : 231 articles publics resynchronisés depuis le même constructeur que les métadonnées. Contrôle des clés étrangères sans anomalie après nettoyage.
- Compte fictif de production et son droit temporaire supprimés. Retour à un seul compte, celui qui préexistait. Aucun achat effectué. Aucun récit de test demandé ni crédit consommé.

## Reprise du test réel explicitement autorisée

Le premier contrôle automatique d’approbation avait rejeté une demande d’histoire réelle susceptible d’engager un coût fournisseur. Romain a ensuite explicitement autorisé ce test par « go ». La recette utilise exclusivement des comptes fictifs, sans achat ni modification de l’abonnement du compte préexistant.

Le premier parcours autorisé a mis en évidence une réponse de planification tronquée par la limite de sortie et un brouillon refusé par les contrôles de structure. Les commits `cc0d44a23576b5cb3a5535fdd0b5178ef0931f7c` et `446c38eb4b795d34aa7adbe10217c3de681c2f18` augmentent la marge des étapes structurées, normalisent les compteurs de la passerelle et permettent une seule réparation du brouillon avant la relecture. Les budgets cumulés, critères de publication et limites de tentatives restent en place. Pour cette seule demande fictive, le délai de reprise après le deuxième échec a été avancé ; aucun compteur d’appels, de tentatives ou de crédits n’a été remis à zéro.

La [génération réelle](https://github.com/romainfalanga/Blog-Auto-G-n-r-sur-la-Psychologie-Humaine/actions/runs/41656560972) a ensuite publié une histoire de 1 372 mots. Les contrôles authentifiés valident un seul crédit consommé (sept → six), la répétition idempotente de la demande, les trois continuités séparées, les citations du dénouement, sept recommandations sans aucun like et la prise en compte des retours. Un retour incrémente la version des sources et remet le portrait en attente, sans changer les trois identités ni le quota. Le compte fictif et son droit temporaire ont été supprimés.

La lecture éditoriale de ce premier résultat n'a cependant pas validé sa qualité : un bloc `JSON_RÉSUMÉ` et du vocabulaire technique subsistaient, et les personnages reprenaient trop directement une scène personnelle fictive. Cela a motivé la correction de la frontière entre analyse personnelle et fiction et une nouvelle recette réelle.

## Correction de la fiction et seconde recette autorisée

Le commit `eb47b67ccf79c9107051a93272224ae3459ae7a5` sépare les originaux personnels du brief de fiction. Une étape analytique produit des orientations à valeurs contrôlées ; le planificateur, le concepteur de personnages et le rédacteur ne reçoivent plus la prose autobiographique. Les souhaits libres restent contrôlés analytiquement, y compris avant l'enregistrement des nouvelles identités. Les identités existantes et leurs archives ne sont pas réécrites.

Le garde de format V2 remplace les anciennes consignes contradictoires. Les marqueurs techniques, certains commentaires de fabrication, le jargon anglais observé et la narration au « je » hors dialogue sont contrôlés avant publication. Un garde supplémentaire refuse les reprises textuelles personnelles reconnaissables, y compris dans les remplacements du relecteur. Il ne constitue pas une garantie absolue contre toute paraphrase.

La revue indépendante a reproduit puis vérifié la fermeture de deux défauts : préférence libre non contrôlée avant création d'identité et réinjection textuelle d'une source par un remplacement éditorial. Les 55 tests Python passent ; la [CI du correctif](https://github.com/romainfalanga/Blog-Auto-G-n-r-sur-la-Psychologie-Humaine/actions/runs/41660769109) est verte, soit 164 tests avec le journal JS et la recherche.

Un second compte entièrement fictif reprend les mêmes deux informations, sans like. Son portrait est prêt en trois appels, ses trois personnages en deux appels, chacun dès la première tentative. Les fiches sont relues indépendamment : Maëline vit dans un foyer partagé, Idris crée entre amis et Soraya anime un collectif d'espaces verts. Les milieux, relations et ressources diffèrent ; aucune reprise de la scène du dessin ni du père du profil de recette n'est constatée dans ces fiches. Les sept crédits restent disponibles après leur préparation.

La première tentative de son épisode est refusée au contrôle de cohérence ; la deuxième passe la relecture mais échoue à la validation du résumé/delta. Aucune de ces tentatives ne publie ni ne consomme de crédit. Le commit `1ef3e5d1ce8db9e81ff8091deb6f4ec6e453110e` ajoute une seule réparation de cette réponse structurée, sans réécrire le récit déjà validé. La cible du résumé devient 800–1 200 caractères avec la même borne dure de 1 800 ; les preuves restent des sous-chaînes exactes et la résolution doit toujours citer le dernier tiers. Les erreurs fournisseur ne sont pas absorbées. La [CI correspondante](https://github.com/romainfalanga/Blog-Auto-G-n-r-sur-la-Psychologie-Humaine/actions/runs/41662588656) passe 168 tests (59 Python, 93 journal JS, 16 recherche).

Pendant cette recette, seul le délai de reprise du compte fictif est ajusté pour permettre les vérifications et attendre la publication du correctif. Les plafonds de trois tentatives, vingt appels et le budget de tokens restent inchangés ; les compteurs cumulés sont conservés, de même que la réservation unique.

La troisième tentative est refusée après application des remplacements éditoriaux (`correction/validation_error`). La demande atteint sa limite et se clôt sans publication ; sa réservation est libérée et les sept crédits sont de nouveau disponibles. Ses seize appels restent comptabilisés. Le commit `851347a1cf5a52d28a9a66d210be8f5261fe2d4d` traite ce chemin : une seule réparation post-édition peut rétablir la conformité du corps, puis une relecture reste obligatoire. Deux tours de remplacements au maximum sont permis ; aucun défaut restant n'est accepté pour publier. Les diagnostics du runner sont limités à une liste blanche de codes, sans contenu personnel ni réponse du fournisseur. Revue indépendante et [CI](https://github.com/romainfalanga/Blog-Auto-G-n-r-sur-la-Psychologie-Humaine/actions/runs/41664102728) validées : 175 tests (66 Python, 93 journal JS, 16 recherche).

Une nouvelle demande synthétique utilise les mêmes personnages et les mêmes sources après clôture de la précédente. Elle ne remet aucun compteur d'une ancienne demande à zéro et ne change pas le canon, qui n'a encore aucun épisode publié.

### Résultat fonctionnel final

La [génération finale](https://github.com/romainfalanga/Blog-Auto-G-n-r-sur-la-Psychologie-Humaine/actions/runs/41664102716) réussit dès la première tentative de cette nouvelle demande, en sept appels. L'épisode de Maëline comprend 1 660 mots, huit sections et trois questions. Les appels authentifiés confirment :

- un seul crédit consommé, six restants et aucune réservation résiduelle ; répéter la demande retourne la même histoire sans second débit ;
- seul le canon de Maëline avance à l'épisode 1 ; les deux autres sagas restent à zéro ;
- chaque citation du delta existe dans le récit et celle du dénouement appartient au dernier tiers ;
- sept recommandations sont proposées avec zéro like ;
- un retour complet est enregistré et relu, la version des sources passe de 3 à 4, le portrait repart en attente ; les trois identités et le quota restent inchangés ;
- le compte fictif et son droit temporaire sont supprimés. La base revient au seul compte préexistant, zéro compte de recette restant, contrôle des clés étrangères sans anomalie.

Métriques de cette dernière histoire : 23 921 tokens d'entrée, 8 262 de sortie, champ fournisseur `observed_cost=0.02192825`. La demande précédemment refusée conserve ses seize appels et `observed_cost=0.04842905` ; ces coûts d'essais ne sont pas masqués par la réussite finale. Aucun achat d'abonnement n'a été effectué.

### Lecture indépendante et dernier garde local

La lecture intégrale confirme la troisième personne, l'identité et les relations fictives cohérentes dans leur ensemble, l'absence de JSON et de jargon anglais, et l'absence de reprise de l'événement du dessin ou du père du profil fictif. L'exercice reste facultatif et le récit n'est pas présenté comme le vécu du lecteur.

Elle relève toutefois des échos de fabrication dans deux titres et une phrase (« scène incarnée », « explications naturelles des pratiques », « aux noms français »). Le dernier ajustement les ajoute au refus local et reformule la consigne de rédaction qui les avait favorisés. Les 67 tests Python passent, avec un test de ces trois reprises et un titre narratif témoin accepté. Ce dernier ajustement est vérifié localement, sans nouveau cycle payant.

Des réserves littéraires subsistent dans l'échantillon : répétitions pédagogiques, quelques formulations trop générales, accord fautif dans une question et petits raccords matériels (pied nu/semelle, appel sortant/entrant). La recette démontre le parcours fonctionnel et les contrôles de débit ; elle ne valide pas entièrement la qualité narrative ni sa constance pour tous les profils. Le texte brut de recette n'est donc pas présenté comme un exemple éditorial approuvé.

## Limites de la vérification

L’encaissement reste désactivé (`billing:false`) faute de bindings Stripe. Le test complet des formulaires privés dans un navigateur connecté reste à réaliser ; les formulaires sont couverts par tests d'interface et appels authentifiés. La qualité littéraire sur plusieurs profils et de longues sagas, ainsi qu'une amélioration générale du classement, restent à évaluer. Le portrait, les trois fiches, la publication, les crédits et les retours ont été vérifiés par les API authentifiées et la chaîne Actions/OpenRouter réelle.
