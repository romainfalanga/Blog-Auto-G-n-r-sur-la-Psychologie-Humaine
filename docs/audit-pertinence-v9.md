# Audit et optimisation de la pertinence — V9

Audit du 4 novembre 2026, sur la base V8 `c0d305c7c81cf4ae6c81c5dfc750c5fdec7a60ad`.

## Périmètre et méthode

Le périmètre couvre les chemins actifs du Worker, de l’interface et du traitement Python : authentification et isolation, déclarations personnelles, proches, événements, mémoire, lectures et retours, portrait et carte, conversation et propositions d’Ambre, personnage et épisodes, ordonnancement et quotas, abonnement, export et suppression. Les générateurs du catalogue, de la navigation et des exemples, les migrations et les workflows ont également été examinés. Les outils historiques de migration ne constituent pas un second moteur actif.

Les points d’entrée sont `journal/src/worker.mjs`, `personal-backend.mjs`, `chat-backend.mjs` et `ui.mjs`. Les contrats de données se trouvent dans `private-context.mjs`, `portrait-language.mjs`, `psychology-map.mjs` et les migrations ; les moteurs dans `portrait-refresh.mjs`, `psychology-guide.mjs`, `reading-intent.mjs`, `recommendations.mjs` et `scripts/generate_personal_articles.py`. `scripts/run_journals.py` orchestre les traitements longs. Le catalogue est généré depuis les sources éditoriales du dépôt.

La revue s’appuie sur les appels réels entre modules, des reproductions synthétiques des défauts avant correction, les tests existants et de nouvelles régressions. Une revue croisée indépendante porte sur les contrats partagés. Aucun ancien portrait ni texte généré n’est traité comme une nouvelle preuve personnelle. Les textes des conversations réelles ne sont pas utilisés comme jeu de test.

## Fonctionnement conservé et plan retenu

1. Fiabiliser l’entrée : préserver les ajouts, les corrections récentes et l’identité structurée actuelle.
2. Fiabiliser le raisonnement : transmettre les contre-indices au vérificateur, conserver les alternatives, annoncer la couverture partielle.
3. Fiabiliser la recherche : séparer besoin, négation, intérêt et choix de lecture ; éviter de compter plusieurs fois le même récit.
4. Fiabiliser les actions : versions concurrentes, navigation, budget, reprise et confirmation de la résiliation.
5. Intégrer, tester les parcours, migrer puis publier ; consigner les limites restantes.

Une action personnelle modifie ses sources et leur version ; le portrait est reconstruit depuis ces sources. La synthèse et la vérification sémantique sont deux appels distincts. Les citations et révisions sont validées par le code ; le second appel vérifie le sens des affirmations, y compris les contre-indices. Les interprétations facultatives rejetées sont retirées, une synthèse rejetée n’est pas publiée. Le chat exploite un instantané authentifié commun et ne peut modifier une fiche qu’après confirmation d’une proposition. Le texte courant du chat peut orienter une recherche immédiate sans devenir automatiquement une déclaration enregistrée.

L’influence d’un mécanisme reste qualitative et contextualisée, distincte de la solidité de ses preuves. Le catalogue sert à explorer des explications concurrentes ; il ne constitue ni un diagnostic ni un inventaire exhaustif de la psychologie humaine. Aucun score universel de personnalité n’est calculé. L’accès gratuit de test demandé par Romain reste activé, avec les quotas de 100 messages et 7 histoires par semaine.

## Défauts observés, corrections et preuves

P1 : conservation des données, exactitude du contexte ou action sensible. P2 : pertinence, performance et fiabilité d’usage.

| Priorité | Défaut reproduit | Correction retenue | Vérification |
|---|---|---|---|
| P1 | Un dernier échange dépassant le budget d’historique fait disparaître tous les échanges antérieurs du prompt | Extrait borné du dernier échange puis poursuite de la sélection ; métadonnées de couverture réelles | Conversation longue et historique partiel |
| P1 | La correction récente en fin d’une mémoire longue n’atteint pas Ambre | Fenêtre pertinente et fin de mémoire réservée, omissions explicites | Correction en fin de mémoire et budget global |
| P1 | Une proposition d’ajout peut remplacer un long texte existant ; une consigne mixte peut viser la mauvaise fiche | Modes `append`/`set`, intention ciblée par champ, fragment d’ajout seul, fusion sur la version courante avec contrôle des limites | Longue relation conservée, négation locale, consignes mixtes, conflit, dépassement et ancienne proposition |
| P1 | Le vérificateur ne voit que les sources citées : une correction non citée peut lui échapper | Contre-contexte original distinct, priorisation des corrections, couverture et budget explicites | Ancienne conclusion réfutée par une correction non citée ; refus sans appel si correction indispensable trop longue |
| P1 | Certaines étiquettes du portrait échappent au contrôle des accords | Même contrôle pour les intitulés d’axes et préférences de lecture | Accords incohérents refusés, citations originales préservées |
| P1 | Une carte V3 absente ou `null` peut être normalisée silencieusement | Tableau explicite exigé dans Worker et Python ; compatibilité des anciens formats conservée | Contrats V3/anciens portraits |
| P1 | La recette longue reproduit un rejet de structure de carte ; les consignes natives omettent des contraintes présentes dans le traitement Python | Contrats alignés pour piste contredite, influence inconnue, preuve et cible d’interaction ; diagnostics `contract.*` fermés, sans texte rejeté | Rejet prudent conservé, cause catégorisée, puis portrait réel validé avec contre-indice et couverture complète |
| P1 | Un retour initial envoyé depuis un deuxième onglet peut écraser le premier | Révision 0 obligatoire pour création, comparaison atomique à l’écriture et brouillon conservé | Deux créations concurrentes et mise à jour obsolète |
| P1 | Un personnage archivé reste modifiable par l’API de retour | Vérification du personnage actif dans la requête atomique | Refus avant et pendant la transaction |
| P1 | Une clé Stripe réutilisée après reprise peut rejouer une ancienne réponse de résiliation | Intention durable distincte du paiement, reprise réseau idempotente et vérification canonique après écriture | Reprise après résiliation, réponse rejouée non confirmée, appels concurrents simulés |
| P1 | Le proxy de traitement privé ne porte pas les mêmes contraintes de conservation que le chat | Paramètres fournisseur `data_collection: deny` et `zdr: true` également sur les traitements privés | Payload du proxy contrôlé ; appel réel vérifié lors de la recette |
| P2 | « Je n’arrive pas à poser mes limites » est traité comme une négation du besoin | Incapacité déclarée distinguée d’un démenti, sans transformer une ambiguïté en affirmation | Limites proposées ; « je ne rumine plus » reste négatif |
| P2 | L’expansion d’une demande négative réintroduit des sujets refusés | Analyse par proposition, expansion des clauses positives uniquement | Négations, doubles négations et phrases mixtes |
| P2 | Une exclusion exprimée dans la question actuelle n’influence pas assez le classement | Signal de requête local, exclusion explicite respectée même avec des likes passés | Exclusion actuelle et absence de mutation du profil |
| P2 | Un ancien sujet à éviter reste bloqué malgré un souhait de lecture ultérieur | Seul un choix de lecture explicite, plus récent et non ambigu, peut le remplacer | Préférence récente admise ; objectif général sans intention de lecture insuffisant |
| P2 | Copier la même déclaration dans quatre collections augmente artificiellement son poids | Regroupement des déclarations identiques, contribution maximale du groupe | Score réel passé de 0,3518 à 0,8235 avant correction ; désormais 0,3518 dans les deux cas |
| P2 | L’importance saisie d’un événement n’atteint pas la recommandation | Métadonnée transmise ; modulation éditoriale limitée entre 0,8 et 1,2 | Importance 1/5 modifie la priorité sans créer une preuve supplémentaire |
| P2 | Des domaines tardifs du guide, notamment sommeil/corps et fonctions exécutives, sont évincés par l’ordre du catalogue | Deux places réservées à la question, hypothèses actives limitées à la moitié, domaines triés par pertinence | Questions sommeil et attention avec les 16 domaines présents |
| P2 | Une piste contredite peut encore recevoir un bonus dans le guide | Bonus réservé aux pistes actives ; statut mixte conservé | Piste contredite sans bonus, alternatives disponibles |
| P2 | La recherche du chat recharge une seconde fois le contexte SQL | Réutilisation de l’instantané authentifié de la requête | Compteur SQL stable et instantané inchangé |
| P2 | Une réponse tardive du chat peut arrêter le suivi de la nouvelle page après navigation | Garde d’instance avant mutation et gestion des minuteries | Navigation aller-retour et réponse retardée |
| P2 | Un traitement impossible pour son budget peut être relancé sans progrès ; certains codes sont mal classés | Arrêt immédiat sur budget, reprise bornée après au moins 24 h, statut fidèle, code de transport conservé, bascule batch pour contre-contexte trop grand | Aucun appel fournisseur au refus ; absence de relance immédiate, reprise après renouvellement, classification testée |

La migration `0013_billing_cancellation_attempts.sql` ajoute uniquement les trois champs d’intention de résiliation. Elle ne change ni les droits actuels ni la facturation. Aucun paiement réel n’est nécessaire à la recette.

## Contrôles et limites de la mesure

Les vérifications couvrent notamment les versions et citations, les corrections, les exclusions de fiction, les quotas et réservations, les baux expirés, l’idempotence, les fuites entre comptes, les exports/suppressions, les parcours UI et les anciennes migrations. Aucune autre défaillance reproductible n’a été trouvée dans les chemins inspectés ; cela ne constitue pas une preuve formelle d’absence de défaut.

Le benchmark éditorial existant contient 30 jugements synthétiques peu denses. Ses résultats restent identiques après ces corrections : NDCG@6 ≈ 0,7886, précision@6 ≈ 0,2389. Les cas ciblés démontrent les corrections de négation, de copie et d’exclusion ; le benchmark ne mesure pas une amélioration clinique ou une satisfaction réelle. Les poids du moteur n’ont pas été ajustés pour gagner artificiellement sur ces 30 cas.

Le moteur de recherche reste lexical, complété par le guide conceptuel ; les paraphrases éloignées peuvent être moins bien retrouvées. Un index sémantique demanderait un corpus d’évaluation indépendant et des coûts mesurés avant remplacement. La vérification par un modèle reste faillible, même avec des citations exactes. Les contextes longs sont bornés, leurs omissions sont annoncées et le traitement batch conserve des budgets finis. Le nom Ambre est une identité de produit ; il ne signifie pas une conscience humaine ou une qualification professionnelle.

## Validation et publication

Validation finale du code : **404 tests automatisés passent** (290 JavaScript du journal, 16 du site public et 98 Python). Les deux générateurs navigation/catalogue sont à jour ; les 17 modules Worker et 8 scripts Python passent le contrôle de syntaxe. Construction Hugo et absence d’erreur de patch vérifiées. Les corrections ont fait l’objet d’une revue croisée et de régressions sur les défauts trouvés pendant cette revue.

Publication initiale : commit `13b3cc12d8856250d6f538b8a9022e615e7689da`, workflows de vérification et de génération réussis ; Netlify `6aeb4f01e8914fb33c3784a8`, état ready, même commit. Le correctif complémentaire de contrat et cette preuve sont enregistrés dans le commit de suivi. Worker final : production `8a06b896f08f4ccfa7f7bcb66953ac15`, recette `1509b62f259640379e8e698075a6ab94`. Migration 0013 appliquée aux deux bases, aucune violation de clé étrangère. Secret fournisseur conservé et accès gratuit toujours actif.

Recette réelle, sur un compte synthétique ensuite supprimé :

- Mémoire de 8 296 caractères avec correction de la situation professionnelle en fin de texte ; relation de 4 305 caractères.
- Réponse d’Ambre obtenue en 13,1 s, correction récente correctement distinguée des anciennes inquiétudes, aucune écriture automatique. Renvoi idempotent de la demande et quota de 99 messages restants vérifiés.
- Proposition `append` contenant uniquement le fragment. Après confirmation, relation de 4 404 caractères avec l’intégralité de l’ancien texte ; renvoi de la confirmation sans doublon ni nouvelle révision.
- La première recette de portrait a reproduit des refus de structure de cartographie. Le diagnostic a été ajouté et les consignes natives alignées sur le contrat Python existant. Après cet alignement, une tentative de synthèse et sa vérification sémantique ont réussi : portrait V3 courant, genre déclaré correct, correction récente citée, contre-indice présent, aucune source pending. Les paramètres de conservation du proxy privé ont donc aussi été exercés avec le fournisseur réel.
- Compte synthétique supprimé par l’API. Contrôle final : un utilisateur réel, cinq proches, aucun événement ni traitement inachevé, portrait courant cohérent avec le genre déclaré et la version des sources. Les échanges réels survenus pendant l’audit n’ont pas été lus ni modifiés.

La facturation reste désactivée pendant l’accès gratuit ; les scénarios de résiliation Stripe ont été testés par simulations du protocole, sans paiement ni abonnement réel.
