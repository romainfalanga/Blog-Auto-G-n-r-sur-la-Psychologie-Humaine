# Interface du plan personnel — 17 novembre 2026

État : implémentation locale livrée pour intégration et recette navigateur. Le plan de référence est la version 3 approuvée de `docs/plan-changement-ambre-2026-11-17.md`. Cette livraison ne constitue ni un déploiement ni une validation clinique.

## Fichiers et raccords

- `journal/src/change-interface.mjs` exporte `CHANGE_INTERFACE_SCRIPT` et `CHANGE_INTERFACE_STYLE`. Le shell les importe et les interpole ; sa modification relève de l’intégration principale.
- `renderChangePlan(host, current)` est appelé par `renderCareMember` dans Mon suivi.
- `journal/src/chat-interface.mjs` affiche les cartes actuelles du plan, les options de conversation et la confirmation de suspension reçue du serveur.
- `journal/src/settings-interface.mjs` conserve les trois onglets « Au prochain message », « Mes informations » et « Réponses précédentes ». Les menus et onglets du portrait ne sont pas modifiés.
- `journal/tests/change-interface.test.mjs` vérifie les comportements avec des fixtures DOM et API. Les valeurs privées sont insérées comme texte, jamais comme HTML.

## Mon plan dans Mon suivi

La page sépare objectifs, étapes, essais et retours, propositions, consignes professionnelles privées et points personnels. Chaque collection utilise sa pagination serveur par 50 éléments. Les limites affichées sont des limites de stockage ; aucun compteur de performance, score ou série quotidienne n’est ajouté. Un objectif rappelle la limite de trois objectifs en cours. Les valeurs de quotas détaillées sont lues depuis l’API.

Un objectif peut être créé, corrigé, mis en pause, repris ou supprimé. Ses étapes peuvent être ajoutées et corrigées. Les essais distinguent une intention à venir, un retour et une annulation. Les retours proposent : fait, en partie, non essayé, arrêté ou incertain ; effet utile, sans effet notable, pire, incertain ou non renseigné. Les dates restent vides et sont transmises comme `null` lorsqu’elles sont inconnues. Une précision libre sur la date est possible.

Un résultat rapporté ne peut pas être remis dans l’état « prévu » par son formulaire. « Créer un nouvel essai » passe par la route de création d’un nouvel objet et conserve l’essai antérieur. La suppression explique les dépendances retirées et la conservation des messages originaux de conversation.

Les consignes professionnelles sont déclaratives, privées et modifiables. Leur état distingue « à prendre en compte », « à vérifier en séance » et « terminée ». Elles peuvent être associées à un objectif lors de leur création, sans exiger une liaison avec un compte professionnel.

## Choix et invitations de retour

Les cartes du chat ne stockent pas une copie du texte dans l’enveloppe de réponse. Elles utilisent `message.change_proposal_ids`, puis lisent `GET /api/plan/proposals/:id`. Une proposition déjà traitée ne conserve que son état visible et le lien vers le plan actuel. Une proposition supprimée est signalée comme indisponible.

« Choisir » et « Modifier » ouvrent un formulaire de choix prérempli. Le bouton final est la décision de la personne. « Pas maintenant » utilise la route `dismiss` ; aucun objectif ou résultat n’est créé. Les routes `accept` et `dismiss` reçoivent `request_key`, `revision`, `epoch` et `entity_revision`. Les corrections sont envoyées dans `edited`.

Créer un premier essai à partir d’une proposition est un choix séparé, décoché par défaut. Autoriser Ambre à revenir sur cet essai est encore un autre choix, également décoché. Une date ou une prochaine conversation pertinente peut être choisie. Le texte rappelle le maximum de deux invitations espacées d’au moins sept jours et l’absence de notification extérieure. Le serveur reste responsable de leur admissibilité effective.

« Suspendre les retours » dans le chat envoie `PUT /api/plan/followups` avec `{enabled:false,target_id:'all'}` ; cet arrêt restrictif reste possible depuis un écran ancien. La commande de chat reconnue par le serveur reçoit `followups_paused:true` : l’interface annonce son enregistrement sans appel à Ambre et ne laisse pas un message optimiste bloqué en attente.

La réactivation demande un choix explicite dans Mon plan ou les Paramètres et transmet `enabled:true`, `confirmed:true`, `revision` et `epoch`. Elle ne remplace pas l’accord propre à chaque essai.

## Points personnels et mémoire

« Préparer mon point personnel » lit `GET /api/plan/preparation`. Le texte proposé est entièrement modifiable avant `POST /api/plan/preparations`, puis modifiable par `PATCH` et supprimable par `DELETE`.

Aucune liaison professionnelle n’est nécessaire. Le point reste privé. Les notes de mémoire apparaissent dans une rubrique distincte avec leur avertissement de provenance non vérifiée. Elles ne sont pas ajoutées par défaut : la personne choisit chaque note à ajouter. Le formulaire indique `include_memory:true` et la `memory_revision` seulement si une note a été ajoutée. Cette provenance reste attachée au point en cas de reformulation ultérieure de la note. Les références du plan sont envoyées dans `dependencies`.

Si les éléments sources d’un point en cours de création changent, le formulaire conserve le texte à copier mais ne réattribue pas automatiquement ce texte à de nouvelles références. Il faut préparer une nouvelle version. Cette restriction évite une provenance trompeuse ou la recréation implicite d’une copie devenue obsolète.

## Conflits et nouvelles tentatives

Les formulaires lisent la version actuelle avant ouverture. Un conflit 409 laisse tous les champs dans la fenêtre. L’utilisateur peut afficher la version enregistrée, comparer, puis choisir explicitement de garder sa saisie sur cette base avant un nouvel enregistrement. Il n’y a ni écrasement ni nouvelle écriture automatique.

Un changement d’`epoch` bloque la reprise : le texte reste disponible à copier, mais l’ancien formulaire ne recrée pas un plan effacé. Une proposition déjà acceptée ou écartée ne peut pas être réadoptée depuis la fenêtre ancienne. Un élément supprimé reste signalé comme tel.

Après une erreur réseau ambiguë, un POST inchangé réutilise sa clé de requête. Modifier les données ou choisir une nouvelle base renouvelle la clé. Pendant l’enregistrement, les champs sont temporairement désactivés pour éviter de perdre une saisie ajoutée après l’envoi. Une erreur les rétablit avec leurs valeurs. Annuler une saisie modifiée demande confirmation ; Échap suit la même règle.

## Paramètres et effacement de conversation

Le contrat d’insight confirmé avec l’intégrateur est :

```text
data.change = current.change = {
  plan, strategies, private_care_instructions, due_trial, instructions_omitted
}
plan contient revision, epoch, proposals_enabled et reminders_paused.
data.methods = {version, common_rules, methods, review_criteria}
selected_turn.snapshot.change
selected_turn.snapshot.analysis.strategy_ids
selected_turn.review.change.criteria
```

Les Paramètres montrent le contexte de plan disponible, les consignes privées retenues et les omissions de consignes. Les traces montrent les stratégies de l’analyse et les huit verdicts lorsqu’ils sont conservés. Une trace incomplète est annoncée comme telle ; aucun verdict n’est inventé. Le catalogue et ses références sont lus sur `GET /api/plan/catalogue`, avec restrictions et niveau de lecture des sources. `reviewed` ne signifie pas validation clinique.

L’effacement de conversation possède une case distincte et décochée « Effacer aussi tout mon plan… ». La requête transmet toujours un booléen `include_plan` explicite. Par défaut, le plan adopté, ses résultats et les consignes privées sont conservés. La confirmation annonce aussi l’effacement des bilans et le besoin de donner à nouveau son accord et de lancer une nouvelle analyse initiale. Le résultat affiche `notice` et `retained` du serveur.

## Vérification locale et recette restante

Vérifié avant livraison : compilation des trois scripts embarqués ; 11 tests spécifiques réussis ; 95 tests UI existants réussis ; contrôle de syntaxe et absence d’erreurs de whitespace dans les fichiers modifiés.

Les fixtures couvrent : lecture sans adoption, conservation de saisie sur conflit, ancien `epoch`, clé de tentative ambiguë, création d’essai distincte, dates inconnues, résultat non essayé/pire, notes de mémoire en opt-in, collections paginées, texte traité sans HTML, suppression de chat sans plan par défaut, suspension sans contrôle ancien et commande de chat sans attente fantôme.

La recette Playwright avec le Worker et SQLite réels est assurée par l’intégration principale, aux largeurs 1440, 390 et 320 pixels. Points à contrôler : affichage et défilement de la fenêtre avec clavier mobile ; focus initial, Échap et retour au déclencheur ; création/correction/pause/suppression ; conflit avec seconde modification ; préparation sans compte professionnel ; choix d’une proposition et accord de retour séparés ; trois onglets Paramètres ; menus du portrait conservés ; purge de conversation avec et sans plan.

Aucune mutation de données réelles, aucun commit, push ou déploiement n’a été effectué par cette livraison UI.
