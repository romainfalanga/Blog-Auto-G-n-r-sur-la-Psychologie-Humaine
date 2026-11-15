# Plan Cartographie V2 — 15 novembre 2026

Retour de l’utilisateur sur la V1 publiée : résultat déjà parlant, mais page trop longue, mobile à améliorer, causes et conséquences manquantes (par exemple le rôle de sauveur ne ressort pas alors qu’il est vécu), besoin de voir pour chaque cause toutes ses conséquences, et d’enregistrer la cartographie dans le profil pour qu’Ambre s’en serve dès son premier message.

## 1. Contenu : beaucoup plus de causes et de conséquences

Seconde vague de recherche, trois agents distincts des précédents, même schéma de fragment, mêmes exigences de sources lues et de forces de preuve :

| Agent | Périmètre |
| --- | --- |
| Rôles et relations | Conséquences relationnelles et de rôle (rôle de sauveur, plaire pour être aimé, codépendance, dépendance affective, hyper-indépendance, peur de l’intimité, difficulté à faire confiance, choix de partenaires, colère, honte, culpabilité chronique, hypervigilance relationnelle) reliées aux facteurs existants et nouveaux. |
| Nouvelles causes | 20 à 25 causes absentes : conflits parentaux sans violence, triangulation, parent indisponible ou absent, divorce, famille recomposée, déménagements répétés, adoption, violences dans la fratrie, punitions corporelles « ordinaires », pression scolaire, emprise religieuse ou sectaire, honte de sa famille, se débrouiller seul trop tôt, ne pas se sentir protégé, se sentir différent, etc., chacune avec ses conséquences. |
| Corps, comportements et réussite | Sommeil, douleurs et santé, poids et alimentation, écrans et jeux, travail compulsif, procrastination, perfectionnisme, sentiment d’imposture, difficultés scolaires et professionnelles, sexualité et intimité. |

Puis relecture indépendante et corrections tracées (`editorial/cartographie/corrections.json`), comme pour la V1.

## 2. Algorithme

- Deux niveaux par cause : « oui » et « oui, souvent ou gravement » (poids × 1,5), pour refléter l’effet dose-réponse.
- Score par conséquence avec rendements décroissants, convergence comptée sur des causes distinctes ; liens définitionnels exclus de la convergence ; critères de classification à part.
- Synthèse par grands domaines (humeur, anxiété et stress, relations et attachement, image de soi, émotions, comportements et corps) avec une barre d’intensité relative.
- « Je me reconnais / pas moi » sur chaque conséquence, enregistré avec la cartographie pour Ambre.

## 3. Page plus courte et plus visuelle

- Un parcours en étapes (un groupe de causes à la fois, barre de progression) au lieu d’une longue liste ; sur ordinateur, résultats à droite ; sur mobile, onglets « Mon enfance » / « Ce qui en ressort » et barre basse fixe avec le nombre de réponses.
- Bouton « Voir ses conséquences » sur chaque cause : panneau listant toutes les conséquences associées à cette cause, avec force et sources.
- Résultats compacts : synthèse par domaines, puis cartes repliables ; détails et sources à la demande.

## 4. Enregistrement et Ambre

- Table D1 `cartographies` (migration `0016`) : réponses, intensités, retours « je me reconnais », date de verrouillage, version. Routes authentifiées `GET/PUT /api/cartographie`, comprises dans l’export et la suppression du compte. Consentement explicite avant enregistrement (donnée de santé) ; sans connexion, rien n’est envoyé.
- « Enregistrer et verrouiller » : la cartographie devient la référence d’Ambre ; « Modifier » la déverrouille.
- Ambre : la cartographie verrouillée (causes cochées, conséquences en convergence, retours de la personne) entre dans le contexte privé et dans les consignes du système comme base d’hypothèses à vérifier, jamais comme diagnostic. Le premier message la mentionne et propose de partir des conséquences reconnues.
- Déploiement : migration D1 puis Worker, en conservant les autres modules, bindings et secrets ; vérification de `/api/cartographie` et du premier message.

## 5. Vérifications

Tests du moteur, de l’API et du prompt ; rendu Netlify ; essai réel sur ordinateur et mobile (Playwright) ; vérification en ligne après fusion.
