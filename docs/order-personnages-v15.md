# Ordre de découverte des personnages

## Plan de mise en œuvre

L’accueil présente directement la recherche et les personnages. Le bloc d’introduction et les initiales décoratives sont retirés. Les catégories et les dates d’articles sont retirées du rendu des listes, des parcours et des pages de lecture. Les métadonnées, les textes sources et les liens existants restent utilisables par le moteur.

L’ordre initial privilégie la largeur des situations psychologiques explorées. Les parcours de rumination, limites, choix, comparaison et régulation émotionnelle ouvrent la liste. Les parcours centrés sur une transition ou une contrainte particulière arrivent ensuite. Cet ordre est un choix éditorial à partir des présentations et des mots-clés des 33 personnages. Il ne mesure aucune prévalence et ne prédit pas à qui une histoire conviendra. L’âge, le genre, l’origine ou l’orientation des personnages ne servent pas de critères de classement.

La recherche conserve son classement de pertinence. Effacer la recherche rétablit l’ordre de découverte. Le fichier de données dédié évite qu’une régénération des profils remplace cet ordre.

| Position | Personnage | Entrée psychologique principale |
| --- | --- | --- |
| 1 | Sophie | Rumination, autocritique et perfectionnisme |
| 2 | Thomas | Limites, peur de décevoir et oubli de soi |
| 3 | Baptiste | Choix, incertitude, regret et valeurs |
| 4 | Lucas | Comparaison, image de soi et sentiment d’imposture |
| 5 | Mei | Regard des autres, expression de soi et désaccords |
| 6 | Élise | Régulation des émotions et ressources ordinaires |
| 7 | Camille | Charge mentale, anticipation et besoins personnels |
| 8 | Nadia | Exigence, contrôle, délégation et fatigue |
| 9 | Rémi | Distraction, démarrage des tâches et priorités |
| 10 | Aline | Surresponsabilité, réciprocité et culpabilité au repos |
| 11 | Patrick | Retenue émotionnelle et demande de soutien |
| 12 | Youssef | Identifier et exprimer ses émotions |
| 13 | Salomé | Pression des normes, regard d’autrui et vie privée |
| 14 | Éloi | Apprentissage, peur de l’erreur et compétence |
| 15 | Stéphane | Validation extérieure et image de réussite |
| 16 | Karim | Équilibre des responsabilités et décisions impulsives |
| 17 | Priya | Évitement émotionnel et place du travail |
| 18 | Inès | Limites intimes et communication dans le couple |
| 19 | Malik | Appartenances, attentes familiales et liens à distance |
| 20 | Maëlle | Récupération et besoins corporels sous horaires contraints |
| 21 | Adrien | Procrastination et appuis personnels après une rupture |
| 22 | Léa | Surinvestissement relationnel et peur de l’abandon |
| 23 | Isabelle | Fatigue à aider et difficulté à recevoir de l’aide |
| 24 | Margot | Quête de sens au cours d’une reconversion |
| 25 | Fabien | Reconstruction et parentalité après une séparation |
| 26 | Samira | Charge mentale liée à l’insécurité matérielle |
| 27 | Djamila | Vigilance relationnelle et reconstruction de la confiance |
| 28 | Hugo | Idéalisation et alternances dans les relations |
| 29 | Vincent | Charge d’aidance, répit et autonomie du proche |
| 30 | Anaïs | Capacités fluctuantes et ajustements face à la maladie |
| 31 | Odile | Deuil, solitude choisie et changements de rôle |
| 32 | Monique | Nouveaux liens et culpabilité après un veuvage |
| 33 | Bernard | Deuil, silences et reprise du dialogue familial |

Les places proches sont des arbitrages éditoriaux révisables. Tous les parcours restent accessibles par la recherche, notamment lorsqu’une situation spécifique est saisie.

## Vérification

La compilation Hugo et les 16 tests de recherche passent. Le navigateur a contrôlé l’accueil, un profil, un article, la liste générale et deux listes issues des catégories sur cinq largeurs de 320 à 1 440 pixels. La recherche par prénom puis sa remise à zéro retrouve le même ordre complet. Les 33 profils compilés ne contiennent plus d’élément d’initiale, de date d’article ou de catégorie dans leur parcours. Les listes générales présentent les articles et une pagination resserrée.
