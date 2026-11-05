# Audit d’Ambre et des données personnelles

## Périmètre examiné

L’audit porte sur les routes du Worker, la constitution du contexte privé, les propositions de conversation, la synthèse du portrait, les recommandations et le guide psychologique. Les parcours d’interface sont examinés dans le plan de livraison V15. Le catalogue public comprend 387 articles et 33 personnages. Cette vérification du logiciel ne constitue pas une évaluation clinique de l’efficacité d’Ambre.

| Composant | Informations accessibles et fonctionnement vérifié |
| --- | --- |
| `worker.mjs` | Authentification et séparation des comptes, accès aux pages privées et API, export, suppression du compte, consentement et abonnement, contrôle d’origine et réponses sans cache. |
| `private-context.mjs` | Présentation actuelle, commentaires des articles aimés, relations décrites, événements et proches associés, mémoire des corrections validées. Une double lecture de version protège la cohérence du contexte. L’identité est séparée des preuves psychologiques. |
| `personal-backend.mjs` | Créations idempotentes, éditions sous révision, limites des collections, événements datés ou approximatifs, déclenchement de la révision du portrait. Les traitements périmés ne peuvent publier après une modification des sources. |
| `chat-backend.mjs` | Consentement, une réponse en cours à la fois, clé d’idempotence et réservation atomique du quota. Publication conditionnée à la version des sources, au bail, à l’accès et au consentement. Échec fournisseur ou réponse invalidée libèrent la réservation. |
| `psychology-framework.mjs` et `psychology-guide.mjs` | Instructions d’écoute et d’exploration, alternatives et contre-exemples, repères du catalogue, limites et orientation humaine. |
| `recommendations.mjs` et catalogue | Classement des articles existants à partir de la demande et des informations autorisées, préférences et exclusions. Les identifiants et destinations proposés sont contrôlés côté serveur. |
| `chat-interface.mjs` et `ui.mjs` | Envoi immédiat, consultation de l’état de réponse, reprise après interruption, conservation du brouillon, confirmation des modifications, dictée dans un texte modifiable. |

Les pages Mon portrait, Mes proches et Ma ligne de vie alimentent les mêmes sources utilisées par Ambre. Les modifications suggérées dans une conversation restent des propositions jusqu’à leur acceptation. Une précision peut affecter plusieurs éléments, mais chaque cible reçoit sa propre proposition sourcée. Une édition manuelle concurrente déclenche un conflit de révision au lieu d’écraser silencieusement les données.

## Recherches et choix d’accompagnement

NICE recommande de partir des priorités de la personne, de communiquer par informations assimilables, de choisir ensemble une suite et de prévoir sa réévaluation. Ces principes orientent ici le choix d’un objectif concret et d’un petit essai accepté, puis l’examen de ce qui a changé. Leur adaptation à un assistant n’établit pas l’efficacité d’un traitement. [NICE NG197, recommandations 1.2.5 à 1.2.17 et 1.4](https://www.nice.org.uk/guidance/ng197/chapter/recommendations)

Le guide SAMHSA décrit les questions ouvertes, les reformulations et les résumés, ainsi que le risque de transformer l’entretien en interrogation. Le guide d’Ambre privilégie désormais une reformulation précise, permet une réponse sans question et demande de vérifier l’effet d’un conseil antérieur avant de le renouveler. Il adapte des principes d’entretien motivationnel décrits dans un contexte d’addictologie. [SAMHSA TIP 35, chapitre 3](https://www.ncbi.nlm.nih.gov/books/NBK571068/)

L’OMS souligne les erreurs possibles des modèles génératifs et le risque de confiance excessive dans leurs résultats. Le contexte informatique donne accès à des informations déclarées et sélectionnées. Il ne fournit ni examen clinique, ni observation directe, ni connaissance exhaustive de l’utilisateur ou de ses proches. Le comportement d’Ambre conserve donc une distinction explicite entre faits rapportés, interprétations et hypothèses, avec une évaluation professionnelle proposée lorsque la souffrance, le retentissement ou une demande de diagnostic le justifient. [OMS, recommandations sur les modèles multimodaux en santé](https://www.who.int/news/item/18-01-2024-who-releases-ai-ethics-and-governance-guidance-for-large-multi-modal-models)

## Défauts observés et corrections

| Observation dans le code | Correction mise en œuvre |
| --- | --- |
| Le contexte de conversation perdait la date, l’âge approximatif et les personnes liées d’un événement lors de sa réduction. | Les sources sélectionnées conservent leur temporalité, leurs liens, leur importance déclarée et leur date de mise à jour. L’index des événements conserve également les repères temporels. |
| Les périodes et les rôles personnalisés des proches disparaissaient de l’index transmis. | Ces champs restent disponibles pour situer la relation. |
| Le portrait était coupé à 1 500 caractères au milieu de son JSON. | Un objet borné expose la synthèse, ses références, les ressources et les incertitudes. Une synthèse écourtée est signalée. |
| Seules les quatre premières hypothèses étaient transmises, même si une autre concernait directement la question. | Les hypothèses sont classées selon la demande et les sources sélectionnées, tout en conservant leur statut et leurs contre-indices. |
| Le guide de conversation complétait ses places avec des repères sans correspondance. | Le guide d’Ambre retient des repères ayant une pertinence textuelle ou un lien avec une hypothèse active. Un contexte vide peut produire un guide vide. La sélection générale du pipeline de portrait garde sa couverture large. |
| La couverture ne signalait pas les proches, événements et hypothèses omis des index. | Les quantités incluses et disponibles sont transmises avec la couverture des sources et de l’historique. |
| Un changement de précision de date par Ambre pouvait conserver d’anciennes dates incompatibles. | Le normaliseur commun efface les champs temporels inactifs, pour les modifications manuelles comme pour les propositions acceptées. |
| La pagination serveur de la ligne de vie suivait les identifiants aléatoires. Le tri visuel ne concernait que les éléments chargés. | Le serveur classe toutes les dates en ordre décroissant avant pagination, puis les moments sans date précise. Les identifiants existants servent toujours de curseurs. Un curseur supprimé invite à recharger. |
| Le message de mémoire pleine demandait de raccourcir un texte sans éditeur accessible. | Il indique la limite et oriente vers les champs modifiables de Mon portrait, À propos de moi. |
| Deux registres de brouillons manquaient dans l’état de l’interface et empêchaient l’édition des notes de lectures. | Les registres utilisés par le formulaire sont initialisés. |
| Quitter une présentation intacte créait un faux brouillon susceptible de masquer ensuite une modification acceptée avec Ambre. | Les brouillons sont alimentés par les événements de saisie. Le texte réellement modifié reste conservé. |

## Validation et limites concrètes

Les tests ciblés exercent les vraies routes et les transactions SQLite avec réponses fournisseur simulées. Ils couvrent notamment la transmission des dates et liens, la sélection d’une hypothèse située après les quatre premières, les corrections partielles de date validées, la pagination de plus de cinquante événements et le retrait des anciens outils de récit.

La conversation affichée contient au maximum les soixante derniers échanges. Le modèle reçoit jusqu’à douze échanges terminés sous un budget de 8 000 caractères, plus un contexte privé limité à 16 000 caractères et un guide réduit. Les omissions sont signalées. Les éléments importants pour le long terme peuvent être proposés dans les sources personnelles, avec accord. Il n’existe pas de mémoire illimitée de toutes les conversations.

La sélection des articles et des repères utilise principalement une pertinence lexicale. Elle peut manquer une paraphrase éloignée et ne mesure aucune prévalence ni probabilité de trouble. Une modification ou suppression depuis la ligne de vie recharge sa première page et réinitialise la pagination. Des dates modifiées dans un autre onglet peuvent nécessiter un rechargement pour revoir l’ordre complet.

La détection déterministe du danger immédiat couvre des expressions explicites et complète les instructions du modèle. Elle ne garantit pas de reconnaître toute situation d’urgence. Les propositions d’enregistrement et les lectures sont suspendues lorsqu’elle s’active. Aucun test simulé ne prouve une fiabilité clinique. Une évaluation avec des professionnels et des utilisateurs reste nécessaire pour revendiquer un bénéfice clinique ou une performance diagnostique.
