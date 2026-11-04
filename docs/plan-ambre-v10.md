# Plan V10 — conversation, capacités et couverture éditoriale

Base de départ : `5eb09cf8cbc5ecb325d973fa1eff5d4e315f5072`. Demande de Romain du 4 novembre 2026. État : implémentation et validation locale terminées ; recette de publication en cours.

## Exigences et preuves attendues

| ID | Résultat | Responsable | Preuve attendue |
|---|---|---|---|
| R1 | Bulle utilisateur immédiate, composeur vidé, attente « Ambre réfléchit… », brouillon suivant indépendant | Interface | POST retardé, échec et retour réseau incertain sans doublon ni perte |
| R2 | Réponse affichée progressivement depuis son début, sans entraîner la lecture en bas | Interface | Réponse longue, défilement manuel conservé, mouvement réduit |
| R3 | Accueil d’Ambre présentant ses capacités exactes et les premières informations à compléter | Backend et interface | Fil neuf, prénom courant, aucun appel ni consommation de quota |
| R4 | Ambre ne propose que des actions actuellement exécutables | Backend | Capacités serveur, schéma/sortie/confirmation protégés, états concurrents |
| R5 | Recherche fiable des thèmes existants, notamment rôle de sauveur, sans fausse affirmation d’absence | Backend et catalogue | Requête explicite, synonymes, liens publics canoniques, catalogue complet |
| R6 | Audit des lacunes puis treize nouveaux personnages et douze histoires chacun | Éditorial | 156 récits originaux, continuité, format et longueur contrôlés |
| R7 | Navigation, recherche, recommandations, carte et contexte d’Ambre intègrent les nouveaux contenus | Intégration | Registre, matrice, pages, exemples et index cohérents ; construction Hugo |
| R8 | Cartographie compréhensible : vécu, hypothèse, preuves, contextes et contre-indices | Interface | Pistes personnelles d’abord, catalogue secondaire, parcours mobile |
| R9 | Mentions d’aide d’Ambre sous forme de notes d’information ; boutons d’ajout centrés sur mobile | Interface | Inspection mobile et absence de liens d’action dans les notes |
| R10 | Dictée française progressive et éditable, sans coût d’API ajouté | Principal | Résultats provisoires/finals, édition, permissions, navigation, limites |
| R11 | Conversation et productions IA du compte de Romain réinitialisées ; saisies personnelles conservées | Principal | Transaction ciblée, sources identiques avant/après, aucun travail ancien publiable |

## Révision du plan contre le code

Le sentiment de gel vient d’un affichage de la bulle après le POST, du verrouillage global du formulaire et d’un état d’attente rendu trop tard. Le changement porte sur l’état client, la réconciliation par clé d’envoi et le défilement. Le JSON contenant texte et propositions reste validé côté serveur avant publication ; la révélation progressive est locale et ne prétend pas être le flux brut du fournisseur.

Le refus de génération personnelle n’arrive aujourd’hui qu’à la confirmation. Une capacité serveur doit donc être transmise à Ambre, appliquée au schéma et à sa sortie, puis revérifiée à l’acceptation. Elle réutilise les véritables prérequis : dix lectures aimées, portrait actuel complet, personnage prêt, accès et quota. Les suggestions de lectures publiques restent distinctes d’une nouvelle histoire personnelle.

Le rôle de sauveur est déjà présent dans le catalogue de mécanismes et les récits de Thomas, mais son alias exact est mal retrouvé. L’absence parmi quelques recommandations ne signifie pas absence dans la bibliothèque. Une recherche conceptuelle globale et vérifiable précède donc toute affirmation d’absence. L’ajout de personnages complète ensuite les domaines moins traités ; il ne remplace pas cette correction.

Les personnages actuels ont 11 à 13 articles, pour une médiane de 12. Douze récits par nouveau personnage donnent une profondeur comparable. L’audit compare les 161 repères existants, les sujets éditoriaux et les scènes réellement publiées. Aucune liste finie ne couvre l’intégralité de la psychologie humaine ; l’objectif est une couverture explicite et équilibrée, avec des pistes concurrentes, des ressources et des contre-exemples.

La revue de la première projection a repéré 23 repères du catalogue sans sujet direct initial ni angle explicite prévu. Le plan a été amélioré en ajoutant des exigences de développement à des épisodes cohérents, sans changer les identités, les sujets ou le nombre de récits : médiation et frontières familiales, retrouvailles, appuis attentionnels, apprentissage par observation, perte sans clôture claire, environnement sensoriel et ressources, notamment. Ces angles doivent être incarnés et expliqués dans les textes ; un tag ou une mention ne vaut pas une couverture. Leur présence réelle sera contrôlée après rédaction, sans prétendre que le catalogue de 161 repères est exhaustif de la psychologie.

Pour la continuité, les articles réellement publiés et la matrice font foi. Les entrées anciennes du registre qui ne correspondent à aucune publication doivent être rapprochées du corpus, sans inventer un épisode absent. Aucun appel à l’outil ou au fournisseur de génération d’articles ne sera utilisé : les nouveaux textes sont rédigés directement, puis contrôlés par des outils déterministes.

La cartographie comporte déjà des informations utiles mais les présente après une grille de domaines et des filtres. Le plan inverse cette hiérarchie : une liste de pistes personnelles et une fiche reliant observation, mots d’origine, portée, autres explications et question suivante. Le catalogue demeure une exploration secondaire. Les niveaux d’influence restent distincts de la confiance ; aucun pourcentage de personnalité n’est ajouté.

## Dictée : choix après recherche

| Solution examinée | Atouts | Limites | Décision |
|---|---|---|---|
| Reconnaissance native du navigateur | Résultats provisoires en direct, français, pas de clé ni facture de transcription pour l’application | Support et qualité variables ; le navigateur peut traiter la voix à distance | Retenue pour la dictée immédiate |
| Reconnaissance native locale | Voix traitée sur l’appareil, fonctionnement hors ligne possible | API récente et disponibilité du français variables | Préférée lorsque le pack français est déjà disponible |
| Whisper dans le navigateur | Audio local, modèle maîtrisé, pas de coût de serveur par minute | Chargement de plusieurs dizaines de Mo au minimum, calcul et latence variables sur mobile, dépendances supplémentaires | Référence évaluée ; pas de chargement imposé pour un simple champ de conversation |
| API cloud de transcription | Qualité et performances possibles selon fournisseur | Coût récurrent, transfert audio et service supplémentaire | Écartée pour respecter la demande gratuite |

Le micro produit uniquement un brouillon. Les révisions de résultats provisoires remplacent leur segment, sans répéter les mots. Toute saisie manuelle arrête la dictée pour préserver la correction. L’écoute s’arrête à l’envoi, à la sortie de page et lorsque la page est masquée ; les événements tardifs ne peuvent pas écrire dans le brouillon suivant. Aucune piste audio n’est enregistrée ni envoyée au backend du site. Le service natif et sa limite de compatibilité sont indiqués dans l’interface.

Le code interdisait actuellement tout microphone via `Permissions-Policy: microphone=()`, dans le Worker et Netlify. Ces deux politiques doivent autoriser le microphone pour l’origine du site ; les autres permissions restent restreintes.

Sources techniques primaires consultées : [spécification Web Speech](https://webaudio.github.io/web-speech-api/), [MDN, utilisation et traitement local](https://developer.mozilla.org/en-US/docs/Web/API/Web_Speech_API/Using_the_Web_Speech_API), [Microsoft, disponibilité du moteur local](https://learn.microsoft.com/en-us/microsoft-edge/web-platform/speech-recognition-api), [Transformers.js, WhisperTextStreamer](https://huggingface.co/docs/transformers.js/v3.8.1/api/generation/streamers), [modèle Whisper base et poids ONNX](https://huggingface.co/onnx-community/whisper-base/tree/main/onnx). La qualité réelle dépend de l’appareil, de la langue, du bruit et du navigateur : aucun taux de reconnaissance non mesuré n’est promis.

## Réinitialisation ciblée

Les métadonnées du compte réel montrent un utilisateur, quatre tours de chat, aucune proposition acceptée, une proposition d’histoire en attente, aucun récit ni retour de récit, un personnage, une lecture aimée et cinq proches. Aucun mécanisme d’archive de retours n’est nécessaire pour ce compte sans récit. Le schéma supprime les retours avec un récit : ce point doit être explicitement vérifié avant toute suppression.

La transaction conservera comptes et accès, profil déclaré, mémoire personnelle, proches, événements, likes et retours manuels. Elle invalidera d’abord les baux des travaux et du chat, puis supprimera conversations, propositions, personnage, portraits générés, caches d’analyse et données de génération devenues inutiles. Les sources personnelles seront comparées avant et après par leurs valeurs ou empreintes. L’analyse fraîche peut ensuite reconstruire le portrait depuis ces mêmes sources, sans réutiliser les productions anciennes.

Cette opération sera faite une seule fois après publication et recette sur un compte synthétique. Elle est autorisée explicitement par la demande de Romain. Aucun secret ni contenu personnel ne sera placé dans le dépôt ou dans ce rapport.

Le choix retenu pour cette remise à neuf ponctuelle est une suppression physique des tours et propositions de chat, qui rend de nouveau les 100 messages disponibles. La comptabilité des appels déjà facturés dans `provider_budgets` est conservée strictement. Le consentement existant et les sources canoniques confirmées restent inchangés. Cette opération exceptionnelle est distincte du bouton d’effacement normal, qui conserve le quota utilisé.

Répétition réalisée sur D1 de recette avec deux comptes synthétiques : profil, mémoire, proche, événement lié, like et consentement strictement identiques après opération ; ancien portrait, dérivés, personnage et chat retirés ; baux de chat et portrait inutilisables ; clés étrangères valides ; second compte inchangé. Le code ponctuel reste hors du produit. La maintenance bloque toutes les API sauf la santé ainsi que le cron durant le précontrôle, la transaction et la comparaison.

## Relecture indépendante et améliorations

La relecture a reproduit une offre impossible formulée seulement en prose, sans proposition structurée. Le contrôle a été étendu aux offres et annonces de création de récit, article ou épisode, sans supprimer les recommandations publiques ni les conditions futures correctement formulées. Les variantes reproduites sont désormais couvertes par les tests fournisseur simulé. Les capacités sont toujours revérifiées après l’appel et à l’acceptation.

Les liens éditoriaux vers les repères sont conservés dans le manifeste des récits et exportés avec leur provenance. La cartographie privilégie ces correspondances revues ; les recommandations les utilisent comme vocabulaire de recherche de faible poids. Ils ne créent jamais une hypothèse personnelle ou une preuve sur le lecteur. Une piste personnelle doit toujours citer ses propres sources explicites, même si plusieurs récits du catalogue illustrent ce mécanisme.

La comparaison des textes rédigés avec la grille a encore distingué cinq angles qui ne pouvaient pas être déduits du seul voisinage des sujets : réponse corporelle au stress, charge qui empêche la récupération, décisions interdépendantes, apaisement stratégique et ambivalence de motivation. Des développements courts et situés ont été demandés dans les épisodes appropriés, avec une limite de 2500 mots. Le plan final porte donc 28 exigences secondaires explicites, en plus des sujets principaux, sans ajouter de récit ni transformer un contexte social ou une identité en cause psychologique.

## Ordre d’intégration et contrôles

1. Corriger les contrats de capacités et la recherche ; travailler en parallèle sur l’interface et la dictée.
2. Valider les treize fiches et leurs arcs, puis rédiger les 156 articles dans des lots indépendants, avec un auteur par personnage.
3. Contrôler chaque lot : texte original, longueur, structure, continuité et formulation psychologique prudente. Agréger registre, matrice et pages, puis reconstruire les index publics.
4. Exécuter les tests du journal et du site, les contrôles Python, la construction Hugo et une revue indépendante des contrats sensibles.
5. Recette de conversation et de cartographie sur desktop/mobile, test des états de dictée et des erreurs, appel réel d’Ambre sur compte synthétique, publication puis réinitialisation du compte réel.

Les contrôles audio simulés prouvent le fonctionnement de l’éditeur et de ses états, pas la qualité acoustique d’un microphone réel. La recette distinguera ces deux niveaux.

## Validation locale finale

156 récits, 305015 mots mesurés, 13 parcours de 12 épisodes ; aucun écart au contrôle éditorial. Catalogue final : 33 personnages et 387 articles. Les 231 fichiers antérieurs ont été comparés octet par octet à la base et restent identiques. Les index, le manifeste et les pages sont cohérents et leur reconstruction est idempotente.

332 tests du journal, 98 tests Python et 16 tests de recherche passent. Construction Hugo réussie ; 156 URLs de récits et 13 parcours à 12 liens vérifiés. Les variantes du sauveur retrouvent le récit dédié d’Aline. La recherche privilégie un repère dans les métadonnées d’un récit plutôt qu’une mention incidente dans sa description. Une revue indépendante des contrats sensibles n’a plus de bug matériel confirmé après correction.
