# Décode ton esprit — Interface application

## Intentions et plan révisé

L'écran doit donner accès à l'action attendue. Réduire le texte d'accueil, les titres répétés et les consignes permanentes. Conserver les informations nécessaires au consentement, au paiement, aux erreurs et aux actions irréversibles dans leur contexte.

Direction : blanc #FFFFFF, fond #FAFAF8, texte #2D2D2D, secondaire #6B6B6B, accent vert #3D8577, bordures #E8E8E4. Police système pour les commandes et titres de l'application ; Georgia reste réservée aux récits. Contenu aligné à gauche, largeur de lecture limitée, espaces plus compacts. Un même menu global. Pas de nouveau décor ou écran promotionnel.

| Exigence | Mise en œuvre | Critère de vérification |
|---|---|---|
| R1 Supprimer le compteur global d'articles aimés | Retirer le badge de l'en-tête et ses mises à jour | Aucun badge de likes dans les titres des pages |
| R2 Accompagnement = conversation | Fil vide avant le premier échange, saisie en bas, messages au centre, quota en haut à droite | Accès direct sans paragraphe d'introduction, historique conservé |
| R3 Plateforme minimaliste | Titres uniques, phrases courtes, aides à la demande, formulaires/listes en premier | Proches et événements ajoutables sans lire un mode d'emploi |
| R4 Portrait structuré | Trois vues : Synthèse, À propos de moi, Lectures | Les contenus ne sont plus empilés ; toutes les données restent modifiables |
| R5 Préserver les fonctions | Brouillons, révisions, consentement, propositions, quotas, liens profonds, facturation | Tests ciblés + contrôle visuel mobile et ordinateur |

Le plan initial a été revu : ne pas déplacer les vues du portrait vers des routes nouvelles (préserver les ancres existantes), ne pas effacer les formulaires lors d'un changement de vue (préserver les brouillons), ne pas masquer le consentement derrière une acceptation implicite, conserver historique et propositions si l'envoi est bloqué. Le compteur de likes reste seulement là où il aide à débloquer Mon personnage ou à compter les résultats de Lectures.

## Réalisation et preuves

Refonte implémentée. Aucun changement de données ni d’algorithme pour cette livraison.

- 144 tests JavaScript du journal et 16 tests de recherche réussis ; compilation Hugo 0.140.0 et cohérence du menu validées.
- Recette navigateur isolée avec données fictives : conversation vide sur ordinateur et dans une vue de 390 × 844 px, saisie en bas sans débordement horizontal observé ; portrait en trois vues ; conservation d’un brouillon entre vues ; liste des proches et ajout accessibles directement.
- Tests fonctionnels ajoutés : absence du badge global, conversation initiale sans faux message, quota et blocages d’envoi, historique/propositions conservés, ancres du portrait et préservation des formulaires, réessai déjà terminé, déduplication d’un message en attente.
- Relecture indépendante : correction du retour au chat après déconnexion, du bouton d’envoi après réponse idempotente, de la priorité des ancres explicites et du message optimiste dupliqué après un échec réseau.
- Limite de la recette mobile : largeur/hauteur contrôlées en iframe ; clavier physique de téléphone non testé. Aucun appel IA payant ni modification du compte utilisateur effectué.

Les explications détaillées restent disponibles dans « Informations et historique » pour le chat et dans les aides du compte. Les informations essentielles au consentement, à l’offre et à la suppression restent explicites. L’ouverture des nouveaux abonnements dépend toujours de la configuration Stripe, hors périmètre de cette refonte.
