# Cartographie et transparence d’Ambre — 16 novembre 2026

Ce chantier remplace les choix d’interface de la V3 et la première page Paramètres, à la demande de Romain. L’éditorial reste en pause. La publication déjà autorisée comprend le Worker puis le site après vérification.

| Exigence | Critère observable | Responsable | État |
|---|---|---|---|
| R1 Portrait | Seulement À propos de moi et Mon portrait dans le menu local | Principal | Publié et vérifié en production |
| R2 Paramètres | Données réelles consultables, sélection explicitée, modèles par étape, détails des nouvelles réponses | Principal + engine_review | Publié ; vérifications détaillées ci-dessous |
| R3 Six thèmes | Six accordéons fermés au chargement, aucun suivant, aucun septième thème | care_review | Publié ; vérifications détaillées ci-dessous |
| R4 Sauvegarde | Aucune case à cocher, sauvegarde automatique pour le compte connecté ; brouillon local sans compte ; erreurs et conflits visibles | care_review | Publié ; vérifications détaillées ci-dessous |
| R5 Mobile | Réponses et ajout d’explication centrés à 320 et 390 px, sans débordement | care_review | Publié ; vérifications détaillées ci-dessous |
| R6 Résultats | Tous les types sans filtre, explication associations/diagnostic, absence du bloc avant de commencer | care_review + clinical_coverage | Publié ; vérifications détaillées ci-dessous |
| R7 Vue d’ensemble | Hiérarchie lisible, absence d’espace vide au-dessus du titre | care_review | Publié ; vérifications détaillées ci-dessous |
| R8 Fond scientifique | Audit chiffré CIM-11/DSM-5-TR/modèles ; corrections sourcées ; limites de couverture explicites | clinical_coverage | Publié ; vérifications détaillées ci-dessous |
| R9 Calcul et Ambre | Classement prudent, cohérence client/serveur, retours négatifs et limites d’âge transmis, tests de non-régression | clinical_coverage + engine_review | Publié ; vérifications détaillées ci-dessous |
| R10 Livraison | Relecture indépendante, tests projet, navigateur mobile/bureau, déploiements vérifiés | Principal | Publié, recette distante et effacement vérifiés |

## Décisions

- Une classification diagnostique n’est pas une table de causalité. Ne pas fabriquer de liens entre les vécus d’enfance et l’intégralité des diagnostics pour afficher une exhaustivité artificielle.
- Référencer précisément le périmètre du catalogue, les éléments exploités par le questionnaire et les lacunes restantes. Ni DSM ni CIM ne constituent un inventaire de tous les mécanismes psychologiques.
- Paramètres est un inspecteur privé : montrer le contenu disponible et réellement transmis, les sélections et les limites. Les traces historiques sans contenu ne seront pas reconstruites a posteriori comme si elles étaient exactes.
- Les traces décrivent les entrées des étapes et leurs statuts ; elles ne conservent aucun raisonnement interne du fournisseur. Lecture limitée au propriétaire ; suppression avec la conversation ou le compte.
- Conserver les ressources anciennes en base tout en retirant le septième thème demandé. L’enregistrement automatique s’applique aux nouvelles saisies du membre connecté ; une suppression ne doit pas déclencher une recréation immédiate.

## Direction visuelle

Conserver l’identité de l’espace personnel : blanc `#FFFFFF`, fond bleu gris `#F2F5F7`, encre `#172D3C`, vert `#205E54`, vert pâle `#E7F1EE`, lignes `#D6E0E6`. Georgia reste réservée au grand titre ; police système pour les données et commandes.

Paramètres devient un dossier de travail avec trois vues : prochain message, informations enregistrées, réponses précédentes. Le contenu personnel est au premier plan ; les étapes sont numérotées uniquement parce qu’elles décrivent une séquence réelle. Les détails longs s’ouvrent à la demande. Les modèles sont affichés près de leur étape. Texte aligné à gauche, largeur de lecture limitée, contrôles accessibles au clavier.

La cartographie utilise les six intitulés de thèmes comme entrées d’accordéon. Le progrès apparaît auprès du thème, sans imposer un parcours. La vue d’ensemble explique les familles de résultats et le sens du classement, sans pourcentages ni probabilités inventés.

## Vérifications prévues

Tests des contrats API et de l’isolation des traces, comparaison du calcul public/serveur, cas cliniques limites, mémoire et reprise des étapes, tests projet existants. Recette Playwright à 1440, 390 et 320 px avec données synthétiques : accordéons, réponses, explications, sauvegarde, conflits, résultats, contenu réellement utilisé par Ambre, modèles et navigation portrait. Vérifier les publications et supprimer les comptes de recette.

## Preuves avant publication

- 454 tests du journal, 31 tests du site, 43 tests personnels et les contrôles du workflow passent. Les deux calculs de cartographie concordent sur 1 711 cas (réponses isolées et toutes les paires).
- 14 scénarios Playwright de cartographie passent : accordéons, non explicite, brouillon, autosauvegarde sérialisée, erreurs, conflit 409, suppression, chargement initial tardif, sources et centrage mobile. Le bouton de précision est vérifié sur le même axe que les réponses.
- Un parcours intégré utilise le vrai Worker avec SQLite et un fournisseur simulé : trois entrées d’appels enregistrées à l’identique, contenu de Paramètres consultable, six thèmes publics Hugo et PUT réels, deux onglets du portrait, absence d’erreur JavaScript et de débordement à 1 440, 390 et 320 px.
- Relecture indépendante : navigation clavier, correction de mémoire, conflit CAS, filtres de recherche et affichage mobile validés. Deux écarts de transmission clinique repérés puis corrigés : preuve coupée avant les 1 100 caractères et métadonnées omises du contexte historique.
- Audit exhaustif des identifiants du projet : 858 codes du registre, 161 repères, 82 sujets de cartographie ; correspondances sémantiques justifiées pour les 161 repères. Les 22 chapitres diagnostiques du DSM-5-TR ont un statut thématique ; cela ne certifie pas l’intégralité des critères ou diagnostics du manuel. Les 425 liens ne sont pas tous relus dans le texte intégral des études : cette limite est documentée, sans allégation de validation clinique complète.

Les limites et références sont dans [l’audit](audit-cartographie-transparence-2026-11-16.md) et [ses correspondances détaillées](audit-cartographie-correspondances-2026-11-16.json). Aucun ajout de liens causaux destiné à remplir artificiellement le catalogue.

Dernière relecture clinique : le résumé de la preuve primaire et son URL sont conservés dans des métadonnées distinctes du texte raccourci. La nuance sur la foi et l’emprise survit à l’extrait. Le surcoût maximal mesuré est de 464 caractères par source ; les textes restent bornés à 6 500 caractères pour huit sources. Les étiquettes distinguent symptômes, groupes de troubles et qualificatifs cliniques.

## Correctifs issus de la recette réelle

La première réponse réelle a respecté « sans conseil ni question », mais seule la contrainte sans conseil était enregistrée. Les formulations liées par « ni » sont désormais reconnues explicitement pour les conseils et les questions ; la sécurité actuelle reste prioritaire. La relecture a également identifié le format canonique `pattern_id` du portrait : chaque piste conserve maintenant un identifiant distinct et le libellé du catalogue, au lieu d’être fusionnée sous un identifiant absent. Deux tests de comportement couvrent ces cas, et une relecture indépendante les a vérifiés.

La recette mobile distante confirme aussi un écart de 14 pixels entre le bord de la carte et le titre Vue d’ensemble, sans espace parasite, à 390 et 320 pixels. Une réponse déjà sélectionnée ne crée plus de brouillon ni de sauvegarde supplémentaire lors d’un nouveau clic ; le scénario de navigation rapide correspondant est couvert en navigateur.

Un autre essai réel a montré qu’une question ordinaire sur la portée de la cartographie pouvait être classée à tort comme risque possible par le petit modèle. Une alerte proposée uniquement par ce modèle passe désormais par un appel conditionnel isolé, ancré dans le message actuel ; il ne reçoit ni index, ni portrait, ni mémoire. Une confirmation actuelle explicite peut joindre le dernier échange de sécurité pertinent. Les signaux du filet déterministe gardent leur priorité. L’analyse initiale est persistée avant ce contrôle, dont les échecs et reprises sont bornés. L’étape, son modèle et ses entrées sont visibles dans Paramètres ; les ressources de violence utilisent le signal confirmé approprié.

## Livraison distante terminée

- Code final `367159faad289b54fa859009e36707c5fa3cf42d`, PR 77, 78 et 79 intégrées ; CI `43872759457` réussie.
- Worker `94323d7713e842fab11cfef3ad5dba0f`, 16/11/2026 13:07:45 UTC : 32 modules vérifiés par SHA-256, aucun écart ; aucune migration et secrets conservés.
- Netlify `6afb003f7e6eb88ee1fb1d23`, même commit, état `ready`, publication 13:08:56 UTC.
- Deux échanges réels terminés après les correctifs, avec analyse et relecture nano, rédaction Sonnet, verdicts validés et absence d’alerte injustifiée. Respect de « sans conseil ni question » et explication non diagnostique contrôlés dans les textes produits. Les entrées exactes des trois étapes de chaque réponse sont consultables ; le contrôle conditionnel supplémentaire n’était pas nécessaire dans ces deux traces et reste testé séparément.
- Paramètres (trois onglets), portrait (deux onglets), cartographie (six volets, sauvegarde réelle, centrage et résultats) vérifiés à 1 440, 390 et 320 px. Aucune erreur JavaScript ni débordement. Export des traces réussi.
- Compte synthétique supprimé par l’API avec vérification de la déconnexion, identifiants locaux effacés. Contrôle D1 ciblé : zéro ligne restante dans les 40 tables liées au compte ; aucune violation de clé étrangère. Aucun dossier réel utilisé.

Les contrôles logiciels et ces échanges ne constituent pas une garantie clinique universelle. L’audit de couverture reste la référence pour les lacunes scientifiques et les limites de classification.
