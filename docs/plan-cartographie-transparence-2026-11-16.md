# Cartographie et transparence d’Ambre — 16 novembre 2026

Ce chantier remplace les choix d’interface de la V3 et la première page Paramètres, à la demande de Romain. L’éditorial reste en pause. La publication déjà autorisée comprend le Worker puis le site après vérification.

| Exigence | Critère observable | Responsable | État |
|---|---|---|---|
| R1 Portrait | Seulement À propos de moi et Mon portrait dans le menu local | Principal | Vérifié en navigateur |
| R2 Paramètres | Données réelles consultables, sélection explicitée, modèles par étape, détails des nouvelles réponses | Principal + engine_review | Vérifié localement ; voir les preuves ci-dessous |
| R3 Six thèmes | Six accordéons fermés au chargement, aucun suivant, aucun septième thème | care_review | Vérifié localement ; voir les preuves ci-dessous |
| R4 Sauvegarde | Aucune case à cocher, sauvegarde automatique pour le compte connecté ; brouillon local sans compte ; erreurs et conflits visibles | care_review | Vérifié localement ; voir les preuves ci-dessous |
| R5 Mobile | Réponses et ajout d’explication centrés à 320 et 390 px, sans débordement | care_review | Vérifié localement ; voir les preuves ci-dessous |
| R6 Résultats | Tous les types sans filtre, explication associations/diagnostic, absence du bloc avant de commencer | care_review + clinical_coverage | Vérifié localement ; voir les preuves ci-dessous |
| R7 Vue d’ensemble | Hiérarchie lisible, absence d’espace vide au-dessus du titre | care_review | Vérifié localement ; voir les preuves ci-dessous |
| R8 Fond scientifique | Audit chiffré CIM-11/DSM-5-TR/modèles ; corrections sourcées ; limites de couverture explicites | clinical_coverage | Vérifié localement ; voir les preuves ci-dessous |
| R9 Calcul et Ambre | Classement prudent, cohérence client/serveur, retours négatifs et limites d’âge transmis, tests de non-régression | clinical_coverage + engine_review | Vérifié localement ; voir les preuves ci-dessous |
| R10 Livraison | Relecture indépendante, tests projet, navigateur mobile/bureau, déploiements vérifiés | Principal | Recette locale validée ; publication et recette distante en cours |

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

- 440 tests du journal, 31 tests du site, 43 tests personnels et les contrôles du workflow passent. Les deux calculs de cartographie concordent sur 1 711 cas (réponses isolées et toutes les paires).
- 13 scénarios Playwright de cartographie passent : accordéons, non explicite, brouillon, autosauvegarde sérialisée, erreurs, conflit 409, suppression, chargement initial tardif, sources et centrage mobile. Le bouton de précision est vérifié sur le même axe que les réponses.
- Un parcours intégré utilise le vrai Worker avec SQLite et un fournisseur simulé : trois entrées d’appels enregistrées à l’identique, contenu de Paramètres consultable, six thèmes publics Hugo et PUT réels, deux onglets du portrait, absence d’erreur JavaScript et de débordement à 1 440, 390 et 320 px.
- Relecture indépendante : navigation clavier, correction de mémoire, conflit CAS, filtres de recherche et affichage mobile validés. Deux écarts de transmission clinique repérés puis corrigés : preuve coupée avant les 1 100 caractères et métadonnées omises du contexte historique.
- Audit exhaustif des identifiants du projet : 858 codes du registre, 161 repères, 82 sujets de cartographie ; correspondances sémantiques justifiées pour les 161 repères. Les 22 chapitres diagnostiques du DSM-5-TR ont un statut thématique ; cela ne certifie pas l’intégralité des critères ou diagnostics du manuel. Les 425 liens ne sont pas tous relus dans le texte intégral des études : cette limite est documentée, sans allégation de validation clinique complète.

Les limites et références sont dans [l’audit](audit-cartographie-transparence-2026-11-16.md) et [ses correspondances détaillées](audit-cartographie-correspondances-2026-11-16.json). Aucun ajout de liens causaux destiné à remplir artificiellement le catalogue.

Dernière relecture clinique : le résumé de la preuve primaire et son URL sont conservés dans des métadonnées distinctes du texte raccourci. La nuance sur la foi et l’emprise survit à l’extrait. Le surcoût maximal mesuré est de 464 caractères par source ; les textes restent bornés à 6 500 caractères pour huit sources. Les étiquettes distinguent symptômes, groupes de troubles et qualificatifs cliniques.
