# Ambre et les deux cartes — 19 novembre 2026

Le module `symptom-map-context.mjs` remplace le résumé des questionnaires v1. Les réponses enregistrées de Mes symptômes sont lues sans accord supplémentaire. Les prérequis de discussion restent ceux de Mon enfance et du profil.

## Contexte

Avec Mes symptômes renseigné : aperçu fixe de 3 500 caractères, aperçu Mon enfance de 3 000 caractères ; index complet par thème et sujet, avec Oui, Non, Je ne sais pas, absence de réponse distincte, précisions personnelles, retours et caractéristiques manquantes. Les thèmes et sujets offrent des parties séparées pour que les extraits puissent conserver chaque réponse plutôt que couper arbitrairement un gros JSON. Sans carte symptômes, le chemin historique reste identique : Mon enfance et ses notes sont disponibles dans l’index, sans ajouter un bloc redondant qui évincerait les sources antérieures.

L’analyse reçoit un résumé symptômes de 600 caractères. Les bilans ont leur propre bloc de 6 000 caractères (réductible à 3 000), aperçu et notes personnelles séparés ; seules les notes effectivement fournies deviennent des citations admissibles pour le digest. Le contexte maximal est de 40 000 caractères, l’enveloppe complète étant aussi plafonnée avec les consignes et le schéma. Les notes absentes de l’extrait sont décomptées.

Paramètres montre l’aperçu actuel et les traces exactes gardent les blocs fournis. L’index ouvre les deux pages séparément.

## Hypothèses et sécurité

Une hypothèse nommée doit préciser ses appuis et les informations manquantes. Aucun diagnostic affirmé, score ou causalité individuelle inventée. Les orientations professionnelles ne sont pas systématiques : danger, forte gêne quotidienne ou demande. Le critère de relecture et la consigne de Mon enfance sont cohérents.

`safety_at` est fixé par le serveur lors du premier Oui de sécurité. Un Oui toujours présent conserve le drapeau même au-delà de 14 jours ; après un retour Non, l’antécédent récent reste pris en compte pendant 14 jours. `safety_asked_at` mémorise uniquement la question effectivement publiée, dans la transaction de la réponse. Les questions de sécurité issues du seul questionnaire sont espacées d’au moins 14 jours ; le danger actuel du message garde son traitement prioritaire. Les ressources 3114 et 15/112 sont garanties même si le modèle pose la question sans les numéros.

Les lectures, appels fournisseur et publications sont protégés contre la modification/suppression de la carte : version, date, safety_at et révision monotone du périmètre de partage. Cela couvre aussi une suppression suivie d’une recréation dans la même seconde. Les bilans conservent la révision avec leur brouillon pour vérifier les reprises. La mise à jour du seul `safety_asked_at` ne modifie pas la révision de contenu.

## Budgets et vérifications

Décision du propriétaire : rédaction au plus +40 %, bilan au plus +20 %. Les références historiques des tests sont conservées. Quand les deux cartes sont fournies, l’entrée du rédacteur est plafonnée à 24 900 caractères et le tour avec réécriture à 42 000 jetons réservés. Sans carte symptômes, le plafond historique de 30 000 et le contexte bilan de 34 000 restent inchangés. Les plafonds financiers quotidiens et hebdomadaires sont inchangés.

Vérifications locales : tests du nouveau contexte, notes sans réponse, incertitude, citations, sécurité et temporisation ; conversation et bilan sur SQLite réel avec fournisseur simulé ; profil maximal avec deux cartes et deux passes de rédaction/relecture. Les tests historiques B/C/D/E de présence des sources, notes datées, événements marquants, plan de changement et réécriture maximale passent sans relever leurs assertions. Compatibilité avant migration 0023 vérifiée. Les contre-tests indépendants de course de publication sont dans `symptom-map-publication-review.test.mjs`.

Ces vérifications ne prouvent pas une recette fournisseur distante. La publication et la recette du site relèvent de la livraison intégrée.
