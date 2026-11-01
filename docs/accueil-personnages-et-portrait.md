# Accueil centré sur les personnages et portrait unifié

Demande du1er novembre2026. Cette version remplace les choix précédents sur les encarts de découverte et les pages séparées.

1. Supprimer tous les encarts « Nos personnages » des pages personnelles, sans toucher aux données.
2. Alléger le début des231articles : titre, prénom, aperçu « Ce que cette histoire explore », puis récit. Retirer fil d’Ariane, badge de catégorie, date et durée visibles. Garder les thèmes après le récit et conserver le texte d’origine.
3. Enrichir les20présentations à partir des descriptions actuelles et du registre fictif. Les caractéristiques deviennent du texte naturel; les repères restent seulement des métadonnées de recherche, filtrage et recommandation.
4. Remplacer les trois rubriques de l’accueil par le catalogue des20personnages, avec les filtres existants. Rediriger l’ancienne liste /personnages/ vers /#personnages, tout en conservant les parcours individuels et les catégories d’articles existantes.
5. Regrouper les lectures aimées et toutes leurs explications modifiables dans Mon portrait. Conserver les deux champs historiques reason/differences. Retirer Mes articles aimés du menu et rediriger les anciens liens vers la collection du portrait en conservant leurs paramètres.
6. Vérifier compilation, couverture231/20, filtres, édition et conservation des notes, isolation des comptes, navigation et compatibilité des anciennes adresses; publier sur le même domaine.

État : implémenté, contrôles locaux réussis et publication en cours. Aucun changement des secrets, des droits d’abonnement, des automatismes de génération ou du contenu des récits n’est prévu.

## Contrôles réalisés

-20présentations de83à91mots, faits vérifiés contre les pages actuelles et le registre des personnages fictifs; aucun corps de récit modifié.
-231articles avec aperçu avant le récit, prénom sous le titre et thèmes après le corps, sans fil d’Ariane, badge, date ou durée visibles dans l’en-tête.
-40tests Node et8contrôles VM réussis, dont conservation des deux notes et des saisies effectuées pendant un enregistrement.
-Recette de l’accueil : filtre « Rumination » réduit les20personnages à Sophie; aucune pastille de caractéristiques.
-Recette du portrait : portrait et collection restent présents ensemble; les deux brouillons sont conservés lors d’un aller-retour entre pages.

-Contrôle mobile à390pixels : accueil filtrable et portrait avec deux notes éditables, aucun débordement horizontal.
