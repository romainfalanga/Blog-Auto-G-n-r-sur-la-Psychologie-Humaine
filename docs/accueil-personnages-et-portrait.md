# Accueil centré sur les personnages et portrait unifié

Demande du 1er novembre 2026. Cette version remplace les choix précédents sur les encarts de découverte et les pages séparées.

1. Supprimer tous les encarts « Nos personnages » des pages personnelles, sans toucher aux données.
2. Alléger le début des 231 articles : titre, prénom, aperçu « Ce que cette histoire explore », puis récit. Retirer fil d’Ariane, badge de catégorie, date et durée visibles. Garder les thèmes après le récit et conserver le texte d’origine.
3. Enrichir les 20 présentations à partir des descriptions actuelles et du registre fictif. Les caractéristiques deviennent du texte naturel; les repères restent seulement des métadonnées de recherche, filtrage et recommandation.
4. Remplacer les trois rubriques de l’accueil par le catalogue des 20 personnages, avec les filtres existants. Rediriger l’ancienne liste /personnages/ vers /#personnages, tout en conservant les parcours individuels et les catégories d’articles existantes.
5. Regrouper les lectures aimées et toutes leurs explications modifiables dans Mon portrait. Conserver les deux champs historiques reason/differences. Retirer Mes articles aimés du menu et rediriger les anciens liens vers la collection du portrait en conservant leurs paramètres.
6. Vérifier compilation, couverture 231/20, filtres, édition et conservation des notes, isolation des comptes, navigation et compatibilité des anciennes adresses; publier sur le même domaine.

État : implémenté et publié sur https://decodetonesprit.fr/. Contrôles locaux et vérifications en ligne réussis. Les secrets, les droits d’abonnement, les automatismes de génération et le contenu des récits sont conservés.

## Contrôles réalisés

- 20 présentations de 83 à 91 mots, faits vérifiés contre les pages actuelles et le registre des personnages fictifs; aucun corps de récit modifié.
- 231 articles avec aperçu avant le récit, prénom sous le titre et thèmes après le corps, sans fil d’Ariane, badge, date ou durée visibles dans l’en-tête.
- 40 tests Node et 8 contrôles VM réussis, dont conservation des deux notes et des saisies effectuées pendant un enregistrement.
- Recette de l’accueil : filtre « Rumination » réduit les 20 personnages à Sophie; aucune pastille de caractéristiques.
- Recette du portrait : portrait et collection restent présents ensemble; les deux brouillons sont conservés lors d’un aller-retour entre pages.

- Contrôle mobile à 390 pixels : accueil filtrable et portrait avec deux notes éditables, aucun débordement horizontal.

- Publication Netlify réussie pour le commit b672c5d9; Worker publié.
- Vérifications en ligne : filtre Rumination → Sophie, aperçu et prénom visibles avant le récit, thèmes à la fin.
- Ancienne liste des personnages : 301 vers /#personnages; parcours individuel : 200. Ancienne collection : 302 vers le portrait avec paramètres conservés. API des likes anonyme : 401.
