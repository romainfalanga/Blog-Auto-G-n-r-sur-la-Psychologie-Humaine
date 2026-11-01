# Accueil, aperçus et pages personnelles

Demande du 1er novembre 2026, reprise après interruption. Le motif interne de la suspension n'est pas accessible. Le dépôt et le déploiement publié ont été retrouvés; les modifications de cette demande n'étaient pas appliquées lors de la reprise.

| Objectif | Mise en œuvre | Vérification |
|---|---|---|
| Accueil allégé | Retirer le bloc « Et si le prochain personnage… »; ajouter une seule phrase cohérente au paragraphe d’introduction | Absence du bloc, phrase et lien dans le paragraphe |
| Comprendre avant de lire | Aperçu court issu de la définition ou de la partie explicative de chaque article, placé avant le récit | Couverture des231articles, source connue et corps Markdown conservé |
| Menu général | Accueil, Nos personnages, Mes histoires, Mes articles aimés, Mon portrait, Mon compte | Même menu mobile/ordinateur sur blog et pages personnelles |
| Pages personnelles distinctes | /mes-histoires/, /mes-articles/, /mon-portrait/, /mon-compte/ avec adresse et titre propres | Ouverture directe, actualisation, précédent/suivant, authentification et anciens liens |
| Retirer le hub | Supprimer la barre d’onglets centrale; rediriger /journal/ vers la page correspondante | Liens article, portrait, paiement et connexion conservés |
| Découverte des personnages | Lien bien visible sous les catégories et sur les quatre pages personnelles | Accès à d’autres lectures, sans gêner les actions courantes |
| Préserver l’existant | Même domaine, données, articles, automatisation et protections d’accès | Compilation, tests pertinents et recette responsive |

Les aperçus reprennent uniquement le texte déjà présent dans la section pédagogique. Ils n'ajoutent aucun diagnostic ni affirmation nouvelle. La partie narrative et les articles d'origine restent conservés. Les quatre pages partagent les composants du site mais disposent chacune d'une route ouvrable et rechargeable; les anciennes adresses restent compatibles.

État : implémentation publiée et vérifiée. Aucune modification des secrets, des abonnements ni suppression des données existantes n'est nécessaire pour ces changements.

## Résultats

- 231 aperçus pédagogiques vérifiés avant le récit, de135 à449caractères, sans modification des corps Markdown. Le build Netlify les régénère pour chaque publication.
- 37 tests Node réussis, 6 scénarios VM de navigation et 5 vérifications indépendantes supplémentaires. Les réponses périmées ne remplacent plus une nouvelle page.
- Accueil, aperçu d’article et nouvelle navigation contrôlés sur ordinateur et dans une vue de390pixels, sans débordement horizontal.
- Publication en deux étapes : proxys préparés et vérifiés avant mise en place des nouveaux liens. Les routes personnelles ont chacune un titre HTTP et gardent les protections des API privées.
- Aucun changement des secrets, aucune migration ni suppression des données de production pour ce chantier.
