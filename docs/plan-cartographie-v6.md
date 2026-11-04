# Décode ton esprit — Cartographie et accès de test

## Plan et relecture

| Attente | Mise en œuvre | Contrôle |
|---|---|---|
| Recherche sans propositions | Suppression du panneau et de ses listeners ; filtrage conservé | Focus et saisie sans popup, lien natif |
| Psychologie des personnages visible | Cinq repères canoniques complets, bio compacte | Contrôle mobile 320/390 et desktop |
| Portrait lisible et actionnable | Synthèse, À propos de moi ouvert, Articles aimés, Ma cartographie | Navigation, brouillons et liens profonds |
| Articles aimés simples | Liste ouvrable au clic, un retour principal explicite, anciennes différences conservées | Sauvegarde et retrait sans filtres |
| Pas de footer au milieu | Footer retiré des pages de l’application | États courts et vides |
| Accès sans paiement | Flag serveur FREE_ACCESS_ENABLED, droits séparés de Stripe | Chat, préparation et génération autorisés, seuils et quotas conservés |
| Révision après intervention | Invalidation par version, job durable, debounce court, exécution native et cron de rattrapage | Ancien résultat refusé après mutation, pas d’appels au simple rendu à jour |
| Compréhension structurée | Catalogue commun de 53 processus/ressources en 10 domaines et hypothèses sourcées | Identifiants fermés, citations exactes, contre-indices et corrections |

Direction visuelle : même palette claire, texte #2D2D2D, accent #3D8577, fonds #FFFFFF/#FAFAF8, séparateurs #E8E8E4. Police système dans l’application ; disposition alignée à gauche, champs regroupés et repères compacts. Les cartes de fonctionnement ont une structure commune observation → fonction possible → alternatives → preuves → question. Les hypothèses actives précèdent les domaines à explorer.

Le plan a été revu pour éviter une simple liste de symptômes, la causalité familiale automatique, les faux pourcentages et l’accumulation artificielle de certitudes. Aucun inventaire ne couvre « toute la psychologie ». Le catalogue est éditorial, extensible, non exhaustif, sans classification automatique de troubles. Une piste non explorée reste inconnue. « Convergent » signifie des sources personnelles de types différents qui se recoupent ; il ne s’agit pas d’une probabilité diagnostique. Les corrections peuvent diminuer ou contredire une piste.

## Sources et méthode

Le [NHS Thought record](https://www.nhs.uk/every-mind-matters/mental-wellbeing-tips/self-help-cbt-techniques/thought-record/) fournit une structure utile : situation, pensées, émotions, éléments favorables/défavorables et alternatives. Les [ressources CCI](https://cci.health.wa.gov.au/Resources/Looking-After-Yourself) documentent notamment perfectionnisme, affirmation de soi et estime de soi. Le [NIMH RDoC](https://www.nimh.nih.gov/research/research-funded-by-nimh/rdoc/about-rdoc) illustre une organisation par domaines de fonctionnement et précise que son cadre de recherche n’est pas un guide diagnostique. Notre catalogue n’est ni une implémentation clinique de ces cadres ni un test validé.

Les sources originales restent canoniques : présentation, relations racontées, événements, corrections et retours explicites sur les articles. Un like seul reste un intérêt. La fiction ne prouve pas le vécu du lecteur. Les pistes utilisent des IDs de catalogue, des citations exactes de champs personnels, des alternatives et une prochaine question. Aucun parent n’est diagnostiqué et aucune intention cachée ne lui est attribuée.

Le même catalogue alimente le chat, l’analyse native et le pipeline Python. Les recommandations peuvent enrichir leur recherche avec des concepts fermés de pistes actuelles, à poids limité et en réutilisant l’ID de la source initiale : une hypothèse ne crée pas une preuve indépendante. Les pistes contredites/mixtes ne renforcent pas les recommandations. La cartographie propose des liens vers les articles et personnages correspondants, ce qui reste une aide à explorer et pas une validation de l’hypothèse.

## Garde-fous opérationnels

L’accès de test n’altère aucun statut Stripe et n’ouvre aucun paiement. Connexion, consentement chat, propriété des données, quotas de 100 messages et 7 histoires par semaine, ainsi que préparation du personnage restent contrôlés côté serveur. L’IA coûte toujours des crédits OpenRouter au propriétaire ; budgets, plafonds et idempotence restent actifs.

Les modifications d’informations invalident les analyses en cours. L’interface montre les états en attente et actualise les vues d’analyse sans remplacer les brouillons. Les retours de cartographie sont ajoutés atomiquement aux corrections existantes et sont idempotents. Le formulaire général de correction envoie une version attendue pour éviter un écrasement concurrent.

## État et vérification

Implémentation terminée et relue. 178 tests JavaScript du journal, 16 tests de recherche et 72 tests Python réussis (266 au total), compilation Hugo et contrôle du catalogue/menu réussis. Recette navigateur : cartographie active et preuves/citations, formulaires ouverts, liste d’articles sans filtres et formulaire au clic ; vue mobile de 390 px. Les onglets ont été disposés en grille 2×2 sur mobile après cette recette. Le clavier matériel d’un téléphone n’a pas été testé.

Relecture indépendante : ajout d’un contrôle sémantique séparé dans le pipeline natif ; vérification du concept et de la fonction hypothétique dans les deux pipelines ; copies d’une même citation neutralisées pour la convergence ; gros contextes déférés au traitement Python sans consommer de tentative ni appeler le modèle. La couverture est calculée côté serveur, jamais recopiée par le modèle.

La recette réelle OpenRouter et les références de publication seront ajoutées après déploiement. Aucun paiement Stripe n’est nécessaire au mode de test.
