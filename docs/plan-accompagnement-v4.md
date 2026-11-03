# Accompagnement conversationnel — plan, relecture et livraison V4

## Exigences et décisions

| ID | Résultat attendu | Critère de vérification |
|---|---|---|
| R1 | Même menu partout | Source unique `data/navigation.json`, mêmes sept liens dans le même ordre, menu mobile compris |
| R2 | Mon personnage minimal tant que verrouillé | Aucun nom, carte, quota ou histoire affiché avant abonnement/essai valide + 10 likes + présentation + portrait à jour |
| R3 | Conversation proactive | Écoute du besoin, questions ouvertes, contexte des proches/événements/portrait, historique privé |
| R4 | Écritures autorisées par la personne | Proposition lisible et modifiable, accepter/refuser, contrôle de révision et propriété côté serveur |
| R5 | Structure d'exploration psychologique | Six domaines, douze patterns, observations, contre-exemples, ressources, questions et étapes ; aucune grille diagnostique |
| R6 | Article demandé depuis la conversation | Proposition d'angle, confirmation puis même file de génération et même quota que Mon personnage |
| R7 | 100 messages et 7 histoires par semaine | Réservations atomiques, idempotence, crédit libéré en cas d'échec, lundi 00:00 Europe/Paris |
| R8 | 29 €/mois, 7 jours gratuits | Moyen de paiement requis, essai après Checkout vérifié, renouvellement automatique, annulation avant échéance |
| R9 | OpenRouter existant | Secret exclusivement serveur, modèle configuré, limites d'entrée/sortie et contrôles de confidentialité |

Le chat est disponible pendant l'essai ou l'abonnement avant les dix likes ; l'histoire reste soumise au portrait prêt et aux dix likes. « Articles » désigne ici les épisodes personnalisés existants avec leur analyse, pas un second catalogue facturé séparément. Les histoires demandées dans la conversation peuvent recevoir un angle approuvé pour répondre au besoin du moment.

Les quotas sont calendaires, sans report, et ne se réinitialisent ni à l'abonnement, ni à la suppression de l'historique. Un essai qui traverse le lundi touche deux semaines calendaires. Un seul essai par compte ; l'historique Stripe est également contrôlé. Ce dispositif ne prétend pas empêcher toute création abusive de comptes différents.

## Plan d'implémentation

1. Unifier la navigation publique Hugo et l'application Worker ; cacher le contenu de Mon personnage dès la garde d'entrée.
2. Centraliser les droits `active`/`trialing` et adapter tous les contrôles SQL, les webhooks, le Checkout et l'annulation.
3. Ajouter les conversations, consentement dédié, propositions et compteur hebdomadaire en migration additive ; conserver les tables personnelles et historiques.
4. Construire le guide de conversation et les contrats JSON ; appliquer les propositions seulement après validation humaine explicite.
5. Réutiliser les validateurs métier et les écritures atomiques avec versions attendues ; réutiliser le service des histoires.
6. Relier l'angle approuvé au pipeline analytique, sans transférer directement une scène intime au rédacteur.
7. Intégrer l'interface, les états d'attente et les réessais, vérifier indépendamment, tester en isolation, puis publier et contrôler la production.

## Relecture du plan et améliorations retenues

- **Deux menus divergent** : une simple copie finirait par diverger encore. Générer le menu Worker depuis le même JSON que Hugo et vérifier le fichier généré en CI.
- **Alex apparaît avant déblocage** : sortir du rendu avant toute récupération de récit ancien, retirer aussi l'ancien personnage fictif du portrait. Ne pas effacer les données.
- **Essai reconnu seulement dans le chat** : remplacer l'ensemble des gardes de génération par un contrat partagé. `trialing` doit avoir une fin d'essai future ; un simple statut local ne suffit pas.
- **Retour Checkout trompeur** : ne jamais accorder les droits depuis un paramètre d'URL. Vérifier la souscription Stripe canonique ; ne pas recycler une ancienne session à 10 €.
- **Essais répétés** : conserver `trial_used_at` et chercher l'historique distant, même après une perte de webhook.
- **Conversation qui invalide sans cesse le portrait** : l'historique reste séparé des sources de personnalisation. Seule une proposition acceptée modifie ces sources et invalide l'analyse précédente.
- **Double clic / réseau perdu** : clé idempotente pour le message et pour l'histoire, un seul message en cours par compte ; récupération par polling.
- **Latence Netlify** : réponse immédiate 202, traitement Worker `waitUntil`, polling de l'état plutôt que connexion HTTP longue.
- **Conflits entre deux onglets** : proposition liée à une révision précise ; refus 409 si les données ont changé, sans écrasement silencieux.
- **Instructions injectées dans le chat** : sorties structurées et validées, champs autorisés, cibles et révisions fixées côté serveur, aucune exécution de code ou URL d'outil du modèle.
- **Confusion entre hypothèse et autobiographie** : citations utilisateur et formulation attribuée ; aucun diagnostic ou causalité familiale affirmée sur la seule base d'un indice.
- **Coûts de réessais** : crédits commerciaux libérés après échec, mais compteur d'appels fournisseur conservé ; budgets bornés et pas de fallback moins protecteur.
- **Suppression qui rend le quota gratuit** : purger le contenu mais conserver les compteurs minimaux de consommation ; suppression du compte en cascade.
- **Article générique malgré une demande précise** : ajouter `requested_focus`, inclus dans l'idempotence et transmis à l'analyste. Le planificateur/rédacteur n'en reçoivent que les orientations abstraites contrôlées.

## Structure psychologique et limites

`journal/src/psychology-framework.mjs` définit six domaines : contexte, émotions, pensées, relations, actions, parcours. Chaque pattern contient signaux possibles, alternatives, questions ouvertes, ressources et petite expérimentation facultative. La boucle d'entretien est : écouter → exemple concret → fonction et effets → contre-exemple → reformulation prudente → permission d'enregistrer → révision.

Cette structure est un guide éditorial de réflexion personnelle, pas un instrument clinique validé, un score de personnalité ou un diagnostic. « Sauveur » et « intellectualisation » restent des hypothèses descriptives à discuter. Le produit se présente comme un assistant IA, sans usurper le titre de psychologue. En cas de détresse aiguë, l'entretien de profil et les propositions sont suspendus au profit de la sécurité et de l'aide humaine. Il ne doit pas renforcer une croyance délirante, conseiller l'arrêt d'un traitement ou encourager une dépendance à l'IA.

Le premier usage du chat demande une activation explicite et explique le traitement via OpenRouter. Les données civiles inutiles sont exclues du contexte ; l'utilisateur peut arrêter l'utilisation et supprimer l'historique. Les modifications déjà acceptées restent dans les pages concernées, où elles sont modifiables. Cette mise en œuvre technique ne constitue pas une certification réglementaire ou clinique.

## Choix du modèle et coûts

Choix initial : `anthropic/claude-sonnet-5.5`, configurable côté serveur par `OPENROUTER_CHAT_MODEL`, sans modifier le modèle existant des histoires. Motivation : qualité de communication et capacité à produire des réponses structurées ; le choix doit être confirmé par des scénarios synthétiques, pas par une prétendue supériorité clinique.

Tarif public consulté le 3 novembre 2026 : 2 USD/million de tokens d'entrée et 10 USD/million de tokens de sortie. Exemple de calcul, pas une facture mesurée : 6 000 tokens d'entrée + 1 000 de sortie = 0,022 USD par tour ; 100 tours = 2,20 USD par semaine, hors histoires, raisonnement supplémentaire, taxes et frais. La longueur du contexte et des réponses est bornée. Les plafonds techniques ne constituent pas une garantie de marge.

Routing exige `data_collection=deny`, `zdr=true` et support des paramètres ; aucune bascule vers un fournisseur moins protecteur en cas d'indisponibilité. Les messages ne sont jamais journalisés dans la console. La clé OpenRouter reste un secret du Worker.

## Sources primaires consultées

- [NHS — Thought record](https://www.nhs.uk/every-mind-matters/mental-wellbeing-tips/self-help-cbt-techniques/thought-record/) : situation, pensées, émotions et recherche d'éléments qui nuancent une pensée.
- [CCI — Self-help resources](https://cci.health.wa.gov.au/Resources/Looking-After-Yourself), [assertivité](https://www.cci.health.wa.gov.au/Resources/Looking-After-Yourself/Assertiveness), [perfectionnisme](https://www.cci.health.wa.gov.au/resources/looking-after-yourself/perfectionism) : ressources de compréhension personnelle ; ne remplacent pas un suivi professionnel.
- [OpenRouter — Sonnet 5.5](https://openrouter.ai/anthropic/claude-sonnet-5.5), [sorties structurées](https://openrouter.ai/docs/guides/features/structured-outputs), [ZDR](https://openrouter.ai/docs/guides/features/zdr).
- [Stripe Checkout API 2024-06-20](https://docs.stripe.com/api/checkout/sessions/create?api-version=2024-06-20) et [gestion des essais](https://docs.stripe.com/billing/subscriptions/trials). Conserver la version API déjà épinglée ; ne pas mélanger le nouvel outil Trial Offers avec les paramètres legacy Checkout.
- [CNIL — chatbots](https://www.cnil.fr/fr/chatbots-les-conseils-de-la-cnil-pour-respecter-les-droits-des-personnes) : information, minimisation, conservation adaptée et traitement des données sensibles.

## État et vérifications

Implémentation terminée et relue. 138 tests JavaScript du journal, 16 tests de recherche et 71 tests Python réussis ; construction Hugo réussie. Neuf contrôles API isolés ont validé consentement, réponse asynchrone, quota, absence de mutation automatique, réessai, proposition modifiée puis acceptée, verrouillage du personnage, export et purge. Stripe n'est pas configuré en production (absence des trois bindings nécessaires constatée le 3 novembre 2026). Les tests simulés pourront vérifier le code de facturation ; un essai réellement souscrit et une facturation Stripe ne doivent pas être annoncés fonctionnels sans configuration et recette Stripe réelles.

### Corrections issues de la vérification indépendante

- Historique, propositions et suppression restent visibles lorsque le quota ou l’abonnement bloquent un nouvel envoi.
- Un échec terminal renouvelle la clé du réessai, une erreur réseau ambiguë conserve la clé initiale.
- Les mises à jour IA utilisent un patch explicite, évitant d’effacer des dates ou préférences absentes du contexte envoyé au modèle.
- Le contexte réserve une place aux différentes familles de sources ; de nombreux likes ne peuvent plus évincer systématiquement les proches ou événements.
- Les citations exactes et l’état antérieur apparaissent dans l’aperçu ; liens vers des proches et importance d’un événement sont modifiables.
- Les contrôles backend conservent leur rôle même si le modèle fournit un objet JSON conforme au schéma.

### Compatibilité réelle du fournisseur

Le catalogue OpenRouter a confirmé `anthropic/claude-sonnet-5.5`, les sorties structurées et les prix. Ce modèle ne supporte pas `temperature` : le paramètre a été retiré. Une première grammaire JSON à six variantes a été rejetée comme trop complexe par le fournisseur ; elle a été remplacée par une enveloppe unique et des changements de champs, tous revalidés côté serveur. Le schéma ne contient plus de `anyOf`. Les erreurs fournisseur ne journalisent ni messages privés ni secrets et ne consomment pas de crédit utilisateur.

Le test réel en production, avec un compte synthétique ensuite supprimé, a validé les neuf contrôles ci-dessus avec Sonnet 5.5 : réponse en 11,1 secondes, 6 638 tokens d'entrée et 504 de sortie, coût fournisseur rapporté de 0,018316 USD. Il ne s'agit pas d'une mesure représentative de tous les échanges. Aucun abonnement Stripe réel ni génération d'histoire payante n'a été déclenché pendant cette recette. Le nombre de comptes est revenu à sa valeur initiale après nettoyage.
