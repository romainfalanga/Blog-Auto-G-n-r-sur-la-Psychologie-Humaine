# Cartographie et accompagnement — V7

## Objectif et décisions

L’accueil conserve les mots-clés des personnages et retire leurs introductions. Mon portrait présente une synthèse qui parle de la personne avec son prénom déclaré. Ma cartographie devient un atlas de domaines et sous-domaines : les pistes personnelles d’abord, puis une exploration du catalogue, avec une seule fiche détaillée à la fois.

Le catalogue est élargi aux processus cognitifs et émotionnels, relations, apprentissages familiaux, sécurité affective, adaptation, fonctions exécutives, corps, récupération, apprentissage, intimité, identité, contexte et ressources. Il ne prétend pas couvrir exhaustivement la psychologie ou constituer une classification diagnostique. Les références documentent des familles de notions, pas une validation clinique de chaque fiche.

## Plan d’implémentation

| Demande | Mise en œuvre | Vérification attendue |
|---|---|---|
| Accueil plus compact | Retrait de la biographie visible ; mots-clés et index de recherche conservés | Construction Hugo, contrôle du HTML publié |
| Cartographie plus lisible | Tuiles de domaines, Mes pistes / Tout explorer, recherche et sous-domaines, index et fiche ; clavier et mobile | Tests DOM, rendu desktop et largeur 390 px |
| Couverture plus large | Catalogue partagé JSON → Worker et Python, anciens IDs conservés ; indices, alternatives, questions et liens | Cohérence des IDs, relations, sources et sous-domaines |
| Prénom et accords | Identité d’affichage séparée des preuves ; remplacement ciblé dans la prose, citations intactes | Changement de prénom sans réanalyse, non-modification des citations et des tiers |
| Message IA en échec | Gestion des erreurs OpenRouter, délai de reprise, quota libéré ; secours fournisseurs sous les mêmes contraintes de données | Tests 429, idempotence, délais, reprise UI, appel réel sur compte de test |
| Accompagnant proactif | Écoute du besoin, une question à la fois, recherche de nuances ; choix d’articles existants parmi des IDs canoniques | Articles cliquables dans le fil, aucune URL inventée ni écriture de profil automatique |

## Stratégie de pertinence

L’atlas complet reste consultable dans l’interface. L’analyse de portrait reçoit l’index de tous les concepts et un guide détaillé de 24 fiches sélectionnées selon les sources, avec diversité des domaines. Le dialogue reçoit huit fiches pertinentes, les hypothèses actuelles et un contexte borné. Ce filtrage est une recherche textuelle, pas une probabilité psychologique. Il n’établit pas que toutes les fiches ont été évaluées. Un concept non détaillé reste accessible dans l’index de l’analyse.

Les preuves restent des déclarations originales avec citation exacte et révision. Un intérêt de lecture ne devient pas une biographie. Les corrections et contre-indices peuvent affaiblir une hypothèse ; plusieurs reprises du même récit ne créent pas une convergence. L’analyse native et le traitement Python conservent leur validation syntaxique et leur vérification sémantique distinctes. Les données civiles d’affichage ne sont pas ajoutées aux preuves ou au contexte du rédacteur de fiction.

Pour les suggestions de lecture, la demande courante est un signal éphémère ajouté au moteur existant, sans mutation du portrait. Le serveur fournit au maximum six candidats du catalogue. Le modèle peut choisir trois IDs au plus ; titres et URLs sont reconstruits depuis ces candidats, puis conservés dans l’historique. La demande d’une lecture et la génération d’une nouvelle histoire restent des opérations distinctes.

## Analyse des failles et corrections

- Agrandir le catalogue sans sélection aurait augmenté coût, latence et risque de saturation : guides bornés, index complet réservé à la synthèse.
- Une longue liste d’accordéons serait inutilisable sur mobile : navigation par domaine, index filtré, détail unique et navigation clavier.
- Personnaliser toute une phrase par des accords globaux modifierait les propos sur les proches : accords limités aux attributs directs, citations et passages entre guillemets préservés.
- Un 429 peut venir du fournisseur ou d’un plafond partagé : aucun changement de modèle ni retry automatique sur ce code. `Retry-After` est respecté ; le quota utilisateur est libéré.
- Un bouton désactivé pendant le délai peut rester bloqué : réactualisation automatique à l’expiration, avec brouillon conservé.
- Une relance après timeout peut facturer une deuxième génération : pas de replay ambigu, deux tentatives au maximum uniquement pour erreurs serveur explicites et dans 23 secondes.
- L’autorisation de secours ne doit pas relâcher les préférences de données : `zdr`, `data_collection:deny`, `require_parameters` conservés.
- Une réponse retardée ne doit pas écrire sur des données modifiées : gardes de consentement, propriétaire, version, lease et idempotence conservées.

## Références techniques

- [OpenRouter : erreurs et Retry-After](https://openrouter.ai/docs/api/reference/errors-and-debugging)
- [OpenRouter : routage et paramètres fournisseurs](https://openrouter.ai/docs/guides/routing/provider-selection)
- [Cloudflare : durée de waitUntil](https://developers.cloudflare.com/workers/runtime-apis/context/)
- Références du catalogue : `docs/catalogue-psychologique-v7.md`.

## Résultats de recette

- Catalogue livré : **161 repères / 16 domaines / 44 sous-domaines**, 25 références institutionnelles, anciens IDs préservés.
- **310 tests passent** : 237 JavaScript (journal et recherche), 73 Python. Générateurs à jour, syntaxe et `git diff --check` valides ; construction Hugo réussie.
- Recette visuelle sur version de test avec données fictives : atlas desktop et iframe de 390 px, recherche « rumination » avec libellé différent, fiche détaillée, prénom et cartes de lecture. Les textes d’en-tête ont encore été réduits après cette relecture.
- Un test réel a révélé qu’une recherche littérale manquait les expressions « je repense » et « refuser ». Ajout d’une correspondance de vocabulaire de lecture, éphémère, sans écrire de trait psychologique.
- **Chat réel OpenRouter réussi** sur compte synthétique en production : réponse en 11,8 s, deux lectures suggérées (liens HTTP 200), une question ouverte, 98 messages restants après deux échanges. Rejeu de la même clé idempotent, aucune écriture automatique dans le profil.
- **Portrait réel recalculé en 20,6 s**, version courante, hypothèse sourcée et prose utilisant `{{prenom}}` ; prénom exposé séparément à l’interface.
- L’échec initial du compte réel était un `provider_http_429_request`, crédit libéré. Aucun message réel du compte de Romain n’a été lu ou modifié. Aucun secret n’a été extrait. Le routage de secours conserve les contraintes de données.
- Un futur plafond fournisseur ne peut pas être empêché par l’application : délai et raison sont affichés, le message n’est pas décompté et l’envoi redevient disponible automatiquement. Aucune promesse d’absence de panne externe.


Le compte synthétique a été supprimé après recette : un seul compte réel reste présent, aucun chat en cours ni portrait en échec. La prévisualisation avec données fictives a été remplacée par le Worker normal.
