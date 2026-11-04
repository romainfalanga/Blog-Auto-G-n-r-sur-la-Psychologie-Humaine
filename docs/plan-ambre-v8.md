# Ambre — identité, profil fiable et formulation contextualisée

## Plan retenu et critères

| Demande | Réalisation prévue | Preuve requise |
|---|---|---|
| Nom Ambre partout | Navigation globale, titre de page, fil, formulaires et consignes ; route existante conservée | Recherche des libellés, rendu réel |
| Psychologue IA de la personne | Identité Ambre stable, prénom actuel dans le contexte séparé, ton direct sans remerciement introductif | Prompt et échange réel |
| Bonnes informations de profil | Contrat commun âge/genre/métier pour chat, analyste natif et Python ; priorité aux déclarations structurées | Tests frontière et contexte, versions concurrentes |
| Mauvais « elle » dans la synthèse | Recalcul des portraits anciens ; nouvelle prose sans pronoms ambigus, citations intactes | Validation V3 et portrait recalculé |
| Corrections par Ambre | Remplacement du formulaire par un lien ; proposition d’ajout de précision confirmée et citée | Validation, CAS mémoire, idempotence, mise à jour du portrait |
| Proches et ligne de vie | Liens courts vers Ambre et propositions créées/modifiées avec accord | Parcours chat → prévisualisation → confirmation |
| Mécanismes de poids différents | Influence qualitative contextualisée, séparée des preuves ; explications concurrentes et liens hypothétiques | Contrats Worker/Python, validation sémantique et UI |
| Accueil ordinateur | Recherche centrée, nouvelle phrase exacte | Hugo et navigateur ; compteur des personnages conservé |

## Architecture et recul

La formulation part des déclarations originales, du contexte présent et des réactions rapportées. Ambre explore une situation précise : interprétation, ressenti, besoin, réaction, effets à court et long terme, répétition, exceptions et appuis. Plusieurs mécanismes peuvent expliquer une même réaction ; une relation ou un événement ne permet pas d’affirmer une cause cachée.

L’idée de curseurs individuels devient une **influence décrite, située et révisable**, jamais un coefficient universel présenté comme mesuré. Deux dimensions restent distinctes : influence (à préciser/discrète/modérée/marquée selon le vécu rapporté), et solidité des indices (piste émergente/convergente/nuancée/contredite). Une forte gêne dans une seule situation peut coexister avec peu de preuves d’un mécanisme stable. Aucune donnée laisse le mécanisme inconnu, pas absent. Les ressources et les contraintes réelles peuvent modifier les réactions.

La cartographie garde les 161 repères et leur exploration ouverte. Les pistes actives peuvent préciser trois contextes au plus et trois liens hypothétiques avec d’autres pistes actives : amplification, atténuation ou autre explication. Chaque influence et lien doit référencer des citations personnelles ; un lien couvre les preuves des deux pistes. Une piste contredite ne peut pas amplifier les autres. Les nouvelles affirmations sont vérifiées sémantiquement contre les sources originales.

La recherche d’articles conserve le moteur existant : les hypothèses sourcées servent de vocabulaire de recherche avec une pondération prudente de l’influence, sans devenir une nouvelle preuve ni multiplier artificiellement les sources. Les quotas, la liste fermée d’articles, les contraintes fournisseurs et les garde-fous de concurrence sont préservés.

Références conceptuelles : [NIMH, approche dimensionnelle RDoC](https://www.nimh.nih.gov/research/research-funded-by-nimh/rdoc/about-rdoc) et [développement et environnement](https://www.nimh.nih.gov/research/research-funded-by-nimh/rdoc/developmental-and-environmental-aspects) ; [NHS England, compréhension partagée et personnalisée](https://www.england.nhs.uk/long-read/mental-health-personalised-care-framework-the-modern-care-programme-approach/). Il s’agit de principes de conception, pas d’une validation clinique de l’application ou de ses coefficients internes.

## Failles trouvées et corrections prévues

1. Le chat ne transmettait pas les informations structurées du profil. Le nouveau contexte d’identité est séparé des sources et prioritaire pour prénom et accords ; âge, genre et métier ne prouvent aucun trait.
2. Le traitement Python retournait `declared_gender` alors que ses consignes lisaient `gender`. Contrat commun explicite et testé avec l’analyste natif.
3. La personnalisation visuelle remplaçait « La personne » mais laissait un « Elle » ultérieur. Aucun remplacement global ne peut distinguer sûrement le sujet d’une mère ou d’une amie. Les anciens portraits sont donc recalculés ; l’analyse V3 utilise `{{prenom}}` et des rôles explicites, avec refus des pronoms ambigus hors citations.
4. Réécrire toute la mémoire pour ajouter une correction risquerait d’effacer des faits. `memory_append` ajoute seulement la précision confirmée, avec citation exacte, version attendue, limite existante de 10 000 caractères et protection contre les doubles ajouts.
5. Le type de proposition est contraint dans SQLite. La migration 0012 recrée la table en conservant les propositions et ajoute le nouveau type ; elle invalide seulement la version des anciens portraits, pas leurs sources.
6. Une génération ne doit pas publier avec un genre ou un contexte modifié entre-temps : gardes de version et de consentement conservées, contexte utilisé enregistré avec l’analyse.
7. Une intensité numérique inventée donnerait une précision trompeuse. Les qualificatifs et les contextes restent lisibles, sans pourcentage clinique ni jauge de personnalité.

## Vérifications et publication

- 351 tests réussis : 261 JavaScript et 90 Python. Générateurs cohérents, construction Hugo et vérifications de syntaxe réussies.
- Relecture indépendante : publication V3 imposée côté serveur même pour un ancien client ; le contexte courant ne peut pas être remplacé par une identité inventée dans une réponse modèle. Date de naissance complète requise dans la citation d’une proposition ; l’âge seul est refusé.
- Préférences d’accords conservées pour tous les genres ; les cas ambigus restent au contrôle sémantique avec le contexte courant, plutôt que des remplacements globaux sur les tiers.
- Prévisualisation avec données synthétiques vérifiée : liens synthèse/proches/ligne de vie vers Ambre, titre du fil et badge IA, influence et mécanismes liés, largeur 390 px.
- Migration 0012 appliquée en recette puis production ; test de conservation sur propositions préexistantes et contrôle des clés étrangères. Conversation existante conservée. Une courte pause de génération a séparé migration et reprise ; aucune génération n’était en cours.
- Échange réel OpenRouter sur compte synthétique : 15,4 s, identité Ambre/prénom actuel, absence de remerciement introductif, trois propositions (précision, proche, événement), aucune écriture avant confirmation. Après confirmation : trois ajouts enregistrés, rejeu sans doublon, profil versionné.
- Le recalcul du portrait complexe a mis en évidence un rejet trop global : une seule hypothèse refusée faisait échouer les autres sections. Le contrôle vérifie désormais aussi les questions et retire uniquement les parties non étayées. Un résumé non soutenu ou l’absence d’axe validé empêchent toujours la publication ; aucun appel modèle supplémentaire. Les diagnostics ne contiennent que des chemins de champs, aucun texte privé. Délai de nouvelle tentative native ramené à 30 s aussi pour le cron ; trois tentatives et budgets maximaux conservés. Les estimations d’influence sont explicitement formulées comme des pistes à préciser.
- Derniers contrôles du portrait recalculé et publication GitHub/Netlify : en cours.
