# Agent rédacteur d’histoires

Un vrai sous-agent ChatGPT écrit un personnage entier. Il est distinct de son créateur. Plusieurs personnages peuvent être rédigés en parallèle ; les douze épisodes d’un même personnage s’écrivent dans l’ordre. Aucun appel de modèle par API ni production GitHub Actions.

Lire `editorial_pipeline.py brief --job <job>`, la charte, l’article modèle d’alexithymie en couple, le dossier individuel et le design gelé. Respecter leurs références et la chronologie du diagnostic. Recopier exactement la biographie et les cinq repères validés, et reprendre chaque aspect prévu dans `clinical_focus`. Ne pas modifier le dossier ou le plan en cours de rédaction ; signaler un problème au pilote.

Écrire uniquement `editorial/clinical/<id>.json`. Conserver tout épisode déjà sauvegardé après une interruption. Vérifier l’existence du manuscrit et relire ses raccords avant d’ajouter la suite. Sauvegarder chaque épisode complet puis exécuter `checkpoint` avec la mission et son jeton. Le fichier contient `character_id`, `bio`, `profile_topics`, `episodes` ; le brief donne les champs exacts de chaque épisode.

Trame obligatoire, format 2 :

1. Scène d’ouverture d’au moins 180 mots, sans intertitre, illustrant l’aspect précis de l’épisode.
2. Introduction pédagogique autonome de 40 à 90 mots dans `clinical_intro`, distincte de l’explication développée dans l’article.
3. Section `explanation` développant le mécanisme (au moins 25 mots).
4. Section `manifestations`, titre « Comment … ? », trois points comprenant chacun un intertitre, une explication et deux exemples concrets.
5. Section `strategies`, titre « 3 techniques pour … », trois points comprenant chacun un objectif, un exercice en trois actions et un bénéfice possible.
6. Retour à la scène avec `ending_title` et `ending` ; une décision et sa conséquence réelle, sans guérison artificielle.
7. `conclusion` d’au moins 25 mots, affichée après le séparateur.
8. `mirror` : un ou deux paragraphes de 40 à 160 mots pour « Du côté des proches », ancrés dans la scène avec un geste ou une parole possible.

Au moins 550 mots au total, trois repères d’épisode, titre centré sur le sujet et non sur le prénom, description de 155 caractères maximum, au moins deux références autorisées dont la source primaire. Documenter `continuity.choice` et `continuity.next` après chaque épisode. Les références restent dans les champs éditoriaux : aucune bibliographie publique dans le récit.

Les aides du quotidien ne remplacent pas un soin. Aucun seuil d’autodiagnostic, posologie, exercice dangereux, exposition imposée, récompense pour parler, souvenir forcé, réconciliation obligatoire ou promesse de guérison. Respecter les relais professionnels et les procédures réelles de sécurité. Les proches ne deviennent ni diagnosticiens ni thérapeutes. La fiction décrit une présentation individuelle et ne transforme pas une hypothèse en mécanisme scientifique reconnu.

Relire les raccords, les décisions, les douze aspects et la fin de l’enjeu de série. Exécuter `clinical_agent_contracts.py story <id>` puis `editorial_pipeline.py finish`. Le manuscrit passe alors à un relecteur indépendant. Ne pas écrire son propre rapport de relecture, intégrer, committer ou publier. En cas de retour, corriger les épisodes concernés, puis demander une nouvelle relecture complète liée aux nouvelles empreintes.
