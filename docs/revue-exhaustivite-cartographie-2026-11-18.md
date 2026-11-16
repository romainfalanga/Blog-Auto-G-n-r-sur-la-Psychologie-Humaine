# Revue critique d'exhaustivité de la Cartographie — 18 novembre 2026

Revue en lecture seule (aucun fichier existant modifié, aucun commit, aucun déploiement). Rédigée au fil de l'eau ; les sections marquées « en cours » sont complétées progressivement.

## 0. Périmètre et méthode

- Jeu examiné : `data/cartographie.json` (schéma 2) au commit `dd23271` — 59 questions, 6 thèmes, 109 cartes actives, 3 identifiants historiques, 493 liens, 260 sources, 11 ressources. Comptes recalculés par parseur JSON, identiques à `docs/integration-cartographie-2026-11-17.md`.
- Code lu : `static/js/cartographie.js` (moteur `compute` + ancien `mount` non utilisé par la page), `static/js/cartographie-ui.js` (interface réelle), `journal/src/cartographie.mjs` (calcul Worker, résumé transmis à Ambre), `layouts/_default/cartographie.html`.
- Documents lus : intégration et relecture finale du 17/11, plan d'exécution Ambre/Cartographie, lots de recherche (parcourus), registre `docs/audit-cartographie-correspondances-2026-11-17.json` (parcouru par parseur).
- Vérifications externes : recherches web ciblées et ouverture de sources ; chaque référence citée ci-dessous comme « ouverte » l'a été pendant cette revue. Aucune référence n'est proposée sans avoir été vue.
- Rappel : les décisions explicites du propriétaire (liste unique triée par nombre de vécus, retrait des introductions, retrait de l'encadré 3114 et du bouton de suppression) sont signalées comme telles ; la revue en mesure les effets sans les annuler d'autorité.

## 1. Constat d'ensemble (à lire d'abord)

Le graphe est sérieux, prudent et très documenté ; la plupart des défauts relevés ne sont pas des erreurs de fond mais des effets d'architecture. Les quatre problèmes les plus importants :

1. **Le tri public récompense la quantité de liens, pas la preuve.** L'interface (`cartographie-ui.js`, `renderResults`) trie par nombre de vécus cochés reliés à la carte, *tous types de liens confondus* (modèles cliniques, définitions et contextes compris). Le moteur calcule pourtant des niveaux (`convergence`, `piste`, `theorique`) que l'interface n'affiche plus (classe CSS seulement) ; Ambre, elle, reçoit l'ordre par familles de preuves. **La personne et Ambre ne voient donc pas le même classement.** Effets mesurés plus bas (§ 4).
2. **Le volume est illisible dès qu'une violence est cochée** : « Violences psychologiques » seule ouvre 49 cartes ; un profil de huit vécus de violence en ouvre 81 ; tout cocher en ouvre 109. Trois questions (violences physiques, sexuelles, psychologiques) portent 151 des 493 liens (31 %).
3. **Les 11 ressources protectrices existent dans les données mais ne sont plus proposées à l'écran** : aucune question sur les expériences positives (PCE/BCE). L'outil ne voit l'enfance que par ses manques.
4. **Plusieurs vécus fréquents et bien documentés sont presque muets** : précarité (3 liens, aucun vers dépression/anxiété), exil/guerre et migration (0 lien), secrets familiaux (0), témoin de violences entre adultes (7 liens dont 4 non empiriques).

