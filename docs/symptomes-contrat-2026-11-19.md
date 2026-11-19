# Contrat de données « Mes symptômes » (19 novembre 2026)

Contrat pour les lots A (API, moteur, migrations), B (interface) et C (Ambre). Il découle du plan `docs/plan-mes-symptomes-2026-11-19.md` (version 1.1, qui fait foi) et de la source unique `editorial/symptomes/symptomes.json`. Toute évolution du contrat passe par une nouvelle `version` du contenu.

## 1. Fichiers

| Fichier | Rôle | Écrit par |
| --- | --- | --- |
| `editorial/symptomes/symptomes.json` | source versionnée : instruments, modules, règles, sécurité, correspondances, statut de relecture | lot 0 (à la main, relu) |
| `editorial/symptomes/licences/` + `MANIFEST.json` | pages de licence et formulaires officiels archivés, empreintes SHA-256 | lot 0 |
| `data/symptomes.json` | copie compacte lue par Hugo (modules `publie: true` seulement) | `scripts/build_symptomes.py` |
| `journal/src/symptomes-data.generated.mjs` | même contenu, `export const SYMPTOMES`, pour le Worker | `scripts/build_symptomes.py` |

`python3 scripts/build_symptomes.py` régénère ; `--check` valide et échoue si un généré est périmé (à ajouter à `.github/workflows/verify.yml` au lot A, à côté de `build_cartographie_module.mjs --check`). Le validateur contrôle : empreintes des archives et absence de fichier hors manifeste ; présence de chaque item officiel dans le texte archivé ; paliers contigus couvrant l'échelle, repères PHQ-9 / GAD-7 à 10, paliers 5/10/15/20 et 5/10/15, entrées PHQ-2 / GAD-2 à 2 ; grappes ITQ couvrant les 18 items ; 23 modules, drapeau `sensible`, une seule question de gêne « Non / Un peu / Nettement » par module, ≤ 30 mots, pas de double négation, question terminée par « ? », repère marqué `repere_non_valide` avec base et code ; formulations interdites dans les textes (« vous avez un trouble », « diagnostic de », « probable », pourcentages, « risque faible ») ; sujets existants, ni retirés ni contextuels, `associated_count` > 0, hors retraits v1.1, hors sujets de sécurité ; ressources 15, 112, 3114, 114, 3919, 119, 0 800 235 236 ; `publie: true` seulement avec un statut de relecture `relu` ou `retenu` et un rapport.

Les générés ne contiennent que les modules publiés ; `unpublished_modules` liste les autres. **Le serveur refuse toute réponse à un module non publié ou à une question inconnue.**

## 2. Identifiants et versions

- Identifiants de questions uniques dans tout le fichier : items d'instruments (`phq9_1`…`phq9_9`, `gad7_1`…, `phq15_a`…`phq15_m`, `phq15_sommeil`, `phq15_fatigue`, `itq_p1`…`itq_c9`, `itq_quand`), question de sécurité `securite_actuelle`, questions des modules.
- `version` du contenu (`2026-11-19.1`) et `instruments.<id>.version` (ex. `phq9-fr-FR-pfizer-201412`) sont enregistrées avec chaque passation. Un changement de traduction, de seuil ou de règle crée une nouvelle version ; les anciennes passations sont relues avec **leur** version (le Worker garde les versions précédentes nécessaires au recalcul, ou stocke le résultat calculé avec la version).

## 3. Forme des réponses (API, lot A)

Une passation = un module, enregistrée dans `symptom_checks` (migration `0028`), jamais écrasée.

```json
{
  "module": "humeur",
  "content_version": "2026-11-19.1",
  "instrument_versions": {"phq9": "phq9-fr-FR-pfizer-201412"},
  "answers": {
    "phq9_1": 2, "phq9_2": 1, "phq9_3": 0, "phq9_4": 3, "phq9_5": 0, "phq9_6": 1, "phq9_7": 0, "phq9_8": 0, "phq9_9": 0,
    "humeur_duree": "non", "humeur_perinatal": "pref", "humeur_gene": "un_peu"
  },
  "declined": false,
  "started_at": 1763542800, "completed_at": 1763543100
}
```

- Valeurs : items d'instruments = entier de l'échelle officielle (`phq_frequence` 0–3, `phq_gene_4_semaines` 0–2, `itq_0_4` 0–4), `itq_quand` = code ; questions du site = code de l'échelle (`non` / `oui` / `nsp`, `non` / `oui` / `pref`, `non` / `une` / `plusieurs`, `non` / `un_peu` / `nettement`).
- Toute question peut rester sans réponse (`null` ou absente) ; « Je préfère ne pas répondre » est toujours possible. **Aucune réponse manquante n'est comptée 0.** Seule exception : `phq15_d` accepte `"na"` (« Je ne suis pas concerné(e) »), compté 0 (point à confirmer, voir § 9).
- `declined: true` = la personne a choisi de ne pas faire le module (état « refusé »).
- Questions « vie entière » (`period: vie`) : posées une fois ; leur dernière réponse est reprise dans les passations suivantes (modifiable, jamais reposée d'office).
- Réutilisation : `phq15_sommeil` et `phq15_fatigue` reprennent `phq9_3` et `phq9_4` de la même session ; sinon ils sont posés avec l'échelle PHQ-9. PHQ-2 et GAD-2 sont les items 1–2 du PHQ-9 et du GAD-7 (jamais posés deux fois).
- Le **serveur recalcule** scores et résultats à partir des réponses et des versions ; il ignore tout score envoyé par le client.

## 4. Calcul des scores

| Instrument | Calcul | Résultat |
| --- | --- | --- |
| PHQ-2 / GAD-2 | somme des items 1–2 (0–6) | entrée positive si ≥ 2 (v1.1) ; le manuel officiel recommande ≥ 3 en usage seul (conservé dans `manual_threshold`) ; l'instrument complet reste proposé sur demande |
| PHQ-9 | somme 0–27 ; aucun score si un item manque | palier : 0–4, 5–9 léger, 10–14 modéré, 15–19 modérément sévère, 20–27 sévère ; au-dessus du repère si ≥ 10 |
| GAD-7 | somme 0–21 | 0–4, 5–9 léger, 10–14 modéré, 15–21 sévère ; repère ≥ 10 |
| PHQ-15 | 1a–1m cotés 0–2 ; sommeil et fatigue recodés 0→0, 1→1, 2→2, 3→2 ; somme 0–30 | 0–4, 5–9 faible, 10–14 moyen, 15–30 élevé ; repère ≥ 10 |
| ITQ | grappe positive si un de ses items ≥ 2 ; TSPT = réviviscence (P1/P2) + évitement (P3/P4) + menace (P5/P6) + atteinte (P7–P9) ; POS = régulation (C1/C2) + concept de soi (C3/C4) + relations (C5/C6) + atteinte (C7–C9) | `tspt_c` si TSPT et POS ; sinon `tspt` si TSPT ; sinon `aucun` ; exclusifs ; aucun résultat si un des 18 items manque ; sommes par grappe conservées, jamais présentées comme seuil |

Le PHQ-9 n'utilise pas sa question finale de difficulté (non notée selon le manuel) : la question de gêne du module la remplace. Les mentions d'origine (`notice`) s'affichent sous chaque instrument Pfizer ; l'ITQ cite Cloitre et al. 2018 et Peraud et al. 2022.

## 5. Évaluation des règles des modules

Chaque module a `entry` (questions ou instrument, `positive_if`), un `deepen` facultatif (`questions`, `instrument`, ou `branches[{if, questions|instrument}]`) et des `rules[{id, kind, when, result_positive, result_negative, subjects}]`.

Expressions (`when`, `positive_if`, `if`) :
- `{"q": id, "in": [codes]}` ou `{"q": id, "gte": n}` (items numériques) ;
- `{"score": "phq2|phq9|gad2|gad7|phq15", "gte": n}` ;
- `{"itq_result": "tspt|tspt_c|aucun|non_passe"}` (`non_passe` = ITQ non proposé car `exposition_evenement` ≠ oui : valeur connue, pas inconnue), `{"itq_cluster": "<grappe>"}` ;
- `{"all": [...]}`, `{"any": [...]}`, `{"not": e}`, `{"atleast": {"n": 2, "of": [...]}}`.

Logique à trois valeurs : une condition est **vraie**, **fausse** ou **inconnue** (réponse absente, `nsp`, `pref`, score non calculable). `all` = fausse si une est fausse, vraie si toutes sont vraies, sinon inconnue ; `any` = vraie si une est vraie, fausse si toutes sont fausses, sinon inconnue ; `atleast` = vraie si ≥ n vraies, fausse si le nombre de vraies plus d'inconnues < n, sinon inconnue ; `not` inverse vrai/faux et garde l'inconnu.

Parcours : si `entry.positive_if` est faux, le module s'arrête (règles du module négatives, sauf instrument demandé volontairement via `deepen_on_request`) ; s'il est vrai, `deepen` puis la question de gêne sont posés. Les règles `kind: entree` ne s'affichent que si l'instrument complet manque (`display: si_phq9_absent`, `si_gad7_absent`) ; `kind: grappe` (`display: croisement_seulement`) ne sert qu'au croisement ; `display: si_positif` = rien n'est affiché si négatif ; `kind: contexte` et `kind: orientation` n'ont jamais de sujet. `display_negative: si_itq_aucun` = le texte négatif ITQ ne s'affiche que si ni `tspt` ni `tspt_c`.

Phrase placée sous un texte positif (`rules[].positive_note`) : `positive_note` (phrase fixe v1.1, règles `instrument` et `repere`), `site_note` (règles `repere_site` : « façon de fonctionner, pas un trouble »), `texte` (`positive_note_text` propre à la règle, ex. restriction alimentaire), `aucune` (orientations, contexte, tabac, risques sexuels). Si `high_note_from` est présent (PHQ-9, GAD-7 : 15) et que le score l'atteint, `fixed_texts.high_note` remplace la phrase fixe. Trois règles positives ou plus dans la page : ajouter `fixed_texts.several_positive`.

## 6. États de résultat

Par règle : `positif`, `negatif`, `indetermine`. Par module :

| État | Condition | Affichage |
| --- | --- | --- |
| `au_dessus` | une règle `instrument` positive | texte positif (placeholders `{score}`, `{level}` ; `{level_clause}` = « , palier « léger » » si 5–9, sinon vide) puis `fixed_texts.positive_note` |
| `en_dessous` | règle `instrument` négative avec score calculé | texte négatif puis `negative_note` |
| `repere_present` | au moins une règle `repere`, `repere_site` ou `orientation` positive | chaque texte positif ; libellé `unvalidated_label` (règles `repere`) ou `unvalidated_label_site` (règles `repere_site`) ; puis `positive_note` ; « plusieurs repères présents » si ≥ 2 règles positives |
| `repere_absent` | toutes les règles évaluables négatives | textes négatifs (ou `module_negative`) puis `negative_note` |
| `indetermine` | aucune règle positive et au moins une inconnue | `fixed_texts.indetermine` |
| `non_explore` | module non commencé ou abandonné | — |
| `refuse` | `declined: true` ou entrée répondue `pref` | — |

La gêne (`<module>_gene`) est affichée à part (`gene_nettement` ; sinon `gene_absente_trouble`, ou `gene_absente` pour les modules de `gene_absente_trouble_except`). Elle conditionne les règles qui la citent explicitement (TOC, deuil, cauchemars, personnalité, sexualité compulsive). Les modules `sensible` affichent `fixed_texts.privacy_sensitive` avant la première question ; `sexualite.intro` est affiché en tête du module. Jamais de nom de trouble comme conclusion, jamais « vous avez », jamais « diagnostic probable ». Toute carte au-dessus du repère montre `what_next` sans défilement.

Délais : `modules[].retest_days` (calculé par le générateur = plus courte période de réponse du module ; PHQ/GAD 14 j, PHQ-15 28 j, ITQ 30 j, repères du mois 30 j, repères 12 mois / 2 ans / « en général » 90 j, vie entière jamais reposée). Refaire avant le délai est permis et affiche `overlap_note`.

## 7. Sécurité

- Page réservée aux adultes (`safety.adult_gate`) : âge ≥ 18 calculé depuis la date de naissance de Mon compte ; sinon texte et ressources `mineur`, **rien n'est enregistré**.
- `securite_actuelle` posée en tête de chaque passation du temps 1 (Non / Oui / Je préfère ne pas répondre), jamais notée.
- Déclencheurs (`safety.triggers`) : `securite_actuelle = oui`, `phq9_9 ≥ 1` (dès la réponse, avant la fin du PHQ-9, quel que soit le score total), `reg_automutilation = oui`, `rel_violences = oui` → **écran bloquant** immédiat : « Merci de l'avoir dit. », phrase `always`, ressources dont `show` contient le type du déclencheur (`suicide`, `automutilation`, `violences`), plus `fil_sante_jeunes` si 18–25 ans, dans l'ordre `display.blocking_screen.order` ; boutons « Continuer le questionnaire », « Parler à Ambre », « Arrêter », sans délai, non fermable d'un clic extérieur. `securite_actuelle = pref` → bloc doux non bloquant.
- Aucun score de risque, aucune phrase « risque faible / élevé », aucune notification externe.
- Messages propres : `blocking_screen.messages.automutilation` et `.violences` ; bouton « Quitter vite cette page » sur l'écran violences ; ressources visibles après « Arrêter » et pendant la suite du questionnaire ; ordre des boutons : Ambre (assistant automatique), Arrêter, Continuer ; bouton Ambre retiré si Ambre ne suit pas le protocole déterministe (`chat-safety.mjs`). Réponse `pref` aux questions d'automutilation ou de violences : bloc doux `soft_block_by_kind`.
- Persistance : chaque déclenchement écrit une ligne d'alerte datée (table de sécurité de `0028`), même si la passation est abandonnée ; bandeau de sécurité en tête des résultats et de l'historique pendant 14 jours.
- Ambre : `safety.reported_at` traité comme antécédent récent ; une question directe de sécurité au plus une fois par 14 jours, avant toute exploration ; la vérification de risque ne reçoit aucun résultat de questionnaire.

## 8. Croisement avec Mon enfance (4 états)

Entrées : résultats des modules (dernière passation par module) et `computeCartography(answers, data, feedback)` de Mon enfance.

État d'un sujet côté symptômes (`subject_modules` donne, pour chaque sujet croisable, les modules publiés qui peuvent le signaler, dans l'ordre des modules) :
- `signale` : une règle positive d'un de ces modules contient le sujet ;
- `non_signale` : tous ces modules sont terminés et aucune règle positive ne le contient ;
- `non_explore` : au moins un de ces modules n'est pas fait (ou `indetermine`) et aucun ne le signale ;
- `refuse` : ces modules sont tous refusés.

Côté enfance : sujet présent dans les sections `top` ou `other`, retour différent de `non`.

Listes :
1. « Présent dans vos deux questionnaires » : `signale` ET présent côté enfance, suivi de `crossing_note`.
2. « Signalé aujourd'hui seulement » : `signale` ET absent côté enfance (y compris retour `non` : le sujet reste visible, sans mention de croisement).
3. « Lié à vos vécus, sans difficulté signalée aujourd'hui » : présent côté enfance ET `non_signale`, avec `crossing_childhood_only_note`. Les sujets `non_explore` ou `refuse` n'y figurent pas.

Libellés : `crossing.display_labels` remplace le libellé de la cartographie pour les sujets listés (jamais « attachement anxieux / évitant »). Suspension : si `modules[].crossing_suspended_if.when` est vrai (violences actuelles dans `relations`), aucun sujet de ce module n'est croisé et son `text` est affiché. La liste 3 est repliée par défaut (`crossing_childhood_only_collapsed`).

Tri : ordre du premier module qui signale (ou peut signaler) le sujet, puis libellé (ordre alphabétique `localeCompare('fr')` côté client, comparaison binaire identique côté serveur — test de parité comme pour la cartographie). Aucun nombre de vécus, aucun niveau de preuve affiché ; aucun lien vécu → sujet créé ; `links` et l'audit de la cartographie ne sont jamais modifiés.

## 9. Points ouverts pour les lots suivants

- `phq15_d` « non concerné » compté 0 : le formulaire anglais marque l'item « WOMEN ONLY », le manuel ne précise pas la cotation ; à confirmer.
- Seuil d'entrée GAD-2 ≥ 2 : décision v1.1, appui bibliographique à relire (Kroenke 2007, résumé tronqué).
- Coquille officielle de l'item ITQ P1 (« se rejour ») recopiée telle quelle ; confirmation demandée au Cn2r (non bloquante).
- Codes CIM-11 des chapitres 07 (sommeil : 7A00, 7B01.1) et 17 (santé sexuelle) à vérifier sur icd.who.int.
- Budget Ambre (≤ 1 200 caractères, +10 %) et champs transmis : inchangés par rapport au plan v1.1 ; le résumé n'envoie jamais de réponse item par item.
