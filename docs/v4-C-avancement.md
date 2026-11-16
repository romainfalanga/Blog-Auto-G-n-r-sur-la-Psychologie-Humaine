# Lot C (Cartographie) — avancement v4.1

Plan : `docs/plan-ambre-v4-2026-11-18.md` (révision v4.1). Source : `docs/revue-exhaustivite-cartographie-2026-11-18.md` §4.3, §8, §9.
Hors lot (décisions du propriétaire) : C2 (scission de `controle`), « Je ne sais pas », échelle Non/Oui pour les événements uniques, saisie des ressources, encadré 3114, bouton de suppression.

## État de départ (vérifié)
- 59 questions, 109 cartes, 493 liens, 260 sources ; empreinte du cœur `f60aaa89…`.
- `node --test tests/cartographie.test.cjs journal/tests/cartographie.test.mjs` : 42/42 ; navigateur 16/16 ; trois `--check` réussis.

## Fait
- **C1 — ordre unique page/Ambre.** `compute` (navigateur) et `computeCartography` (Worker) trient : cartes contextuelles en dernier, puis nombre de vécus cochés reliés par une association empirique (forte, modérée, limitée ; définitions exclues), puis familles distinctes, force maximale, identifiant. Champs ajoutés : `associated_factors`, `associated_count`, `other_links`. L'interface n'a plus de tri propre ; en-tête replié = « N vécus associés · force maximale mesurée · M autres repères » ; carte contextuelle = « Repère de contexte » (plus de « Preuves solides ») ; corps : associations mesurées, puis « Autres repères, non comptés ». Fenêtre « Liens étudiés » : forces empiriques puis modèles/définitions/CIM marqués « non compté ». Ambre : `display_order` (10 premiers sujets, même ordre) et `display_count` dans le résumé et le prompt (`cartographyPrompt`, ajout d'une ligne) ; `convergence`, `pistes`, preuves et points de départ suivent l'ordre affiché, les sujets reconnus restent sélectionnés en premier sous le plafond par catégorie. `interpretation.ranking` réécrit.
- **C3 — forces et formulations** (`corrections.json`, 24 surcharges motivées) : 19 liens « modérés » à source unique de cohorte → « limitée » (dont Soares ×5, harmonisé avec « santé physique », et Kosty) ; harcèlement → emploi → « limitée » (Brimblecombe et Takizawa = même cohorte NCDS 1958, vérifié) ; négligence affective → conduites suicidaires → « limitée » (même Angelakis 2020 que « tentatives ») ; violences psychologiques → conduites suicidaires « forte » → « modérée » (composantes modérée + limitée) ; violences sexuelles → anxiété « forte » → « modérée » (Hailes 2019 relu, qualité AMSTAR 1/5) ; isolement → substances **retiré** (Lacey 2014 relu : issue CRP, tabac médiateur ; aucun reciblage). Exil → 6B40 et 6B41 en contexte de classification (CDDR OMS anglais ouvert et lu). Thèmes « avant 18 ans », « Fessées, tapes ou gifles éducatives », « un parent », exemples neutres pour l'événement traumatique, doublon orientation retiré de « discrimination ».
- **C4 — ajouts vérifiés** (`liens-revue-v4-2026-11-18.json`) : témoin de violences → externalisation (Vu 2016, résumé seul, texte fermé → « limitée » au lieu de « modérée ») ; précarité → dépression (Gilman 2003, résumé, une cohorte → « limitée »). **Non appliqués** : violences sexuelles → 6B60 (Hailes réutilise Ludwig 2018, déjà seule source) ; carte « préoccupation pour l'apparence » (Longobardi 2022, texte intégral fermé) ; question « violences amoureuses à l'adolescence » (Exner-Cortens 2013 lisible, mais ajout d'une question irréversible : décision du propriétaire) ; note « parentalité » (aucun emplacement d'affichage, formulation sensible).
- **C5** : mention « Aucune association suffisamment documentée n'est retenue pour l'instant. » sous les questions sans lien (migration, secrets de famille) ; `aria-live` retiré de la liste, région `role=status` invisible limitée au compteur (« N réponses · M sujets affichés »).
- **C6** : lien discret « Ce que la carte ne couvre pas » sous la liste, ouvrant la fenêtre existante (11 domaines de la revue §9, sources). Aucune introduction ni encadré.
- Audit : entrée `post_integration_reviews` (lot_C_v4_1) et `final_validation` mises à jour, `independent_review_of_this_delta: pending` ; `audit_cartographie.py` relancé. Nouvelle empreinte `13ad65d2…` ; 59 questions, 109 cartes, 496 liens, 261 sources ; forces 5 fortes, 178 modérées, 182 limitées, 113 théoriques, 18 critères.
- Tests : `journal/tests/v4-C-cartographie.test.mjs` (simulation ≈ 2 000 profils, prompt réel), `tests/v4-C-cartographie.test.cjs` ; assertions existantes adaptées (ordre affiché, exil en contexte CIM, reconnaissance sous plafond).

- Navigateur : `tests/v4-C-cartographie-browser.cjs` (6 scénarios : top 5 page = top 5 Ambre, en-têtes contextuels sans force, autres repères à part, questions sans lien, compteur seul annoncé, lien « Ce que la carte ne couvre pas » à 1 440 et 320 px) ; `tests/cartographie-browser.cjs` adapté (ordre lu sur `data-associated`, cartes contextuelles en dernier).

## Mesures (2 000 profils aléatoires par taille, données publiées ; script de mesure hors dépôt)
| Cases | Écart top 5 page/Ambre (avant → après) | Cartes contextuelles avant une non contextuelle | Cartes sans association empirique (non contextuelles) dans le top 5 | Dépression ou anxiété dans le top 3 |
| ---: | ---: | ---: | ---: | ---: |
| 3 | 0,83 → 0 | 83 % → 0 % | 0,53 → 0,40 | 84 % → 91 % |
| 6 | 1,79 → 0 | 97 % → 0 % | 0,55 → 0,02 | 94 % → 98 % |
| 10 | 1,77 → 0 | 100 % → 0 % | 0,37 → 0 | 99 % → 100 % |
| 15 | 1,75 → 0 | 100 % → 0 % | 0,29 → 0 | 100 % → 100 % |

Profil « parentification » : avant Sauveur, Abnégation, Codépendance, Besoins (modèles) ; après Anxiété, Dépression, Codépendance. Tout coché : première carte suicidaire (contextuelle) du 4e au 86e rang. Effet de généricité (dépression/anxiété en tête) accru : la pondération de spécificité (revue §4.3 point 4) n'est pas dans le plan v4.1.

## Vérifications finales
- `node --test tests/*.test.cjs` 39/39 ; `npm --prefix journal test` 606/607 (échec `chat.test.mjs` « explicit self identity corrections… », propositions/citations, hors lot C, en cours chez S) ; navigateur 16/16 et 6/6 ; tous les `--check` de `verify.yml` réussis ; `git diff --check` propre.
- Tailles : données 382 675 octets (budget 600 000 pour la page), module Worker 294 795 octets (budget 500 000). `display_order` ajoute ≈ 1 000 caractères au contexte d'Ambre.

## Reste
- Relecture indépendante du delta de liens (`independent_review_of_this_delta: pending` dans `correspondances-2026-11-17.json`).
- Lot S : mentionner `display_order` dans `CHILDHOOD_CARTOGRAPHY_GUIDANCE` ; libellés Paramètres (`display_order`, `display_count`, `associated_count`).
- Propriétaire : C2, « Je ne sais pas », échelle Non/Oui des événements, question « violences amoureuses à l'adolescence » (Exner-Cortens 2013), carte « préoccupation pour l'apparence » (texte intégral de Longobardi 2022 à obtenir), note « parentalité », pondération de spécificité, description de `content/cartographie.md` (« conséquences » → « sujets associés »).

## Corrections après relecture indépendante (`docs/relecture-cartographie-v4-2026-11-18.md`)
- Résumés n° 17 (harcèlement → agoraphobie) et exil → 6B41 : textes du rapport repris tels quels. N° 22 (violences psychologiques → conduites suicidaires) et n° 28 (précarité → dépression) : textes du rapport légèrement raccourcis, car ils dépassaient le plafond de 350 caractères du générateur (détail consigné dans `correspondances-2026-11-17.json`).
- Justification n° 22 reformulée : « forte » est réservée à des méta-analyses concordantes de force au moins modérée ; les agrégats violences physiques et sexuelles restent « forte ». Libellé « Fessées, tapes ou gifles données pour éduquer ». Aucune force modifiée.
- `independent_review_of_this_delta` : « accepté après corrections ». Empreinte `ce03dcd3…`. Audit et `--check` relancés ; tests 39/39, Worker Cartographie 35/35, navigateur 16/16 et 6/6.
