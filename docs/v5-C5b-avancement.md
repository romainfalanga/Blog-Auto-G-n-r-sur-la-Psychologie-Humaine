# Lot C5b — avancement (Cartographie : exhaustivité v5)

Source : `docs/relecture-cartographie-v5-2026-11-19.md` §8 (relecture indépendante), proposition archivée `editorial/cartographie/propositions/liens-exhaustivite-v5-2026-11-19.json` (inchangée).

## Fait
- **Fragment définitif** `editorial/cartographie/liens-exhaustivite-v5-2026-11-19.json` (format plan 2026-11-17 §6.3 ; aucune clé `proposed_*` ni `out_of_scope`) : 11 sources (dont `deng2024_anxiete_sociale`, déplacée), 4 sujets (`ten_difficultes_personnalite` contextuel, `ten_impulsivite` et `ten_violences_amoureuses_jeunes` avec les descriptions corrigées, `ten_preoccupation_apparence`), **22 liens**, tous `limitee` / `empirical_association`, textes du §3. Le §8 annonce « 21 » liens mais énumère 22 numéros (27 − 5 refusés) : la liste explicite a été suivie.
- **`corrections.json`** : `att_anxieux` (seule mention de « dépendance affective ») ; `pat_social_isolation` (libellé « Solitude ou isolement subis », description) ; liens existants : `harcelement → icd_6b04` (+ Deng 2024, texte §5-5, limitee), `traumatisme_ponctuel → anx_generale` / `dep_generale` (+ Rubens 2018, entrées existantes mises à jour), `negligence_physique → icd_6a60` (limitee, §7), `negligence_emotionnelle → icd_6a60` (limitee par cohérence, à la demande du coordinateur ; validation du propriétaire signalée par la relecture), `parent_trouble → icd_6b05` (texte, moderee), `deces_proche → icd_6b40` et `→ icd_6a20` (textes).
- **« Ce que la carte ne couvre pas »** (`liens-revue-v4-2026-11-18.json`) : item personnalité (§5-8) ; nouvel item « Traits psychopathiques » (refus motivé §2.1) ; item exil (§5-10 + refus des comparaisons réfugiés/psychoses) ; « Mécanismes de défense » et « Dépendance affective » inchangés.
- **Registre** `couverture-referentiels.json` : 18 entrées `lot: v5` (16 de la proposition avec les ajustements §6, + `traits_psychopathiques`, `defenses_maturite_globale`), lot `v5` avec empreintes du fragment, du rapport et de la relecture ; `source_records` complétés ; compteurs recalculés (1 585 décisions).
- **Audit** : `correspondances-2026-11-17.json` → `final_validation` revalidée sur le nouveau graphe (`revalidated_by: lot_C5b_v5`) ; `docs/audit-cartographie-correspondances-2026-11-17.json` régénéré.
- **Données** : 59 facteurs, 113 sujets (25 contextuels), 518 liens, 272 sources. `data/cartographie.json` 406 611 octets (< 600 000), module Worker 314 252 octets (< 500 000).
- **Seuils du classement re-simulés** : N = 43 (inchangé), sujets fréquents inchangés. 2 000 profils : dépression/anxiété 1re 33 %, top 3 64 %, top 5 76 %, tête sur un seul lien limité 4 %, solide derrière limité 0 % ; profils typés identiques au C5a final.
- **Test** : `tests/v5-C-integration.test.cjs` (sujets et liens acceptés présents et limités, refusés absents, corrections, mention unique, hors périmètre). Aucun test figé sur les nombres à ajuster.

## Non intégré (décision de la relecture)
Cartes `ten_traits_psychopathiques` et `ten_defenses_peu_adaptees` et leurs liens ; liens proposés exil → psychoses, violences sexuelles → psychopathie, violences psychologiques → autocritique (refusés) ; exil → souffrance dans l'enfance et violences physiques → violences amoureuses (en attente) ; corrections codépendance et défenses (refusées) ; champ `also_known_as` (évolution de code reportée).

## Lacunes à conserver (état de couverture)
Textes fermés de Longobardi 2022, Liu 2023 et Lawrence 2019 ; tableaux S2–S3 d'Eltanamly 2021 ; Credé 2023 non lu ; décision du propriétaire sur `negligence_emotionnelle → icd_6a60` (appliquée en limitee) ; trouble du deuil prolongé absent du catalogue.

## Vérifications
`python3 scripts/build_cartographie.py --check`, `node scripts/build_cartographie_module.mjs --check`, `python3 scripts/audit_cartographie.py --check` et les autres `--check` de `verify.yml` : OK. `node --test tests/*.test.cjs` 41/41 ; trois suites navigateur OK. `npm --prefix journal test` : tous les tests de cartographie et d'Ambre passent ; 24 échecs dans `ui-v2.test.mjs` (interface du chat), fichier et code en cours de modification par A5-P1 (non touchés ici).
