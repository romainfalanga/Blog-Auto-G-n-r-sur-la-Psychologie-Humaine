# Lot A5-P2 — formulaires et observations

## Mise en ligne du 17 novembre 2026 — état actuel

- Confirmation explicite du propriétaire reçue après le refus initial du contrôle automatique.
- PR 91 fusionnée : `57ed975c418e161e69d53f65566e4c444325432f` ; source `36011eeee70ded42ad0480ae0c07c95285e9a771`. CI GitHub `44071907530` réussie. Empreinte d’arbre locale et distante identique : `ed0fe4dac662b507e1ff7e9c141c8f5e72817031`.
- Point D1 avant migration : `000000a5-000006c9-00005126-72ff6de7275583f636ab438d8e1cdd08`. Migration 0024 appliquée une seule fois puis inscrite ; 24 migrations, aucune violation de clé étrangère.
- Worker publié le 17 novembre à 11:55 UTC, identifiant `845db15084ec40f48ad98bae0b342f09`. Les 44 modules ont été relus après déploiement : zéro différence SHA-256. Les 7 liaisons, le secret `OPENROUTER_API_KEY` et le cron `* * * * *` sont conservés.
- Netlify `6afc40c77420eb243264a72f`, ready, même commit de fusion, publié à 11:56:27 UTC.
- La recette réelle avec comptes synthétiques est en cours ; les résultats et suppressions sont consignés dans la note finale de reprise. Le compte réel du propriétaire n’est jamais utilisé.

Les paragraphes de préparation ci-dessous sont historiques ; cet état courant fait foi.

Reprise depuis `f9aa647` (PR 90), le 17 novembre 2026. Le patch `docs/wip/a5-p2-en-cours.patch` a été relu puis appliqué comme point de départ ; ce fichier reste une archive du travail interrompu.

| Exigence | Responsable | État / preuve |
|---|---|---|
| R1 Champs facultatifs proches, moments et portrait | Interface + intégration | Implémentés, contrôles en cours |
| R2 Compatibilité avant 0024, révisions, export et suppression | Intégration | Tests ciblés en cours |
| R3 Insécurité déclarée → protection sans confrontation, 3919 si pertinent | Intégration | Test moteur et suivi pronominal ajoutés |
| R4 Observations proposées, enregistrées uniquement après validation individuelle | Observations | En cours ; circuit des propositions existant |
| R5 Budget par tour ≤ +10 % | Intégration | Référence historique inchangée ; profil P2 rempli ajouté |
| R6 Relecture indépendante, tests complets et recette réelle | Relecture + intégration | À faire |
| R7 Publication D1 → Worker complet → fusion main | Principal | À faire |
| R8 Questions de sens du questionnaire et plafond quotidien | Propriétaire | En attente, aucune modification |

## État distant constaté avant modification

- D1 : `d1_migrations` contient exactement 0001 à 0023. Les colonnes 0024 sont absentes. Aucune migration rejouée.
- Worker `decode-journal` : 7 liaisons, secret `OPENROUTER_API_KEY`, compatibilité `2026-10-01`, cron `* * * * *` présents.
- Le chantier éditorial et les données de Cartographie ne sont pas modifiés.

## Corrections du patch initial

- Les champs remplis peuvent constituer une source même sans ancien récit.
- La protection d'un proche explicitement nommé est conservée au tour suivant avec un pronom, sans l'attribuer à une autre personne nommée.
- Les tests de migrations tiennent compte de 0024 sans modifier le script historique de baseline.

Ce journal sera complété avec les preuves finales avant livraison.

## Validation finale locale

- `npm --prefix journal test` : **785/785**, dont nouveaux tests formulaires, consentement individuel, refus de confrontation dans un bilan et budget (référence historique inchangée).
- `node --test tests/*.test.cjs` : **41/41**.
- Navigateurs Cartographie existants : **16 + 6** scénarios réussis.
- Parcours nouveaux formulaires avec vrai Worker/SQLite : **6** scénarios à 390 et 1440 px, sans erreur JS ni débordement.
- Tests Python personnels : **43/43** ; toutes les commandes de `.github/workflows/verify.yml` passent.
- Relecture indépendante : 253 tests ciblés puis contre-vérification du refus de confrontation du bilan, verdict non bloquant. Corrections : suivi de sécurité sur plusieurs tours pronominaux ; type des choix ; nouvelle proposition après modification d’une fiche ; budget par étape.
- Compensation : enveloppe 6 500 → 6 300 caractères, propositions/observations rappelées déduites du même budget. Aucun appel supplémentaire ; 28 tests de budget passent.

## Publication

Non déployé à ce stade. Le contrôle automatique de permissions a refusé la création du commit distant malgré la consigne du texte joint (« Publie directement » et « Sauvegarde régulièrement dans Git ») et le dépôt revérifié. Une confirmation explicite est demandée une fois le résultat prêt. Aucune migration D1 ni mise à jour Worker/main n’a été effectuée. L’arbre GitHub préparé n’est rattaché à aucune branche publiée.

## Décisions reçues du propriétaire

Oui : « Je ne sais pas », Non/Oui pour événements uniques, séparer parents stricts et culpabilisation/manipulation. Non : rétablir « Ce qui m’a aidé », ajouter violences amoureuses à l’adolescence, confirmer le lien négligence affective → troubles bipolaires. Limite quotidienne maintenue désactivée. Ces changements de questionnaire sont préparés dans un lot isolé après A5-P2, en préservant les anciennes réponses.
