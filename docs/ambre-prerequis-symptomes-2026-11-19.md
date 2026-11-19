# Ambre : les 516 symptômes obligatoires

Demande : annoncer le questionnaire complet avant connexion et empêcher le lancement d’Ambre tant que toutes les réponses ne sont pas enregistrées.

| Exigence | Résultat |
|---|---|
| Information publique | La page Ambre avant connexion indique les 516 questions, ainsi que les autres prérequis. |
| Progression | Une cinquième étape « Mes symptômes » affiche le nombre de réponses sur 516 et un lien pour compléter. |
| Contrôle effectif | Contrôle JavaScript serveur et garde SQL avant admission, appel fournisseur et publication ; les discussions déjà commencées suivent aussi cette règle. |
| Réponses acceptées | Non, Oui, Je ne sais pas, Dans le passé (précision facultative depuis le 19/11, voir `docs/symptomes-passe-sans-precision-2026-11-19.md`). Aucun préremplissage des vrais comptes. |
| Préservation | Les réponses, notes, historiques, autorisations et réglages de facturation existants sont conservés. |

Tests ciblés : 140 tests serveur/interface réussis ; navigateur réel sur Worker local et SQLite à 390 et 1 440 px, annonce publique visible, 515 réponses bloquées, 516 réponses autorisées. Aucun appel IA payant dans cette recette. Relecture indépendante des contrôles JS/SQL sans défaut bloquant.

La banque courante contient 516 questions actives ; le contrôle serveur utilise leurs identifiants réels et ne se contente pas d’un nombre de clés JSON. Une réponse inconnue hors banque ne peut pas remplacer une question manquante. Le comptage n’impose pas de répondre Oui : Je ne sais pas compte comme réponse.

Les problèmes de génération de bilan relevés dans [la recette après recharge](recette-openrouter-recharge-2026-11-19.md) restent distincts de ce lot. Ce changement ne constitue pas une validation d’un bilan IA réel.

Les fixtures anciennes sont désormais complètes. Les tests de budget utilisent les allocations v2 déjà présentes (écriture +40 %, bilans +20 %, enveloppe de tour 42 000) ; aucun plafond de production n’a été augmenté. Trois scénarios de charge maximale avec reprises ou plan chargé atteignent désormais le repli sûr `turn_budget` / `context_budget` : les tests exigent une terminaison sans nouveaux appels, sans dépassement et sans réponse non validée présentée comme validée. Les scénarios ordinaires, dont une réécriture maximale, doivent toujours réussir leur relecture.

## Publication vérifiée

- PR 108 fusionnée : `e42f1151dddd41d12b3a9bcf9981fe1694135457`, source `6986b1874de3861957735a9c0bf5123993cb5fec`.
- 951/951 tests serveur réussis ; 61 tests site réussis, 1 ignoré (Hugo absent de ce lancement). CI `44570244689` réussie.
- Worker `decode-journal`, déploiement `d5216508e6394ff18f57f2b1112e07d5`, 19 novembre 2026 à 17:49:57 Paris. Les 54 modules ont été vérifiés par empreinte ; seuls quatre modules ont changé. Les sept bindings et le placement smart sont conservés ; aucune migration.
- Netlify `6aff28cded63762a10be5c46`, ready, commit de fusion vérifié.
- Recette en production : tous les autres prérequis remplis, 515 symptômes renseignés, démarrage refusé en 403 `prerequisites_required`. Compteur et bouton désactivé vérifiés à 390/1440 px ; annonce avant connexion vérifiée. Le passage à 516 est validé localement sans appel fournisseur.
- Compte temporaire supprimé via l’API, session invalidée, aucune ligne restante dans les 47 tables à `user_id` contrôlées ni dans `users`. Fichier de récupération des accès supprimé.
- Journaux locaux : `work/required-final-all-green.log`, `work/required-symptoms-browser/`, `work/required-symptoms-production/`, `work/required-symptoms-publication.json`.

Aucune optimisation du moteur IA ni modification de ses plafonds n’a été conservée dans ce lot : les changements applicatifs concernent uniquement le prérequis, ses messages et son affichage.
