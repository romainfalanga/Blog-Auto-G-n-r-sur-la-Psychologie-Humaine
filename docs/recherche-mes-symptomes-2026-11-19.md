# « Mes symptômes » : recherche et conception — 19 novembre 2026

Document de recherche et de conception. **Aucun code modifié, rien commité ni déployé.** Rédigé au fil de l'eau : les sections se complètent à mesure que les sources sont ouvertes. Proposition de données : `editorial/symptomes/proposition-2026-11-19.json`.

Règles de lecture :
- « lu » = page officielle ou article ouvert et lu en entier ; « résumé » = seul le résumé, la notice ou un extrait a été lu ; « non ouvert » = information non vérifiée ici, à confirmer avant usage.
- Aucun item d'instrument n'est recopié dans ce document ou dans la proposition JSON, sauf mention explicite d'une licence lue qui le permet. Les items sont désignés par leur numéro et leur thème.
- Les seuils sont des **repères publiés**, jamais des diagnostics.
- Règle absolue héritée du plan du 17 novembre (§ 0) : aucun lien n'est créé sans source ; une absence de preuve est un résultat valide. Ici, aucun croisement « symptôme ↔ sujet » n'ajoute de lien vécu → sujet ; il ne fait que mettre en regard deux déclarations de la personne.

## 1. État de départ lu dans le dépôt

- `data/cartographie.json` : 113 sujets (`outcomes`), 7 domaines d'affichage (`perceptions`, `humeur`, `anxiete`, `relations`, `soi`, `emotions`, `comportements`), 4 natures (`trouble`, `schema`, `attachement`, `tendance`). Liste complète utilisée pour le tableau de correspondance (§ 5).
- Stockage existant : table `cartographies` (migration `0016`), une ligne par membre, `answers`, `feedback`, `consent` obligatoire (données de santé), `locked_at`, `version`.
- Ambre : `journal/src/ambre-engine.mjs` importe `cartographyIndexEntries` (`journal/src/cartographie.mjs`) ; résumé borné (`summarizeCartography`, `maxConvergence=15`, `maxPistes=10`) ; sécurité déterministe dans `chat-safety.mjs` (3114, 15, 112, 17) et vérification de risque sans index ni catalogue de troubles.
- Mon enfance : réponses Non / Oui / Je ne sais pas ; niveaux de preuve et codes CIM/DSM retirés du rendu (reprise du 17 novembre). La même discipline d'affichage est proposée ici.

## 2. Instruments : licences et conditions (journal de lecture)

Chaque fiche indique la source effectivement ouverte et le niveau de lecture. « Statut proposé » : `retenu` (utilisable sur un site gratuit avec les mentions), `retenu-sous-reserve` (autorisation écrite à demander ou clause à vérifier avant mise en ligne), `exclu`.

### 2.1 PHQ-9, PHQ-2, GAD-7, GAD-2, PHQ-15 (Pfizer / Spitzer, Kroenke, Williams, Löwe)
- Source : https://www.phqscreeners.com/select-screener — lu (page de téléchargement). Citation : « no permission is required to reproduce, translate, display or distribute them ». La page renvoie à des conditions d'utilisation Pfizer non lues (texte absent de la page) ; la PHQ-15 n'est pas nommée explicitement dans la phrase relevée.
- Traductions : le site distribue les traductions officielles ; la version française n'a pas été ouverte ici (à télécharger et vérifier avant intégration ; ne pas retraduire soi-même).
- Mesures (connaissance standard des publications d'origine, à vérifier dans les PDF officiels avant implémentation) : PHQ-9 = 9 items, 2 dernières semaines, 0–3, total 0–27, repères 5/10/15/20 ; PHQ-2 = items 1–2, repère ≥ 3 ; GAD-7 = 7 items, 2 semaines, 0–3, total 0–21, repères 5/10/15 ; GAD-2 = repère ≥ 3 ; PHQ-15 = 15 symptômes somatiques, 4 semaines, 0–2, repères 5/10/15.
- Point de sécurité : l'item 9 du PHQ-9 porte sur les pensées de mort ou d'automutilation ; toute réponse > 0 déclenche le parcours de sécurité (§ 7).
- Statut proposé : `retenu` (marque/mention « PHQ » et copyright Pfizer conservés tels qu'imprimés sur le formulaire officiel ; pas de modification des items).

### 2.2 PCL-5 (National Center for PTSD, VA)
- Source : https://www.ptsd.va.gov/professional/assessment/adult-sr/ptsd-checklist.asp — lu. « This measure was developed by staff at VA's National Center for PTSD and is in the public domain and not copyrighted. » Usage visé : « qualified health professionals and researchers » (recommandation éthique APA, pas une restriction de licence).
- 20 items, mois écoulé (version semaine disponible), 0–4, total 0–80 ; repère « probable PTSD » 31–33 selon la page ; règle de grappes B/C/D/E avec items ≥ 2.
- Version française : Ashbaugh et al. 2016, PLOS ONE, https://pmc.ncbi.nlm.nih.gov/articles/PMC5056703/ — résumé (via résultat de recherche) ; α = .94, test-retest r = .89, échantillon étudiant.
- Exige une exposition à un événement (critère A) : la page doit d'abord demander, sans détail, si un événement très éprouvant a eu lieu, sans faire raconter.
- Statut proposé : `retenu`, en indiquant que la personne n'est pas évaluée par un professionnel.

### 2.3 ITQ — International Trauma Questionnaire (TSPT et TSPT complexe CIM-11)
- Source : https://www.traumameasuresglobal.com/ — lu (accueil). « The ITQ is freely available in the public domain to all interested parties. »
- Version française : PDF du Cn2r https://cn2r.fr/wp-content/uploads/2025/04/ITQ_VF_Cn2r.pdf — texte extrait localement, seulement l'en-tête et la notice (items non recopiés) : « Cloitre et al. ©2018 », « Traduction française et validation par Peraud et al. (2022) », 18 items, mois écoulé, cotation par grappes (ré-expérience, évitement, menace ; troubles de l'organisation de soi) avec un item ≥ 2 par grappe et une atteinte fonctionnelle. Validation : Peraud et al. 2022, Eur J Psychotraumatol, DOI 10.1080/20008066.2022.2152109 — résumé.
- La version enfant/adolescent du Cn2r est marquée « Traduction non validée » : non retenue (le site s'adresse aux adultes).
- Statut proposé : `retenu` pour TSPT/TSPT-C CIM-11 (préféré au PCL-5 pour l'alignement CIM-11 ; le PCL-5 reste l'option DSM-5-TR). Mentions : Cloitre et al. 2018 ; Peraud et al. 2022 ; Cn2r.

### 2.4 C-SSRS (Columbia, échelle de sévérité du risque suicidaire)
- Source : https://cssrs.columbia.edu/the-columbia-scale-c-ssrs/cssrs-for-communities-and-healthcare/ — lu. « You do not need special permission from us to take the Columbia Protocol tools and use them in your setting… » ; « The core Columbia Protocol questions do not vary ». Une version d'auto-repérage existe, décrite comme remplie « for professional review » (destinée à être relue par un professionnel). Copyright Research Foundation for Mental Hygiene (2008) relevé sur un PDF tiers (résumé de recherche, non ouvert).
- Analyse : la C-SSRS est conçue pour un triage avec relecture humaine. Sur un site sans professionnel en aval, présenter un score de « niveau de risque » ferait croire à une évaluation. **Proposition : ne pas l'implémenter comme questionnaire noté.** Le site garde son filet déterministe (`chat-safety.mjs`) et le parcours de sécurité (§ 7) déclenché par l'item 9 du PHQ-9 ou une question filtre unique ; il oriente vers 3114 / 15 / 112. Statut : `exclu` en tant qu'échelle notée (raison : usage prévu avec relecture professionnelle, risque de fausse réassurance).

### 2.5 AUDIT, AUDIT-C (OMS)
- Source secondaire lue : https://camcops.readthedocs.io/en/stable/tasks/audit.html (cite le manuel OMS 2001, WHO/MSD/MSB/01.6a) : le document « may, however, be freely reviewed, abstracted, reproduced, and translated, in part or in whole but not for sale or for use in conjunction with commercial purposes » ; AUDIT-C « available for use in the public domain » (cite SAMHSA). Manuel OMS lui-même non ouvert ici.
- Version française validée : Gache et al. 2005 (Paris, Genève, Lausanne) — résumé ; repères de dépistage de consommation excessive ≥ 7 (hommes) / ≥ 6 (femmes) en soins primaires. Fiche INSPQ (https://www.inspq.qc.ca/node/16257, résumé de recherche) : domaine public, source à mentionner, français disponible.
- 10 items, 12 derniers mois ; zones OMS 0–7 / 8–15 / 16–19 / 20–40 (manuel, à vérifier dans le PDF OMS avant implémentation). AUDIT-C = 3 premiers items.
- **Point juridique :** clause « pas d'usage commercial ». Le site a un module de facturation (migrations `0008_billing_trial`, `0026_monthly_usage`), aujourd'hui gratuit et sans paiement activé. Si un abonnement payant est activé, la page devra rester hors offre payante ou faire l'objet d'une autorisation OMS. Statut : `retenu-sous-reserve` (gratuité de la page).

### 2.6 DAST-10 (Skinner, CAMH)
- Source secondaire lue : https://camcops.readthedocs.io/en/stable/tasks/dast.html : « Copyright 1982 by Harvey A. Skinner, PhD, and the Centre for Addiction and Mental Health » ; « You may reproduce this instrument for non-commercial use (clinical, research, training purposes) as long as you credit the author… ».
- « Usage non commercial (clinique, recherche, formation) » : un outil d'auto-repérage grand public n'entre pas clairement dans ces trois finalités. Validation française du DAST-10 non trouvée ici.
- Statut : `retenu-sous-reserve` (écrire au CAMH ; sinon remplacer par des repères non validés paraphrasant les critères CIM-11 de l'usage nocif / dépendance, § 4).

### 2.7 PDSS / PDSS-SR (Shear)
- Source secondaire lue : https://camcops.readthedocs.io/en/stable/tasks/pdss.html : « Reproduction of this task/scale is not permitted (or such permission is not clear) » ; « Copyright © M. Katherine Shear ». La PDSS-SR n'y est pas traitée.
- Statut : `exclu` en l'absence d'autorisation écrite ; le trouble panique est couvert par une question filtre et des repères non validés paraphrasant la CIM-11 6B01.

### 2.8 Y-BOCS (Goodman et al.)
- Source secondaire lue : https://camcops.readthedocs.io/en/stable/tasks/ybocs.html : « YBOCS copyright © Wayne K. Goodman and colleagues, 1986 » ; « Distribution of the YBOCS is not permitted without specific authorization ». De plus, échelle conçue pour un entretien clinique.
- Statut : `exclu` ; l'OCI-R est étudiée à la place (§ 2.9).

### 2.9 OCI-R (Foa et al. 2002)
- Sources : PubMed https://pubmed.ncbi.nlm.nih.gov/12501574/ — résumé (18 items, 6 sous-échelles, analyses ROC en faveur d'un usage de repérage) ; validation française Zermatten et al. 2006, Eur Rev Appl Psychol, https://orbi.uliege.be/handle/2268/245436 — résumé (583 étudiants, structure à 6 facteurs confirmée, patients TOC non testés).
- Licence : **aucune déclaration de licence trouvée** ; l'article est sous copyright APA, ce qui ne dit rien de l'usage de l'échelle.
- Statut : `retenu-sous-reserve` (demande écrite aux auteurs). En attendant : question filtre et repères non validés paraphrasant la CIM-11 6B20.

### 2.10 ISI — Index de sévérité de l'insomnie (Morin)
- Source : page ePROVIDE (Mapi Research Trust) https://eprovide.mapi-trust.org/isi-insomnia-severity-index — **non lue** (redirection vers une page de maintenance le 19 novembre) ; selon le résultat de recherche, Mapi distribue l'ISI pour le compte du Pr Morin et toutes les questions de conditions d'utilisation et de traductions lui sont adressées.
- Statut : `exclu` en l'état (licence gérée par un distributeur, conditions non publiques) ; repère non validé paraphrasant les critères d'insomnie chronique (CIM-11 7A00, chapitre sommeil-éveil) à la place.

### 2.11 SPIN / Mini-SPIN (Connor, Davidson et al.)
- Sources : résultats de recherche seulement (plateforme tierce : copyright des auteurs et de Duke University ; Mini-SPIN = 3 items issus de la SPIN, Connor et al. 2001, Depression and Anxiety). Aucune page officielle de licence trouvée.
- Statut : `exclu` faute de licence vérifiable ; repères non validés CIM-11 6B04.

### 2.12 YSQ-S3 (Young, schémas précoces inadaptés)
- Sources : résultats de recherche ; aucune page de licence officielle (Schema Therapy Institute) trouvée. NovoPsych présente sa propre échelle MSS-YSQ comme alternative open source « overcoming a limiting factor of the YSQ and derivatives » (résultat de recherche), ce qui confirme indirectement que le YSQ n'est pas libre. Longueur variable selon les sources (90 ou 114 items).
- Alternative étudiée : licence NovoPsych https://novopsych.com/open-source-license-psychometric-assessment/ — lue. Usage clinique et non commercial libre ; usage commercial gratuit jusqu'à 1 000 passations par an et, intégré à un site ou une application, 100 utilisateurs par an ; clause 6 : interdiction d'usage dans un service concurrent et d'utiliser les résultats pour « design, train, fine-tune, evaluate, or improve software or models that administer or report on psychometric assessments for third parties » ; partage à l'identique des œuvres dérivées. Pas de version française validée trouvée.
- Statut : YSQ-S3 `exclu` (droits non publics) ; MSS-YSQ `exclu` pour ce site (plafond de 100 utilisateurs pour un site, clause 6 incompatible avec la transmission à une IA qui commente les résultats, pas de version française). **Les 18 schémas restent couverts par Mon enfance (sujets `sch_*`) ; la page Mes symptômes ne cherche pas à les « mesurer ».** Elle peut proposer, par schéma, une seule phrase de reconnaissance paraphrasant la définition publiée de Young et al. (2003), marquée « repère non validé », sans score.

### 2.13 EAT-26 (Garner)
- Source : https://www.eat-26.com/permission/ — lu. Copyright ; « all fees and royalties have been waived » pour les utilisateurs individuels ; exception : « commercial WEB platforms that charge user fees » (licence obligatoire) ; la mise en ligne ou le lien exigent une demande de lettre d'autorisation. Traductions non abordées.
- Statut : `retenu-sous-reserve` (lettre d'autorisation à demander à Eating Attitudes LLC ; version française à vérifier). Le SCOFF est préféré pour l'entrée (§ 2.18).

### 2.14 ECR-R (Fraley, Waller, Brennan 2000)
- Source : https://labs.psychology.illinois.edu/~rcfraley/measures/ecrr.htm — lu. « The scales were published in a scientific journal for use in the public domain » pour la recherche non commerciale ; « You may not use the scales for commercial purposes without permission. » 36 items (anxiété 18, évitement 18), moyenne par dimension ; l'auteur déconseille fortement toute catégorisation : « There is no "natural" or "correct" way to assign people to attachment categories or styles. »
- Version française : Favez et al. 2016, Swiss Journal of Psychology, https://archive-ouverte.unige.ch/unige:86146 — résumé (600 participants).
- Conséquence de conception : **aucun seuil** ; deux dimensions affichées comme position relative, jamais « style d'attachement ». Statut : `retenu-sous-reserve` (site gratuit non commercial, mais pas de la recherche : courriel à l'auteur recommandé).

### 2.15 ASRS v1.1 (OMS / Kessler, TDAH de l'adulte)
- Sources : https://www.hcp.med.harvard.edu/ncs/asrs.php — **non lu** (erreur 503, puis blocage du pare-feu du site) ; informations tirées des résultats de recherche : le dépisteur à 6 questions est « free and does not require any formal permission or approval », à condition de citer l'article de 2005 et de garder la mention de copyright ; la liste à 18 items demande une autorisation (gérée par l'équipe du Dr Adler, NYU) ; un PDF français du dépisteur existe sur le site de Harvard ; titulaires du copyright mentionnés de façon incohérente (NYU / Kessler / Harvard / OMS).
- **Exclusion de périmètre** : la cartographie interdit les sujets du neurodéveloppement (`icd_6a00`…`icd_6a05`, plan du 17 novembre § 1). La page Mes symptômes reprend la même règle : pas de module TDAH, pas d'autisme. Statut : `exclu (périmètre)` ; à réexaminer seulement sur décision du propriétaire, avec la lettre de permission.

### 2.16 DAST-10, CAST, ICJE / PGSI (substances, cannabis, jeu)
- CAST (OFDT, Legleye) : article PMC6877270 https://pmc.ncbi.nlm.nih.gov/articles/PMC6877270 — lu (via résumé détaillé) : 6 items, 0–4, total 0–24, population française 15–75 ans (Baromètre santé 2014) ; repère CAST ≥ 5 pour un trouble DSM-5 modéré ou sévère, ≥ 8 pour un trouble sévère. Aucune licence indiquée dans l'article ; banque COPOLAD https://fase2.copolad.eu/en/banco-de-instrumentos/ficha/135 : « disponible gratuitement » (résumé). Statut : `retenu-sous-reserve` (confirmer par écrit auprès de l'OFDT ; mention OFDT/Legleye).
- ICJE (version française du PGSI, Ferris et Wynne 2001, CCSA) : https://www.larevuedupraticien.fr/outil/jeux-dargent-et-de-hasard-indice-canadien-du-jeu-excessif — lu (fiche) : 9 items, 12 derniers mois, 0–3, total 0–27 ; repères 0 / 1–2 / 3–7 / ≥ 8. L'ANJ recommande ses niveaux aux opérateurs (guide ANJ, résumé). **Licence du CCSA non trouvée.** Statut : `retenu-sous-reserve` (demande au CCSA).
- DAST-10 : voir § 2.6.

### 2.17 MDQ (Hirschfeld 2000) ; PQ-B et PQ-16 (prodromes psychotiques)
- MDQ : validation française Weber Rouget et al. 2005, J Affect Disord 88:103–108 — résumé (via résultat de recherche) ; licence **non trouvée**. Le MDQ explore des épisodes passés sur la vie entière et sa sensibilité au type II est faible en français. Statut : `retenu-sous-reserve` ; sur la page, il ne sert jamais de repère « bipolaire » isolé : il ouvre un message d'orientation vers un médecin.
- PQ-B : traduction française (Larøi, Laloyaux, Liège) hébergée par le laboratoire Loewy (UCSF) — fichier non ouvert ; licence non indiquée. PQ-16 : version française fPQ16, Lejuste et al. 2021, L'Encéphale — résumé : validée chez 100 jeunes **en demande de soins**, repère 4 items positifs (sensibilité 91 %, spécificité 60 %). Hors population clinique, la valeur prédictive positive serait faible.
- Statut : `exclu` comme score affiché (risque d'alarme et de faux positifs sur un site grand public ; population de validation non comparable). Les expériences perceptives inhabituelles sont abordées par deux repères non validés paraphrasant la CIM-11 6A20 et une orientation sans score (§ 4).

### 2.18 SCOFF ; DES-II ; MSI-BPD ; TAS-20 ; DERS-16 ; Rosenberg
- SCOFF (Morgan et al. 1999, BMJ 319:1467) — version française SCOFF-F, Garcia et al. 2010, https://pubmed.ncbi.nlm.nih.gov/20509759/ — résumé : 5 items oui/non, repère ≥ 2, sensibilité 94,6 %, spécificité 94,8 % en population étudiante. Licence non trouvée ; la traduction a été soumise aux auteurs. Statut : `retenu-sous-reserve`.
- DES-II (Carlson & Putnam) : résultats de recherche seulement — copyright des auteurs, autorisation de copie pour « research and clinical use » ; version française Darves-Bornoz et al. 1999, Can J Psychiatry 44:271–275 (victimes de viol) — résumé ; DES-Taxon non documenté ici. Statut : `retenu-sous-reserve` (usage grand public hors « recherche et clinique » : demande écrite).
- MSI-BPD (Zanarini 2003) : validation française chez l'adolescent, Mirkovic et al. 2020, BMC Psychiatry, https://pmc.ncbi.nlm.nih.gov/articles/PMC7216676/ — résumé : traduction faite avec l'accord de la Pr Zanarini ; repère optimal ≥ 5 chez l'adolescent, ≥ 7 (repère des auteurs) très spécifique mais peu sensible ; Le Corff et al. 2021 (étudiants) — résumé. Licence non publiée. Statut : `retenu-sous-reserve`. PAI-BOR : éditeur PAR, licence payante (non vérifiée ici) → `exclu`.
- TAS-20 (Bagby, Parker, Taylor 1994) : validation française Loas et al. 1995/1996 — références trouvées dans une liste, non ouvertes ; **licence non trouvée**. Statut : `retenu-sous-reserve`. Le site possède déjà un modèle d'alexithymie en couple (charte éditoriale) : la page en reprendrait seulement trois repères, sans score, en attendant l'autorisation.
- DERS-16 (Bjureberg et al. 2016, J Psychopathol Behav Assess) — résumé ; **aucune validation française de la DERS-16 trouvée** (DERS-36 : Dan-Glauser & Scherer 2013 ; Côté et al. 2013 ; DERS-SF adolescents 2025). Statut : `exclu` en l'état (pas de version française validée de la forme courte).
- Rosenberg (RSES, 10 items) : page https://socy.umd.edu/about-us/using-rosenberg-self-esteem-scale — **non lue** (503) ; selon le résultat de recherche, la page indique que l'échelle est « now in the public domain » et utilisable sans frais ni notification. Validation française (Vallières & Vallerand 1990) non vérifiée ici. Statut : `retenu-sous-reserve` (relire la page UMD et l'article français avant intégration).

### 2.19 BDI-II, BAI (Pearson)
- Conditions Q-global de Pearson (résultat de recherche) : « You may not, under any circumstance, copy or reproduce the text of any Test question without Pearson's prior written permission. » Licences payantes par passation. Statut : `exclu`.

### 2.20 Classifications : CIM-11 et CDDR
- CIM-11 : https://icd.who.int/docs/icd-api/license — lu : « ICD-11 is licensed under the Creative Commons Attribution-NoDerivs 3.0 IGO license (CC BY-ND 3.0 IGO) ». Selon le document de licence (résultat de recherche) : pas d'« adaptations » ; les correspondances entre classifications demandent un accord écrit de l'OMS. **Conséquence :** citer les codes et intitulés officiels tels quels avec attribution ; ne pas publier une version modifiée de la classification ; le tableau « symptômes ↔ sujets » du site relie des repères du site à des codes, il ne constitue pas une table de correspondance entre classifications.
- CDDR (OMS 2024, ISBN 978-92-4-007726-3) : https://www.who.int/publications-detail-redirect/9789240077263 — lu (fiche) : licence affichée « CC BY-NC-ND 3.0 IGO » avec un lien vers BY-NC-SA ; **incohérence à vérifier dans le PDF**. Dans les deux cas : usage non commercial, attribution. Les repères non validés du site **paraphrasent** les exigences essentielles, sans recopier le texte, avec la citation de la CDDR ; si la licence est ND, une paraphrase courte et indépendante reste préférable à toute traduction suivie.
- DSM-5-TR : texte sous copyright de l'APA (pas de licence ouverte) ; seules des paraphrases de critères, sans reproduction, sont envisageables. Les repères de la page s'appuient d'abord sur la CIM-11.

### 2.21 Seuils vérifiés dans les résumés d'origine (Europe PMC, 19 novembre)
- PHQ-9, Kroenke et al. 2001, PMID 11556941 — résumé lu : « PHQ-9 scores of 5, 10, 15, and 20 represented mild, moderate, moderately severe, and severe depression » ; ≥ 10 : sensibilité et spécificité 88 % pour l'épisode dépressif majeur.
- PHQ-2, Kroenke et al. 2003, PMID 14583691 — résumé lu : deux dernières semaines, 0–3 ; ≥ 3 = repère optimal de dépistage (sensibilité 83 %, spécificité 92 %).
- PHQ-15, Kroenke et al. 2002, PMID 11914441 — résumé lu : 15 symptômes cotés 0–2 ; 5, 10, 15 = sévérité somatique faible, moyenne, élevée.
- GAD-7, Spitzer et al. 2006, PMID 16717171 — résumé lu : un repère optimise sensibilité 89 % / spécificité 82 % ; **la valeur 10 et les paliers 5/10/15 ne figurent pas dans le résumé** : à confirmer dans l'article intégral ou le guide officiel PHQ avant implémentation. GAD-2 (Kroenke et al. 2007, PMID 17339617) : résumé tronqué, repère ≥ 3 **à confirmer**.
- Français : PHQ-9 anglais/français équivalents au niveau du score total (Canada, sclérodermie), PMID 23251676, PLOS ONE 2012 — résumé ; GAD-7 validé chez des personnes épileptiques à Marseille, PMID 26994447 — résumé. **Aucune validation française en population générale adulte n'a été ouverte ici pour le PHQ-9 ni le GAD-7** : lacune à instruire.

## 3. Synthèse : instruments proposés

| Instrument | Domaine | Items | Période | Repère publié (jamais diagnostic) | Français | Statut |
| --- | --- | --- | --- | --- | --- | --- |
| PHQ-2 → PHQ-9 | humeur | 2 → 9 | 2 semaines | PHQ-2 ≥ 3 ; PHQ-9 5/10/15/20 | traduction officielle Pfizer (à télécharger) | retenu |
| GAD-2 → GAD-7 | anxiété | 2 → 7 | 2 semaines | ≥ 3 ; 5/10/15 (à confirmer) | idem | retenu |
| PHQ-15 | symptômes physiques | 15 | 4 semaines | 5/10/15 | idem | retenu |
| ITQ | trauma (CIM-11 TSPT, TSPT-C) | 18 | mois écoulé | règle par grappes (item ≥ 2) | Peraud et al. 2022 (Cn2r) | retenu |
| PCL-5 | trauma (DSM-5-TR), option | 20 | mois écoulé | 31–33 | Ashbaugh et al. 2016 | retenu (option) |
| AUDIT-C → AUDIT | alcool | 3 → 10 | 12 mois | AUDIT ≥ 7 H / ≥ 6 F (Gache 2005) | validé | sous réserve (non commercial) |
| CAST | cannabis | 6 | 12 mois | ≥ 5 ; ≥ 8 | conçu en français (OFDT) | sous réserve |
| ICJE (PGSI) | jeux d'argent | 9 | 12 mois | 3–7 ; ≥ 8 | usage ANJ/OFDT | sous réserve |
| SCOFF | alimentation | 5 | non précisée (à vérifier) | ≥ 2 | Garcia 2010 | sous réserve |
| DES-II | dissociation | 28 | non précisée (à vérifier) | à documenter | Darves-Bornoz 1999 | sous réserve |
| MSI-BPD | personnalité | 10 | à vérifier | ≥ 7 (≥ 5 adolescents) | Mirkovic 2020 | sous réserve |
| ECR-R | attachement | 36 | relations en général | **aucun seuil** (dimensions) | Favez 2016 | sous réserve |
| TAS-20 | alexithymie | 20 | — | à documenter | Loas 1995/1996 | sous réserve |
| Rosenberg | estime de soi | 10 | — | aucun seuil recommandé ici | à vérifier | sous réserve |
| MDQ | bipolaire | 13 + 2 | vie entière | orientation seulement | Weber Rouget 2005 | sous réserve |
| OCI-R | TOC | 18 | dernier mois (à vérifier) | à documenter | Zermatten 2006 | sous réserve |

« Sous réserve » : la page fonctionne d'abord avec les repères non validés du § 4 ; l'instrument remplace le repère seulement après réception d'une autorisation écrite archivée dans `editorial/symptomes/autorisations/` (dossier à créer à ce moment-là).

## 4. Couverture

### 4.1 Par bloc de la CIM-11, chapitre 06 (et chapitres voisins)

| Bloc CIM-11 | Repérage proposé | Commentaire |
| --- | --- | --- |
| 6A0 neurodéveloppement | **aucun** | exclu du site par règle (cartographie) ; mention « non couvert » sur la page |
| 6A2 schizophrénie et psychoses | 2 repères non validés + orientation, sans score | PQ-B/PQ-16 exclus (§ 2.17) |
| 6A4 catatonie | aucun | syndrome médical, hors auto-questionnaire |
| 6A6 bipolaires | 1 repère non validé (épisode d'énergie haute) → MDQ sous réserve | jamais de repère isolé ; orientation médicale |
| 6A7 dépressifs | PHQ-2 → PHQ-9 ; repère « humeur basse quasi permanente depuis 2 ans » pour 6A72 | 6E20 périnatal : question de contexte (grossesse/naissance < 12 mois), pas de score spécifique (EPDS non étudié ici : lacune) |
| 6B0 anxieux | GAD-2 → GAD-7 ; repères 6B01, 6B02, 6B04, 6B05 | 6B03 phobie spécifique, 6B06 mutisme : non demandés (pas de sujet publié) |
| 6B2 TOC et apparentés | repères 6B20, 6B21, 6B23 → OCI-R sous réserve | 6B22 référence olfactive, 6B24 accumulation, 6B25 comportements répétitifs centrés sur le corps : lacunes (pas de sujet publié) |
| 6B4 liés au stress | ITQ (6B40, 6B41) ; repères 6B42 deuil prolongé, 6B43 adaptation | 6B44/6B45 : diagnostics de l'enfance, non demandés |
| 6B6 dissociatifs | repères (dépersonnalisation, amnésie, symptômes neurologiques fonctionnels) → DES-II sous réserve | 6B64 identité : jamais suggéré par un score ; orientation spécialisée |
| 6B8 alimentation | SCOFF → repères 6B80, 6B81, 6B82 | 6B83 ARFID, 6B84 pica, 6B85 rumination : lacunes |
| 6C0 élimination | aucun | pas de sujet publié |
| 6C2 détresse corporelle | PHQ-15 + repère de préoccupation excessive (6C20) | orientation médicale systématique : un symptôme physique se fait d'abord examiner |
| 6C4 substances | AUDIT-C → AUDIT ; CAST ; repère tabac quotidien ; DAST-10 sous réserve | autres produits : repères CIM-11 de l'usage nocif et de la dépendance |
| 6C5 comportements addictifs | ICJE (6C50) ; repères 6C51 jeu vidéo | usage des écrans hors 6C51 : repère non validé |
| 6C7 contrôle des impulsions | repères 6C72 comportement sexuel compulsif, 6C73 explosions de colère | — |
| 6C9 comportement perturbateur et dyssocial | non demandé à l'adulte | sujet `ten_externalisation` couvert par le repère colère/impulsivité |
| 6D1 personnalité | MSI-BPD sous réserve ; 3 repères sur la sévérité (soi, relations) | la CIM-11 classe par sévérité ; « motif borderline » jamais déduit seul |
| 6D3 paraphiliques, 6D5 factices, 6D7–6D8 neurocognitifs | aucun | hors périmètre (pas de sujet, risques) |
| 6E2 périnatal | contexte (§ 6A7) | — |
| 07 sommeil-éveil | repères insomnie chronique et cauchemars (ISI exclu) | codes chapitre 07 à vérifier sur icd.who.int avant publication |
| 17 santé sexuelle | 1 repère non validé de difficulté sexuelle avec détresse | orientation médicale |

### 4.2 Par catégorie de sujets de `data/cartographie.json`
- `trouble` (26 sujets) : repérés par un instrument ou des repères CIM-11, sauf 6B44/6B45 (enfance).
- `schema` (18) : **pas de mesure** (YSQ non libre) ; une phrase de reconnaissance par schéma, Oui / Un peu / Non, sans score, marquée « repère non validé », ouverte seulement depuis les modules « relations » et « soi ».
- `attachement` (3) : ECR-R sous réserve (2 dimensions, sans style) ; la désorganisation du jeune enfant n'est pas demandée.
- `tendance` (66) : repères non validés pour les fonctionnements actuels (rumination, autocritique, perfectionnisme, colère…), questions de sécurité pour suicide, automutilation et violences subies ; les états de santé physique et les vécus passés ne sont pas demandés (liste au § 5).

## 5. Tableau de correspondance modules ↔ sujets de `data/cartographie.json`

Généré par script à partir de la liste des 113 sujets (aucun oubli, aucun doublon, aucun identifiant inconnu vérifiés). Rôle : `symptome` = le module peut signaler ce sujet ; `securite` = question de sécurité jamais notée ; `non_demande` = sujet laissé à Mon enfance ou hors page.

| Module | Sujets (rôle symptome) |
| --- | --- |
| `humeur` — Humeur basse et perte d'intérêt | `dep_generale`, `icd_6a72`, `ten_symptomes_depressifs_perinataux`, `ten_symptomes_premenstruels`, `ten_rumination`, `ten_inferences_negatives` |
| `bipolaire` — Périodes d'énergie ou d'humeur inhabituellement hautes | `icd_6a60` |
| `anxiete` — Inquiétude et tension | `anx_generale`, `ten_hypervigilance`, `ten_nevrosisme` |
| `panique` — Crises de peur soudaines et évitement des lieux | `icd_6b01`, `icd_6b02` |
| `anxiete_sociale` — Peur du regard des autres | `icd_6b04` |
| `separation` — Peur d'être séparé des proches | `icd_6b05` |
| `toc_apparentes` — Pensées qui s'imposent, rituels, apparence, santé | `icd_6b20`, `ten_preoccupation_apparence`, `icd_6b23` |
| `trauma_stress` — Après un événement très éprouvant ; deuil ; adaptation | `icd_6b40`, `icd_6b41`, `icd_6b43`, `icd_6b42` |
| `dissociation` — Sentiment de détachement, absences, symptômes fonctionnels | `ten_dissociation`, `icd_6b60`, `icd_6b64` |
| `somatique` — Symptômes physiques qui pèsent | `icd_6c20`, `ten_douleurs_persistantes`, `ten_fibromyalgie`, `ten_migraine` |
| `alimentation` — Alimentation, poids et image du corps | `icd_6b81`, `icd_6b80`, `icd_6b81_boulimie`, `icd_6b82` |
| `alcool` — Alcool | `icd_6c40_2`, `ten_usage_substances` |
| `cannabis` — Cannabis | `ten_usage_cannabis` |
| `autres_substances` — Tabac et autres produits | `ten_tabagisme_persistant` |
| `jeu_ecrans` — Jeux d'argent, jeux vidéo, écrans | `ten_jeu_argent_problematique`, `ten_ecrans` |
| `sexualite` — Vie sexuelle | `ten_sexualite_compulsive`, `ten_difficultes_sexuelles`, `ten_risques_sexuels` |
| `sommeil` — Sommeil et cauchemars | `ten_sommeil` |
| `perceptions` — Perceptions et pensées inhabituelles | `icd_6a20`, `ten_traits_schizotypiques` |
| `regulation` — Émotions qui débordent, colère, impulsivité | `ten_dysregulation_emotionnelle`, `ten_colere`, `ten_impulsivite`, `ten_externalisation`, `sch_controle_insuffisant`, `ten_procrastination`, `sch_droits_excessifs` |
| `alexithymie` — Reconnaître et dire ses émotions et ses besoins | `ten_alexithymie`, `pat_emotional_suppression`, `sch_surcontrole`, `pat_needs_expression` |
| `personnalite` — Façons durables d'être en relation avec soi et les autres | `icd_6d11_5`, `ten_difficultes_personnalite`, `ten_narcissisme_vulnerable`, `ten_narcissisme_grandiose` |
| `relations` — Liens proches, attachement, conflits | `att_anxieux`, `att_evitant`, `pat_abandonment_fear`, `ten_peur_intimite`, `ten_codependance`, `pat_appeasement`, `pat_conflict`, `pat_overresponsibility`, `ten_qualite_relationnelle`, `pat_social_isolation`, `sch_abandon`, `sch_abnegation`, `sch_assujettissement`, `sch_isolement`, `sch_fusion`, `sch_mefiance`, `sch_recherche_approbation`, `sch_dependance`, `sch_carence_affective` |
| `soi` — Image de soi, exigences, honte et culpabilité | `ten_estime_soi`, `ten_honte`, `pat_guilt`, `pat_self_criticism`, `pat_perfectionism`, `pat_impostor`, `pat_conditional_worth`, `ten_mode_parent_exigeant`, `sch_exigences_elevees`, `sch_imperfection`, `sch_echec`, `sch_negativite`, `sch_punition`, `sch_vulnerabilite`, `ten_surcompensation_schema` |

| Sujet | Rôle | Module / raison |
| --- | --- | --- |
| `ten_idees_suicidaires` | securite | `humeur` — parcours de sécurité (§ 7), jamais un score |
| `ten_conduites_suicidaires` | securite | `humeur` — parcours de sécurité (§ 7), jamais un score |
| `ten_tentatives_suicide` | securite | `humeur` — parcours de sécurité (§ 7), jamais un score |
| `ten_automutilations_non_suicidaires` | securite | `regulation` — parcours de sécurité (§ 7), jamais un score |
| `ten_violences_couple` | securite | `relations` — parcours de sécurité (§ 7), jamais un score |
| `att_desorganisation_enfant` | non_demande | catégorie de la petite enfance, non mesurable chez l'adulte |
| `icd_6b44` | non_demande | trouble réactionnel de l'attachement : diagnostic de l'enfance, non repérable par auto-questionnaire adulte |
| `icd_6b45` | non_demande | engagement social désinhibé : diagnostic de l'enfance |
| `ten_souffrance_enfance` | non_demande | porte sur l'enfance ; relève de Mon enfance |
| `ten_difficultes_scolaires` | non_demande | porte sur la scolarité passée |
| `ten_recompense_immediate_jeunes` | non_demande | repère de recherche chez les jeunes, pas un symptôme |
| `ten_violences_amoureuses_jeunes` | non_demande | vécu passé de l'adolescence, pas un symptôme |
| `pat_parentification` | non_demande | vécu de l'enfance, déjà dans Mon enfance |
| `ten_revictimisation` | non_demande | risque statistique de violences subies : jamais présenté comme un symptôme de la personne |
| `ten_traits_personnalite_antisociale` | non_demande | dimension de recherche ; pas d'auto-repérage proposé (risque de stigmatisation, pas d'instrument libre) |
| `ten_diabete_type2` | non_demande | état de santé physique, pas un symptôme psychologique |
| `ten_maladies_cardiovasculaires` | non_demande | état de santé physique |
| `ten_sante_physique` | non_demande | état de santé physique |
| `ten_surpoids` | non_demande | mesure corporelle ; ne pas demander poids/IMC sur cette page |
| `ten_difficultes_professionnelles` | non_demande | situation sociale ; abordée seulement par la question de retentissement |

## 6. Conception du parcours

### 6.1 Principes
- Page `/mes-symptomes/` dans l'espace personnel, à côté de Mon enfance. Titre de section : « Ce que je vis en ce moment ». Bandeau permanent : « Ces questionnaires repèrent des difficultés ; ils ne posent pas de diagnostic. Seul un professionnel peut le faire, en entretien. »
- **Dépistage en deux temps.** Temps 1 : une question de sécurité pour tous, puis 1 à 3 questions d'entrée par module (≈ 30 questions, 5 à 7 minutes). Temps 2 : l'approfondissement d'un module ne s'ouvre que si son entrée est positive, et la personne choisit de le faire maintenant ou plus tard. Ordre de proposition : humeur, anxiété, trauma, puis les modules positifs par ordre de retentissement déclaré.
- Les entrées utilisent les formes courtes publiées quand elles existent (PHQ-2, GAD-2, AUDIT-C, SCOFF, question d'exposition de l'ITQ) ; sinon un repère non validé de 1 à 2 questions, rédigé par le site en paraphrasant la CIM-11 (CDDR).
- **Échelles d'origine respectées** : libellés et nombre de modalités de chaque instrument, texte français officiel importé à l'identique, ordre des items inchangé, pas de reformulation. Les repères du site utilisent Non / Oui / Je ne sais pas (cohérent avec Mon enfance) ou une échelle de fréquence explicitement marquée comme propre au site.
- **Période de référence affichée** en tête de chaque bloc (« Au cours des 2 dernières semaines », « Au cours du dernier mois », « Au cours des 12 derniers mois », « Depuis au moins 2 ans »).
- Chaque module se termine par une question de retentissement (« Ces difficultés vous gênent-elles nettement dans votre vie ? ») : un repère positif sans gêne déclarée est affiché comme tel.
- « Je préfère ne pas répondre » toujours possible ; une réponse manquante n'est jamais comptée comme 0 : le score n'est pas calculé (règles de cotation d'origine ; la PCL-5 et l'ITQ n'ont pas de règle d'imputation sur le site).
- Mon enfance n'oriente pas l'ordre des questions (pour ne pas suggérer des symptômes) ; le croisement n'intervient qu'à l'affichage des résultats.

### 6.2 Résultats
- Par module : « Vos réponses sont **au-dessus** / **en dessous** du repère publié pour [instrument, auteurs, année] » ; jamais « vous avez », jamais le nom d'un trouble comme conclusion. Le palier publié (ex. PHQ-9 « modéré ») peut être donné en clair avec la phrase « ce palier décrit l'intensité des symptômes, pas un diagnostic ».
- Repères non validés : « Vos réponses ressemblent à certains éléments décrits par la CIM-11 pour … (repère non validé) ».
- En dessous du repère : message neutre, sans réassurance excessive (« Cela n'exclut pas une difficulté ; si elle vous gêne, parlez-en »).
- Chaque carte de résultat propose : en parler à un médecin généraliste ou à un psychologue (dispositif de remboursement, lien institutionnel à vérifier), en parler avec Ambre, refaire plus tard.
- Aucune page de résultat ne montre un repère élevé sans un bloc « Et maintenant ? » visible sans défilement ; si la sécurité est positive, ce bloc est remplacé par le bloc de sécurité (§ 7).

### 6.3 Croisement avec Mon enfance
- Entrées : sujets de `computeCartography` (sections `top` et `other`, retours `oui`/`non`) et sujets signalés par les modules au-dessus du repère (table du § 5, rôle `symptome`).
- Sorties en trois listes : **« Ce qui ressort des deux côtés »** (sujet signalé par un module ET relié à au moins un vécu déclaré, retour différent de `non`) ; « Ce qui ressort aujourd'hui seulement » ; « Ce que la recherche associe à vos vécus, sans difficulté signalée aujourd'hui » (texte : « ces associations sont statistiques ; ne pas les retrouver chez vous est fréquent »).
- Tri de la première liste : module au-dessus du repère avec retentissement d'abord, puis nombre de vécus associés (entiers), puis ordre alphabétique — algorithme en entiers, implémenté deux fois (client et `journal/src/`) avec un test de parité, comme la cartographie.
- **Le croisement ne crée aucun lien vécu → sujet**, ne modifie ni `links` ni l'audit des correspondances, et n'affiche aucun niveau de preuve (décision du 17 novembre). Phrase obligatoire : « Ce rapprochement aide à réfléchir ; il ne montre pas que vos vécus expliquent ce que vous ressentez. »
- Un sujet marqué `non` dans Mon enfance reste visible côté symptômes, sans mention de croisement.

## 7. Sécurité

- **Question de sécurité pour tous**, en tête du temps 1, rédigée par le site (repère non validé, non notée) : idées de suicide ou de se faire du mal « en ce moment ou ces derniers jours ». Réponses : Non / Oui / Je préfère ne pas répondre.
- **Déclencheurs** : cette question à Oui ; item 9 du PHQ-9 > 0 ; repère d'automutilation actuelle (module `regulation`) ; question de violences subies actuellement (module `relations`).
- **Conduite** : interruption immédiate du questionnaire par un écran plein (pas une fenêtre qu'on ferme d'un clic) : « Merci de l'avoir dit. » ; si danger immédiat : **15** ou **112** ; **3114** (numéro national de prévention du suicide, 24 h/24, 7 j/7, gratuit) ; **114** par SMS pour les personnes sourdes ou malentendantes ; rejoindre une personne de confiance ; pour les violences : **17**, **3919**, **119** (mineurs). Ces numéros sont déjà présents dans le code (`chat-safety.mjs`, pages du site). Puis choix : « Continuer le questionnaire », « Parler à Ambre », « Arrêter ». « Je préfère ne pas répondre » affiche une version douce du même bloc, sans blocage.
- **Aucun score de risque** n'est calculé ni affiché (C-SSRS non retenue, § 2.4). Le bloc ne dit jamais « risque faible ».
- **Résultats** : tant qu'un déclencheur date de moins de 14 jours, le bloc de sécurité figure en tête de la page de résultats et de l'historique.
- **Ambre** : le déclencheur est transmis comme fait daté (`safety.reported_at`), traité comme un **antécédent récent** par la règle existante d'ANALYSIS_SYSTEM (« une ancienne note de sécurité… est un antécédent, jamais la preuve d'un danger actuel ») : vigilance, une question directe de sécurité au premier message suivant, ressources si confirmation ; jamais d'exploration du questionnaire avant cela. La vérification de risque (`RISK_CHECK_SYSTEM`) ne reçoit toujours aucun résultat de questionnaire.
- **Aucune notification externe** (ni courriel, ni tiers) : le site n'a pas de professionnel de garde ; l'interface le dit (« Ce site ne peut pas intervenir en urgence »).

## 8. Ce que reçoit Ambre

Résumé borné, construit côté serveur, au plus 1 200 caractères (≈ 350 jetons), placé avec le contexte de Mon enfance et soumis au même découpage budgétaire (`withoutChangeDetails`, omissions tracées) :

```json
{"symptoms":{"as_of":"2026-11-19","consent":true,
 "domains":[{"id":"humeur","instrument":"PHQ-9","period":"2 semaines","band":"au-dessus","level":"modéré","date":"2026-11-19","impact":true},
            {"id":"panique","instrument":"repere_cim11","band":"repere_positif","date":"2026-11-19","validated":false}],
 "below":["anxiete","alcool"],"not_assessed":["sommeil"],"declined":["sexualite"],
 "convergent_subjects":["dep_generale","icd_6b01"],
 "safety":{"reported_at":"2026-11-19"},
 "rules":"repères, pas de diagnostic ; ne pas nommer de trouble comme conclusion ; ne pas recalculer"}}
```

- Pas de réponses item par item, pas de texte d'item (droits et minimisation). Au plus 8 modules (les plus récents au-dessus du repère d'abord) et 10 sujets convergents (identifiants, résolus en libellés par l'index existant `cartographyIndexEntries`).
- Consignes à ajouter au système d'Ambre : citer l'instrument et la date, rappeler la période de référence si elle est dépassée (« ce questionnaire date de 6 semaines »), proposer de refaire, ne jamais dire « vous avez un trouble », orienter vers un professionnel quand un repère est au-dessus avec retentissement.
- Ambre ne peut **ni** remplir ni modifier un questionnaire ; elle peut proposer un module (lien), comme les notes de cartographie actuelles.

## 9. Stockage, consentement, historique

- Nouvelle migration (numéro suivant, `0027_symptom_checks.sql`, additive) :
  - `symptom_checks(id TEXT PK, user_id TEXT REFERENCES users(id) ON DELETE CASCADE, module TEXT, instrument TEXT, instrument_version TEXT, answers TEXT json_valid object, score INTEGER NULL, band TEXT NULL, impact INTEGER NULL, period_days INTEGER, completed_at INTEGER, consent INTEGER CHECK(consent=1))` — une ligne par passation (historique), jamais d'écrasement.
  - `symptom_safety(user_id, check_id, reported_at)` séparée pour l'accès déterministe de la sécurité.
- **Consentement** explicite et distinct de Mon enfance (données de santé, RGPD art. 9) avant tout enregistrement ; sans consentement, calcul uniquement dans le navigateur, rien n'est envoyé, Ambre ne reçoit rien. Retrait du consentement = suppression proposée.
- **Export** : ajouté à l'export du compte (JSON, une entrée par passation avec instrument, période, score, repère, source). **Suppression** : par passation, par module, ou tout ; incluse dans la suppression de compte (contrôle « 45 tables » à étendre).
- **Dans le temps** : historique par module (date, score, palier) ; « refaire » proposé quand la période de référence est écoulée (14 jours PHQ/GAD, 30 jours ITQ/PCL-5, 3 mois pour les échelles sur 12 mois) ; un nouveau passage avant ce délai est permis mais marqué « période en partie chevauchante ». Variation affichée seulement pour les échelles qui publient un changement significatif (ex. PCL-5 : baisse de 10 points, page VA) ; sinon, pas d'interprétation de l'évolution. Ambre reçoit la dernière passation par module, plus la précédente si elle a moins de 6 mois.
- Données instrumentales versionnées (`instrument_version`) : un changement de traduction ou de seuil ne réinterprète pas les anciens résultats.

## 10. Format de données proposé

Fichier : `editorial/symptomes/proposition-2026-11-19.json` (≈ 32 Ko, JSON valide, 113 sujets répartis sans oubli). Clés :
- `rules`, `scales` (échelles du site ; les échelles d'instruments sont marquées `officielle`) ;
- `safety` : question universelle (rédigée par le site, non notée), déclencheurs, ressources, affichage, transmission à Ambre ;
- `sources` (type, citation, URL, niveau de lecture `lu` / `resume` / `notice_seulement` / `fiche_seulement`) et `licences` (statut, titulaire, citation de licence exacte, conditions) ;
- `instruments` : nom, mesure, nombre d'items, période (texte et jours), échelle, `items` = références `PHQ-9 #n` avec `text: null` et `text_source` (document officiel à archiver avec empreinte), cotation, repères publiés et leur source, items de sécurité, statut ;
- `domains` : **exemple complet pour 3 modules** — `humeur` (PHQ-2 → PHQ-9, repère dysthymie, contexte périnatal), `trauma_stress` (question d'exposition → ITQ, option PCL-5, repères deuil prolongé et adaptation), `panique` (5 repères non validés CIM-11 6B01/6B02 et règles) ; chacun avec codes CIM-11, équivalents DSM-5-TR (noms seulement), entrée, approfondissement, retentissement, textes de résultat, sujets reliés ;
- `domains_to_complete` : les 20 autres modules avec sujets et instrument prévu ;
- `crosswalk` : `symptom` (sujet → module), `safety`, `not_asked` (sujet → raison) ;
- `crossing`, `ambre_summary`, `storage`.

Pour l'intégration : un générateur `scripts/build_symptomes.py --check` (à écrire) validerait que chaque identifiant de `crosswalk` existe dans `data/cartographie.json`, que chaque repère non validé porte `validated:false` et une base CIM-11, qu'aucun instrument `retenu-sous-reserve` n'est publié sans fichier d'autorisation, que chaque `text_source` a une empreinte, et que les textes de résultat refusent « vous avez », « diagnostic de » et les pourcentages (comme `summary` de la cartographie).

## 11. Instruments exclus et points juridiques

| Instrument | Raison |
| --- | --- |
| BDI-II, BAI | Pearson : reproduction des items interdite sans autorisation écrite ; licences payantes |
| PAI-BOR | éditeur commercial (PAR), licence payante (non vérifiée en détail) |
| Y-BOCS | « Distribution of the YBOCS is not permitted without specific authorization » ; échelle d'entretien |
| PDSS / PDSS-SR | « Reproduction … is not permitted (or such permission is not clear) » |
| ISI | licence gérée par Mapi Research Trust, conditions non publiques |
| SPIN / Mini-SPIN | droits Duke / auteurs, aucune licence publique trouvée |
| YSQ-S3 | droits non publics ; MSS-YSQ (NovoPsych) incompatible (100 utilisateurs par site, clause 6 sur l'IA, pas de français) |
| DERS-16 | pas de version française validée de la forme courte trouvée |
| C-SSRS (comme score) | conçue pour une relecture professionnelle ; risque de fausse réassurance |
| PQ-B / PQ-16 (comme score) | validés en population demandeuse de soins ; faux positifs et alarme sur un site grand public |
| ASRS (et tout module neurodéveloppement) | hors périmètre du site (règle de la cartographie) |

Points juridiques :
1. **Droit d'auteur des items** : un questionnaire publié reste une œuvre protégée sauf déclaration contraire ; la publication scientifique d'un article ne vaut pas licence. Les validations en accès libre (CC BY) couvrent l'article, pas l'instrument.
2. **Non commercial** : AUDIT (OMS), DAST-10, ECR-R, CDDR (BY-NC). Le site est gratuit, mais un module de facturation existe : si un abonnement payant est activé un jour, ces modules doivent rester gratuits et hors offre, ou faire l'objet d'autorisations.
3. **Pas de modification** : PHQ/GAD (pratique d'usage), C-SSRS (« core questions do not vary »), CIM-11 (CC BY-ND : pas d'adaptation) ; les repères du site sont des textes propres, courts, qui citent leur base sans la traduire.
4. **Marques et noms** : « PHQ », « PRIME-MD » sont liés à Pfizer ; utiliser les noms d'instruments comme références bibliographiques, sans logo ni suggestion de partenariat ou de validation par les titulaires.
5. **Correspondances** : la licence CIM-11 exclut les tables de correspondance entre classifications sans accord écrit de l'OMS ; le site n'en publie pas (il relie ses propres repères à des codes).
6. **Données de santé** : consentement explicite distinct, minimisation (pas de réponses item par item à Ambre), export et effacement ; hébergement et analyse d'impact (AIPD) à vérifier avec le registre existant du site.
7. **Dispositif médical** : un logiciel qui produit une conclusion diagnostique individuelle peut relever du règlement européen 2017/745 ; la formulation « repère publié, pas de diagnostic », l'absence de recommandation thérapeutique individualisée et l'absence de score de risque suicidaire limitent ce risque. **Avis juridique recommandé avant publication** (non instruit ici).

## 12. Lacunes et suites

- Lire en entier : conditions Pfizer, guide officiel PHQ (seuils GAD-7/GAD-2), manuel AUDIT OMS (zones), page ASRS de Harvard, page Rosenberg (UMD), licence CDDR (BY-NC-ND ou BY-NC-SA), document de licence CIM-11.
- Écrire pour autorisation : CAMH (DAST-10), OFDT (CAST), CCSA (ICJE), auteurs de SCOFF, OCI-R, TAS-20, MSI-BPD, MDQ, DES-II, ECR-R ; Eating Attitudes LLC (EAT-26).
- Validations françaises en population générale adulte non trouvées ou non ouvertes : PHQ-9, GAD-7, PHQ-15, DAST-10, OCI-R adulte clinique, Rosenberg.
- Non étudiés : EPDS (périnatal), PG-13-R (deuil prolongé), instruments de dépendance au tabac, au jeu vidéo et aux écrans, CIDI (questions filtres de l'OMS : licence non vérifiée, non utilisée).
- Codes du chapitre 07 (sommeil) et 17 (santé sexuelle) à vérifier sur icd.who.int.
- Relecture clinique indépendante des repères non validés (paraphrases CIM-11), comme pour les épisodes éditoriaux : les contrôles de format ne remplacent pas la relecture des faits.
- Aucune modification de `data/cartographie.json`, des générateurs, d'Ambre ni des migrations n'a été faite.
