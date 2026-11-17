# Relecture indépendante — widget Ambre et intégration

17 novembre 2026. Relecture effectuée en lecture seule des fichiers applicatifs. Les corrections mentionnées ci-dessous ont été réalisées par leurs propriétaires pendant cette relecture, pas par le relecteur. Aucun déploiement.

Périmètre : `static/js/ambre-widget.js`, `static/css/ambre-widget.css`, montage dans `worker.mjs` et `baseof.html`, génération des assets partagés, bootstrap serveur et tests existants. Lecture ponctuelle de `ui.mjs` pour vérifier le contrat du bridge enfant, sans modifier ce fichier.

## Findings corrigés pendant la relecture

### P1 — ancien compte encore visible au retour dans un onglet déjà ouvert

La version initiale du widget ne vérifiait la session qu’à l’ouverture et via BroadcastChannel. Son panneau déjà ouvert pouvait continuer à afficher le compte A après un changement de cookie/session hors de ces notifications. Le handler de visibilité historique de l’UI était limité aux lectures et à `tab === 'portrait'`.

Correction propriétaire observée : le widget masque l’iframe au passage caché et demande `open` au retour visible ; l’enfant relit `/api/me` avant `/api/chat`. La garde de session enfant invalide les données du compte précédent.

**Vérification réelle ajoutée par la relecture :** Worker actuel, SQLite en mémoire, sessions A et B et messages distincts, dans Chromium. Après remplacement du cookie A par B sans BroadcastChannel, les événements visibility/focus entraînent exactement `/api/me`, `/api/chat` et le DOM affiche `MESSAGE PRIVE B` à la place de A. La fermeture/réouverture affiche encore B. Script et trace dans `review-widget/session-worker.cjs` et `review-widget/focus-session-worker.log`.

### P1 — ancienne page privée conservée si la vérification de session échoue

Le handler initial `dte:session-invalidated` faisait seulement `refresh().catch(function(){})`. Après changement de session signalé par l’iframe, un GET `/api/me` en erreur pouvait laisser l’ancienne page A derrière le panneau ; la fermeture révélait cette page au compte B.

Correction propriétaire observée : `clearEmbeddedSession()`, `state.me = null` et remplacement immédiat du DOM avant le GET ; l’erreur affiche une vérification de session à relancer.

**Vérification réelle ajoutée :** page compte rendue par le vrai Worker, cookie A→B, notification d’invalidation puis `/api/me` simulé en 503. L’adresse A disparaît, aucun ancien `#account-email` ne subsiste ; « Vérifier ma session » après rétablissement affiche B. Voir `review-widget/invalidation-worker.cjs`.

### P2 — Tab sortait du dialogue vers l’interface du navigateur

Reproduction initiale : depuis le dernier bouton de l’iframe, Tab produisait `document.activeElement === document.body` et `document.hasFocus() === false`. Le dialogue natif seul ne suffisait pas à assurer une boucle entre les documents.

Correction propriétaire observée : messages directionnels de focus entre les extrémités de l’enfant et la barre parent, sous les mêmes gardes source/origine/version.

**Vérification réelle ajoutée :** avec le véritable chat Worker, Tab depuis le dernier contrôle enfant place le focus sur « Ouvrir Ambre en pleine page » et `document.hasFocus() === true`. Shift+Tab revient au bouton Envoyer enfant. Voir `review-widget/focus-worker.cjs`, lignes `FOCUS_AFTER_TAB` et `FOCUS_AFTER_SHIFT_TAB` de la trace.

## Contrôles satisfaisants à la lecture

- Messages entrants limités à l’origine exacte, à `frame.contentWindow`/`window.parent`, au canal et à la version. Une iframe remplacée ne peut plus envoyer de commande acceptée via son ancienne fenêtre. Envoi à une origine explicite, jamais `*`.
- Aucune URL iframe fournie par message ; seul `/mon-accompagnement/?embedded=1` est chargé. Les navigations du bridge restent sur l’origine locale et une liste de chemins.
- Exception framing limitée aux réponses HTML 200 de la route Ambre avec `embedded=1` ; routes ordinaires et API restent en DENY / `frame-ancestors 'none'`.
- Le script widget injecté par le Worker reçoit le même nonce par réponse que l’application. Le script n’est pas ajouté dans le document embarqué, et son propre garde interdit le montage en iframe.
- Les assets sont canoniques dans `static/`, repris par la génération puis les shells Hugo et Worker ; la commande `build_navigation.mjs --check` contrôlait cette cohérence au moment de l’examen.
- Le widget ne précharge pas le chat à la lecture d’un article. Sa fermeture conserve la même iframe ; l’enfant dispose la voix sur `close` et capture son brouillon. Les messages privés ne sont pas injectés dans les pages publiques par le bootstrap.
- Le Worker sert `Cache-Control: private, no-store`. Le transformateur Cartographie conserve en plus Vary Cookie et supprime les validateurs HTTP.

## Recettes encore nécessaires / limites

1. **Clavier mobile réel, surtout depuis un article Hugo.** Les tests du widget varient largeur/hauteur mais n’ouvrent pas de clavier logiciel. Le panneau repose sur `100dvh` sans adaptation `visualViewport`. Le shell Worker a `interactive-widget=resizes-content`, contrairement au parent Hugo : le viewport de l’iframe ne garantit pas à lui seul que le composeur reste au-dessus du clavier du document parent. Ce point reste une limite de validation ; je n’ai pas prétendu l’avoir reproduit sur appareil mobile.
2. **En-têtes reçus via le proxy Netlify.** Le vrai Worker a été chargé en iframe par Chromium avec ses en-têtes réels, avec succès. Le proxy Netlify distant n’a pas été traversé pour cette version non publiée. Le `X-Frame-Options = DENY` global de `netlify.toml` impose de vérifier les en-têtes réellement reçus sur l’URL embarquée et les routes ordinaires, sans supposer que les tests Worker prouvent le comportement du proxy.
3. **Envoi en vol, dictée et dialogue enfant.** Les mocks initiaux ne couvrent ni POST en cours à fermeture/réouverture ni microphone actif ni Échap avec une vraie modale enfant. Ajouter ces parcours au test intégré ; les preuves de focus ci-dessus ne remplacent pas ces vérifications.
4. **Bridge négatif complet.** Le test existant couvre surtout un message de même origine provenant du parent lui-même. Compléter origine étrangère, autre iframe, version inconnue et chemins refusés. L’implémentation rejette origine/source/version, mais transmet librement les query/hash des chemins autorisés ; une allowlist des paramètres attendus rendrait le contrat d’audit plus strict.
5. **Cache du Worker.** `private, no-store` est présent ; `Vary: Cookie` n’est pas ajouté par le Worker, contrairement au transformateur Cartographie. Ce n’est pas une preuve de fuite avec le no-store actuel, mais le contrat d’audit demandait cette variation explicite.

## Fiabilité des preuves

Les trois harness supplémentaires utilisent le vrai Worker et toutes les migrations SQLite locales. Les comptes et textes sont synthétiques ; aucun modèle ni réseau externe n’est appelé. Les scripts de preuve portent un chemin workspace absolu et ne sont pas proposés comme suites CI définitives.

Les premiers essais Chromium ont crashé avec la reconnaissance vocale native headless disponible. Avec `SpeechRecognition` et `webkitSpeechRecognition` désactivés, comme dans les suites navigateur existantes, les parcours ont abouti. Cela limite les conclusions sur la voix ; ce crash n’est pas présenté comme un bug produit prouvé.

Bilan de la relecture : les trois problèmes concrets signalés ont reçu des corrections observées et des vérifications ciblées. Aucun nouveau défaut bloquant confirmé dans le périmètre relu ; les limites de recette ci-dessus restent ouvertes avant d’affirmer une validation complète mobile/proxy/voix.
