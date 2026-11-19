// Personal space styles. Tokens, header, footer, titles, buttons, fields and windows come from
// static/css/site-system.css (SITE_SYSTEM_CSS); this file only lays out the private pages.
export const WORKSPACE_STYLE = String.raw`
main { position: relative; min-width: 0; }
.app-head, #app-messages, #view > *, .journal-welcome { max-width: var(--page); }
.app-head { margin: 0 0 var(--s5); }
.app-head h1 { margin: 0; }
.app-head .page-back { margin: calc(-1 * var(--s3)) 0 var(--s1); padding: 0; }
.page-description { margin: var(--s1) 0 0; color: var(--muted); font-size: var(--t-sm); }
.panel, .narrow, .personal-section { min-width: 0; }
.muted, .helper, .field-hint, small { color: var(--muted); }
.helper, .field-hint, small { font-size: var(--t-xs); line-height: 1.4; }
.field-hint { display: block; margin-top: 6px; font-weight: 400; }
.status { margin: var(--s3) 0 0; font-size: var(--t-sm); color: var(--muted); }
.status:empty { display: none; }
.status.error, .error { color: var(--danger); }
.status.success { color: var(--accent); }
.notice { margin: var(--s4) 0; padding: var(--s3) var(--s4); border-left: 2px solid var(--accent); background: var(--accent-wash); font-size: var(--t-sm); }
.notice.warn { border-color: var(--danger); background: var(--surface); }
.notice > :last-child { margin-bottom: 0; }
.actions { display: flex; flex-wrap: wrap; align-items: center; gap: var(--s3); margin-top: var(--s5); }
.row { display: flex; flex-wrap: wrap; align-items: baseline; justify-content: space-between; gap: var(--s2) var(--s4); }
.form-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: var(--s4) var(--s5); }
.form-grid .full { grid-column: 1 / -1; }
.field { min-width: 0; margin: 0 0 var(--s4); }
.form-grid .field { margin: 0; }
.field > label > :where(input, select, textarea), label.field > :where(input, select, textarea) { margin-top: 6px; font-weight: 400; }
.check { display: flex; align-items: flex-start; gap: var(--s3); min-height: 44px; margin: 0; padding: 10px 0; font: 400 var(--t-md)/1.5 var(--font-ui); }
.pill { color: var(--muted); font-size: var(--t-xs); }
.page-loading { min-height: 180px; margin: 0; }
blockquote { margin: var(--s3) 0; padding-left: var(--s4); border-left: 2px solid var(--line); }

/* One way to fold and unfold: a chevron on the right. */
details > summary { list-style: none; }
:where(details) > :where(summary) { display: flex; align-items: center; min-height: 44px; font-weight: 600; }
details > summary::-webkit-details-marker { display: none; }
:where(.disclosure, .privacy-details, .settings-disclosure, .change-section, .care-section details, .proposal-before, .profile-group, .change-chat-options) > summary {
  position: relative;
  display: flex;
  align-items: center;
  min-height: 48px;
  padding: var(--s2) var(--s6) var(--s2) 0;
  font: 600 var(--t-md)/1.35 var(--font-ui);
  overflow-wrap: anywhere;
}
:where(.disclosure, .privacy-details, .settings-disclosure, .change-section, .care-section details, .proposal-before, .profile-group, .change-chat-options) > summary::after {
  content: '›' / '';
  position: absolute;
  right: var(--s1);
  top: 50%;
  color: var(--muted);
  font: 400 22px/1 var(--font-ui);
  transform: translateY(-50%);
  transition: transform .12s;
}
:where(.disclosure, .privacy-details, .settings-disclosure, .change-section, .care-section details, .proposal-before, .profile-group, .change-chat-options)[open] > summary::after { transform: translateY(-50%) rotate(90deg); }

/* Empty state: one sentence, one action. */
.workspace-empty { padding: var(--s2) 0; }
.workspace-empty .empty-glyph { display: none; }
.workspace-empty h2 { margin: 0 0 var(--s1); font-size: var(--t-md); }
.workspace-empty p { max-width: 54ch; margin: 0; color: var(--muted); }

/* Signed out. */
.journal-welcome p { color: var(--muted); }
#auth-form > button[type=submit] { width: 100%; margin-top: var(--s2); }
.auth-switch { margin: var(--s4) 0 0; font-size: var(--t-sm); }
.auth-switch .text { font-size: inherit; }
#auth-role-toggle { font-size: var(--t-sm); }

/* People and life events: content on the page, separated by rules. */
.source-toolbar { display: flex; flex-wrap: wrap; align-items: center; gap: var(--s1) var(--s3); }
.source-toolbar:empty { display: none; }
.source-toolbar .text-link { padding: 0; }
.source-editor-host:empty { display: none; }
.source-editor { margin: 0 0 var(--s5); }
.source-editor > summary {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-height: 44px;
  padding: 0 18px;
  border-radius: var(--r-control);
  background: var(--accent);
  color: #fff;
  font: 600 var(--t-md)/1.2 var(--font-ui);
}
@media (hover: hover) { .source-editor > summary:hover { background: color-mix(in srgb, var(--accent) 86%, #000); } }
.source-editor:not([open]) > summary:active { background: color-mix(in srgb, var(--accent) 80%, #000); transform: scale(.98); }
.source-editor[open] > summary { min-height: 0; padding: 0; background: none; color: var(--ink); font: 600 20px/1.3 var(--font-ui); pointer-events: none; }
.source-editor form { margin-top: var(--s4); }
.source-editor[open] { padding-bottom: var(--s4); border-bottom: 1px solid var(--line); }
.form-more { grid-column: 1 / -1; border-top: 1px solid var(--line); }
.form-more > .form-grid { padding: var(--s2) 0 var(--s4); }
.source-editor form > .actions {
  position: sticky;
  bottom: 0;
  z-index: 2;
  margin: var(--s5) calc(-1 * var(--gutter)) 0;
  padding: var(--s3) var(--gutter);
  border-top: 1px solid var(--line);
  background: var(--bg);
}
.source-editor form > .status { margin-bottom: var(--s3); }
.people-options { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 0 var(--s5); }
.people-options > :not(.check) { grid-column: 1 / -1; }
.source-comparison p, .source-note, .source-context-row p, .self-profile-saved, .source-feedback { white-space: pre-wrap; overflow-wrap: anywhere; }
.source-meta { margin: 0; color: var(--muted); font-size: var(--t-xs); line-height: 1.4; }
.source-tag { color: var(--muted); font-weight: 400; }
.source-note { margin: var(--s2) 0 0; max-width: 68ch; }
.source-context { display: grid; gap: var(--s3); margin-top: var(--s4); }
.source-context-row h4 { margin: 0 0 2px; color: var(--muted); font: 400 var(--t-sm)/1.45 var(--font-ui); }
.source-context-row p { margin: 0; }
.source-actions { margin-top: var(--s3); gap: 0 var(--s4); }
.event-foot .text, .timeline-note .text, .source-note-editor > summary, .person-timeline-body > .text-link { font-size: var(--t-sm); }

.people-directory { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: var(--s3); }
/* A person: the name and the role, one or two lines of the relation, a chevron. */
.person-card {
  position: relative;
  display: grid;
  grid-template-columns: minmax(0, 1fr);
  align-items: start;
  gap: var(--s3);
  min-width: 0;
  padding: var(--s4) var(--s7) var(--s4) var(--s4);
  border: 1px solid var(--line);
  border-radius: var(--r-card);
  background: var(--surface);
  transition: border-color .12s, background-color .12s, transform .08s;
}
.person-card:has(:focus-visible) { border-color: var(--accent); }
@media (hover: hover) { .person-card:hover { border-color: var(--accent); } }
.person-card:target { border-color: var(--accent); }
/* Touch: an immediate answer under the finger, then a quiet wait if the page takes time. */
.person-card:has(.person-open:active), .person-card:has(.person-open[data-pending]) { border-color: var(--line-strong); background: var(--accent-wash); }
.person-card:has(.person-open:active) { transform: scale(.99); }
.person-head { overflow-wrap: anywhere; }
.person-card h3 { display: inline; margin: 0; }
.person-head .source-tag::before { content: ' · ' / ''; }
.person-open { color: inherit; text-decoration: none; }
.person-open::after { content: ''; position: absolute; inset: 0; border-radius: var(--r-card); }
.person-open:focus-visible { outline: none; }
.person-open:focus-visible::after { outline: 2px solid var(--accent); outline-offset: 2px; }
.person-card::after { content: '›' / ''; position: absolute; right: var(--s4); top: 50%; color: var(--muted); font: 400 22px/1 var(--font-ui); transform: translateY(-50%); }
html.is-navigating .person-card:has(.person-open[data-pending])::after { content: '' / ''; width: 14px; height: 14px; margin-top: -7px; border: 2px solid var(--line-strong); border-top-color: var(--accent); border-radius: 50%; transform: none; animation: pending-turn .7s linear infinite; }
@keyframes pending-turn { to { transform: rotate(1turn); } }
.person-card .person-line { display: -webkit-box; -webkit-box-orient: vertical; -webkit-line-clamp: 2; margin: var(--s1) 0 0; overflow: hidden; color: var(--muted); font-size: var(--t-sm); line-height: 1.45; }
/* On a computer the list uses the full content width: three columns, the add button at its right edge. */
@media (min-width: 900px) {
  main:has(#people-content .people-directory) :where(.app-head, #view > .source-panel) { max-width: none; }
  main:has(#people-content .people-directory) .source-editor-host:has(> .source-editor:not([open])) { right: 0; }
  #people-content .people-directory { grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: var(--s4); }
  #people-content .person-card { min-height: 104px; }
  #people-content .source-editor[open] { max-width: var(--page); }
}

.event-timeline { position: relative; display: grid; grid-template-columns: minmax(0, 1fr); gap: var(--s4); padding-left: var(--s5); }
.event-timeline::before { content: ''; position: absolute; inset: 14px auto 0 3px; width: 1px; background: var(--line-strong); }
.timeline-year { position: relative; margin: var(--s2) 0 calc(-1 * var(--s2)); font: 400 var(--t-lg)/1.3 var(--font-title); }
.event-timeline .timeline-year::before { content: ''; position: absolute; top: .5em; left: calc(-1 * var(--s5)); width: 7px; height: 7px; border-radius: 50%; background: var(--accent); }
.undated-events { display: grid; gap: var(--s4); margin-top: var(--s6); padding-top: var(--s5); border-top: 1px solid var(--line); }
/* Each moment is a card on the line, its notes nested under a thin rule. */
.event-card { min-width: 0; padding: var(--s4); border: 1px solid var(--line); border-radius: var(--r-card); background: var(--surface); }
.event-card:target, .event-card:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.event-card > .source-body > .source-note { margin-top: var(--s2); }
.event-card .event-foot { margin-top: var(--s3); padding-top: var(--s2); border-top: 1px solid var(--line); }
.event-card .event-foot .timeline-note { margin: var(--s2) 0; padding: var(--s1) 0 0 var(--s3); border-top: 0; border-left: 2px solid var(--line); }
.event-card h4 { margin: 0; overflow-wrap: anywhere; }
.event-card .source-meta a, .event-people a { color: inherit; text-decoration: underline; }
.event-foot { display: flex; flex-wrap: wrap; align-items: flex-start; gap: 0 var(--s4); margin-top: var(--s2); }
.event-foot .event-notes, .event-foot .source-actions { display: contents; }
.event-foot .timeline-note, .event-foot .event-notes > .helper, .event-foot .event-notes > .status { flex: 1 1 100%; }
.event-foot .event-notes > .source-note-editor[open] { flex: 1 1 100%; order: 2; }
.timeline-entry, .timeline-note { min-width: 0; margin: 0; padding: var(--s3) 0; border-top: 1px solid var(--line); }
.timeline-entry h4 { margin: 0; }
.timeline-entry .timeline-note { margin-left: var(--s4); }
.timeline-note > .text.danger, .timeline-note > .source-note-editor { display: inline-block; vertical-align: top; margin-right: var(--s4); }
.timeline-note > .source-note-editor[open] { display: block; margin-right: 0; }
.source-note-editor > summary { display: inline-flex; align-items: center; min-height: 44px; color: var(--accent); font: 600 var(--t-sm)/1.2 var(--font-ui); border-radius: var(--r-control); }
.source-note-editor > summary:hover { text-decoration: underline; }
.source-note-editor form { display: grid; gap: var(--s2); margin: var(--s1) 0 var(--s3); }
.source-note-editor textarea { min-height: 48px; field-sizing: content; max-height: 40dvh; }
.source-note-editor form > button { justify-self: start; margin-top: var(--s1); }
.ambre-observation { padding-left: var(--s4); border-left: 2px solid var(--accent); }
/* Person page: the information in one card, then the story as a vertical line. */
.person-info { padding: var(--s5); border: 1px solid var(--line); border-radius: var(--r-card); background: var(--surface); }
.person-info h2, .person-timeline h2 { margin: 0 0 var(--s3); overflow-wrap: anywhere; }
.person-info h2 { font-size: var(--t-lg); }
.person-info .source-note:first-of-type { margin-top: 0; }
.person-info > .source-actions { margin-top: var(--s4); padding-top: var(--s3); border-top: 1px solid var(--line); gap: var(--s2) var(--s5); }
.person-info .source-editor-host:not(:empty) { margin-top: var(--s5); }
.person-info .source-editor[open] { padding-bottom: 0; border-bottom: 0; }
.person-timeline { margin-top: var(--s6); }
.person-timeline h2 { margin-bottom: var(--s5); }
.person-frise { position: relative; margin: 0; padding: 0 0 0 var(--s6); list-style: none; }
.person-frise::before { content: ''; position: absolute; top: 8px; bottom: var(--s5); left: 6px; width: 2px; border-radius: 1px; background: var(--line); }
.frise-step { position: relative; min-width: 0; padding-bottom: var(--s6); overflow-wrap: anywhere; }
.frise-step:last-child { padding-bottom: var(--s2); }
.frise-step::before { content: ''; position: absolute; top: 4px; left: calc(-1 * var(--s6)); width: 14px; height: 14px; border: 2px solid var(--accent); border-radius: 50%; background: var(--accent); box-sizing: border-box; }
.frise-note::before, .frise-observation::before { background: var(--bg); }
.frise-when, .frise-note .timeline-note > .source-meta:first-child { margin: 0 0 2px; color: var(--muted); font: 600 var(--t-xs)/1.5 var(--font-ui); letter-spacing: .01em; }
.frise-observation > .frise-when { color: var(--accent); }
.frise-title { margin: 0; font: 600 var(--t-lg)/1.35 var(--font-ui); }
.frise-title a { color: var(--ink); text-decoration-color: var(--line-strong); }
.frise-title a:hover { color: var(--accent); }
.frise-step > .source-note { margin: var(--s1) 0 0; }
.frise-step .timeline-note { padding: 0; border-top: 0; }
.frise-step .timeline-note > .source-note { margin-top: 0; }
.frise-event .event-notes { margin-top: var(--s3); }
.frise-event .event-notes > .timeline-note { margin: 0 0 var(--s2); padding: var(--s2) 0 0 var(--s3); border-left: 2px solid var(--line); }
.frise-event .event-notes > .timeline-note > .source-meta:first-child { margin-bottom: 2px; }
.frise-event .event-notes > .source-note-editor > summary { font-weight: 600; }
.frise-observation > .text { font-size: var(--t-sm); }
.person-timeline-body > .frise-more { margin: 0 0 var(--s4) var(--s6); }
/* Migration 0027: « Anecdotique / Marquant ». One segmented control for the form and the filter;
   a marking moment has a full dot and a stronger title, an anecdotal one a hollow dot, a moment
   « à préciser » a dashed dot and two buttons to choose. Targets stay at 44 px. */
.segmented { display: inline-flex; flex-wrap: wrap; max-width: 100%; border: 1px solid var(--line-strong); border-radius: var(--r-control); background: var(--surface); overflow: hidden; }
.segmented > .segment, .segmented > .segment-button { position: relative; display: inline-flex; align-items: center; justify-content: center; flex: 1 1 0; min-width: 0; min-height: 44px; margin: 0; padding: 0 var(--s4); border: 0; border-radius: 0; background: var(--surface); color: var(--ink); font: 600 var(--t-sm)/1.2 var(--font-ui); cursor: pointer; }
.segmented > * + * { border-left: 1px solid var(--line-strong) !important; }
.segment > input { position: absolute; inset: 0; width: 100%; height: 100%; margin: 0; opacity: 0; cursor: pointer; }
.segment:has(input:checked), .segmented > .segment-button[aria-pressed="true"] { background: var(--accent); color: #fff; }
.segment:has(input:focus-visible) { outline: 2px solid var(--accent); outline-offset: -4px; box-shadow: inset 0 0 0 4px #fff; }
@media (hover: hover) { .segment:not(:has(input:checked)):hover, .segmented > .segment-button[aria-pressed="false"]:hover { background: var(--accent-wash); } }
.segmented > .segment-button[aria-pressed="false"]:not(:disabled):active { background: var(--accent-wash); }
.significance-field .segmented { display: flex; width: min(100%, 360px); }
.significance-field:has(input:user-invalid) .segmented { border-color: var(--danger); }
.moment-filter { flex: none; }
.moment-filter > .segment-button { min-width: 104px; }
.source-toolbar:has(.moment-filter) { justify-content: space-between; gap: var(--s3); margin-bottom: var(--s4); }
.source-toolbar .moment-filter { margin-left: auto; }
.person-timeline-body > .moment-filter { margin: 0 0 var(--s5); }
.significance-view { margin-top: var(--s2); }
.significance-tag { display: inline-flex; align-items: center; gap: 6px; color: var(--muted); font: 600 var(--t-xs)/1.4 var(--font-ui); letter-spacing: .01em; }
.significance-tag::before { content: '' / ''; flex: none; width: 9px; height: 9px; border: 1.5px solid var(--line-strong); border-radius: 50%; box-sizing: border-box; }
.significance-view.is-major .significance-tag { color: var(--accent); }
.significance-view.is-major .significance-tag::before { border-color: var(--accent); background: var(--accent); }
.significance-view.is-unset .significance-tag::before { border-style: dashed; }
.significance-choice { display: flex; flex-wrap: wrap; align-items: center; gap: var(--s2) var(--s3); padding: var(--s2) var(--s3); border: 1px dashed var(--line-strong); border-radius: var(--r-control); background: var(--bg); }
.significance-choice > .significance-tag { flex: 1 1 100%; color: var(--ink); }
.significance-choice > .segment-button { flex: 1 1 0; min-width: 120px; min-height: 44px; padding: 0 var(--s3); font-size: var(--t-sm); }
.significance-view > .status:empty { display: none; }
.event-timeline > .event-card { position: relative; }
.event-timeline > .event-card::before { content: '' / ''; position: absolute; top: 22px; left: -25px; width: 9px; height: 9px; border: 1.5px solid var(--line-strong); border-radius: 50%; background: var(--bg); box-sizing: border-box; }
.event-timeline > .event-card.is-major::before { top: 20px; left: -26px; width: 11px; height: 11px; border: 2px solid var(--accent); background: var(--accent); }
.event-timeline > .event-card.is-unset::before { border-style: dashed; }
.event-timeline > .event-card:not(.is-major, .is-anecdotal, .is-unset)::before { display: none; }
.event-card.is-major { border-color: color-mix(in srgb, var(--accent) 35%, var(--line)); }
.event-card.is-major h4 { font-weight: 700; }
.event-card.is-anecdotal { background: color-mix(in srgb, var(--surface) 55%, var(--bg)); }
.event-card.is-anecdotal h4 { font-weight: 500; }
.frise-event.is-major .frise-title { font-weight: 700; }
.frise-event.is-anecdotal::before, .frise-event.is-unset::before { top: 6px; left: calc(-1 * var(--s6) + 2px); width: 10px; height: 10px; border: 1.5px solid var(--line-strong); background: var(--bg); }
.frise-event.is-unset::before { border-style: dashed; }
.frise-event.is-anecdotal .frise-title { font-weight: 500; font-size: var(--t-md); }
.frise-event.is-anecdotal .frise-title a { color: color-mix(in srgb, var(--ink) 82%, var(--muted)); }
.frise-event .significance-view { margin: var(--s1) 0 0; }
.pending-moments { margin-top: var(--s6); padding-top: var(--s5); border-top: 1px solid var(--line); }
.pending-moments > .helper { margin: 0 0 var(--s4); }
.undated-events.pending-moments > .helper { margin: calc(-1 * var(--s2)) 0 0; }
.frise-pending-title { margin: 0 0 var(--s1); font: 400 var(--t-lg)/1.3 var(--font-title); }
.moment-filter-empty, .moment-filter-hidden { margin: var(--s4) 0 0; font-size: var(--t-sm); }
@media (forced-colors: active) {
  .segment:has(input:checked), .segmented > .segment-button[aria-pressed="true"] { background: Highlight; color: HighlightText; forced-color-adjust: none; }
  .segment:has(input:focus-visible) { outline-color: CanvasText; }
  .event-timeline > .event-card.is-major::before, .significance-view.is-major .significance-tag::before { background: CanvasText; border-color: CanvasText; forced-color-adjust: none; }
}
@media (max-width: 599px) {
  .source-toolbar .moment-filter { width: 100%; margin-left: 0; }
  .moment-filter > .segment-button { flex: 1 1 0; }
  .person-timeline-body > .moment-filter { display: flex; width: 100%; }
}
/* Under the line: « Ajouter une note » and « Ajouter un moment » side by side; an open note form takes the full width below. */
.person-timeline-adds { display: flex; flex-wrap: wrap; align-items: flex-start; gap: var(--s3); margin-top: var(--s2); padding-top: var(--s5); border-top: 1px solid var(--line); }
.person-timeline-adds > .person-note-add[open] { flex: 1 1 100%; order: 2; }
.person-moment-add::before { content: '+' / ''; margin-right: 0; color: var(--accent); font-weight: 400; font-size: 20px; line-height: 1; }
.moment-back { min-height: 44px; font-size: var(--t-sm); }
.person-note-add > summary {
  display: inline-flex;
  min-height: 44px;
  padding: 0 18px;
  border: 1px solid var(--line-strong);
  border-radius: var(--r-control);
  background: var(--surface);
  color: var(--ink);
  font: 600 var(--t-md)/1.2 var(--font-ui);
}
.person-note-add > summary::before { content: '+' / ''; margin-right: var(--s2); color: var(--accent); font-weight: 400; font-size: 20px; }
.person-note-add > summary:hover { text-decoration: none; }
.person-note-add > summary:active { background: var(--accent-wash); transform: scale(.98); }
@media (hover: hover) { .person-note-add > summary:hover { background: var(--accent-wash); } }
.person-note-add[open] > summary { border-color: transparent; background: none; padding-left: 0; }
.person-note-add form { max-width: 560px; }
.pagination-button { margin-top: var(--s5); }

/* Account: one column, one rule between sections. */
.account-row { min-width: 0; padding: var(--s6) 0; border-top: 1px solid var(--line); overflow-wrap: anywhere; }
.account-row:first-child { padding-top: 0; border-top: 0; }
.account-row h2 { margin: 0 0 var(--s3); font-size: var(--t-lg); }
.account-row p { margin: 0 0 var(--s2); }
.account-row .actions { margin-top: var(--s3); }
.account-row .actions .text { padding: 0; }
.account-quiet { font-size: var(--t-sm); }
.price { font: 400 var(--t-xl)/1.2 var(--font-title); }
.price small { font: 400 var(--t-sm)/1 var(--font-ui); }
.privacy-details { margin-top: var(--s3); border-top: 1px solid var(--line); color: var(--muted); font-size: var(--t-sm); }
.privacy-details > summary { color: var(--ink); }
.self-identity-fields { margin-bottom: 0; }

/* Conversation: the page is the conversation, the composer is the only fixed element. */
/* Before the start (« Avant de commencer »), the page scrolls like every other page: no fixed
   height, no inner scroller, the footer in place. Only a started conversation fills the screen. */
.shell:has(> .conversation-page .chat-thread:not([hidden])) { padding-block: 0; }
body:has(.conversation-page .chat-thread:not([hidden])) .site-footer { display: none; }
.conversation-page { display: flex; flex-direction: column; }
.conversation-page:has(.chat-thread:not([hidden])) { height: calc(100dvh - var(--header-height) - 1px); min-height: 0; }
.conversation-page #app-messages:empty { display: none; }
.conversation-page #view { flex: 1; display: flex; flex-direction: column; min-height: 0; max-width: none; }
.conversation { position: relative; flex: 1; display: flex; flex-direction: column; min-height: 0; max-width: var(--page); }
.conversation-toolbar { flex: none; display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: var(--s1) var(--s4); padding: 0 0 var(--s3); border-bottom: 1px solid var(--line); }
.conversation-page:has(.chat-thread:not([hidden])) .conversation-toolbar { padding-top: var(--s4); }
.conversation-toolbar h1 { margin: 0; font-size: clamp(var(--t-xl), 4vw, var(--t-2xl)); }
.conversation-title-divider { display: none; }
.conversation-role { display: block; margin-top: 2px; color: var(--muted); font: 400 var(--t-xs)/1.4 var(--font-ui); letter-spacing: 0; }
.message-quota { flex: none; margin: 0 0 0 auto; color: var(--muted); font-size: var(--t-xs); line-height: 1.4; text-align: right; white-space: nowrap; font-variant-numeric: tabular-nums; }
.message-quota strong { font-weight: 400; }
.conversation-toolbar .chat-point { flex: none; margin-left: auto; }
.conversation-toolbar .chat-point + .message-quota { margin-left: 0; }
.chat-point-note { margin: 0; padding: var(--s2) 0; border-bottom: 1px solid var(--line); color: var(--muted); font-size: var(--t-xs); }
.conversation-scroll { padding: var(--s4) 0 0; }
/* One scrolling area once the conversation runs, with a thin, quiet bar kept off the text. */
.conversation-page:has(.chat-thread:not([hidden])) .conversation-scroll {
  flex: 1;
  min-height: 0;
  margin-right: calc(-1 * var(--s3));
  padding: var(--s4) var(--s3) var(--s4) 0;
  overflow-y: auto;
  overscroll-behavior-y: contain;
  scroll-padding: var(--s4);
  scrollbar-width: thin;
  scrollbar-color: var(--line-strong) transparent;
}
.conversation-scroll::-webkit-scrollbar { width: 6px; }
.conversation-scroll::-webkit-scrollbar-thumb { border-radius: 3px; background: var(--line-strong); }
.conversation-scroll::-webkit-scrollbar-track { background: transparent; }
.chat-thread[hidden], .chat-composer[hidden], .journey-panel[hidden] { display: none; }
.chat-message { max-width: 92%; margin: var(--s4) 0; scroll-margin-top: var(--s4); }
.chat-message h3 { margin: 0 0 var(--s1); color: var(--muted); font: 400 var(--t-xs)/1.4 var(--font-ui); }
.chat-user { width: fit-content; margin-left: auto; padding: var(--s3) var(--s4); border-radius: var(--r-card); background: var(--accent-wash); }
.chat-content { margin: 0; line-height: 1.65; white-space: pre-wrap; overflow-wrap: anywhere; }
.chat-noted { margin: var(--s3) 0 0; padding-top: var(--s2); border-top: 1px solid var(--line); color: var(--muted); font-size: var(--t-xs); }
.chat-noted-label { font-weight: 600; }
.chat-noted .text-link, .chat-review-link { padding: 0; font-size: var(--t-sm); }
.chat-review-link { display: flex; width: fit-content; margin-top: var(--s1); }
.chat-thinking { display: flex; align-items: center; gap: var(--s2); min-height: 44px; color: var(--muted); font-size: var(--t-sm); }
.chat-thinking h3 { position: absolute; width: 1px; height: 1px; overflow: hidden; clip: rect(0 0 0 0); }
.thinking-dots { display: inline-flex; gap: 3px; }
.thinking-dots i { width: 4px; height: 4px; border-radius: 50%; background: var(--accent); animation: ambre-thinking 1.2s ease-in-out infinite; }
.thinking-dots i:nth-child(2) { animation-delay: .15s; }
.thinking-dots i:nth-child(3) { animation-delay: .3s; }
@keyframes ambre-thinking { 0%, 70%, 100% { opacity: .35; } 35% { opacity: 1; } }
#chat-proposals:empty { display: none; }
#chat-proposals > form { margin: var(--s5) 0; padding: var(--s4) 0 0; border-top: 1px solid var(--line); }
.chat-composer { flex: none; padding: var(--s3) 0 max(var(--s3), env(safe-area-inset-bottom)); border-top: 1px solid var(--line); }
.composer-row { display: flex; align-items: flex-end; gap: var(--s2); }
.chat-composer textarea { flex: 1; min-width: 0; resize: none; height: 48px; min-height: 48px; max-height: 160px; padding: 11px var(--s3); field-sizing: content; }
.chat-composer button { flex: none; min-height: 48px; }
.chat-composer button.chat-microphone { width: 48px; padding: 0; border-color: var(--line-strong); background: var(--surface); color: var(--accent); }
.chat-microphone .stop-glyph, .chat-microphone:is([data-voice-phase="starting"], [data-voice-phase="listening"], [data-voice-phase="finishing"]) .microphone-glyph { display: none; }
.chat-microphone:is([data-voice-phase="starting"], [data-voice-phase="listening"], [data-voice-phase="finishing"]) .stop-glyph { display: block; }
.chat-composer button.chat-microphone[data-voice-phase="listening"] { border-color: var(--danger); color: var(--danger); }
.chat-memory-notice, .chat-voice-status, .chat-composer .status, .chat-turn-state .status { margin: var(--s2) 0 0; color: var(--muted); font-size: var(--t-xs); line-height: 1.4; }
.chat-memory-notice { margin: 0 0 var(--s2); }
.chat-composer .status.error, .chat-turn-state .status.error { color: var(--danger); }
.chat-voice-status:empty, .chat-turn-state:empty { display: none; }
.chat-turn-state button { min-height: 44px; padding: 0; font-size: var(--t-sm); }
.chat-gate { flex: none; padding: var(--s3) 0; border-top: 1px solid var(--line); font-size: var(--t-sm); }
.chat-gate p { margin: 0; }
.chat-gate .check { font-size: var(--t-sm); }
.chat-gate .actions { margin-top: var(--s2); }
.chat-new-response { position: absolute; bottom: 96px; left: 50%; transform: translateX(-50%); }
.journey-panel h2 { margin: var(--s2) 0 var(--s3); }
.journey-panel h2:focus { outline: none; }
.journey-panel p { max-width: 62ch; }
/* Before the start: Ambre's name and role, one sentence, four steps with their state, then « Commencer ». */
.conversation-page:has(.journey-panel[data-step="intro"]:not([hidden])) .conversation-toolbar { padding-bottom: var(--s4); }
.conversation-page:has(.journey-panel[data-step="intro"]:not([hidden])) .conversation-toolbar h1 { font-size: clamp(var(--t-2xl), 4vw, var(--t-3xl)); }
.conversation-page:has(.journey-panel[data-step="intro"]:not([hidden])) .conversation-role { margin-top: var(--s1); font-size: var(--t-sm); }
.journey-panel[data-step="intro"] { padding-top: var(--s2); }
.journey-panel[data-step="intro"] h2 { margin-bottom: var(--s2); }
.journey-lead { margin: 0 0 var(--s5); color: var(--ink); }
.journey-steps { margin: 0 0 var(--s5); border: 1px solid var(--line); border-radius: var(--r-card); background: var(--surface); overflow: hidden; }
.journey-progress { display: flex; flex-wrap: wrap; align-items: center; gap: var(--s1) var(--s3); max-width: none !important; margin: 0; padding: var(--s3) var(--s4); border-bottom: 1px solid var(--line); font-size: var(--t-sm); }
.journey-progress-label { font-weight: 600; }
.journey-progress-count { margin-left: auto; color: var(--muted); font-variant-numeric: tabular-nums; }
.journey-meter { display: flex; gap: 4px; }
.journey-meter i { width: 22px; height: 4px; border-radius: 2px; background: var(--line); }
.journey-meter i.is-done { background: var(--accent); }
.journey-sources { margin: 0; padding: 0; list-style: none; }
.journey-source {
  position: relative;
  display: grid;
  grid-template-columns: 28px minmax(0, 1fr) auto auto;
  grid-template-areas: "mark text state link";
  align-items: center;
  gap: 2px var(--s3);
  min-height: 64px;
  padding: var(--s3) var(--s4);
  border-top: 1px solid var(--line);
  transition: background-color .12s;
}
.journey-source:first-child { border-top: 0; }
.journey-step-mark { grid-area: mark; display: inline-flex; align-items: center; justify-content: center; width: 28px; height: 28px; border: 1.5px solid var(--line-strong); border-radius: 50%; color: var(--muted); font: 600 var(--t-sm)/1 var(--font-ui); font-variant-numeric: tabular-nums; }
.journey-source.is-done .journey-step-mark { border-color: var(--accent); background: var(--accent); color: #fff; }
.journey-source-text { grid-area: text; display: flex; flex-direction: column; min-width: 0; }
.journey-source-name { font-weight: 600; }
.journey-source-detail { color: var(--muted); font-size: var(--t-sm); line-height: 1.4; }
.journey-source-state { grid-area: state; justify-self: start; padding: 2px var(--s2); border-radius: var(--r-control); background: var(--accent-wash); color: var(--ink); font-size: var(--t-xs); line-height: 1.5; white-space: nowrap; }
.journey-source.is-done .journey-source-state { padding: 0; background: none; color: var(--muted); }
.journey-source .journey-link { grid-area: link; flex: none; }
.journey-source .text-link { min-height: 40px; font-size: var(--t-sm); }
.journey-source .button.journey-link { min-height: 40px; padding: 0 var(--s4); font-size: var(--t-sm); }
.journey-source .journey-link::after { content: ''; position: absolute; inset: 0; }
.journey-source .journey-link:focus-visible { outline: none; }
.journey-source:has(.journey-link:focus-visible) { outline: 2px solid var(--accent); outline-offset: -2px; }
.journey-source:has(.journey-link:active) { background: var(--accent-wash); }
@media (hover: hover) { .journey-source:hover { background: color-mix(in srgb, var(--accent-wash) 55%, var(--surface)); } }
.journey-launch { display: flex; flex-wrap: wrap; align-items: center; gap: var(--s2) var(--s4); }
.journey-launch #journey-start { min-height: 48px; padding-inline: var(--s6); font-size: var(--t-md); }
.journey-launch .journey-hint { flex: 1 1 240px; margin: 0; font-size: var(--t-sm); }
.journey-panel .actions { margin-top: var(--s4); }
.journey-panel .journey-note { max-width: none; margin-top: var(--s6); padding-top: var(--s3); border-top: 1px solid var(--line); color: var(--muted); font-size: var(--t-xs); line-height: 1.5; }
@media (max-width: 599px) {
  .journey-source { grid-template-columns: 28px minmax(0, 1fr) auto; grid-template-areas: "mark text link" "mark state link"; padding: var(--s3); }
  .journey-steps { margin-inline: 0; }
  .journey-progress { padding-inline: var(--s3); }
  .journey-meter { display: none; }
  .journey-launch { flex-direction: column; align-items: stretch; }
  .journey-launch #journey-start { width: 100%; }
  .journey-launch .journey-hint { flex: none; }
}

/* Reviews. */
.review-item { padding: var(--s5) 0; border-top: 1px solid var(--line); }
.review-item:first-child { padding-top: 0; border-top: 0; }
.review-item .review-title { margin: 0 0 var(--s1); }
.review-meta { margin: 0 0 var(--s2); color: var(--muted); font-size: var(--t-xs); }
.review-body p { max-width: 68ch; margin: 0 0 var(--s3); line-height: 1.7; }
.review-excerpt { color: var(--muted); }
.review-item .actions { margin-top: var(--s2); }
.review-digest, .review-feedback { max-width: 68ch; margin: var(--s5) 0 0; padding: var(--s4) 0 0; border-top: 1px solid var(--line); }
.review-feedback-choices { display: flex; flex-wrap: wrap; gap: var(--s2); margin: 0 0 var(--s3); }
.review-feedback-choices button[aria-pressed=true] { border-color: var(--accent); background: var(--accent); color: #fff; }

/* Follow-up (Mon suivi) and read-only dossiers. */
.care-section { margin: 0; padding: var(--s6) 0; border-top: 1px solid var(--line); }
.care-section:first-child { padding-top: 0; border-top: 0; }
.care-section h2 { margin: 0 0 var(--s3); }
.care-form, .care-scopes { display: grid; gap: var(--s3); }
.care-form > button, .care-section > button { justify-self: start; }
.care-form + .helper { margin: var(--s3) 0 0; }
.care-scopes { grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); margin: var(--s4) 0; }
.care-card, .care-read-item { padding: var(--s4) 0; border-top: 1px solid var(--line); overflow-wrap: anywhere; }
.care-card > button, .care-section > button { margin: var(--s2) var(--s3) var(--s2) 0; }
.care-code:empty { display: none; }
.care-code { padding: var(--s4); border: 1px solid var(--line); border-radius: var(--r-card); background: var(--surface); overflow-wrap: anywhere; }
.care-code strong { display: block; font: 600 var(--t-lg)/1.4 ui-monospace, monospace; user-select: all; }
.care-prose { white-space: pre-wrap; overflow-wrap: anywhere; }
.care-log { padding: var(--s3) 0; border-bottom: 1px solid var(--line); font-size: var(--t-sm); }
.care-section button { max-width: 100%; white-space: normal; }

/* Settings (Paramètres). */
#view > .settings-workspace { max-width: var(--page-wide); }
.settings-toolbar { display: flex; align-items: center; justify-content: space-between; gap: var(--s4); margin-bottom: var(--s5); border-bottom: 1px solid var(--line); }
.settings-tabs { display: flex; flex-wrap: wrap; gap: 0 var(--s5); min-width: 0; }
button.settings-tab { flex: none; min-height: 48px; padding: 0; border: 0; border-bottom: 2px solid transparent; border-radius: 0; background: none; color: var(--muted); font-size: var(--t-sm); }
button.settings-tab:hover { background: none; color: var(--ink); }
button.settings-tab[aria-selected=true] { border-bottom-color: var(--accent); color: var(--ink); }
.settings-toolbar > button { flex: none; }
.settings-pane > *, .settings-section > * { max-width: var(--page); }
.settings-section { margin: 0 0 var(--s6); }
.settings-section > h2 { margin: 0 0 var(--s3); }
/* A sentence never sticks to the fold or the button above it. */
.settings-workspace :where(.change-section, .settings-disclosure) + :where(p),
.settings-workspace :where(button) + :where(p),
.settings-workspace :where(button) + .status + :where(p) { margin-top: var(--s3); }
.settings-intro, .settings-extra > p, .settings-footnote { color: var(--muted); font-size: var(--t-sm); }
.settings-context, .settings-source-area { margin: var(--s5) 0; padding: var(--s5) 0 0; border-top: 1px solid var(--line); }
.settings-context > h3, .settings-source-area h3 { margin: 0 0 var(--s2); }
.settings-disclosure { min-width: 0; margin: 0; border-bottom: 1px solid var(--line); }
.settings-disclosure > :not(summary) { margin-bottom: var(--s4); }
.settings-disclosure .settings-disclosure { margin-left: var(--s3); border-bottom: 0; border-top: 1px solid var(--line); }
.settings-disclosure .settings-disclosure > summary { min-height: 44px; font-size: var(--t-sm); }
.settings-value { min-width: 0; font-size: var(--t-sm); }
.settings-value-item + .settings-value-item { margin-top: var(--s3); padding-top: var(--s3); border-top: 1px solid var(--line); }
.settings-data-row { display: grid; grid-template-columns: minmax(100px, .65fr) minmax(0, 1.7fr); gap: var(--s2) var(--s5); padding: var(--s2) 0; }
.settings-data-label { color: var(--muted); font-size: var(--t-xs); overflow-wrap: anywhere; }
.settings-prose { max-width: 68ch; margin: 0; line-height: 1.7; white-space: pre-wrap; overflow-wrap: anywhere; }
.settings-json { max-height: 480px; padding: var(--s3); overflow-y: auto; border: 1px solid var(--line); border-radius: var(--r-control); background: var(--surface); font-size: var(--t-xs); white-space: pre-wrap; overflow-wrap: anywhere; word-break: break-word; }
.settings-pipeline { margin: var(--s5) 0; padding: 0; list-style: none; counter-reset: pipeline; }
.settings-step { position: relative; margin: 0; padding: 0 0 var(--s5) var(--s6); counter-increment: pipeline; }
.settings-step::before { content: counter(pipeline) '.'; position: absolute; left: 0; top: 0; color: var(--muted); font: 600 var(--t-md)/1.35 var(--font-ui); }
.settings-step h3 { margin: 0 0 var(--s2); }
.settings-step > p { max-width: 68ch; font-size: var(--t-sm); }
.settings-model { display: flex; flex-wrap: wrap; align-items: baseline; gap: 2px var(--s2); margin: 0 0 var(--s2); color: var(--muted); font-size: var(--t-xs); }
.settings-model > strong { font-weight: 400; overflow-wrap: anywhere; }
.settings-footnote { padding-top: var(--s4); border-top: 1px solid var(--line); }
.settings-source-area .field { max-width: 560px; }
.settings-source-area > button { margin-top: var(--s4); }
.settings-source-content { padding-left: var(--s4); border-left: 2px solid var(--line); }
.settings-links { display: flex; flex-wrap: wrap; gap: 0 var(--s5); }
.settings-links .text-link, .settings-source-content .text-link, .settings-section > .text-link { padding: 0; }
.settings-extra > .settings-value { padding-top: var(--s4); }
#view .settings-turn-layout { display: grid; grid-template-columns: 230px minmax(0, 1fr); align-items: start; gap: var(--s5); max-width: none; }
.settings-turn-list { display: grid; gap: 0; max-height: 580px; overflow-y: auto; }
button.settings-turn-button { display: block; width: 100%; min-height: 48px; padding: var(--s3); border: 0; border-bottom: 1px solid var(--line); border-radius: 0; background: none; color: var(--ink); font: 400 var(--t-sm)/1.45 var(--font-ui); text-align: left; }
button.settings-turn-button:hover { background: var(--accent-wash); }
button.settings-turn-button[aria-pressed=true] { background: var(--accent-wash); box-shadow: inset 2px 0 0 var(--accent); }
.settings-turn-button > span { display: block; margin-top: 2px; color: var(--muted); font-size: var(--t-xs); }
.settings-turn-detail { min-width: 0; }
.settings-turn-detail:empty { display: none; }
.settings-turn-detail > h3 { margin: 0 0 var(--s4); }
.settings-message { margin: 0 0 var(--s4); padding: var(--s3) var(--s4); border-left: 2px solid var(--accent); background: var(--accent-wash); white-space: pre-wrap; overflow-wrap: anywhere; }
.settings-turn-outcome { color: var(--muted); font-size: var(--t-sm); }
.settings-stage-content h4 { margin: var(--s4) 0 var(--s3); font-size: var(--t-sm); }
.settings-stage-content > .settings-value > .settings-disclosure { margin-left: 0; }
.settings-empty { color: var(--muted); }
.settings-rules { padding-left: var(--s5); font-size: var(--t-sm); }
.settings-rules li { margin-bottom: var(--s2); }
.engine-steps li { margin: var(--s3) 0; }

@media (min-width: 600px) {
  /* The add button sits beside the page title while its form is closed. */
  .source-editor-host:has(> .source-editor:not([open])) { position: absolute; top: 0; right: calc(100% - var(--page)); margin: 0; }
  .source-editor-host:has(> .source-editor:not([open])) .source-editor { margin: 0; }
  main:has(.source-editor:not([open])) .app-head { padding-right: 220px; }
  .person-page .source-editor-host:has(> .source-editor:not([open])) { position: static; }
  main:has(.person-page) .app-head { padding-right: 0; }
  .source-editor form > .actions { position: static; margin-inline: 0; padding-inline: 0; border-top: 0; }
}
@media (max-width: 1100px) and (min-width: 600px) {
  .source-editor-host:has(> .source-editor:not([open])) { right: max(0px, calc(100% - var(--page))); }
}
@media (max-width: 599px) {
  .form-grid, .people-directory, .people-options { grid-template-columns: minmax(0, 1fr); }
  .person-timeline-adds > * { flex: 1 1 100%; }
  .person-timeline-adds > .person-note-add:not([open]) > summary { display: flex; justify-content: center; }
  .person-info { padding: var(--s4); }
  .person-info .source-editor form > .actions { margin-inline: calc(-1 * var(--s4)); padding-inline: var(--s4); border-radius: 0 0 var(--r-card) var(--r-card); background: var(--surface); }
  .source-editor:not([open]) > summary { display: flex; width: 100%; }
  form > .actions > :where(button, .button):not(.text) { flex: 1 1 auto; }
  .settings-toolbar { flex-wrap: wrap; gap: 0 var(--s4); }
  #view .settings-turn-layout { grid-template-columns: minmax(0, 1fr); }
  .settings-turn-list { max-height: 200px; }
  .settings-data-row { grid-template-columns: minmax(0, 1fr); gap: 2px; }
  .settings-disclosure .settings-disclosure { margin-left: var(--s2); }
  .chat-message { max-width: 96%; }
}
@media (max-height: 560px) {
  .conversation-toolbar { padding-block: var(--s2); }
  .chat-composer textarea { max-height: 100px; }
}
`;
