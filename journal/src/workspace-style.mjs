export const WORKSPACE_STYLE = String.raw`
:root {
  --ink: #172d3c;
  --muted: #526575;
  --paper: #fff;
  --wash: #f2f5f7;
  --line: #d6e0e6;
  --accent: #205e54;
  --accent-wash: #e7f1ee;
  --focus: #286967;
  --error: #962b40;
  --app-width: 1040px;
}
body { background: var(--wash); line-height: 1.55; }
.shell { max-width: 1160px; padding-inline: 32px; }
:where(a,button,summary,input,textarea,select):focus-visible {
  outline: 2px solid var(--focus);
  outline-offset: 3px;
}
.app-head,.narrow,#app-messages { max-width: var(--app-width); margin-inline: auto; }
.app-head { margin-block: clamp(30px,5vw,64px) 25px; }
.app-head h1 {
  color: var(--ink);
  font: 500 clamp(2.25rem,4.5vw,3.65rem)/1.03 Georgia,"Times New Roman",serif;
  letter-spacing: -.045em;
}
.page-description { max-width: 65ch; margin: 12px 0 0; color: var(--muted); font-size: .97rem; }
.panel {
  padding: clamp(20px,4vw,40px);
  border: 1px solid var(--line);
  border-radius: 16px;
  background: var(--paper);
  box-shadow: 0 2px 4px #172d3c07;
}
.section-heading,h2 { color: var(--ink); font-size: clamp(1.2rem,2.4vw,1.45rem); font-weight: 650; }
.panel-intro { color: var(--muted); max-width: 62ch; margin: -5px 0 24px; font-size: .9rem; }
.helper,.muted { color: var(--muted); }
button,.button { background: var(--accent); min-height: 44px; }
button:hover,.button:hover { background: #184b44; }
button.secondary,.button.secondary { background: var(--paper); color: var(--accent); border: 1px solid var(--line); }
button.secondary:hover,.button.secondary:hover { background: var(--accent-wash); }
button.text { min-height: 44px; color: var(--accent); }
button.text.danger { color: var(--error); }
button.text:hover { background: transparent; }
input,textarea,select { border-color: var(--line); }
.notice { background: var(--accent-wash); border-color: var(--accent); border-radius: 0 8px 8px 0; }
.portrait-state {
  display: flex;
  flex-wrap: wrap;
  align-items: baseline;
  gap: 0 14px;
  margin-top: 18px;
  padding-top: 12px;
  border-top: 1px solid var(--line);
  color: var(--muted);
  font-size: .83rem;
}
.portrait-state:empty { display: none; }
.portrait-state button.text { padding: 0; font-size: inherit; }
#portrait-body > div > p,#portrait-body .portrait-reading > p { max-width: 74ch; }
/* A single, directly editable presentation. */
#self-description { display: block; min-height: 320px; font-size: 1rem; }
.self-profile-saved { white-space: pre-wrap; overflow-wrap: anywhere; max-height: 240px; overflow-y: auto; }
.page-loading { min-height: 180px; margin: 0; padding-block: 12px; }
.liked-article { padding-block: 18px; }

.workspace-empty { padding: clamp(16px,4vw,35px) 0; }
.workspace-empty .empty-glyph {
  display: grid;
  place-items: center;
  width: 45px;
  height: 45px;
  border-radius: 12px;
  background: var(--accent-wash);
  color: var(--accent);
  margin-bottom: 17px;
}
.empty-glyph svg { width: 24px; height: 24px; }
.workspace-empty h2 { margin: 0 0 7px; }
.workspace-empty p { color: var(--muted); max-width: 54ch; margin: 0; }
.workspace-empty .actions { margin-top: 18px; }

.source-panel { padding: clamp(18px,3.6vw,34px); }
.source-toolbar { display: flex; align-items: center; flex-wrap: wrap; gap: 6px 12px; margin-bottom: 14px; }
.source-panel .source-toolbar h2 { margin: 0; font-size: clamp(1.18rem,2.6vw,1.45rem); }
.source-tag {
  display: inline-flex;
  align-items: center;
  width: fit-content;
  padding: 3px 8px;
  border-radius: 6px;
  background: #eaf0f3;
  color: var(--muted);
  font-size: .75rem;
  font-weight: 650;
  overflow-wrap: anywhere;
}
.source-editor { margin: 0 0 20px; }
.source-editor > summary {
  align-items: center;
  gap: 8px;
  min-height: 44px;
  padding: 9px 14px;
  background: var(--accent);
  border-radius: 8px;
  color: var(--paper);
}
.source-editor[open] > summary { background: var(--accent-wash); color: var(--accent); }
.source-plus { font-size: 1.15rem; line-height: 1; }
.source-editor form {
  margin-top: 13px;
  padding: clamp(16px,3vw,26px);
  border: 1px solid var(--line);
  border-radius: 12px;
  background: var(--wash);
}
.source-editor .form-grid .field { min-width: 0; }
.source-editor .actions { gap: 8px; }
.source-context { display: grid; gap: 12px; margin-top: 18px; }
.source-context-row { min-width: 0; }
.source-context-row h4 { margin: 0 0 3px; font-size: .8125rem; font-weight: 600; color: var(--muted); }
.source-context-row p { margin: 0; white-space: pre-wrap; overflow-wrap: anywhere; }
.source-comparison { width: 100%; }
.source-comparison p { white-space: pre-wrap; overflow-wrap: anywhere; }
.people-directory { display: grid; grid-template-columns: repeat(2,minmax(0,1fr)); gap: 12px; }
.source-item {
  padding: 17px 18px;
  border: 1px solid var(--line);
  border-radius: 12px;
  background: var(--paper);
  min-width: 0;
  scroll-margin-top: 94px;
}
.source-item:target { outline: 2px solid var(--accent); outline-offset: 2px; }
.source-item p { margin: 0; }
.source-item h3,.source-item h4 { margin: 0; font-size: 1.03rem; overflow-wrap: anywhere; }
.source-meta,.event-people { display: flex; flex-wrap: wrap; align-items: center; gap: 5px 9px; margin: 6px 0 0; color: var(--muted); font-size: .82rem; }
.source-item .source-note { margin-top: 10px; max-width: 70ch; }
.source-item .source-actions { margin: 3px 0 -6px; gap: 16px; }
.source-actions button.text { padding: 0; text-decoration: none; font-size: .83rem; font-weight: 650; }
.source-actions button.text:hover { text-decoration: underline; }
.source-item .source-actions button.danger:not(:hover):not(:focus-visible) { color: var(--muted); }
.person-observations { border-top: 1px solid var(--line); margin-top: 12px; }
.person-observations summary { list-style: none; min-height: 44px; padding: 10px 0; color: var(--accent); font-size: .9rem; font-weight: 650; }
.person-observations summary::-webkit-details-marker { display: none; }
.person-observations summary::before { content: '+' / ''; display: inline-block; width: 1rem; }
.person-observations[open] summary::before { content: '−' / ''; }
.person-observation-content :is(h4, h5) { margin: 10px 0 5px; font-size: .9rem; }
.person-observation-content h5 { color: var(--muted); }
.person-observation-map, .person-observation-notes { list-style: none; margin: 0; padding: 0; }
.person-observation-map { display: flex; flex-wrap: wrap; gap: 6px; }
.person-observation-map li { padding: 3px 8px; background: var(--accent-wash); border-radius: 5px; font-size: .77rem; }
.person-observation-map strong { margin-left: .45rem; }
.person-observation-notes li + li { border-top: 1px solid var(--line); }
.person-observation-notes blockquote { margin: 0; padding-left: .65rem; border-left: 2px solid var(--line); font-size: .875rem; overflow-wrap: anywhere; }
.person-observation-notes button { font-size: .76rem; }
.event-timeline {
  position: relative;
  display: grid;
  grid-template-columns: minmax(0,1fr);
  gap: 10px;
  padding-left: 23px;
  margin-left: 5px;
}
.event-timeline::before { content: ""; position: absolute; inset: 16px auto 12px 0; width: 2px; background: var(--line); }
.timeline-year { position: relative; margin: 8px 0 0; color: var(--accent); font-size: .89rem; }
.event-timeline .timeline-year::before {
  content: "";
  position: absolute;
  top: .43em;
  left: -27px;
  width: 10px;
  height: 10px;
  background: var(--accent);
  border: 2px solid var(--paper);
  border-radius: 50%;
}
.event-card .source-meta { margin: 0 0 7px; }
.event-date { font-weight: 700; color: var(--accent); }
.event-meaning { padding-left: 12px; border-left: 2px solid var(--line); margin-top: 12px; font-size: .89rem; }
.event-meaning > span,.event-people > span { color: var(--muted); font-size: .75rem; font-weight: 700; }
.event-people { margin-top: 10px; }
.event-people a { color: var(--accent); min-height: 36px; }
.undated-events { display: grid; gap: 10px; margin-top: 24px; }
.undated-events .timeline-year { color: var(--muted); }

.conversation-page {
  display: flex;
  flex-direction: column;
  max-width: var(--app-width);
  height: calc(100dvh - 69px);
  min-height: 300px;
  margin-inline: auto;
  padding-block: clamp(8px, 2.2vh, 20px);
}
.conversation-page #view { flex: 1; min-height: 0; display: flex; flex-direction: column; }
.conversation { border-radius: 16px; box-shadow: 0 2px 4px #172d3c07; }
.conversation-toolbar {
  display: flex;
  justify-content: space-between;
  align-items: center;
  flex: none;
  gap: 12px;
  padding: 12px clamp(12px, 3vw, 24px);
  border-bottom: 1px solid var(--line);
  background: var(--wash);
}
.conversation-toolbar h1 {
  margin: 0;
  font: 650 clamp(.875rem, 2vw, 1.0625rem)/1.3 system-ui, -apple-system, "Segoe UI", sans-serif;
  letter-spacing: 0;
}
.message-quota {
  flex: none;
  margin: 0;
  color: var(--muted);
  font-size: .75rem;
  line-height: 1.25;
  text-align: right;
  white-space: nowrap;
}
.message-quota strong { color: var(--ink); font-variant-numeric: tabular-nums; }
.chat-message.chat-user { background: var(--accent-wash); }
.chat-composer textarea { border-color: var(--line); }
.chat-memory-notice { margin: 0 0 6px; color: var(--muted); font-size: 13px; }
.account-panel { display: grid; grid-template-columns: repeat(2,minmax(0,1fr)); gap: 0 26px; }
.account-row { min-width: 0; overflow-wrap: anywhere; border-color: var(--line); }
.account-row:first-child { grid-column: 1 / -1; }
.account-row:nth-child(2) { grid-row: span 2; }
.account-row h3 { font-size: 1rem; }
.journal-welcome { padding: clamp(24px,5vw,50px); background: var(--paper); border: 1px solid var(--line); border-radius: 16px; }

@media (min-width: 601px) {
  .source-panel [id$="-content"] { display: grid; grid-template-columns: minmax(0,1fr) auto; align-items: start; column-gap: 16px; }
  .source-toolbar { min-height: 44px; }
  .source-editor-host:has(.source-editor[open]) { grid-column: 1/-1; }
  .people-directory,.event-timeline,.undated-events,.source-panel .workspace-empty,.pagination-button { grid-column: 1/-1; }
}
@media (max-width: 840px) {
  .shell { padding-inline: 19px; }
  .conversation-page { height: calc(100dvh - 65px); }
}
@media (max-width: 600px) {
  .shell { padding-inline: 15px; }
  .app-head { margin-block: 28px 19px; }
  .page-description { font-size: .875rem; margin-top: 9px; }
  .people-directory,.account-panel { grid-template-columns: minmax(0,1fr); }
  .account-row:nth-child(2) { grid-row: auto; }
  .account-row .row { flex-wrap: wrap; gap: 5px; }
  .source-item { padding: 13px 12px; }
  .source-editor > summary { display: flex; width: 100%; justify-content: center; }
  .event-timeline { padding-left: 16px; }
  .event-timeline .timeline-year::before { left: -20px; }
  .workspace-empty .actions { align-items: stretch; }

}
@media (max-width: 380px) {
  .workspace-empty .actions > * { width: 100%; }
}
@media (max-width: 480px) {
  .conversation-title-divider { display: none; }
  .conversation-role { display: block; color: var(--muted); font-size: .8125rem; font-weight: 550; }
}
@media (max-height: 560px) {
  .conversation { border-radius: 10px; }
}
/* Journey with Ambre: inline welcome, written reviews and notes on the map. */
.journey-panel { width: 100%; max-width: 680px; margin: 0 auto 20px; padding: clamp(4px, 2vw, 18px); }
.journey-panel[hidden], .chat-composer[hidden], .chat-thread[hidden] { display: none; }
@media (max-width: 480px) {
 .journey-source { grid-template-columns: minmax(0, 1fr) auto; }
 .journey-source .text-link { grid-column: 1 / -1; justify-self: start; margin-top: 0; }
}
.journey-panel h2 { margin: 0 0 14px; font-size: clamp(1.25rem, 4vw, 1.6rem); line-height: 1.25; letter-spacing: -.3px; }
.journey-panel h2:focus { outline: none; }
.journey-panel p { margin: 0 0 14px; font-size: .9375rem; line-height: 1.6; }
.journey-panel .muted, .journey-panel .helper { font-size: .8125rem; color: var(--muted); }
.journey-sources { list-style: none; margin: 18px 0; padding: 0; border-top: 1px solid var(--line); }
.journey-source { display: grid; grid-template-columns: minmax(0, 1fr) auto auto; gap: 6px 14px; align-items: center; padding: 12px 0; border-bottom: 1px solid var(--line); }
.journey-source-text { display: flex; flex-direction: column; min-width: 0; }
.journey-source-name { font-weight: 600; font-size: .9375rem; }
.journey-source-detail { font-size: .75rem; color: var(--muted); }
.journey-source-state { font-size: .75rem; padding: 3px 9px; border-radius: 999px; background: #f6efe6; color: #6f4e22; white-space: nowrap; }
.journey-source.is-done .journey-source-state { background: #e6f3ef; color: #24594e; }
.journey-source .text-link { margin-left: 0; font-size: .8125rem; }
.journey-panel .check { margin: 4px 0 6px; font-size: .875rem; }
.journey-panel .actions { margin-top: 18px; }
.journey-panel button:disabled { opacity: .55; cursor: not-allowed; }
.journey-panel .status:empty { display: none; }
.journey-wait { color: var(--muted); }
.conversation-toolbar .chat-point { flex: none; margin-left: auto; padding: 7px 14px; font-size: .8125rem; }
.conversation-toolbar .chat-point:disabled { opacity: .55; cursor: not-allowed; }
.chat-point-note { margin: 0; padding: 6px clamp(12px, 3vw, 24px); font-size: .75rem; color: var(--muted); background: var(--wash); border-bottom: 1px solid var(--line); text-align: right; }
.chat-noted { margin: 10px 0 0; padding-top: 9px; border-top: 1px dashed var(--line); font-size: .8125rem; line-height: 1.55; color: var(--muted); }
.chat-noted-label { font-weight: 600; color: #316e62; }
.chat-noted .text-link, .chat-review-link { margin-left: 0; font-size: .8125rem; }
.chat-review-link { display: block; margin-top: 10px; }
.chat-content { white-space: pre-wrap; }
.review-list { display: grid; gap: 0; }
.review-item { padding: 22px 0; border-bottom: 1px solid var(--line); }
.review-item:first-child { padding-top: 4px; }
.review-item .review-title { margin: 0 0 4px; font-size: 1.25rem; letter-spacing: -.2px; }
.review-meta { margin: 0 0 10px; font-size: .8125rem; color: var(--muted); }
.review-body p { margin: 0 0 12px; font-size: .9375rem; line-height: 1.7; max-width: 68ch; }
.review-excerpt { color: var(--muted); }
.review-item .actions { margin-top: 10px; }
.review-digest, .review-feedback { margin: 18px 0 0; padding: 16px 0 0; border-top: 1px solid var(--line); max-width: 68ch; }
.review-digest h3, .review-feedback h3 { margin: 0 0 8px; font-size: 1rem; }
.review-feedback-choices { display: flex; flex-wrap: wrap; gap: 8px; margin: 0 0 12px; }
.review-feedback-choices button { min-height: 44px; }
.review-feedback-choices button[aria-pressed=true] { background: var(--accent-wash); border-color: #a7c4bd; }
@media (max-width: 600px) {
  .journey-source { grid-template-columns: minmax(0, 1fr) auto; }
  .journey-source .text-link { grid-column: 1 / -1; }
  .conversation-toolbar { flex-wrap: wrap; }
}

/* Follow-up, read-only dossiers and transparent conversation settings. */
.care-section { max-width: 860px; margin: 26px auto; padding: clamp(16px,3vw,28px); background: white; border: 1px solid var(--line); border-radius: 14px; }
.care-section h2 { font-size: 1.25rem; }
.care-form, .care-scopes { display: grid; gap: 12px; }
.care-scopes { grid-template-columns: repeat(auto-fit,minmax(180px,1fr)); margin: 16px 0; }
.care-scopes legend { margin-bottom: 10px; }
.care-card, .care-read-item { padding: 18px 0; border-top: 1px solid var(--line); overflow-wrap: anywhere; }
.care-card > button, .care-section > button { margin: 6px 10px 6px 0; }
.care-card summary { font-weight: 600; padding: 8px 0; }
.care-code:empty { display: none; }
.care-code { padding: 18px; background: var(--wash); border-radius: 10px; overflow-wrap: anywhere; }
.care-code strong { display: block; font-family: monospace; font-size: 1.15rem; user-select: all; }
.care-prose { white-space: pre-wrap; overflow-wrap: anywhere; }
.care-log { padding: 10px 0; border-bottom: 1px solid var(--line); font-size: .875rem; }
.engine-steps li { margin: 12px 0; }
.care-section textarea { min-height: 85px; }
@media(max-width:600px){.care-section{margin:18px 0}.care-card input{min-width:0;max-width:100%}.care-section button{max-width:100%;white-space:normal}}
/* The member's own data and the actual inputs used by Ambre. */
.settings-workspace { max-width: 1040px; padding-bottom: 32px; }
.settings-toolbar { display: flex; align-items: center; justify-content: space-between; gap: 16px; border-bottom: 1px solid var(--line); margin-bottom: 28px; }
.settings-tabs { display: flex; flex-wrap: wrap; gap: 4px 20px; }
button.settings-tab { border-radius: 0; background: transparent; color: var(--muted); font-size: .875rem; padding: 14px 0; border-bottom: 3px solid transparent; }
button.settings-tab[aria-selected=true] { color: var(--accent); border-bottom-color: var(--accent); }
.settings-pane[hidden],.settings-source-list [hidden] { display: none; }
.settings-section { margin: 0 0 36px; }
.settings-section > h2 { margin: 0 0 12px; font-size: clamp(1.35rem,2.4vw,1.75rem); letter-spacing: -.025em; }
.settings-intro { max-width: 74ch; color: var(--muted); margin: 0 0 24px; font-size: .925rem; line-height: 1.7; }
.settings-context { background: var(--paper); padding: clamp(16px,3vw,28px); border: 1px solid var(--line); border-radius: 12px; margin: 24px 0 30px; }
.settings-context > h3 { margin: 0 0 12px; font-size: 1rem; }
.settings-disclosure { border-bottom: 1px solid var(--line); margin: 0; min-width: 0; }
.settings-disclosure > summary { position: relative; cursor: pointer; padding: 16px 30px 16px 0; font-size: .9rem; line-height: 1.5; font-weight: 600; list-style: none; overflow-wrap: anywhere; }
.settings-disclosure > summary::-webkit-details-marker { display: none; }
.settings-disclosure > summary::after { content: '+'; position: absolute; right: 2px; top: 16px; color: var(--accent); }
.settings-disclosure[open] > summary::after { content: '−'; }
.settings-disclosure > :not(summary) { margin-bottom: 18px; }
.settings-disclosure .settings-disclosure { border-bottom-color: #e7edef; margin-left: 12px; }
.settings-disclosure .settings-disclosure > summary { font-size: .83rem; font-weight: 550; padding-block: 12px; }
.settings-disclosure .settings-disclosure > summary::after { top: 12px; }
.settings-value { min-width: 0; font-size: .875rem; }
.settings-value-item + .settings-value-item { border-top: 1px solid var(--line); padding-top: 12px; margin-top: 12px; }
.settings-data-row { display: grid; grid-template-columns: minmax(100px,.65fr) minmax(0,1.7fr); gap: 8px 20px; padding: 8px 0; }
.settings-data-label { font-size: .8rem; color: var(--muted); font-weight: 550; overflow-wrap: anywhere; }
.settings-prose { white-space: pre-wrap; overflow-wrap: anywhere; margin: 0; line-height: 1.7; max-width: 76ch; }
.settings-json { white-space: pre-wrap; overflow-wrap: anywhere; word-break: break-word; font-size: .75rem; max-height: 480px; overflow-y: auto; background: var(--wash); padding: 14px; border: 1px solid var(--line); border-radius: 6px; }
.settings-pipeline { list-style: none; counter-reset: pipeline; padding: 0; margin: 24px 0; }
.settings-step { counter-increment: pipeline; position: relative; margin: 0; padding: 0 0 28px 48px; }
.settings-step:not(:last-child)::after { content: ''; position: absolute; top: 34px; bottom: 4px; left: 15px; border-left: 1px solid #a7c4bd; }
.settings-step::before { content: counter(pipeline); position: absolute; left: 0; top: 0; width: 32px; height: 32px; display: grid; place-items: center; border: 1px solid #a7c4bd; border-radius: 50%; color: var(--accent); font-weight: 650; font-size: .875rem; background: var(--accent-wash); }
.settings-step h3 { margin: 0 0 8px; line-height: 32px; font-size: 1.06rem; }
.settings-step > p { max-width: 72ch; font-size: .9rem; line-height: 1.7; }
.settings-model { display: flex; align-items: baseline; flex-wrap: wrap; gap: 3px 12px; margin: 0 0 10px; font-size: .8rem; }
.settings-model > span { color: var(--muted); }
.settings-model > strong { color: var(--accent); font-weight: 600; overflow-wrap: anywhere; }
.settings-footnote { color: var(--muted); font-size: .8rem; max-width: 80ch; border-top: 1px solid var(--line); padding-top: 18px; }
.settings-source-area { margin: 28px 0; background: var(--paper); border: 1px solid var(--line); border-radius: 12px; padding: clamp(16px,3vw,28px); }
.settings-source-area h3 { margin: 0 0 8px; }
.settings-source-area .field { max-width: 560px; margin-bottom: 16px; }
.settings-source-area > button { margin-top: 18px; }
.settings-source-content { border-left: 2px solid #a7c4bd; padding-left: 16px; }
.settings-source-content .text-link,.settings-section > .text-link { display: inline-block; margin: 10px 0; }
.settings-links { display: flex; flex-wrap: wrap; gap: 10px 24px; padding: 12px 0; }
.settings-links .text-link { margin: 0; }
.settings-extra { padding: 4px 0 12px; }
.settings-extra > p { color: var(--muted); max-width: 72ch; font-size: .875rem; }
.settings-extra > .settings-value { padding-top: 18px; }
.settings-turn-layout { display: grid; grid-template-columns: 230px minmax(0,1fr); gap: 28px; align-items: start; }
.settings-turn-list { display: grid; gap: 5px; max-height: 580px; overflow-y: auto; padding-right: 5px; }
button.settings-turn-button { display: block; width: 100%; padding: 14px 12px; text-align: left; background: transparent; color: var(--ink); border: 1px solid transparent; border-bottom-color: var(--line); border-radius: 6px; font-size: .8rem; line-height: 1.5; }
button.settings-turn-button[aria-pressed=true] { background: var(--accent-wash); border-color: #a7c4bd; }
.settings-turn-button > span { display: block; color: var(--muted); font-size: .7rem; margin-top: 5px; font-weight: 400; }
.settings-turn-detail { min-width: 0; background: var(--paper); border: 1px solid var(--line); border-radius: 12px; padding: clamp(16px,2.4vw,26px); }
.settings-turn-detail:empty { display: none; }
.settings-turn-detail > h3 { margin: 0 0 16px; font-size: 1rem; }
.settings-message { margin: 0 0 16px; padding: 12px 16px; border-left: 3px solid #92b8ac; background: var(--accent-wash); white-space: pre-wrap; overflow-wrap: anywhere; font-size: .95rem; }
.settings-turn-outcome { font-size: .83rem; color: var(--muted); }
.settings-stage-content h4 { font-size: .85rem; margin: 18px 0 12px; }
.settings-stage-content > .settings-value > .settings-disclosure { margin-left: 0; }
.settings-empty { padding: 20px; background: var(--paper); border: 1px dashed var(--line); color: var(--muted); border-radius: 8px; font-size: .9rem; }
.settings-rules { padding-left: 22px; font-size: .875rem; }
.settings-rules li { margin-bottom: 9px; }
#settings-memory { scroll-margin-top: 90px; }
#settings-memory .field { margin-bottom: 18px; }
@media(max-width:740px){
 .settings-toolbar { align-items: flex-start; gap: 8px; }
 .settings-tabs { gap: 0 16px; }
 button.settings-tab { font-size: .8rem; padding-block: 12px; }
 .settings-toolbar > button { font-size: .75rem; padding: 12px 0; }
 .settings-turn-layout { grid-template-columns: minmax(0,1fr); gap: 16px; }
 .settings-turn-list { max-height: 200px; }
 .settings-data-row { grid-template-columns: minmax(0,1fr); gap: 2px; }
 .settings-step { padding-left: 42px; }
 .settings-step h3 { font-size: .97rem; }
 .settings-disclosure .settings-disclosure { margin-left: 6px; }
}
@media(max-width:380px){
 .settings-toolbar { display: block; }
 .settings-toolbar > button { padding-block: 6px; min-height: 36px; }
 .settings-tabs { gap: 0 12px; }
 .settings-context,.settings-source-area,.settings-turn-detail { padding: 14px; }
}
`;
