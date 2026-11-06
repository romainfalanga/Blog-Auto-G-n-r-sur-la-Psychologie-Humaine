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

.portrait-tabs {
  display: grid;
  grid-template-columns: repeat(3,minmax(0,1fr));
  gap: 5px;
  margin: 0 auto 16px;
  padding: 4px;
  border: 1px solid var(--line);
  border-radius: 12px;
  background: #eaf0f3;
}
.portrait-tabs a {
  display: grid;
  place-items: center;
  min-height: 44px;
  padding: 8px;
  border-radius: 8px;
  color: var(--muted);
  font-size: .875rem;
  font-weight: 600;
  text-align: center;
  text-decoration: none;
}
.portrait-tabs a[aria-current="page"] {
  background: var(--paper);
  color: var(--accent);
  box-shadow: 0 1px 3px #172d3c17;
}
.portrait-tabs a:hover { color: var(--accent); }
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
#self-profile-form .profile-group {
  padding: 0 0 21px;
  margin: 0 0 21px;
  border-bottom: 1px solid var(--line);
}
#self-profile-form .profile-group summary { color: var(--ink); font-weight: 700; padding: 8px 0; }
#self-profile-form .profile-group textarea { min-height: 116px; }
#self-profile-form .self-presentation textarea { min-height: 136px; }
.profile-save { padding-top: 3px; }
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
.source-count,.source-tag {
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
  .portrait-tabs { grid-template-columns: repeat(2,minmax(0,1fr)); }
  .portrait-tabs [data-portrait-tab="synthesis"] { grid-column: 1/-1; }
  .portrait-tabs a { padding: 7px 5px; font-size: .8125rem; }
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
`;
