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
.site-header {
  background: var(--paper);
  border-color: var(--line);
  backdrop-filter: none;
}
.header-inner { max-width: 1160px; min-height: 72px; padding-inline: 32px; }
.site-logo { display: flex; align-items: center; gap: 10px; color: var(--ink); white-space: nowrap; }
.brand-sign {
  display: inline-grid;
  place-items: center;
  width: 34px;
  height: 34px;
  border-radius: 10px;
  background: var(--accent-wash);
  color: var(--accent);
  flex: none;
}
.brand-sign svg { width: 21px; height: 21px; }
.site-header a:focus-visible,.site-header button:focus-visible { outline-color: var(--focus); }
.site-header .site-navigation { background: var(--paper); border-color: var(--line); }
.nav-links a[aria-current="page"] { background: var(--accent-wash); color: var(--accent); font-weight: 700; }
.nav-links a, .nav-links #header-action button { color: var(--muted); }
.nav-links a:hover { color: var(--accent); }
@media (min-width: 780px) {
  .header-inner { gap: 16px; }
  .site-header .site-menu-toggle { display: none; }
  .site-header .site-navigation { display: block; position: static; padding: 0; border: 0; box-shadow: none; }
  .site-header .nav-links { flex-direction: row; align-items: center; gap: 2px; }
  .site-header .nav-links a,.site-header .nav-links #header-action button {
    display: flex;
    align-items: center;
    min-height: 40px;
    padding: 8px 10px;
    width: auto;
    border: 0;
    border-radius: 8px;
    white-space: nowrap;
    font-size: .875rem;
  }
}
@media (min-width: 780px) and (max-width: 940px) {
  .header-inner { padding-inline: 22px; gap: 10px; }
  .site-logo { font-size: 18px; gap: 7px; }
  .brand-sign { width: 29px; height: 29px; }
  .site-header .nav-links a,.site-header .nav-links #header-action button { padding-inline: 7px; font-size: .8125rem; }
}
@media (max-width: 779px) {
  .site-header .nav-links a,.site-header .nav-links #header-action button {
    min-height: 44px;
    padding: 10px 12px;
    border: 0;
    border-radius: 8px;
  }
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
  display: flex;
  gap: 5px;
  margin: 0 auto 16px;
  padding: 4px;
  border: 1px solid var(--line);
  border-radius: 12px;
  background: #eaf0f3;
}
.portrait-tabs a {
  min-height: 44px;
  padding: 10px 17px;
  border-radius: 8px;
  color: var(--muted);
  font-weight: 600;
}
.portrait-tabs a[aria-current="page"] {
  background: var(--paper);
  color: var(--accent);
  box-shadow: 0 1px 3px #172d3c17;
}
.portrait-tabs a:hover { color: var(--accent); }
.portrait-state {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 19px;
  color: var(--muted);
  font-size: .83rem;
}
.portrait-state::before { content: ""; width: 7px; height: 7px; flex: none; border-radius: 50%; background: var(--accent); }
.portrait-state:empty { display: none; }
.portrait-state.warn { display: flex; flex-wrap: wrap; color: var(--error); }
.portrait-state.warn::before { background: currentColor; }
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
.person-card { display: grid; grid-template-columns: auto minmax(0,1fr); align-items: start; gap: 12px; }
.person-initial {
  display: grid;
  place-items: center;
  width: 42px;
  height: 42px;
  border-radius: 10px;
  background: var(--accent-wash);
  color: var(--accent);
  font: 600 1.1rem Georgia,serif;
}
.source-meta,.event-people { display: flex; flex-wrap: wrap; align-items: center; gap: 5px 9px; margin: 6px 0 0; color: var(--muted); font-size: .82rem; }
.source-item .source-note { margin-top: 10px; max-width: 70ch; }
.source-item .source-actions { margin: 3px 0 -6px; gap: 16px; }
.source-actions button.text { padding: 0; text-decoration: none; font-size: .83rem; font-weight: 650; }
.source-actions button.text:hover { text-decoration: underline; }
.source-item .source-actions button.danger:not(:hover):not(:focus-visible) { color: var(--muted); }
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

.conversation-page { max-width: var(--app-width); }
.conversation-page .app-head { margin: 20px 0 12px; }
.conversation-page .app-head h1 { font-size: clamp(1.55rem,3vw,2.15rem); }
.conversation { border-radius: 16px; box-shadow: 0 2px 4px #172d3c07; }
.conversation-toolbar { background: var(--wash); }
.chat-message.chat-user { background: var(--accent-wash); }
.chat-composer textarea { border-color: var(--line); }
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
@media (max-width: 779px) {
  .shell,.header-inner { padding-inline: 19px; }
  .header-inner { min-height: 64px; }
  .site-logo { font-size: 1.13rem; }
  .brand-sign { width: 30px; height: 30px; }
}
@media (max-width: 600px) {
  .shell,.header-inner { padding-inline: 15px; }
  .app-head { margin-block: 28px 19px; }
  .page-description { font-size: .875rem; margin-top: 9px; }
  .portrait-tabs { display: grid; grid-template-columns: repeat(2,minmax(0,1fr)); }
  .portrait-tabs a { display: grid; place-items: center; padding: 7px 5px; }
  .people-directory,.account-panel { grid-template-columns: minmax(0,1fr); }
  .account-row:nth-child(2) { grid-row: auto; }
  .account-row .row { flex-wrap: wrap; gap: 5px; }
  .source-item { padding: 13px 12px; }
  .source-editor > summary { display: flex; width: 100%; justify-content: center; }
  .event-timeline { padding-left: 16px; }
  .event-timeline .timeline-year::before { left: -20px; }
  .workspace-empty .actions { align-items: stretch; }
  .conversation-page { height: calc(100dvh - 76px); }
}
@media (max-width: 380px) {
  .workspace-empty .actions > * { width: 100%; }
}
@media (max-height: 560px) {
  .conversation-page .app-head { margin-block: 8px; }
  .conversation { border-radius: 10px; }
}
`;
