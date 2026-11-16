import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';

const source = new URL('../data/change_strategies.json', import.meta.url);
const target = new URL('../journal/src/change-strategies.generated.mjs', import.meta.url);
const researchPath = 'docs/recherche-strategies-changement-2026-11-17.md';
const research = new URL('../' + researchPath, import.meta.url);
const families = new Map([
  ['C', ['TCC', 9]], ['D', ['TCD', 6]], ['A', ['ACT', 5]],
  ['SCH', ['schemes', 4]], ['K', ['compassion', 3]],
  ['M', ['entretien_motivationnel', 4]], ['G', ['objectifs_et_suivi', 5]],
]);
const expectedIds = new Map([...families].flatMap(([prefix, [family, count]]) =>
  Array.from({length: count}, (_, i) => [prefix + String(i + 1).padStart(2, '0'), family])));
const blockedContexts = ['safety', 'stabilize', 'repair', 'listen', 'no_advice', 'closed_topic'];
const criticalFields = ['id', 'restriction', 'permission', 'blocked_contexts', 'requires_safe_interpersonal_context'];
const sourceKinds = new Set(['clinical_guideline', 'public_service_material', 'narrative_review',
  'professional_guidance', 'institutional_summary', 'institutional_guidance', 'meta_analysis']);
const readingLevels = new Set(['integral', 'excerpt', 'summary', 'search_excerpt']);
// Version 1 records the access actually obtained, not merely what a source offers.
const reviewedReadingLevels = new Map([
  ['S01', ['search_excerpt']], ['S03', ['integral']], ['S10', ['integral']],
  ['S12', ['summary']], ['S15', ['summary', 'excerpt']], ['S18', ['search_excerpt']],
]);
const restrictions = new Set(['self_help_support', 'prepare_session_only', 'disabled']);
const statuses = new Set(['draft', 'reviewed', 'disabled']);
const fail = message => { throw new Error(`Invalid change strategies: ${message}`); };
const record = value => value !== null && typeof value === 'object' && !Array.isArray(value);
function text(value, label, max = 8000) {
  if (typeof value !== 'string' || !value.length || value.trim() !== value || value.length > max || value.includes('\0')) {
    fail(`${label} must be nonempty, trimmed text of at most ${max} characters`);
  }
}
function strings(value, label, min = 1) {
  if (!Array.isArray(value) || value.length < min) fail(`${label} must contain at least ${min} values`);
  for (const item of value) text(item, label);
  if (new Set(value).size !== value.length) fail(`${label} contains duplicate values`);
}
function https(value, label) {
  text(value, label, 2000);
  let url;
  try { url = new URL(value); } catch { fail(`${label} is not a URL`); }
  if (url.protocol !== 'https:' || url.username || url.password) fail(`${label} must be a public HTTPS URL`);
}
function hasEvery(value, required, label) {
  strings(value, label);
  for (const item of required) if (!value.includes(item)) fail(`${label} is missing ${item}`);
}

/** Documentary consistency and critical restrictions; not a clinical validation. */
export function validateChangeStrategies(data) {
  if (!record(data) || data.version !== 1) fail('version must be 1');
  const rules = data.common_rules;
  if (!record(rules)) fail('common_rules is required');
  for (const key of ['scope', 'review_status', 'adaptation_note', 'permissions_full_text',
    'exclusions_full_text', 'measurement_full_text', 'compact_rule', 'safety_rule',
    'interpersonal_rule', 'no_automatic_commitment', 'professional_rule', 'followup_rule',
    'research_ref', 'plan_ref', 'review_ref']) text(rules[key], `common_rules.${key}`);
  if (!record(rules.permission_levels)) fail('permission_levels is required');
  for (const level of ['P0', 'P1', 'P2', 'P3']) text(rules.permission_levels[level], `permission_levels.${level}`);
  hasEvery(rules.blocked_contexts, blockedContexts, 'common_rules.blocked_contexts');
  hasEvery(rules.compact_required_fields, criticalFields, 'common_rules.compact_required_fields');
  if (rules.restricted_strategies?.C09 !== 'prepare_session_only') fail('C09 must remain preparation-only in common rules');
  if (rules.selection_limits?.candidate_count !== 6 || rules.selection_limits?.detailed_count !== 2 ||
      rules.selection_limits?.developed_actions_per_response !== 1) fail('selection limits must be 6 candidates, 2 details and 1 developed action');
  if (rules.research_ref !== researchPath) fail('common research reference is not the reviewed dossier');

  if (!Array.isArray(data.sources) || data.sources.length !== 18) fail('the reviewed version requires 18 sources');
  const sources = new Map();
  for (const item of data.sources) {
    if (!record(item) || !/^S(?:0[1-9]|1[0-8])$/.test(item.id)) fail('invalid source id');
    if (sources.has(item.id)) fail(`duplicate source ${item.id}`);
    for (const key of ['title', 'publisher', 'read_passages', 'scope_and_limits']) text(item[key], `${item.id}.${key}`);
    https(item.url, `${item.id}.url`);
    if (item.additional_urls !== undefined) {
      strings(item.additional_urls, `${item.id}.additional_urls`);
      for (const url of item.additional_urls) https(url, `${item.id}.additional_urls`);
    }
    if (!sourceKinds.has(item.kind)) fail(`${item.id} has an unknown source kind`);
    strings(item.reading_level, `${item.id}.reading_level`);
    if (item.reading_level.some(level => !readingLevels.has(level))) fail(`${item.id} has an unknown reading level`);
    if (JSON.stringify(item.reading_level) !== JSON.stringify(reviewedReadingLevels.get(item.id) || ['excerpt'])) {
      fail(`${item.id} reading level differs from the access documented for version 1`);
    }
    const date = Date.parse(item.consulted_at);
    if (typeof item.consulted_at !== 'string' || !/^\d{4}-\d{2}-\d{2}$/.test(item.consulted_at) ||
        Number.isNaN(date) || new Date(date).toISOString().slice(0, 10) !== item.consulted_at) fail(`${item.id} has an invalid consultation date`);
    sources.set(item.id, item);
  }
  if (!Array.isArray(data.strategies) || data.strategies.length !== expectedIds.size) fail('the reviewed version requires 36 strategies');
  const strategies = new Map();
  for (const item of data.strategies) {
    if (!record(item) || !expectedIds.has(item.id)) fail('unknown strategy id');
    if (strategies.has(item.id)) fail(`duplicate strategy ${item.id}`);
    if (item.family !== expectedIds.get(item.id)) fail(`${item.id} has an incorrect family`);
    if (!statuses.has(item.status)) fail(`${item.id} has an unknown documentary status`);
    for (const key of ['label', 'permission', 'small_step', 'measure']) text(item[key], `${item.id}.${key}`, 1200);
    if (!/\bP[0-3]\b/.test(item.permission)) fail(`${item.id} does not specify a permission level`);
    for (const key of ['intent_tags', 'indications', 'exclusions', 'source_refs']) strings(item[key], `${item.id}.${key}`);
    if (item.intent_tags.some(tag => !/^[a-z][a-z_]*$/.test(tag))) fail(`${item.id} has an invalid intent tag`);
    if (!restrictions.has(item.restriction)) fail(`${item.id} has an unknown restriction`);
    if (item.status === 'disabled' && item.restriction !== 'disabled') fail(`${item.id} is disabled but its restriction is not`);
    if (item.id === 'C09' && !['prepare_session_only', 'disabled'].includes(item.restriction)) fail('C09 cannot become an autonomous exposure exercise');
    hasEvery(item.blocked_contexts, rules.blocked_contexts, `${item.id}.blocked_contexts`);
    if (typeof item.requires_safe_interpersonal_context !== 'boolean') fail(`${item.id} needs a boolean interpersonal precondition`);
    if (['C07', 'C08', 'D06'].includes(item.id) && !item.requires_safe_interpersonal_context) fail(`${item.id} requires a safe interpersonal context`);
    if (item.source_refs.some(id => !sources.has(id))) fail(`${item.id} references an unknown source`);
    if (item.research_ref !== `${researchPath}#${item.id.toLowerCase()}`) fail(`${item.id} does not reference its research entry`);
    strategies.set(item.id, item);
  }
  return data;
}

export function generateChangeStrategies(data) {
  validateChangeStrategies(data);
  return '// Generated by scripts/build_change_strategies.mjs; edit data/change_strategies.json.\n' +
    '// reviewed denotes documentary review, not clinical validation. Do not truncate restrictions.\n' +
    'export const CHANGE_STRATEGIES = ' + JSON.stringify(data, null, 2) + ';\n';
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  if (process.argv.slice(2).some(arg => arg !== '--check')) throw new Error('Usage: node scripts/build_change_strategies.mjs [--check]');
  const data = JSON.parse(fs.readFileSync(source, 'utf8'));
  const output = generateChangeStrategies(data);
  const dossier = fs.readFileSync(research, 'utf8');
  for (const item of data.strategies) {
    if (!dossier.includes(`<a id="${item.id.toLowerCase()}"></a>`)) fail(`${item.id} research anchor is missing`);
  }
  if (process.argv.includes('--check')) {
    if (!fs.existsSync(target) || fs.readFileSync(target, 'utf8') !== output) {
      throw new Error('Change strategies are out of date; run node scripts/build_change_strategies.mjs');
    }
  } else fs.writeFileSync(target, output);
}
