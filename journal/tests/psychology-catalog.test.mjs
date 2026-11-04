import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {PSYCHOLOGY_CATALOG} from '../src/psychology-catalog.generated.mjs';
import {generatePsychologyCatalog,validatePsychologyCatalog} from '../../scripts/build_psychology_catalog.mjs';
const source=JSON.parse(fs.readFileSync(new URL('../../data/psychology_catalog.json',import.meta.url),'utf8'));
const legacyDomains={
 thinking:['rumination','intellectualization','uncertainty','catastrophizing','mind_reading','all_or_nothing','confirmation','reframing'],
 emotions:['emotional_identification','emotional_suppression','emotional_overload','anger_needs','guilt','shame','body_stress'],
 self:['self_criticism','conditional_worth','impostor','comparison','self_compassion','identity_transition'],
 relationships:['overresponsibility','boundaries','reassurance','conflict','reciprocity','trust'],
 family:['parent_expectations','family_roles','parentification','family_loyalty','emotional_learning','family_boundaries'],
 attachment:['abandonment_fear','closeness_distance','relational_security'],
 action:['perfectionism','avoidance','overcontrol','withdrawal','impulsive_reaction'],
 context:['overwork','grief_change','stress_resources','social_belonging'],
 needs:['autonomy','values_alignment','needs_expression'],
 resources:['support_seeking','emotional_flexibility','resilience','positive_connection','self_awareness']
};
const changed=mutate=>{const data=structuredClone(source);mutate(data);return data;};

test('expanded source and Worker catalog have exactly the same generated content',()=>{
 assert.deepEqual(PSYCHOLOGY_CATALOG,source);
 assert.equal(source.version,2);
 assert.equal(source.patterns.length,161);
 assert.equal(source.domains.length,16);
 assert.equal(source.subdomains.length,44);
 assert.equal(source.sources.length,25);
 assert.equal(generatePsychologyCatalog(source),fs.readFileSync(new URL('../src/psychology-catalog.generated.mjs',import.meta.url),'utf8'));
});

test('all 53 legacy pattern identifiers retain their domain and editorial evidence fields',()=>{
 let count=0;
 for(const [domain,ids] of Object.entries(legacyDomains))for(const id of ids){
  const pattern=source.patterns.find(item=>item.id===id);
  assert.ok(pattern,`legacy ${id} is preserved`);
  assert.equal(pattern.domain,domain);
  assert.ok(pattern.signals.length>=2&&pattern.alternatives.length>=2&&pattern.questions.length>=2);
  assert.ok(pattern.source_refs.length&&pattern.related.length);
  count++;
 }
 assert.equal(count,53);
});

test('every subdomain is used under the declared domain and sources are traceable primary families',()=>{
 const subdomains=new Map(source.subdomains.map(item=>[item.id,item]));
 const refs=new Map(source.sources.map(item=>[item.id,item]));
 const allowedHosts=new Set(['www.nhs.uk','www.cci.health.wa.gov.au','www.nimh.nih.gov','www.nih.gov','www.who.int','www.nice.org.uk']);
 assert.equal(validatePsychologyCatalog(source),source);
 for(const pattern of source.patterns){
  assert.equal(subdomains.get(pattern.subdomain).domain,pattern.domain);
  assert.ok(['process','resource','context'].includes(pattern.kind));
  for(const id of pattern.source_refs){
   assert.ok(refs.has(id));
   const ref=refs.get(id);
   assert.ok(allowedHosts.has(new URL(ref.url).hostname));
   assert.ok(ref.publisher&&ref.scope&&ref.consulted_at);
  }
 }
 for(const subdomain of source.subdomains)assert.ok(source.patterns.some(item=>item.subdomain===subdomain.id));
 assert.equal(source.provenance.mapping,'editorial_family');
 assert.match(source.provenance.note,/ne signifie pas/);
});

test('common labels remain processes and explicitly declared health remains contextual',()=>{
 const patterns=new Map(source.patterns.map(item=>[item.id,item]));
 for(const id of ['impostor','overresponsibility'])assert.equal(patterns.get(id).kind,'process');
 for(const id of ['health_declared','illness_adjustment'])assert.equal(patterns.get(id).kind,'context');
 assert.match(patterns.get('impostor').description,/pas un diagnostic/);
 assert.match(patterns.get('overresponsibility').description,/pas un syndrome clinique/);
 assert.match(patterns.get('health_declared').description,/sans les confirmer ni les déduire/);
 assert.match(source.scope,/non exhaustif/);
 assert.match(source.provenance.health_context,/aucune déduction de diagnostic/);
});

for(const [collection,matcher] of [['domains',/domains has duplicate id/],['subdomains',/subdomains has duplicate id/],['sources',/sources has duplicate id/],['patterns',/patterns has duplicate id/]]){
 test(`generator rejects duplicate ${collection} identifiers`,()=>{
  assert.throws(()=>validatePsychologyCatalog(changed(data=>data[collection].push(structuredClone(data[collection][0])))),matcher);
 });
}

test('generator refuses unknown domains, cross-domain subdomains and invalid relation or source IDs',()=>{
 for(const [mutate,matcher] of [
  [data=>{data.patterns[0].domain='missing_domain';},/unknown domain/],
  [data=>{data.subdomains[0].domain='missing_domain';},/unknown domain/],
  [data=>{data.patterns[0].subdomain='emotions_awareness';},/invalid subdomain/],
  [data=>{data.patterns[0].related=['missing_pattern'];},/invalid related references/],
  [data=>{data.patterns[0].related=[data.patterns[0].id];},/invalid related references/],
  [data=>{data.patterns[0].source_refs=['missing_source'];},/unknown source references/],
  [data=>{data.patterns[0].kind='diagnosis';},/invalid kind/],
  [data=>{data.patterns[0].questions=[];},/questions needs at least 2/]
 ])assert.throws(()=>validatePsychologyCatalog(changed(mutate)),matcher);
});

test('generator rejects empty editorial records and broken provenance',()=>{
 assert.throws(()=>validatePsychologyCatalog(changed(data=>{data.patterns[0].description=' ';})),/description must be/);
 assert.throws(()=>validatePsychologyCatalog(changed(data=>{data.sources[0].scope='';})),/scope must be/);
 assert.throws(()=>validatePsychologyCatalog(changed(data=>{data.provenance.mapping='clinically_validated';})),/editorial_family/);
 assert.throws(()=>validatePsychologyCatalog(changed(data=>{data.sources[0].url='http://example.com';})),/must use https/);
});
