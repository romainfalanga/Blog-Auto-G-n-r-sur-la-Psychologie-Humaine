import {snapshotPage} from './review-snapshot.mjs';
import {serializeSharedPlan} from './change-plan.mjs';
// The current sharing choice governs every historic read. Unknown future sections
// are denied by default; the full original snapshot is never returned to a pro.
const SCOPES={profile:'portrait',portrait:'portrait',childhood:'cartography',symptoms:'symptoms',people:'people',people_notes:'people',observations:'people',events:'life_events',event_notes:'life_events',event_people:'life_events',conversation:'conversation',previous_reviews:'reviews',plan:'plan',memory:'memory',personal_memory:'memory',professional_assessments:'diagnoses'};
export function sharedSnapshotPage(row,scopes,options={}){
 const allowed=new Set(scopes),sections={};
 for(const [key,value] of Object.entries(row.payload.sections||{})){
  if(!allowed.has(SCOPES[key]))continue;
  if(key==='event_people'&&!allowed.has('people'))continue;
  if(key==='plan')sections[key]=value.map(r=>serializeSharedPlan({document:{goals:[],steps:[],trials:[],...r.document_json},revision:r.revision,updated_at:r.updated_at}));
  else sections[key]=value;
 }
 const result=snapshotPage({...row,payload:{...row.payload,sections},assessment:allowed.has('diagnoses')?row.assessment:null},options);
 return {...result,complete:false,scope_limited:true,sharing_scopes:[...allowed],archive_complete:true};
}
