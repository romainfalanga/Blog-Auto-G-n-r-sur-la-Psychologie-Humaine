const {test}=require('node:test');
const assert=require('node:assert/strict');
const engine=require('../static/js/character-search.js');
const entries=engine.prepare([
 {id:'a',name:'Sophie',profile:'Développeuse, perfectionniste, elle rumine au travail.',keywords:['Rumination','Perfectionnisme'],stories:'Elle hésite après une critique en réunion.'},
 {id:'b',name:'Léa',profile:'Étudiante, peur de l’abandon dans ses relations.',keywords:['Peur de l’abandon'],stories:'Elle rencontre Sophie.'},
 {id:'c',name:'Nadia',profile:'Épuisement professionnel et difficulté à déléguer.',keywords:['Perfectionnisme','Besoin de contrôle'],stories:'Une décision pour sa famille.'}
]);
const ids=q=>engine.search(entries,q).map(r=>r.entry.id);
test('un prénom exact passe avant les mentions dans les histoires',()=>assert.equal(ids('Sophie')[0],'a'));
test('accents, apostrophes et casse sont normalisés',()=>assert.deepEqual(ids('LÉA'),['b']));
test('les mots se combinent même à distance',()=>assert.deepEqual(ids('travail réunion'),['a']));
test('les variantes et formulations proches retrouvent les repères',()=>{assert.deepEqual(ids('pensées en boucle'),['a']);assert.deepEqual(ids('je rumine au travail'),['a']);assert.deepEqual(ids('burn-out'),['c']);});
test('une faute légère conserve les synonymes du thème',()=>assert.deepEqual(new Set(ids('perfectioniste')),new Set(['a','c'])));
test('tous les critères sont requis et un mot absent ne donne aucun résultat',()=>{assert.deepEqual(ids('rumination abandon'),[]);assert.deepEqual(ids('zzzzzzz'),[]);});
test('vider la recherche restaure les personnages dans leur ordre initial',()=>assert.deepEqual(ids(''),['a','b','c']));
test('profession, repères et contexte des histoires sont recherchables',()=>{assert.deepEqual(ids('développeuse'),['a']);assert.deepEqual(ids('contrôle'),['c']);assert.deepEqual(ids('famille'),['c']);});
test('les formulations courantes de rumination conservent leur sens',()=>{
 for(const query of ['je pense trop','je réfléchis trop','je n’arrête pas de penser','je n’arrête pas de ruminer','pensées qui tournent en boucle']) assert.deepEqual(ids(query),['a'],query);
});
test('les mots composés sont normalisés aussi dans les données',()=>{
 const items=engine.prepare([{id:'a',name:'Sophie',keywords:['auto-critique interne','lâcher-prise','burn-out']}]);
 for(const query of ['auto critique interne','lâcher prise','burnout']) assert.equal(engine.search(items,query)[0]?.entry.id,'a',query);
});
