(function (root) {
  'use strict';
  function normalize(text) {
    return String(text || '').toLowerCase().replace(/œ/g, 'oe').replace(/æ/g, 'ae').normalize('NFD').replace(/[\u0300-\u036f]/g, '').replace(/[^a-z0-9]+/g, ' ').trim();
  }
  var stop = new Set(('a au aux avec ce ces cet cette chez comme d de des du dans elle elles en est et il ils j je l la le les leur leurs lui ma mais me mes moi mon ne nos notre nous on ou par pas plus pour qu que quel quelle qui s sa sans se ses soi son sur t ta te tes toi ton tout tres tu un une vos votre vous y ai suis avoir etre arrive sens sentiment ressentir ressent souvent toujours').split(' '));
  var groups = [
    ['rumination','ruminer','rumine','suranalyse','surpensee','overthinking','ruminations'],
    ['perfectionnisme','perfectionniste','perfectionnistes','irreprochable'],
    ['imposteur','imposture','imposteurs','illegitime','illegitimite'],
    ['epuisement','epuise','epuisee','burnout','burn out'],
    ['anxiete','anxieux','anxieuse','angoisse','angoisses'],
    ['solitude','seul','seule','seuls','seules','isolement','isole'],
    ['culpabilite','culpabilise','culpabiliser','coupable'],
    ['separation','rupture','divorce','separe','separee','divorcee'],
    ['deuil','veuf','veuve','veuvage'],
    ['timidite','timide','timides'],
    ['creativite','creatif','creative','creatifs'],
    ['travail','professionnel','professionnelle','boulot','emploi'],
    ['parents','parent','parental','parentale','parentalite'],
    ['enfants','enfant','fils','fille'],
    ['abandon','abandonne','abandonnee'],
    ['reconnaissance','validation','approbation'],
    ['autocritique','auto critique','autocritiques'],
    ['controle','controler','maitrise','maitriser'],
    ['limites','limite'],['decevoir','deception','decois'],
    ['confiance','confiante','confiant'],['empathie','empathique'],
    ['hypersensible','hypersensibilite','sensible','sensibilite'],
    ['impulsivite','impulsif','impulsive','impulsives'],
    ['procrastination','procrastiner','procrastine','reporter'],
    ['sauveur','sauver','sauvetage'],['dependance','dependant','dependante']
  ];
  var aliases = new Map();
  groups.forEach(function (group) { group.forEach(function (word) { aliases.set(word, group); }); });
  // Apply the same phrase normalization to queries and the indexed vocabulary.
  function phrases(text) {
    return normalize(text)
      .replace(/\b(?:n )?arrete (?:pas|plus) de (?:penser|reflechir|ruminer)\b/g, 'rumination')
      .replace(/\b(?:pense|penser|reflechis|reflechir) (?:beaucoup )?trop\b/g, 'rumination')
      .replace(/\bpensees? (?:qui )?(?:tournent |tourne )?en boucle\b/g, 'rumination')
      .replace(/\bburn out\b/g, 'burnout')
      .replace(/\bauto critique\b/g, 'autocritique')
      .replace(/\blacher prise\b/g, 'lacherprise');
  }
  function tokens(text) { return Array.from(new Set(phrases(text).split(' ').filter(Boolean))); }
  function concepts(query) {
    return tokens(query).filter(function (word) { return !stop.has(word) && !['n','m','c','recherche','cherche','personnage','theme','themes'].includes(word); });
  }
  function close(a, b) {
    if (a.length < 5 || b.length < 5 || Math.abs(a.length - b.length) > 1) return false;
    var prev = Array.from({ length: b.length + 1 }, function (_, i) { return i; });
    for (var i = 1; i <= a.length; i++) {
      var row = [i];
      for (var j = 1; j <= b.length; j++) row[j] = Math.min(row[j-1]+1, prev[j]+1, prev[j-1]+(a[i-1]===b[j-1]?0:1));
      prev = row;
    }
    return prev[b.length] <= 1;
  }
  function match(term, words, fuzzy) {
    var variants = aliases.get(term) || [term];
    if (fuzzy && !aliases.has(term)) {
      var nearby = groups.find(function(group) { return group.some(function(word) { return close(term, word); }); });
      if (nearby) variants = [term].concat(nearby);
    }
    var best = 0;
    variants.forEach(function (variant) {
      if (variant.includes(' ')) return;
      if (words.includes(variant)) best = Math.max(best, variant === term ? 1 : .85);
      else if (variant.length >= 3 && words.some(function (w) { return w.startsWith(variant); })) best = Math.max(best, .7);
      else if (fuzzy && words.some(function (w) { return close(variant, w); })) best = Math.max(best, .45);
    });
    return best;
  }
  function prepare(entries) {
    return entries.map(function (entry, order) {
      return Object.assign({}, entry, { order: order, fields: [tokens(entry.name), tokens(entry.profile), tokens((entry.keywords || []).join(' ')), tokens(entry.stories)] });
    });
  }
  function search(entries, query) {
    var terms = concepts(query);
    if (!terms.length) return entries.map(function (entry) { return { entry: entry, score: 0, terms: [] }; });
    return entries.map(function (entry) {
      var matched = [], score = 0;
      terms.forEach(function (term) {
        var value = Math.max.apply(null, entry.fields.map(function (field, i) { return match(term, field, false) * [18, 8, 10, 2][i]; }));
        if (!value) value = Math.max.apply(null, entry.fields.map(function (field, i) { return match(term, field, true) * [18, 8, 10, 2][i]; }));
        if (value) { matched.push(term); score += value; }
      });
      if (normalize(entry.name) === normalize(query)) score += 100;
      return { entry: entry, score: score, terms: matched };
    }).filter(function (r) { return r.terms.length === terms.length; }).sort(function (a, b) { return b.score-a.score || a.entry.order-b.entry.order; });
  }
  var engine = { normalize: normalize, concepts: concepts, prepare: prepare, search: search };
  if (typeof module !== 'undefined' && module.exports) module.exports = engine;
  if (!root.document) return;
  var input = document.getElementById('character-search'), index = document.getElementById('character-search-index');
  if (!input || !index) return;
  var entries = prepare(JSON.parse(index.textContent)), grid = document.getElementById('character-results');
  var cards = new Map(Array.from(grid.querySelectorAll('[data-character-id]')).map(function (card) { return [card.dataset.characterId, card]; }));
  var clear = document.getElementById('character-search-clear');
  function update() {
    var query=input.value.trim(), results=search(entries,query), shown=new Set(results.map(function(r){return r.entry.id;}));
    cards.forEach(function(card,id){card.hidden=!shown.has(id);});
    results.forEach(function(r){var card=cards.get(r.entry.id);if(!card)return;grid.append(card);});
    document.getElementById('character-count').textContent=results.length+' personnage'+(results.length>1?'s':'')+(query?' pour cette recherche':' à explorer');
    document.getElementById('character-search-empty').hidden=!!results.length;
    clear.hidden=!query;
  }
  input.addEventListener('input',update);
  input.addEventListener('keydown',function(e){if(e.key==='Escape'||e.key==='Enter')e.preventDefault();});
  clear.addEventListener('click',function(){input.value='';update();input.focus();});
  update();
})(typeof window !== 'undefined' ? window : {});
