const test=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const os=require('node:os');
const path=require('node:path');
const {spawnSync}=require('node:child_process');
const root=path.resolve(__dirname,'..');
const candidates=[process.env.HUGO_BIN,path.join(root,'work/bin/hugo'),'hugo'].filter(Boolean);
const hugo=candidates.find(bin=>spawnSync(bin,['version'],{encoding:'utf8'}).status===0);
let fixtureRoot;
function render(){
 fixtureRoot=fs.mkdtempSync(path.join(os.tmpdir(),'public-references-'));
 for(const folder of ['content','layouts/_default','layouts/partials'])fs.mkdirSync(path.join(fixtureRoot,folder),{recursive:true});
 fs.writeFileSync(path.join(fixtureRoot,'hugo.toml'),'baseURL="https://example.test/"\n[markup.goldmark.renderer]\nunsafe=true\n');
 fs.copyFileSync(path.join(root,'layouts/partials/public-content.html'),path.join(fixtureRoot,'layouts/partials/public-content.html'));
 fs.writeFileSync(path.join(fixtureRoot,'layouts/_default/single.html'),'{{ partial "public-content.html" .Content }}');
 fs.writeFileSync(path.join(fixtureRoot,'layouts/_default/list.html'),'{{ partial "public-content.html" .Content }}');
 fs.writeFileSync(path.join(fixtureRoot,'content/article.md'),`---
title: Article de test
---
Le texte de **Dupont** décrit un exemple. Les [travaux de **Dupont**](https://pubmed.ncbi.nlm.nih.gov/12345/) sont nuancés. En [1973](https://doi.org/10.1000/example), un résultat a été discuté. La parole de Mila reste une citation : « je préfère attendre ».

Un résultat[1] est à nuancer (Dupont, 2020). Une citation liée ([Dupont, 2020](https://doi.org/10.1000/example)) ne sert pas de navigation. [2](#ref-2)

Le rendez-vous reste prévu (Paris, 2020). Elle lit <cite>Les années</cite> et conserve 2<sup>3</sup> sur son carnet.

<code>[1] (Dupont, 2020) Source: préserver le code</code>

Un détail[^reference] reste dans la phrase.

## Les références à vérifier

Cette partie du récit et ses références professionnelles restent intactes.

## Aide et navigation

[Contacter le 3114](https://3114.fr/) ; [aide en cas de violence](https://arretonslesviolences.gouv.fr/besoin-d-aide) ; [droits et soutien](https://www.service-public.gouv.fr/particuliers/vosdroits/F12544) ; [urgence médicale](https://www.ameli.fr/assure/sante/themes/douleur-thoracique/reconnaitre-agit-urgence).

[Feuille pratique du NHS](https://www.nhs.uk/every-mind-matters/mental-wellbeing-tips/self-help-cbt-techniques/thought-record/) ; [guide d'exercices OMS](https://www.who.int/publications/i/item/9789240003927) ; [Mila](/personnages/mila/) ; [3](#etape-3).

<p><strong>Source :</strong> Dupont, 2020, une référence bibliographique isolée.</p>

## Références bibliographiques

Dupont, 2020. Une étude. [Texte](https://doi.org/10.1000/example)

### Une sous-section des références

Cette bibliographie doit être retirée.

## Continuer la lecture

Cette conclusion doit rester, avec [le prochain article](/posts/suivant/).

[^reference]: Une référence en note de bas de page.
`);
 fs.writeFileSync(path.join(fixtureRoot,'content/character.md'),`---
title: Personnage de test
---
<div class="parcours-step"><h2>Un parcours intact</h2><p>Une [expression] ordinaire.</p><a class="parcours-link" href="/posts/episode/">Lire</a></div>
<p>Les <a href='https://www.nature.com/articles/example'><em>travaux de Martin</em></a> apportent une nuance.</p>
<section role="doc-bibliography"><h2>Bibliographie</h2><ol><li>Martin, 2021.</li></ol></section>
<p>Après cette section, le récit continue. <cite>(Martin et al., 2021)</cite></p>
`);
 const built=spawnSync(hugo,['--source',fixtureRoot],{encoding:'utf8',timeout:60000});assert.equal(built.status,0,built.stderr||built.stdout);
 return {article:fs.readFileSync(path.join(fixtureRoot,'public/article/index.html'),'utf8'),character:fs.readFileSync(path.join(fixtureRoot,'public/character/index.html'),'utf8')};
}

test('all public Content entry points use the same display-only reference filter',()=>{
 for(const file of ['layouts/_default/single.html','layouts/_default/list.html','layouts/personnages/single.html','layouts/partials/character-directory.html'])assert.match(fs.readFileSync(path.join(root,file),'utf8'),/partial "public-content\.html" \.Content/,file);
});

test('real Hugo rendering removes bibliography and citation calls while retaining prose, help and navigation',{skip:!hugo&&'Set HUGO_BIN to run the Hugo rendering regression'},t=>{
 t.after(()=>{if(fixtureRoot)fs.rmSync(fixtureRoot,{recursive:true,force:true})});const {article,character}=render();
 for(const html of [article,character])assert.doesNotMatch(html,/href=["']https?:\/\/(?:pubmed\.ncbi\.nlm\.nih\.gov|doi\.org|www\.nature\.com)/);
 assert.match(article,/Les travaux de <strong>Dupont<\/strong> sont nuancés/);assert.match(article,/En 1973, un résultat/);
 assert.match(article,/« je préfère attendre »|&ldquo; je préfère attendre &rdquo;/);assert.match(article,/Le rendez-vous reste prévu \(Paris, 2020\)/);assert.match(article,/<cite>Les années<\/cite>/);assert.match(article,/2<sup>3<\/sup>/);
 assert.match(article,/<code>\[1\] \(Dupont, 2020\) Source: préserver le code<\/code>/);
 const withoutCode=article.replace(/<code>[\s\S]*?<\/code>/g,'');assert.doesNotMatch(withoutCode,/\[1\]|\(Dupont, 2020\)|href="#ref-2"|footnote|bibliographie doit être retirée|Une étude\.|référence bibliographique isolée/);
 assert.match(article,/Les références à vérifier/);assert.match(article,/Cette conclusion doit rester/);
 for(const url of ['https://3114.fr/','https://arretonslesviolences.gouv.fr/besoin-d-aide','https://www.service-public.gouv.fr/particuliers/vosdroits/F12544','https://www.ameli.fr/assure/sante/themes/douleur-thoracique/reconnaitre-agit-urgence','https://www.nhs.uk/every-mind-matters/mental-wellbeing-tips/self-help-cbt-techniques/thought-record/','https://www.who.int/publications/i/item/9789240003927','/personnages/mila/','#etape-3','/posts/suivant/'])assert.ok(article.includes('href="'+url+'"'),url);
 assert.match(character,/<a class="parcours-link" href="\/posts\/episode\/">Lire<\/a>/);assert.match(character,/<em>travaux de Martin<\/em>/);assert.match(character,/Après cette section, le récit continue/);assert.doesNotMatch(character,/doc-bibliography|Martin et al\., 2021/);
});
