
// Local analytical actions, deterministic facts, and offline file exports.
function download(name,text,type="text/csv;charset=utf-8"){
  const url=URL.createObjectURL(new Blob([text],{type}));
  const a=document.createElement("a");a.href=url;a.download=name;document.body.appendChild(a);a.click();a.remove();
  setTimeout(()=>URL.revokeObjectURL(url),1000);
}
function button(label,action,parent){
  const b=document.createElement("button");b.className="btn sec";b.textContent=label;b.onclick=action;parent.appendChild(b);return b;
}
function updateVisible(){
  const count=mode==="ego" && egoCy ? egoCy.nodes().filter(n=>byId.has(n.id())).length : cy.nodes(":visible").length;
  $("#visibleCount").textContent="Показано "+count+" из "+DATA.nodes.length+" узлов"+(mode==="ego"?" (окружение)":"");
}
Object.entries(DATA.csv).forEach(([name,text])=>button(name,()=>{
 if(location.protocol.startsWith("http")){const a=document.createElement("a");a.href="/download/"+name;a.download=name;document.body.appendChild(a);a.click();a.remove();}
 else download(name,text);
},$("#downloads")));
$("#runInfo").textContent="Период: "+(S.date_min||"—")+" — "+(S.date_max||"—")+
  ". Порог исходной выгрузки: 5 000 ₸. Только наблюдаемая сеть. Компонент: "+S.components_all+
  " / без изолятов: "+S.components_without_isolates+". Run: "+M.manifest.run_id;
updateVisible();
function nodeSummary(n){
  return "# Узел "+n.id+"\n\nRun: "+M.manifest.run_id+"\n\nРоль: "+RU[n.r]+
    "; соответствие: "+n.rs+"; приоритет: "+n.p+"\n\n"+n.ev+"\n\n"+n.why+
    "\n\nВход: "+n.ink+" KZT / "+n.intx+" операций; выход: "+n.outk+" KZT / "+n.outtx+
    " операций.\n\nПриоритет: "+n.raw+" × "+n.sf+" / "+n.norm+"; изолят: "+(n.ind+n.outd===0)+
    "\n\nВремя: "+n.fast_status+"; сопоставлено "+n.matched+" из "+n.eligible+", покрытие "+n.coverage+
    "\n\nИсходящие depth=4 неизвестны. Роли — эвристики; пути не доказывают движение конкретных денег.\n";
}
function addNodeActions(n){
  const box=$("#nodeActions");
  button("Добавить к сравнению",()=>{
    const ids=new Set($("#queryGids").value.split(/[\s,;]+/).filter(Boolean));ids.add(n.id);
    $("#queryGids").value=[...ids].join(", ");showTab("analysis");
  },box);
  button("Пути от seed",()=>runQuery("trace_paths_from_seeds",{gid:n.id,max_hops:4,limit:10}),box);
  button("Полная справка",()=>runQuery("get_node_report",{gid:n.id}),box);
  button("Скачать справку",()=>{
 if(location.protocol.startsWith("http")){const a=document.createElement("a");a.href="/api/node-report/"+n.id+".md";a.download="node-"+n.id+".md";document.body.appendChild(a);a.click();a.remove();}
 else download("node-"+n.id+".md",nodeSummary(n),"text/markdown;charset=utf-8");
},box);
}
$("#t-analysis").innerHTML='<h3>Локальная аналитика</h3><p class="note">Запросы по полному набору, независимо от фильтров графа. Работают через python serve.py без ключа LLM.</p>'+
 '<label for="queryGids">Выбранные gid (2–20 для общих получателей)</label><textarea id="queryGids" style="width:100%;min-height:70px" placeholder="Полные gid через запятую"></textarea>'+
 '<select id="queryMode"><option value="direct">Прямые получатели всех выбранных</option><option value="reachable">Общая достижимость ≤4 рёбер</option></select>'+
 '<div class="btns"><button class="btn" id="queryCommon">Найти общих получателей</button><button class="btn sec" id="queryClear">Очистить выбор</button></div><div id="queryResult" aria-live="polite"></div>';
$("#queryClear").onclick=()=>{$("#queryGids").value="";};
$("#queryCommon").onclick=()=>runQuery("find_common_recipients",{
  gids:$("#queryGids").value.split(/[\s,;]+/).filter(Boolean),
  mode:$("#queryMode").value,max_hops:$("#queryMode").value==="direct"?1:4
});
function highlightPaths(paths){
  setMode("net");clearHL();
  const ids=new Set(paths.flatMap(p=>p.gids));
  const pairs=new Set(paths.flatMap(p=>p.edges.map(e=>e.src+"|"+e.dst)));
  cy.batch(()=>{
    cy.elements().addClass("faded");
    cy.nodes().forEach(el=>{if(ids.has(el.id()))el.removeClass("hide faded").addClass("nb");});
    cy.edges().forEach(el=>{if(pairs.has(el.source().id()+"|"+el.target().id()))el.removeClass("faded hide").addClass("path");});
  });
  if(ids.size)cy.fit(cy.nodes().filter(el=>ids.has(el.id())),50);
  updateVisible();
}
function factHTML(f){
  const r=f.value;
  const link=g=>byId.has(g)?'<a class="lnk" data-g="'+esc(g)+'">'+esc(g)+'</a>':esc(g);
  if(f.kind==="node")return '<h3>Узел '+link(r.gid)+'</h3><p>'+esc(r.evidence)+'</p><p>'+esc(r.why)+'</p>'+
    '<p>Роль: '+esc(r.role)+'; приоритет: '+r.priority_score+'; соответствие: '+r.role_score+'</p>'+
    '<p>Время: '+esc(r.temporal.fast_status)+', сопоставлено '+(r.temporal.fast_matched_kzt??"—")+
    ' / '+r.temporal.fast_eligible_kzt+' KZT; покрытие '+pct(r.temporal.fast_coverage)+'.</p>'+
    '<p>Сработавшие правила: '+r.roles.filter(x=>x.eligible).map(x=>esc(x.role)+' ('+x.score.toFixed(3)+')').join(", ")+'</p>';
  if(f.kind==="common_recipients")return '<p>Режим: '+(r.mode==="direct"?"прямые переводы":"достижимость")+
    '; результатов: '+r.total_results+'. Совпадение всех выбранных источников.</p>'+
    r.recipients.map(n=>'<p>'+link(n.gid)+' — '+esc(n.role)+', от '+n.matched_sources+' из '+n.total_sources+'</p>').join("");
  if(f.kind==="seed_paths")return '<p>Достижим из '+r.reachable_seed_count+' seed в пределах поиска. Показано '+
    r.paths.length+' путей.</p>'+r.paths.map(p=>'<p>'+p.gids.map(link).join(" → ")+
    '<br><span class="note">'+p.edges.map(e=>kzt(e.sum_kzt)).join(" → ")+'</span></p>').join("");
  if(f.kind==="cluster")return '<h3>Кластер '+r.cluster_id+'</h3><p>'+esc(r.hypothesis)+'</p><p>'+r.n_nodes+
    ' узлов, '+r.n_seed+' seed; '+kzt(r.sum_kzt_internal)+'. Ключевые: '+r.top_gids.map(link).join(", ")+'</p>';
  return '<details><summary>Факты: '+esc(f.kind)+'</summary><pre style="white-space:pre-wrap">'+esc(JSON.stringify(r,null,2))+'</pre></details>';
}
function bindFactLinks(container){
  container.querySelectorAll("[data-g]").forEach(a=>a.onclick=()=>selectNode(a.dataset.g));
}
async function runQuery(action,args){
  showTab("analysis");const box=$("#queryResult");box.textContent="Выполняю запрос…";
  if(!location.protocol.startsWith("http")){
    box.textContent="Для запросов запустите python serve.py. Статические карточки и скачивание CSV доступны без сервера.";return;
  }
  try{
    const response=await fetch("/api/query",{method:"POST",headers:{"Content-Type":"application/json"},
      body:JSON.stringify({action,args,run_id:M.manifest.run_id})});
    const j=await response.json();if(!response.ok)throw Error(j.error);
    box.innerHTML=j.facts.map(f=>'<div class="ev">'+factHTML(f)+'</div>').join("")+
      '<p class="note">'+(j.truncated?"Выдача ограничена. ":"")+j.limitations.map(esc).join(" ")+'</p>'+
      '<details><summary>Источник результата</summary><p class="note">Run '+esc(j.run_id)+'; '+j.evidence_refs.map(esc).join(", ")+'</p></details>';
    if(j.result.markdown)button("Скачать полную справку",()=>download("node-"+j.result.gid+".md",j.result.markdown,"text/markdown;charset=utf-8"),box);
    bindFactLinks(box);if(j.result.paths)highlightPaths(j.result.paths);
  }catch(error){box.textContent="Запрос не выполнен: "+error.message;}
}
function renderAIFacts(j){
  if(j.facts?.length){
    const box=document.createElement("div");box.innerHTML='<h3>Факты инструментов</h3>'+j.facts.map(factHTML).join("");
    bindFactLinks(box);$("#chat").appendChild(box);
  }
  for(const [key,label] of [["hypotheses","Гипотезы"],["limitations","Ограничения"]]){
    if(j[key]?.length){const p=document.createElement("p");p.className="note";p.textContent=label+": "+j[key].join("; ");$("#chat").appendChild(p);}
  }
  if(j.audit?.length){
    const audit=document.createElement("details"),summary=document.createElement("summary");
    summary.textContent="Журнал вызовов";audit.appendChild(summary);
    j.audit.forEach(item=>{
      const line=document.createElement("p");line.className="note";
      line.textContent=item.name+" "+JSON.stringify(item.arguments)+" — "+item.duration_ms+" мс; "+item.evidence_refs.join(", ");
      audit.appendChild(line);
      const detail=document.createElement("details"),title=document.createElement("summary"),body=document.createElement("pre");
      title.textContent="Результат: "+(item.status||"ok")+(item.truncated?" (ограничен)":"");
      body.style.whiteSpace="pre-wrap";body.textContent=JSON.stringify(item.result,null,2);
      detail.append(title,body);audit.appendChild(detail);
    });
    $("#chat").appendChild(audit);
  }
}
$("#t-clusters").querySelectorAll(".cl").forEach(d=>{
  button("Справка кластера",e=>{e.stopPropagation();runQuery("get_cluster_report",{cluster_id:+d.dataset.c});},d);
});
$("#t-res").innerHTML='<p class="note">Исключение узлов из копии наблюдаемого графа. Прямое удаление и распад связей не моделируют адаптацию участников или предотвращённый ущерб.</p>'+
 '<table><tr><th>Стратегия</th><th>Удалено</th><th>Крупнейшая компонента</th><th>Фрагменты ≥3</th><th>Доля оборота</th></tr>'+
 DATA.resilience.map(r=>'<tr><td>'+esc(r.strategy)+'</td><td>'+r.n_removed+'</td><td>'+r.giant_nodes+'</td><td>'+
 r.fragments_3plus+'</td><td>'+pct(r.giant_turnover_share)+'</td></tr>').join("")+'</table>';
$("#t-method").innerHTML='<h3>Роли</h3><p class="note">Первая выполненная роль по фиксированному порядку. Score — соответствие эвристике, не вероятность.</p>'+
 '<table>'+M.rules.map(([r,t])=>'<tr><td>'+badge(r)+'</td><td>'+esc(t)+'</td></tr>').join("")+'</table>'+
 '<h3>Приоритет</h3><p class="note">Взвешенная сумма компонент × seed-поправка / максимум по набору. Все множители показаны в карточке. Изоляты: 0. При равенстве сортируем gid.</p>'+
 '<h3>Время</h3><p class="note">FIFO 1–2 календарных дня; сумма используется один раз. Знаменатель — вход с полным двухдневным окном до '+S.period_end+
 '. Активность одного дня отдельно. Совместимость не доказывает трассировку средств.</p>'+
 '<h3>Модель обрыва</h3><p class="note">Статус: '+esc(M.trunc.status)+'. CV AUC '+
 (M.trunc.cv_auc==null?"недоступен":M.trunc.cv_auc.toFixed(3))+
 ' на наблюдаемых исходящих depth=1–3. Это не точность ролей. Перенос на depth=4 не проверен. Модель не назначает terminal.</p>'+
 '<h3>Ограничения</h3><p class="note">Выход больше входа: возможны внешние поступления или начальный остаток. Пути от seed структурные; смешивание — модель. Циклов со строгим порядком дат: '+
 M.n_return_cycles+' из '+M.n_cycles+'; те же деньги не идентифицированы. Лимит циклов достигнут: '+
 (M.manifest.optional.cycles_truncated?"да":"нет")+'.</p>';

(function(){
 const rows=DATA.resilience,base=rows.find(r=>r.strategy==="исходная сеть");
 const names=[...new Set(rows.filter(r=>r.strategy!=="исходная сеть").map(r=>r.strategy))];
 const palette=["#d7263d","#7b4fd6","#1b9aaa","#98a2b3"];
 const maxN=Math.max(1,...rows.map(r=>r.n_removed)),maxV=Math.max(1,base.giant_nodes);
 const x=n=>30+n/maxN*330,y=n=>170-n/maxV*145;
 let svg='<h3>Связность после исключения узлов</h3><svg viewBox="0 0 380 200" width="100%"><line x1="30" y1="170" x2="365" y2="170" stroke="#d0d5dd"/>';
 names.forEach((name,i)=>{
  const values=[[0,base.giant_nodes],...rows.filter(r=>r.strategy===name).map(r=>[r.n_removed,r.giant_nodes])];
  svg+='<polyline fill="none" stroke="'+palette[i]+'" stroke-width="2" points="'+values.map(([n,v])=>x(n)+','+y(v)).join(' ')+'"/>';
 });
 svg+='<text x="30" y="190" font-size="10">0</text><text x="340" y="190" font-size="10">'+maxN+'</text></svg>';
 svg+='<p class="note">'+names.map((name,i)=>'<span style="color:'+palette[i]+'">'+esc(name)+'</span>').join(" · ")+'</p>';
 $("#t-res").insertAdjacentHTML("afterbegin",svg);
 $("#t-method").insertAdjacentHTML("beforeend",'<h3>Веса приоритета</h3><table>'+Object.entries(M.weights).map(([k,w])=>'<tr><td>'+esc(k)+'</td><td>'+w+'</td></tr>').join("")+'</table>');
})();
