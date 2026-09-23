// Workspace controls. No external UI dependencies or network resources.
function setAssistantOpen(open){
  const dock=$("#assistantDock");clearTimeout(setAssistantOpen.timer);
  $("#aiLauncher").setAttribute("aria-expanded",String(open));
  if(open){dock.classList.remove("is-closing");dock.hidden=false;$("#aiq").focus({preventScroll:true});}
  else{dock.classList.add("is-closing");setAssistantOpen.timer=setTimeout(()=>{dock.hidden=true;dock.classList.remove("is-closing");},reducedMotion.matches?0:160);$("#aiLauncher").focus({preventScroll:true});}
}
$("#aiLauncher").onclick=()=>setAssistantOpen($("#aiLauncher").getAttribute("aria-expanded")!=="true");
$("#aiClose").onclick=()=>setAssistantOpen(false);
function saveEconomy(){
  try{localStorage.setItem('moneygraph-economy',String(economy));}catch{}
  $("#renderStatus").textContent=economy?"Экономный: подпись выбранного узла, без движения камеры":"Стандартный: плавный фокус и подписи";
}
$("#economyMode").checked=economy;
$("#economyMode").onchange=e=>{economy=e.target.checked;saveEconomy();applyLabels();if(mode==="ego"&&egoCenter)showEgo(egoCenter);};
saveEconomy();decorateIcons();
// Canvas labels are redrawn once the embedded font is ready.
document.fonts.ready.then(()=>{cy.style().update();if(egoCy)egoCy.style().update();});
function setPanelOpen(open){
  const wasOpen=!$("#right").hidden,previousWidth=cy.width(),previousPan=cy.pan();
  $("#right").hidden=!open;
  $("#app").classList.toggle("panel-open",open);
  $("#app").classList.remove("filters-open");
  $("#mobileFilters").setAttribute("aria-expanded","false");
  requestAnimationFrame(()=>{
    cy.resize();
    if(wasOpen&&!open)cy.pan({x:previousPan.x+(cy.width()-previousWidth)/2,y:previousPan.y});
    if(egoCy){egoCy.resize();if(mode==="ego")egoCy.fit(undefined,40);}
  });
}
function notifyUser(text){
  const el=$("#toast");el.textContent=text;el.hidden=false;
  clearTimeout(notifyUser.timer);notifyUser.timer=setTimeout(()=>{el.hidden=true;},3000);
}
async function requestJSON(url,payload,timeoutMs=12000){
  const control=new AbortController(),timer=setTimeout(()=>control.abort(),timeoutMs);
  try{
    const response=await fetch(url,{signal:control.signal,...(payload===undefined?{}:{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(payload)})});
    const result=await response.json();if(!response.ok)throw Error(result.error||"Сервер не выполнил запрос");return result;
  }catch(error){
    if(error.name==="AbortError")throw Error("Время ожидания истекло. Попробуйте ещё раз или продолжите работу с графом.");
    throw error;
  }finally{clearTimeout(timer);}
}
function resetFilters(){
  roleOn.clear();M.roles.forEach(r=>roleOn.add(r));
  $("#rolef").querySelectorAll("input").forEach(e=>{e.checked=true;});
  $("#clsel").value="-1";$("#topn").value="0";$("#seedonly").checked=false;
  setMode("net");clearHL();applyFilters();applyLabels();cy.fit(cy.nodes(":visible"),65);
}
$("#closePanel").onclick=()=>setPanelOpen(false);
$("#openExports").onclick=()=>showTab("export");
$("#openMethod").onclick=()=>showTab("method");
$("#openResilience").onclick=()=>showTab("res");
$("#startAnalysis").onclick=$("#emptyTop").onclick=()=>showTab("top");
$("#mobileFilters").onclick=()=>{
  const open=!$("#app").classList.contains("filters-open");
  setPanelOpen(false);$("#app").classList.toggle("filters-open",open);
  $("#mobileFilters").setAttribute("aria-expanded",String(open));
};
$("#drawerShade").onclick=()=>{$("#app").classList.remove("filters-open");$("#mobileFilters").setAttribute("aria-expanded","false");};
$("#resetFilters").onclick=$("#emptyReset").onclick=resetFilters;
$("#resetView").onclick=()=>{selected=null;resetFilters();setPanelOpen(false);history.replaceState(null,"",location.pathname+location.search);};
$("#clearSearch").onclick=()=>{q.value="";matches=[];renderSugg();q.focus();};
const activeGraph=()=>mode==="ego"&&egoCy?egoCy:cy;
$("#zoomIn").onclick=()=>{const g=activeGraph();g.zoom({level:Math.min(g.maxZoom(),g.zoom()*1.35),renderedPosition:{x:g.width()/2,y:g.height()/2}});};
$("#zoomOut").onclick=()=>{const g=activeGraph();g.zoom({level:Math.max(g.minZoom(),g.zoom()/1.35),renderedPosition:{x:g.width()/2,y:g.height()/2}});};
document.addEventListener("keydown",e=>{
  if(e.key==="Escape"){
    if(!sugg.hidden){sugg.hidden=true;q.setAttribute("aria-expanded","false");}
    else if($("#aiLauncher").getAttribute("aria-expanded")==="true")setAssistantOpen(false);
    else setPanelOpen(false);
  }
  const row=e.target.closest('[role="button"][data-g],[role="button"][data-c]');
  if(row&&e.target===row&&(e.key==="Enter"||e.key===" ")){e.preventDefault();row.click();}
});
// Resize canvases when responsive panels change the available graph area.
let resizeFrame;
new ResizeObserver(()=>{cancelAnimationFrame(resizeFrame);resizeFrame=requestAnimationFrame(()=>{if(mode==="net")cy.resize();else if(egoCy)egoCy.resize();});}).observe($("main"));
