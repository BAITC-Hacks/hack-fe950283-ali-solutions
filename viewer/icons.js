// Original inline SVG icons: no icon font, CDN or network dependency.
const ICONS = {
  network:'<circle cx="5" cy="6" r="3"/><circle cx="19" cy="8" r="3"/><circle cx="11" cy="19" r="3"/><path d="m8 6 8 2M6 9l4 7m7-5-4 5"/>',
  rank:'<path d="M5 20V12h4v8m2 0V4h4v16m2 0V8h4v12M3 20h19"/>',
  groups:'<rect x="3" y="3" width="7" height="7" rx="2"/><rect x="14" y="3" width="7" height="7" rx="2"/><rect x="3" y="14" width="7" height="7" rx="2"/><rect x="14" y="14" width="7" height="7" rx="2"/>',
  merge:'<path d="M4 5h3c5 0 4 7 9 7h4M4 19h3c5 0 4-7 9-7m0-4 4 4-4 4"/>',
  download:'<path d="M12 3v12m-5-5 5 5 5-5M4 16v4h16v-4"/>',
  filter:'<path d="M4 7h16M4 17h16"/><circle cx="9" cy="7" r="2"/><circle cx="16" cy="17" r="2"/>',
  close:'<path d="m6 6 12 12M6 18 18 6"/>',
  plus:'<path d="M12 5v14M5 12h14"/>',
  minus:'<path d="M5 12h14"/>',
  focus:'<path d="M9 3H3v6m12-6h6v6M3 15v6h6m12-6v6h-6"/><circle cx="12" cy="12" r="3"/>',
  labels:'<path d="M3 6V4h18v2M12 4v16m-4 0h8"/>',
  reset:'<path d="M3 10a9 9 0 1 1 2 8M3 4v6h6"/>',
  back:'<path d="m10 5-7 7 7 7M3 12h18"/>',
  book:'<path d="M12 5v16M3 3c4 0 6 0 9 2 3-2 5-2 9-2v16c-4 0-6 0-9 2-3-2-5-2-9-2Z"/>',
  shield:'<path d="m12 3 8 3v6c0 5-8 9-8 9s-8-4-8-9V6Z"/><path d="m8 12 3 3 5-6"/>',
  path:'<circle cx="5" cy="5" r="2"/><circle cx="19" cy="19" r="2"/><path d="M7 5h9a4 4 0 0 1 0 8H8a3 3 0 0 0 0 6h9"/>',
  report:'<path d="M14 3H5v18h14V8Zm0 0v5h5M8 12h8m-8 4h6"/>',
  compare:'<path d="M4 5h10m-4-4 4 4-4 4M20 19H10m4-4-4 4 4 4M4 13v6m-3-3h6M20 5v6m-3-3h6"/>',
  spark:'<path d="m12 3 2.5 6.5L21 12l-6.5 2.5L12 21l-2.5-6.5L3 12l6.5-2.5ZM20 2v4m-2-2h4"/>',
  send:'<path d="m3 3 18 9-18 9 4-9Zm4 9h14"/>',
  speed:'<path d="M4 18a10 10 0 1 1 16 0M12 12l5-5M3 12h2m14 0h2M12 2v2"/><circle cx="12" cy="12" r="2"/>',
  info:'<circle cx="12" cy="12" r="9"/><path d="M12 11v6m0-10v.1"/>',
  eye:'<path d="M2 12s4-7 10-7 10 7 10 7-4 7-10 7S2 12 2 12Z"/><circle cx="12" cy="12" r="3"/>',
};
function icon(name){return '<svg class="ico" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.65" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'+(ICONS[name]||ICONS.info)+'</svg>';}
function decorateIcons(root=document){
  root.querySelectorAll('[data-icon]').forEach(el=>{
    if(el.querySelector('.ico'))return;
    const label=el.getAttribute('aria-label')||el.textContent.trim();
    el.title=el.title||label;el.setAttribute('aria-label',label);
    el.innerHTML=icon(el.dataset.icon)+'<span class="button-label">'+esc(el.textContent.trim())+'</span>';
  });
}
