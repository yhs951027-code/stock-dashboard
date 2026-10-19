// 오프라인에서도 열리도록 앱 파일을 저장해 둔다. 앱 화면은 네트워크 우선(업데이트 바로 반영), 글꼴은 저장본 우선.
const CACHE='shorts-v1';
const SHELL=['./','./index.html','./manifest.json','./icon-192.png','./icon-512.png','./icon-maskable.png'];

self.addEventListener('install',e=>{
  e.waitUntil(caches.open(CACHE).then(c=>c.addAll(SHELL)).then(()=>self.skipWaiting()));
});
self.addEventListener('activate',e=>{
  e.waitUntil(caches.keys().then(ks=>Promise.all(ks.filter(k=>k!==CACHE).map(k=>caches.delete(k)))).then(()=>self.clients.claim()));
});
self.addEventListener('fetch',e=>{
  const req=e.request;
  if(req.method!=='GET') return;
  const url=new URL(req.url);
  if(url.hostname==='fonts.googleapis.com' || url.hostname==='fonts.gstatic.com'){
    e.respondWith(caches.open(CACHE).then(async c=>{
      const hit=await c.match(req);
      if(hit) return hit;
      const res=await fetch(req);
      if(res.ok || res.type==='opaque') c.put(req,res.clone());
      return res;
    }));
    return;
  }
  if(url.origin!==location.origin) return;
  e.respondWith(
    fetch(req).then(res=>{
      if(res.ok){ const copy=res.clone(); caches.open(CACHE).then(c=>c.put(req,copy)); }
      return res;
    }).catch(()=>caches.match(req,{ignoreSearch:true}).then(r=>r || (req.mode==='navigate'? caches.match('./index.html') : Response.error())))
  );
});
