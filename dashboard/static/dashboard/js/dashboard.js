const $=s=>document.querySelector(s),css=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const DAY=864e5,utc=(y,d)=>Math.floor(Date.UTC(y,0,d)/DAY);
let ROWS=[],YS=[2022,2023,2024,2025],AOI=null,REAL=false,S={},ST={},AN=[],PR=[],MODE='hm',P={},tm,HEAT=[],HMX=1,HOV=null,SEL=null,BK='',BC=null,WS=[],LIVE=false,LT=0,DR=null,LSEEN=new Set(),CH=[],V;
const TIP='Hover the calendar for details. Outlined cells = unusual activity.';
const mo=d=>new Date(Date.UTC(2023,0,d)).toLocaleDateString('en-GB',{day:'numeric',month:'short',timeZone:'UTC'});
const inA=(o,a=AOI)=>!a||(o.lat>=a.la0&&o.lat<=a.la1&&o.lon>=a.lo0&&o.lon<=a.lo1);
const pal=()=>{for(const k of['mut','ln','acc','mod','vir','hm','ink','card'])P[k]=css('--'+k)};
const f=v=>v==v?Math.round(v):'n/a';
let LAND=[
[-168,66,-162,70,-141,70,-115,68,-95,72,-82,69,-80,63,-94,59,-82,55,-79,52,-78,62,-70,60,-62,58,-56,52,-66,45,-70,42,-76,38,-81,31,-80,26,-83,29,-90,30,-97,27,-97,21,-90,21,-87,21,-88,16,-83,10,-79,9,-83,8,-86,12,-92,15,-105,20,-110,24,-112,29,-115,30,-110,23,-117,33,-121,35,-124,40,-124,47,-130,55,-140,60,-152,60,-165,55,-158,58,-165,62],
[-73,78,-60,82,-30,83,-20,75,-22,70,-40,65,-44,60,-52,65,-58,75],
[-80,9,-72,12,-62,10,-52,5,-50,0,-35,-6,-39,-14,-41,-22,-48,-26,-54,-34,-58,-38,-65,-42,-68,-52,-70,-55,-74,-50,-73,-40,-71,-30,-70,-18,-76,-14,-81,-5,-80,0,-78,5],
[-10,36,-9,43,-2,44,-5,48,2,51,8,54,10,57,5,60,10,64,18,70,28,71,40,67,45,68,60,70,70,73,80,73,100,77,115,74,130,72,142,72,160,70,170,70,180,68,180,65,165,60,160,55,156,51,143,59,135,55,140,48,130,42,127,38,126,35,122,40,118,38,122,32,120,25,110,21,106,20,109,12,105,9,100,13,100,6,103,1,98,8,98,16,94,17,92,22,86,20,80,15,78,8,73,17,70,22,67,25,57,25,56,27,50,30,48,29,51,25,56,24,59,22,52,16,44,12,43,16,35,28,34,31,36,36,28,37,26,40,23,38,19,41,14,45,18,40,16,38,12,42,8,44,3,43,-2,37],
[-17,21,-10,30,-6,36,10,37,11,33,20,31,32,31,34,28,39,20,43,12,51,12,42,-2,40,-10,40,-16,35,-25,32,-29,27,-34,20,-35,17,-30,12,-17,13,-6,9,4,-5,5,-13,8,-17,14],
[114,-22,122,-18,130,-12,136,-12,137,-16,142,-11,146,-19,153,-26,150,-37,141,-38,135,-35,129,-32,115,-34,114,-26],
[44,-25,50,-15,49,-12,44,-17],[130,31,135,34,141,38,142,44,140,41,133,35],[-5,50,1,51,-2,57,-6,58],
[109,1,117,7,119,1,116,-4,110,-3],[95,5,104,-2,106,-6,100,-1],[131,-1,141,-3,150,-10,141,-9],[166,-46,174,-41,178,-38,172,-35,170,-44]
].map(a=>a.map((v,i)=>v));let DRAW=LAND;
function inLand(lo,la){for(const p of LAND){let c=false;for(let i=0,j=p.length-2;i<p.length;j=i,i+=2){const xi=p[i],yi=p[i+1],xj=p[j],yj=p[j+1];if((yi>la)!=(yj>la)&&lo<(xj-xi)*(la-yi)/(yj-yi)+xi)c=!c}if(c)return true}return false}
const hs=(a,b)=>{let h=Math.imul((a*7919+b*104729)|0,2654435761);h^=h>>>15;h=Math.imul(h,2246822519);h^=h>>>13;return(h>>>0)/4294967296};
function fac(la,lo){let f=.2+1.8*hs(Math.round(la/5),Math.round(lo/5));
 if((lo>-17&&lo<60&&la>14&&la<32)||(lo>75&&lo<110&&la>35&&la<48)||(la>60&&lo>-75&&lo<-12))f*=.05;return f}
function lam(la,lo,d,y){let p,a,w;
 if(la>48){p=200;a=3;w=30}else if(la>25){p=220;a=2;w=35}else if(la>0){p=35;a=8;w=40}else if(la>-30){p=240;a=9;w=40}else{p=30;a=2;w=30}
 let dd=Math.abs(d-p);dd=Math.min(dd,365-dd);return fac(la,lo)*(.05+a*Math.exp(-((dd/w)**2)/2))*y}
function gen(b){
 const sd=Math.round((b.la0+b.la1)*50)*31+Math.round((b.lo0+b.lo1)*50);let a=sd;
 const r=()=>{a|=0;a=a+0x6D2B79F5|0;let t=Math.imul(a^a>>>15,1|a);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296};
 const g=()=>{let u=0;for(let i=0;i<6;i++)u+=r();return(u-3)*1.414};
 const cla=(b.la0+b.la1)/2,clo=(b.lo0+b.lo1)/2,H=b.la1-b.la0,ar=H*(b.lo1-b.lo0)/25,fz=fac(cla,clo),out=[],cl=(v,l,h)=>Math.min(h,Math.max(l,v));
 const hubs=[0,1,2,3].map(()=>[b.la0+r()*H,b.lo0+r()*H]);
 YS.forEach((y,yi)=>{const bs=hs(sd,yi+5)<.45?100+(hs(sd,yi+9)*180|0):-99;
  for(let d=1;d<=365;d++){
   let l=lam(cla,clo,d,[1,1.3,.8,1.15][yi])*ar;if(d>=bs&&d<bs+10)l+=10*ar*fz;
   const n=Math.min(400,Math.max(0,Math.round(l+Math.sqrt(l)*g()))),day=utc(y,d);
   for(let i=0;i<n;i++){
    const h=hubs[r()*4|0],la=cl(h[0]+g()*H/6,b.la0,b.la1),lo=cl(h[1]+g()*H/6,b.lo0,b.lo1),fr=Math.exp(2.2+g()*.8),k=1+(r()*3|0);
    if(y>2022&&r()<.9)for(let j=0;j<k;j++)out.push({s:'V',y,day,lat:la+g()*.003,lon:lo+g()*.003,frp:fr/k,minute:720});
    if(r()<Math.min(.9,fr/25))out.push({s:'M',y,day,lat:la+g()*.004,lon:lo+g()*.004,frp:fr*.7,minute:735})}}});
 return out}
function csvLine(line){
 const out=[];let v='',q=false;
 for(let i=0;i<line.length;i++){const c=line[i];if(c==='"'){if(q&&line[i+1]==='"'){v+='"';i++}else q=!q}else if(c===','&&!q){out.push(v);v=''}else v+=c}
 out.push(v);return out
}
function parse(t,s){
 const L=t.replace(/^\uFEFF/,'').split(/\r?\n/).filter(x=>x.trim().length);
 if(!L.length)throw Error('The selected CSV is empty.');
 const H=csvLine(L[0]).map(x=>x.trim().toLowerCase()),ix=n=>H.indexOf(n);
 const ia=ix('latitude'),io=ix('longitude'),id=ix('acq_date'),ifr=ix('frp'),ic=ix('confidence'),it=ix('acq_time'),isat=ix('satellite');
 if(ia<0||io<0||id<0)throw Error('Missing latitude / longitude / acq_date columns (use a FIRMS CSV).');
 const out=[];
 for(let i=1;i<L.length;i++){
  const p=csvLine(L[i]);if(p.length<H.length)continue;
  const c=ic>=0?p[ic].trim().toLowerCase():'';
  // MODIS confidence is numeric (0-100); VIIRS confidence is commonly l/n/h.
  if(c){if(['l','low'].includes(c))continue;if(!isNaN(Number(c))&&Number(c)<30)continue}
  const [Y,M,D]=p[id].trim().split('-').map(Number);if(!Y||!M||!D)continue;
  const lat=Number(p[ia]),lon=Number(p[io]);if(!Number.isFinite(lat)||!Number.isFinite(lon))continue;
  let mins=NaN;
  if(it>=0&&p[it].trim()!==''){const z=p[it].trim().replace(/\D/g,'').padStart(4,'0');if(z.length<=4){const hh=Number(z.slice(0,-2)),mm=Number(z.slice(-2));if(hh<24&&mm<60)mins=hh*60+mm}}
  const fv=ifr>=0&&p[ifr].trim()!==''?Number(p[ifr]):NaN;
  out.push({s,y:Y,day:Math.floor(Date.UTC(Y,M-1,D)/DAY),lat,lon,frp:Number.isFinite(fv)&&fv>=0?fv:NaN,minute:mins,satellite:isat>=0?p[isat].trim():''});
 }
 return out
}

const CL=-60,CT=85,CN=5,NX=72,NY=29;let LC=[];V={lo0:-180,lo1:180,la0:CL,la1:CT};
function heat(){
 if(!LC.length)for(let j=0;j<NY;j++)for(let i=0;i<NX;i++)LC.push(inLand(-180+CN*i+2.5,CL+CN*j+2.5));
 HEAT=[];HMX=1e-9;const cnt=new Array(NX*NY).fill(0);
 if(REAL)for(const o of ROWS){const i=Math.floor((o.lon+180)/CN),j=Math.floor((o.lat-CL)/CN);if(i>=0&&i<NX&&j>=0&&j<NY)cnt[j*NX+i]++}
 for(let j=0;j<NY;j++)for(let i=0;i<NX;i++){let v=0;
  if(REAL)v=cnt[j*NX+i];else if(LC[j*NX+i]){const la=CL+CN*j+2.5,lo=-180+CN*i+2.5;for(let d=1;d<365;d+=20)v+=lam(la,lo,d,1)/18}
  HEAT.push(v);if(v>HMX)HMX=v}BK=''}
function base(x,dp,w,h,X,Y,sx,W){
 x.scale(dp,dp);const dk=parseInt(P.card.slice(1,3),16)<128;
 let g=x.createLinearGradient(0,0,0,h);g.addColorStop(0,dk?'#0f3347':'#c9e8f5');g.addColorStop(1,dk?'#081a27':'#7fbfe0');x.fillStyle=g;x.fillRect(0,0,w,h);
 const st=W>120?30:W>50?10:5;x.lineWidth=.5;x.strokeStyle=dk?'rgba(255,255,255,.1)':'rgba(255,255,255,.55)';x.fillStyle=dk?'rgba(255,255,255,.5)':'rgba(20,60,90,.65)';x.font='10px system-ui';
 for(let lo=Math.ceil(V.lo0/st)*st;lo<=V.lo1;lo+=st){x.beginPath();x.moveTo(X(lo),0);x.lineTo(X(lo),h);x.stroke();x.fillText(lo+'°',X(lo)+3,h-4)}
 for(let la=Math.ceil(V.la0/st)*st;la<=V.la1;la+=st){x.beginPath();x.moveTo(0,Y(la));x.lineTo(w,Y(la));x.stroke();x.fillText(la+'°',3,Y(la)-3)}
 const C=dk?['#3a4448','#3d5143','#2d5a3a','#4a6535','#6e5f3a','#6a5a35','#2f6a3a','#1f5a35','#2f6339','#6a5c36','#5f5a37','#3b5a3a','#444c50']:['#eef2f3','#c9dcc0','#7fae6a','#a9c274','#e4cf94','#d8c27c','#6fb25a','#3f9a4f','#68ae57','#d9c585','#c9c080','#8fb577','#dfe6e8'],
  F=[0,.1,.21,.31,.39,.48,.55,.6,.655,.74,.81,.9,1],lg=x.createLinearGradient(0,Y(CT),0,Y(CL));F.forEach((f,i)=>lg.addColorStop(f,C[i]));
 x.save();x.shadowColor='rgba(0,0,0,.3)';x.shadowBlur=8;x.fillStyle=lg;x.beginPath();
 for(const p of DRAW){for(let i=0;i<p.length;i+=2)i?x.lineTo(X(p[i]),Y(p[i+1])):x.moveTo(X(p[i]),Y(p[i+1]));x.closePath()}
 x.fill();x.restore();x.strokeStyle=dk?'rgba(255,255,255,.3)':'rgba(40,60,50,.5)';x.lineWidth=.6;x.stroke();
 HEAT.forEach((v,k)=>{if(v<=0)return;const t=Math.sqrt(v/HMX),cx=X(-180+CN*((k%NX)+.5)),cy=Y(CL+CN*(((k/NX)|0)+.5)),r=Math.max(CN*sx*.95,8),hu=48-48*t,gr=x.createRadialGradient(cx,cy,0,cx,cy,r);
  gr.addColorStop(0,`hsla(${hu},100%,55%,${.35+.6*t})`);gr.addColorStop(.5,`hsla(${hu},100%,50%,${.18+.3*t})`);gr.addColorStop(1,`hsla(${hu},100%,50%,0)`);x.fillStyle=gr;x.fillRect(cx-r,cy-r,2*r,2*r)});
 if(REAL&&ROWS.length<6e4)for(const o of ROWS){const px=X(o.lon),py=Y(o.lat);if(px<0||py<0||px>w||py>h)continue;x.fillStyle=o.s==='V'?'#ffd43b':'#ff5a1f';x.fillRect(px-1,py-1,2,2)}}
function drawMap(){
 const c=$('#map'),dp=devicePixelRatio||1,w=c.clientWidth,h=w*(CT-CL)/360,W=V.lo1-V.lo0,sx=w/W,sy=h/(V.la1-V.la0),X=lo=>(lo-V.lo0)*sx,Y=la=>(V.la1-la)*sy;
 c.width=w*dp;c.height=h*dp;c.style.height=h+'px';
 const key=[w,dp,V.lo0,V.la1,W,P.card,DRAW.length,ROWS.length].join('|');
 if(key!==BK||!BC){BK=key;BC=document.createElement('canvas');BC.width=w*dp;BC.height=h*dp;base(BC.getContext('2d'),dp,w,h,X,Y,sx,W)}
 const x=c.getContext('2d');x.drawImage(BC,0,0);x.scale(dp,dp);
 if(HOV){x.strokeStyle='rgba(255,255,255,.95)';x.lineWidth=1.5;x.setLineDash([5,3]);x.strokeRect(X(HOV.lo-HOV.s/2),Y(HOV.la+HOV.s/2),HOV.s*sx,HOV.s*sy);x.setLineDash([])}
 if(AOI){const a=[X(AOI.lo0),Y(AOI.la1),(AOI.lo1-AOI.lo0)*sx,(AOI.la1-AOI.la0)*sy];x.strokeStyle='rgba(255,255,255,.85)';x.lineWidth=4.5;x.strokeRect(...a);x.strokeStyle=P.hm;x.lineWidth=2.5;x.strokeRect(...a)}}

const GRID_DEG=.02,FRP_BIN_MIN=90,FRP_MATCH_MIN=180;
function aggregateSensor(rows,sensor){
 const cells=new Map();
 for(const o of rows){if(o.s!==sensor||!inA(o))continue;
  const k=Math.floor(o.lat/GRID_DEG)+':'+Math.floor(o.lon/GRID_DEG)+':'+o.day;
  let e=cells.get(k);if(!e){e={day:o.day,frpBins:new Map()};cells.set(k,e)}
  if(Number.isFinite(o.frp)&&o.frp>=0){
   const bin=Number.isFinite(o.minute)?Math.floor(o.minute/FRP_BIN_MIN):0;
   let b=e.frpBins.get(bin);if(!b){b={frp:0,minutes:[]};e.frpBins.set(bin,b)}
   b.frp+=o.frp;if(Number.isFinite(o.minute))b.minutes.push(o.minute)
  }
 }
 for(const e of cells.values()){
  let best=null;
  for(const [bin,b] of e.frpBins){if(!best||b.frp>best.frp){best={frp:b.frp,minute:b.minutes.length?b.minutes.reduce((a,x)=>a+x,0)/b.minutes.length:bin*FRP_BIN_MIN+FRP_BIN_MIN/2}}}
  e.frp=best?best.frp:NaN;e.minute=best?best.minute:NaN;
  delete e.frpBins;
 }
 return cells
}
function median(a){if(!a.length)return NaN;const b=[...a].sort((x,y)=>x-y),m=b.length>>1;return b.length%2?b[m]:(b[m-1]+b[m])/2}
function harm(){
 // File date spans are used as nominal coverage. This assumes each uploaded archive
 // is complete for its requested date range; FIRMS detection rows alone cannot prove it.
 const cov={M:[Infinity,-Infinity],V:[Infinity,-Infinity]};
 for(const o of ROWS){const c=cov[o.s];c[0]=Math.min(c[0],o.day);c[1]=Math.max(c[1],o.day)}
 const C={M:aggregateSensor(ROWS,'M'),V:aggregateSensor(ROWS,'V')};
 const overlapStart=Math.max(cov.M[0],cov.V[0]),overlapEnd=Math.min(cov.M[1],cov.V[1]);
 const inOverlap=d=>d>=overlapStart&&d<=overlapEnd;
 const dm={},dv={},du={},fm={},fv={};let rawM=0,rawV=0;
 for(const o of ROWS)if(inA(o)){if(o.s==='M')rawM++;else rawV++}
 let overlapM=0,overlapV=0,overlapUnion=0,matchedCells=0;
 for(const [k,e] of C.M){dm[e.day]=(dm[e.day]||0)+1;if(inOverlap(e.day)){overlapM++;if(C.V.has(k))matchedCells++}}
 for(const [k,e] of C.V){dv[e.day]=(dv[e.day]||0)+1;if(inOverlap(e.day))overlapV++}
 for(const [k,e] of C.M){if(inOverlap(e.day)){overlapUnion++;if(!C.V.has(k)){} }}
 for(const [k,e] of C.V){if(inOverlap(e.day)&&!C.M.has(k))overlapUnion++}
 const kM=overlapM?overlapUnion/overlapM:NaN,kV=overlapV?overlapUnion/overlapV:NaN;
 // FRP scale: robust median MODIS/VIIRS ratio for spatially and temporally paired cell-days.
 const logRatios=[];let frpPairs=0;
 for(const [k,m] of C.M){const v=C.V.get(k);if(!v||!inOverlap(m.day)||!Number.isFinite(m.frp)||!Number.isFinite(v.frp)||m.frp<=0||v.frp<=0)continue;
  if(Number.isFinite(m.minute)&&Number.isFinite(v.minute)&&Math.abs(m.minute-v.minute)>FRP_MATCH_MIN)continue;
  logRatios.push(Math.log(m.frp/v.frp));frpPairs++
 }
 const frpScale=logRatios.length?Math.exp(median(logRatios)):NaN;
 // Index cell-day aggregates by date once; avoid rescanning every detection for every day.
 const byDay={M:new Map(),V:new Map()};
 for(const sensor of ['M','V'])for(const [k,e] of C[sensor]){if(!byDay[sensor].has(e.day))byDay[sensor].set(e.day,[]);byDay[sensor].get(e.day).push([k,e])}
 // Daily output arrays. Values are NaN outside nominal sensor coverage, not zero.
 S={};let totalActivity=0,frpDays=0;
 for(const y of YS){const z=S[y]={raw:[],mod:[],vir:[],hm:[],frpM:[],frpV:[],frpRaw:[],frpHM:[]};
  for(let d=1;d<=366;d++){
   const day=utc(y,d);if(new Date(day*DAY).getUTCFullYear()!==y){for(const q in z)z[q].push(NaN);continue}
   const hasM=day>=cov.M[0]&&day<=cov.M[1],hasV=day>=cov.V[0]&&day<=cov.V[1];
   let frM=0,frV=0,frRaw=0,frH=0,hasFrM=false,hasFrV=false;
   const mCells=byDay.M.get(day)||[],vCells=byDay.V.get(day)||[],m=mCells.length,v=vCells.length;
   const vMap=new Map(vCells),keys=new Set([...mCells.map(x=>x[0]),...vCells.map(x=>x[0])]),union=keys.size;
   for(const [k,e] of mCells)if(Number.isFinite(e.frp)){frM+=e.frp;hasFrM=true}
   for(const [k,e] of vCells)if(Number.isFinite(e.frp)){frV+=e.frp;hasFrV=true}
   const hm=hasM&&hasV?union:hasM?(Number.isFinite(kM)?m*kM:NaN):hasV?(Number.isFinite(kV)?v*kV:NaN):NaN;
   // MODIS is the FRP reference. VIIRS-only FRP is converted to MODIS-equivalent MW.
   for(const k of keys){const a=C.M.get(k),b=vMap.get(k);const x=a&&Number.isFinite(a.frp)?a.frp:NaN,yv=b&&Number.isFinite(b.frp)?b.frp:NaN;
    const timeComparable=!(Number.isFinite(a&&a.minute)&&Number.isFinite(b&&b.minute))||Math.abs(a.minute-b.minute)<=FRP_MATCH_MIN;
    if(Number.isFinite(x)&&Number.isFinite(yv)&&Number.isFinite(frpScale)&&timeComparable)frH+=(x+yv*frpScale)/2;
    else if(Number.isFinite(x))frH+=x;
    else if(Number.isFinite(yv)&&Number.isFinite(frpScale))frH+=yv*frpScale;
   }
   if(hasFrM&&hasFrV)frRaw=frM+frV;else if(hasFrM)frRaw=frM;else if(hasFrV)frRaw=frV;else frRaw=NaN;
   z.mod.push(hasM?m:NaN);z.vir.push(hasV?v:NaN);z.raw.push(hasM&&hasV?m+v:hasM?m:hasV?v:NaN);z.hm.push(hm);if(Number.isFinite(hm))totalActivity+=hm;
   z.frpM.push(hasM&&hasFrM?frM:NaN);z.frpV.push(hasV&&hasFrV?frV:NaN);
   z.frpRaw.push(hasM||hasV?frRaw:NaN);
   const frpHasAny=(hasM&&hasFrM)||(hasV&&hasFrV&&Number.isFinite(frpScale));
   z.frpHM.push(frpHasAny?frH:NaN);if(frpHasAny)frpDays++;
  }
 }
 ST={m:rawM,v:rawV,tot:totalActivity,frpDays,kM,kV,frpScale,frpPairs,matchedCells,overlapM,overlapV,overlapUnion};
 AN=[];
 if(YS.length>2){for(const y of YS)for(let d=0;d<366;d++){const v=S[y].hm[d];if(!(v>=5))continue;const a=[];
   for(const q of YS){if(q===y)continue;for(let j=Math.max(0,d-7);j<=Math.min(365,d+7);j++){const w=S[q].hm[j];if(Number.isFinite(w))a.push(w)}}
   if(a.length<10)continue;
   const mu=a.reduce((p,q)=>p+q,0)/a.length,sd=Math.sqrt(a.reduce((p,q)=>p+(q-mu)**2,0)/a.length),z=(v-mu)/Math.max(sd,1+Math.sqrt(mu));
   if(z>=2.5)AN.push({y,d:d+1,v,m:mu,z})}
  AN.sort((a,b)=>b.z-a.z);const keep=[];for(const a of AN)if(!keep.some(k=>k.y===a.y&&Math.abs(k.d-a.d)<7))keep.push(a);AN=keep.slice(0,8)}
 const all=[];for(const y of YS)for(const v of S[y].hm)if(Number.isFinite(v))all.push(v);all.sort((a,b)=>a-b);
 const T=Math.max(3,all[Math.floor(all.length*.75)]||3);
 PR=[];for(let d=0;d<366;d++){let h=0,t=0;for(const y of YS)for(let j=Math.max(0,d-7);j<=Math.min(365,d+7);j++){const v=S[y].hm[j];if(Number.isFinite(v)){t++;if(v>=T)h++}}PR.push(t?h/t:0)}
}

function cv(id,h){const c=$(id),dp=devicePixelRatio||1,w=c.clientWidth;c.width=w*dp;c.height=h*dp;c.style.height=h+'px';const x=c.getContext('2d');x.scale(dp,dp);return[x,w,h]}
function axis(x,w,h,L){x.fillStyle=P.mut;x.font='11px system-ui';x.strokeStyle=P.ln;x.lineWidth=1;
 for(let m=0;m<12;m++){const px=L+(w-L-6)*(1+Math.round(m*30.5))/366;x.fillText('JFMAMJJASOND'[m],px,h-4);x.beginPath();x.moveTo(px,0);x.lineTo(px,h-16);x.stroke()}}
function drawCal(){
 const rh=24,[x,w,h]=cv('#cal',YS.length*rh+20),L=44,cw=(w-L)/366,k=MODE;
 x.font='12px system-ui';let mx=1;for(const y of YS)for(const v of S[y][k])if(v>mx)mx=v;
 YS.forEach((y,r)=>{x.globalAlpha=1;x.fillStyle=P.mut;x.fillText(y,2,r*rh+16);
  S[y][k].forEach((v,i)=>{if(v!=v)return;const X=L+i*cw,ww=Math.max(cw-.3,.7);
   x.globalAlpha=1;x.fillStyle=P.ln;x.fillRect(X,r*rh,ww,rh-3);
   if(v>0){x.globalAlpha=Math.min(1,.12+Math.sqrt(v/mx)*.88);x.fillStyle=P.acc;x.fillRect(X,r*rh,ww,rh-3)}})});
 x.globalAlpha=1;x.strokeStyle=P.ink;x.lineWidth=1.5;
 for(const a of AN){const r=YS.indexOf(a.y);x.strokeRect(L+(a.d-1)*cw-2,r*rh-1,cw+4,rh-1)}
 x.fillStyle=P.mut;x.font='11px system-ui';for(let m=0;m<12;m++)x.fillText('JFMAMJJASOND'[m],L+cw*Math.round(m*30.5),h-4)}
function drawFrp(){
 const rh=24,[x,w,h]=cv('#frpCal',YS.length*rh+20),L=44,cw=(w-L)/366;
 let mx=0;for(const y of YS)for(const v of S[y].frpHM)if(Number.isFinite(v))mx=Math.max(mx,v);
 YS.forEach((y,r)=>{x.globalAlpha=1;x.fillStyle=P.mut;x.font='12px system-ui';x.fillText(y,2,r*rh+16);
  S[y].frpHM.forEach((v,i)=>{if(!Number.isFinite(v))return;const X=L+i*cw,ww=Math.max(cw-.3,.7);x.globalAlpha=1;x.fillStyle=P.ln;x.fillRect(X,r*rh,ww,rh-3);if(v>0&&mx>0){x.globalAlpha=Math.min(1,.12+Math.sqrt(v/mx)*.88);x.fillStyle=P.vir;x.fillRect(X,r*rh,ww,rh-3)}})});
 x.globalAlpha=1;x.fillStyle=P.mut;x.font='11px system-ui';for(let m=0;m<12;m++)x.fillText('JFMAMJJASOND'[m],L+cw*Math.round(m*30.5),h-4);
 $('#frpTip').textContent=mx>0?`Colour intensity = harmonized daily peak-FRP sum (MW); maximum in this view: ${mx.toFixed(1)} MW. Values are sums of per-cell peak observations, not simultaneous total power or daily energy.`:'No usable FRP values in the selected data/area, or no paired calibration is available for VIIRS-only values.'
}
function drawWin(){
 const[x,w,h]=cv('#win',150),L=34,thr=+$('#pt').value;axis(x,w,h,L);
 const X=i=>L+(w-L-6)*(i+1)/366,Y=v=>h-18-(h-26)*v,segs=WS=[];let st=-1;
 PR.forEach((p,i)=>{if(p>=thr&&st<0)st=i;if((p<thr||i==365)&&st>=0){const e=p<thr?i-1:i;if(e-st>=4)segs.push([st,e]);st=-1}});
 x.fillStyle=P.acc;x.globalAlpha=.22;for(const[a,b]of segs)x.fillRect(X(a),0,X(b)-X(a),h-18);x.globalAlpha=1;
 x.strokeStyle=P.hm;x.lineWidth=2;x.beginPath();PR.forEach((p,i)=>i?x.lineTo(X(i),Y(p)):x.moveTo(X(i),Y(p)));x.stroke();
 x.strokeStyle=P.acc;x.lineWidth=1;x.setLineDash([4,3]);x.beginPath();x.moveTo(L,Y(thr));x.lineTo(w-6,Y(thr));x.stroke();x.setLineDash([]);
 $('#wins').innerHTML=segs.length?segs.map(([a,b])=>`<li>${mo(a+1)} – ${mo(b+1)} <span class="mut">(peak probability ${Math.max(...PR.slice(a,b+1)).toFixed(2)})</span></li>`).join(''):'<li>No window reaches this probability.</li>'}

const fmt=(v,p,n)=>Math.abs(v).toFixed(1)+'°'+(v<0?n:p);
function analyze(){
 pal();drawMap();
 if(!AOI){$('#sel').textContent='Click the map to choose a place.';return}
 harm();const s=AOI.lo1-AOI.lo0;
 $('#sel').textContent=`Selected area: ${fmt((AOI.la0+AOI.la1)/2,'N','S')}, ${fmt((AOI.lo0+AOI.lo1)/2,'E','W')} · ${s.toFixed(0)}°×${s.toFixed(0)}°`;
 $('#stats').innerHTML=[['MODIS raw detections',ST.m],['VIIRS 375 m raw detections',ST.v],['Harmonized active cell-days',Math.round(ST.tot)],['MODIS activity scale',Number.isFinite(ST.kM)?ST.kM.toFixed(2)+'×':'n/a'],['VIIRS activity scale',Number.isFinite(ST.kV)?ST.kV.toFixed(2)+'×':'n/a'],['VIIRS→MODIS FRP factor',Number.isFinite(ST.frpScale)?ST.frpScale.toFixed(2)+'×':'not calibrated'],['Paired FRP cell-days',ST.frpPairs],['Days with usable harmonized FRP',ST.frpDays]]
  .map(([a,b])=>`<div class="stat"><b>${typeof b=='number'?b.toLocaleString():b}</b><span class="mut">${a}</span></div>`).join('');
 $('#anom').innerHTML=AN.length?AN.map(a=>`<li><b>${mo(a.d)} ${a.y}</b>: ${Math.round(a.v)} harmonized cell‑days vs ~${a.m.toFixed(1)} typical (z = ${a.z.toFixed(1)})</li>`).join(''):'<li>No unusual days found here.</li>';
 $('#tip').textContent=TIP;drawCal();drawFrp();drawWin()}
function pick(la,lo){
 const s=+$('#sz').value;
 if(!REAL&&!LIVE&&!inLand(lo,la)){$('#msg').textContent='That spot is open water. Pick a land area.';return}
 AOI={la0:la-s/2,la1:la+s/2,lo0:lo-s/2,lo1:lo+s/2};
 if(LIVE){ROWS=[];LSEEN.clear();REAL=true;YS=[new Date().getUTCFullYear()];analyze();liveFetch();return}
 if(!REAL){ROWS=gen(AOI);$('#msg').textContent='Demo mode: detections for this area come from a seasonal model, not real observations.'}
 analyze()}
const mp=$('#map'),at=e=>{const b=mp.getBoundingClientRect();return{lo:V.lo0+(e.clientX-b.left)/b.width*(V.lo1-V.lo0),la:V.la1-(e.clientY-b.top)/b.height*(V.la1-V.la0)}};
function setV(a,t,nw){const nh=nw*(CT-CL)/360;a=Math.max(-180,Math.min(180-nw,a));t=Math.max(CL+nh,Math.min(CT,t));V={lo0:a,lo1:a+nw,la0:t-nh,la1:t};drawMap()}
function zoom(f,cx=.5,cy=.5){const W=V.lo1-V.lo0,H=V.la1-V.la0,nw=Math.min(360,Math.max(6,W/f)),nh=nw*(CT-CL)/360;setV(V.lo0+W*cx-nw*cx,V.la1-H*cy+nh*cy,nw)}
function zoomTo(la,lo,s){setV(lo-s/2,la+s*(CT-CL)/720,s)}
mp.onclick=e=>{if(DR&&DR.m>4)return;const p=at(e);pick(p.la,p.lo)};
mp.onpointerdown=e=>{DR={x:e.clientX,y:e.clientY,v:{...V},m:0}};
mp.onpointerup=()=>setTimeout(()=>DR=null,0);
mp.onwheel=e=>{e.preventDefault();const b=mp.getBoundingClientRect();zoom(e.deltaY<0?1.4:1/1.4,(e.clientX-b.left)/b.width,(e.clientY-b.top)/b.height)};
$('#zi').onclick=()=>zoom(1.6);$('#zo').onclick=()=>zoom(1/1.6);$('#zr').onclick=()=>{V={lo0:-180,lo1:180,la0:CL,la1:CT};drawMap()};
mp.onpointermove=e=>{
 if(DR&&e.buttons){const b=mp.getBoundingClientRect(),dx=e.clientX-DR.x,dy=e.clientY-DR.y;DR.m=Math.max(DR.m,Math.abs(dx)+Math.abs(dy));
  if(DR.m>4){const W=DR.v.lo1-DR.v.lo0,H=DR.v.la1-DR.v.la0;setV(DR.v.lo0-dx/b.width*W,DR.v.la1+dy/b.height*H,W);return}}
 const p=at(e);HOV={...p,s:+$('#sz').value};drawMap();
 const i=Math.floor((p.lo+180)/CN),j=Math.floor((p.la-CL)/CN),v=HEAT[j*NX+i];
 $('#hov').textContent=`${fmt(p.la,'N','S')}, ${fmt(p.lo,'E','W')} · ${inLand(p.lo,p.la)||REAL?'click to analyze':'open water'}${v>0?' · cell activity '+(v/HMX*100).toFixed(0)+'% of the maximum':''}`};
mp.onpointerleave=()=>{HOV=null;drawMap()};
$('#sz').oninput=()=>$('#szv').textContent=$('#sz').value+'°';
$('#pt').oninput=()=>{pal();drawWin()};
document.querySelectorAll('[name=md]').forEach(r=>r.onchange=()=>{MODE=r.value;pal();drawCal()});
$('#cal').onpointermove=e=>{if(!AOI)return;const b=e.target.getBoundingClientRect(),cw=(b.width-44)/366,y=YS[Math.floor((e.clientY-b.top)/24)],i=Math.floor((e.clientX-b.left-44)/cw);
 if(y==null||i<0||i>365){$('#tip').textContent=TIP;return}const s=S[y];
 $('#tip').textContent=`${y} · ${mo(i+1)}: harmonized activity ${f(s.hm[i])} cell-days · MODIS ${f(s.mod[i])} · VIIRS ${f(s.vir[i])}`;$('#frpTip').textContent=`${y} · ${mo(i+1)}: harmonized FRP ${Number.isFinite(s.frpHM[i])?s.frpHM[i].toFixed(1)+' MW':'n/a'} · MODIS ${Number.isFinite(s.frpM[i])?s.frpM[i].toFixed(1)+' MW':'n/a'} · VIIRS ${Number.isFinite(s.frpV[i])?s.frpV[i].toFixed(1)+' MW':'n/a'} (raw sensor scales)`};
const PRE=[['Amazon arc',-9,-55],['Sahel',12,2],['Congo basin',-4,22],['California',37,-120],['Siberia',58,110],['N. Australia',-13,132],['Greece',39,22]];$('#pre').innerHTML=PRE.map(([n,a,o])=>`<button class="g" data-a="${a}" data-o="${o}">${n}</button>`).join('');
$('#pre').onclick=e=>{const b=e.target.closest('button');if(b){pick(+b.dataset.a,+b.dataset.o);zoomTo(+b.dataset.a,+b.dataset.o,45)}};
$('#demo').onclick=()=>{stopLive();REAL=false;YS=[2022,2023,2024,2025];ROWS=[];AOI=null;heat();$('#msg').textContent='Demo mode: click land to generate that place’s detections (VIIRS exists only from 2023).';analyze();pick(-13,132)};
$('#run').onclick=async()=>{try{const a=$('#fm').files[0],b=$('#fv').files[0];if(!a||!b)throw Error('Choose both a MODIS and a VIIRS FIRMS CSV first.');
 const r=parse(await a.text(),'M').concat(parse(await b.text(),'V'));if(!r.length)throw Error('No usable rows found.');
 stopLive();REAL=true;ROWS=r;YS=[...new Set(r.map(o=>o.y))].sort((a,b)=>a-b);AOI=null;heat();
 $('#msg').textContent=`Real data: ${a.name} + ${b.name} (files stay in your browser). The map now shows your detections; click anywhere to analyze.`;analyze()}catch(e){$('#msg').textContent=e.message}};
addEventListener('resize',()=>{clearTimeout(tm);tm=setTimeout(analyze,150)});
matchMedia('(prefers-color-scheme:dark)').onchange=analyze;

/* ---- Live NASA FIRMS feed ---- */
const KEYK='hotsync.key',AIK='hotsync.ai';
const store=(k,v)=>{try{$('#rem').checked?localStorage.setItem(k,v):localStorage.removeItem(k)}catch(e){}};
try{const k=localStorage.getItem(KEYK),a=localStorage.getItem(AIK);if(k){$('#key').value=k;$('#rem').checked=true}if(a){$('#aik').value=a;$('#rem').checked=true}}catch(e){}
$('#key').onchange=$('#aik').onchange=$('#rem').onchange=()=>{store(KEYK,$('#key').value.trim());store(AIK,$('#aik').value.trim())};
function stopLive(){const was=LIVE;LIVE=false;clearInterval(LT);LT=0;if(was)$('#lst').textContent='Live feed stopped. Data already loaded stays on screen.'}
async function liveFetch(){
 if(!LIVE||!AOI)return;const A=AOI,k=$('#key').value.trim(),cp=(v,m)=>Math.max(-m,Math.min(m,v)).toFixed(3),
  box=[cp(A.lo0,180),cp(A.la0,90),cp(A.lo1,180),cp(A.la1,90)].join(','),ds=n=>new Date(Date.now()-n*DAY).toISOString().slice(0,10),st=$('#lst');
 st.textContent='Fetching latest detections…';
 try{let add=0;
  for(const[src,s]of[['MODIS_NRT','M'],['VIIRS_SNPP_NRT','V']])for(const n of[9,4]){
   const r=await fetch(`https://firms.modaps.eosdis.nasa.gov/api/area/csv/${encodeURIComponent(k)}/${src}/${box}/5/${ds(n)}`),tx=await r.text();
   if(A!==AOI||!LIVE)return;
   if(!r.ok||!/latitude/i.test(tx.split('\n')[0]))throw Error(tx.trim().slice(0,140)||'HTTP '+r.status);
   if(tx.trim().split('\n').length>1)for(const o of parse(tx,s)){const q=s+o.lat+o.lon+o.day+o.minute;if(!LSEEN.has(q)){LSEEN.add(q);ROWS.push(o);add++}}}
  REAL=true;YS=[...new Set(ROWS.map(o=>o.y))].sort((a,b)=>a-b);if(!YS.length)YS=[new Date().getUTCFullYear()];heat();analyze();
  st.textContent=`Live · ${ROWS.length.toLocaleString()} detections in this area (${add} new) · updated ${new Date().toLocaleTimeString()}`
 }catch(e){st.textContent=e instanceof TypeError?'Could not reach NASA FIRMS (offline, or the browser blocked the request).':'FIRMS error: '+e.message}}
function armLive(){clearInterval(LT);const m=+$('#ivl').value;LT=m?setInterval(liveFetch,m*6e4):0}
$('#go').onclick=()=>{if(!$('#key').value.trim()){$('#lst').textContent='Paste your FIRMS MAP_KEY first (free, takes a minute).';return}
 store(KEYK,$('#key').value.trim());LIVE=true;REAL=true;ROWS=[];LSEEN.clear();YS=[new Date().getUTCFullYear()];armLive();
 if(AOI)liveFetch();else{$('#lst').textContent='Live feed ready. Click the map to choose an area.';heat();analyze()}};
$('#stop').onclick=stopLive;$('#ivl').onchange=()=>{if(LIVE)armLive()};

/* ---- Assistant ---- */
const cl=$('#cl'),say=(r,t)=>{const d=document.createElement('div');d.className='m '+r;d.textContent=t;cl.appendChild(d);cl.scrollTop=1e9;return d},NA='Pick an area on the map first (or say "show Greece").';
const sum=()=>!AOI?NA:`${$('#sel').textContent}. ${REAL?'Real/live':'Demo (synthetic)'} data: ${ST.m} MODIS + ${ST.v} VIIRS raw detections → ${Math.round(ST.tot)} harmonized active cell-days. ${AN.length?`Most unusual day: ${mo(AN[0].d)} ${AN[0].y} (${Math.round(AN[0].v)} vs ~${AN[0].m.toFixed(1)} typical).`:'No unusual days stand out.'} ${WS.length?`First critical window: ${mo(WS[0][0]+1)} – ${mo(WS[0][1]+1)}.`:'No burning window clears the threshold.'}`;
function ctx(){return{area:AOI&&$('#sel').textContent,data:REAL?'real or live FIRMS':'synthetic demo model',years:YS,stats:{modis:ST.m,viirs:ST.v,harmonizedCellDays:Math.round(ST.tot||0),frpScale:ST.frpScale},unusual:AN.slice(0,5).map(a=>({day:mo(a.d)+' '+a.y,value:Math.round(a.v),typical:+a.m.toFixed(1)})),windows:WS.map(([a,b])=>({from:mo(a+1),to:mo(b+1)})),threshold:+$('#pt').value}}
function local(q){q=q.toLowerCase();
 const pr=PRE.find(p=>q.includes(p[0].toLowerCase().split(/[ .]/).filter(w=>w.length>3)[0]));
 if(pr&&/go|show|jump|take|look|zoom|move|fire|check/.test(q)){pick(pr[1],pr[2]);zoomTo(pr[1],pr[2],45);return'Jumped to '+pr[0]+'. '+sum()}
 if(/unusual|anomal|spike|abnormal|outlier/.test(q))return!AOI?NA:AN.length?'Unusual days (vs the same season in other years):\n'+AN.slice(0,5).map(a=>`• ${mo(a.d)} ${a.y}: ${Math.round(a.v)} cell-days vs ~${a.m.toFixed(1)} typical (z ${a.z.toFixed(1)})`).join('\n'):'No unusual days stand out here.';
 if(/window|critical|season|when|peak|risk/.test(q))return!AOI?NA:WS.length?`Critical windows at probability ≥ ${$('#pt').value}:\n`+WS.map(([a,b])=>`• ${mo(a+1)} – ${mo(b+1)} (peak ${Math.max(...PR.slice(a,b+1)).toFixed(2)})`).join('\n'):'No window reaches the current threshold. Try lowering the probability slider.';
 if(/frp|power|intens|energy/.test(q)){if(!AOI)return NA;const mx=Math.max(0,...YS.flatMap(y=>S[y].frpHM.filter(Number.isFinite)));return`Peak harmonized FRP day: ${mx.toFixed(1)} MW (a sum of per-cell peak observations, not daily energy). VIIRS→MODIS FRP factor: ${Number.isFinite(ST.frpScale)?ST.frpScale.toFixed(2)+'×':'not calibrated yet'}.`}
 if(/key|live|firms|api|continuous|real data/.test(q))return'Open "Live NASA FIRMS feed" under the map, paste a free MAP_KEY (firms.modaps.eosdis.nasa.gov/api/map_key), choose a refresh interval and press Start live. Then click the map: HotSync pulls the last 10 days of MODIS + VIIRS detections for that area and keeps adding new ones.';
 if(/modis|viirs|sensor|harmon|method|how|scale/.test(q))return'HotSync puts MODIS and VIIRS on the same ~2 km grid and UTC day, counts unique active cell-days, and uses the overlap period to scale each sensor to a common baseline.'+(AOI?` Current scales: MODIS ${Number.isFinite(ST.kM)?ST.kM.toFixed(2)+'×':'n/a'}, VIIRS ${Number.isFinite(ST.kV)?ST.kV.toFixed(2)+'×':'n/a'}.`:'');
 if(/summar|status|overview|stat|here|area|going on|hello|hi\b/.test(q))return sum();
 return'I can summarize the selected area, list unusual days or burning windows, explain FRP and the harmonization, jump to a region ("show Greece"), or help set up the live feed. Add a Claude API key above for open-ended questions.'}
async function claude(){const r=await fetch('https://api.anthropic.com/v1/messages',{method:'POST',headers:{'content-type':'application/json','x-api-key':$('#aik').value.trim(),'anthropic-version':'2023-06-01','anthropic-dangerous-direct-browser-access':'true'},
 body:JSON.stringify({model:'claude-sonnet-4-6',max_tokens:600,system:'You are the assistant inside HotSync, a wildfire hot-spot explorer that harmonizes MODIS and VIIRS detections. Answer in under 120 words using CONTEXT. If the data is a synthetic demo, say so when it matters. CONTEXT: '+JSON.stringify(ctx()),messages:CH.slice(-12)})});
 const j=await r.json();if(!r.ok)throw Error(j.error&&j.error.message||r.status);return j.content.map(c=>c.text||'').join('')}
async function send(q){q=q.trim();if(!q)return;say('u',q);
 if($('#aik').value.trim()){CH.push({role:'user',content:q});const w=say('b','…');
  try{const a=await claude();CH.push({role:'assistant',content:a});w.textContent=a}catch(e){CH.pop();w.textContent='Claude request failed ('+e.message+'). Local answer:\n'+local(q)}return}
 say('b',local(q))}
$('#cb').onclick=()=>{const p=$('#cp');p.hidden=!p.hidden;if(!p.hidden){if(!cl.children.length)say('b','Hi! Ask me about the selected area: summary, unusual days, burning windows, FRP, or say "show Greece" to jump there.');$('#ci').focus()}};
$('#cx').onclick=()=>$('#cp').hidden=true;
$('#cg').onclick=()=>{const i=$('#ci');send(i.value);i.value=''};$('#ci').onkeydown=e=>{if(e.key==='Enter')$('#cg').click()};
$('#cs').onclick=e=>{const b=e.target.closest('button');if(b)send(b.dataset.q)};
$('#demo').click();
/* ---- Detailed coastlines and borders (falls back to the built-in outline if offline) ---- */
function topo(t){const[sx,sy]=t.transform.scale,[tx,ty]=t.transform.translate,arcs=t.arcs.map(a=>{let x=0,y=0;return a.map(([dx,dy])=>[(x+=dx)*sx+tx,(y+=dy)*sy+ty])}),
 ring=r=>{const o=[];for(const i of r){const a=i<0?arcs[~i].slice().reverse():arcs[i];for(const p of o.length?a.slice(1):a)o.push(p[0],p[1])}return o},out=[];
 for(const g of t.objects.countries.geometries){if(g.type==='Polygon')g.arcs.forEach(r=>out.push(ring(r)));else if(g.type==='MultiPolygon')g.arcs.forEach(p=>p.forEach(r=>out.push(ring(r))))}return out}
(async()=>{for(const u of['110m','50m']){try{const r=topo(await(await fetch(`https://cdn.jsdelivr.net/npm/world-atlas@2.0.2/countries-${u}.json`)).json());
 if(u==='110m'){LAND=r;LC=[];DRAW=r;heat()}else DRAW=r;BK='';drawMap()}catch(e){break}}})();
