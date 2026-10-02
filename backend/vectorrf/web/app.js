const $ = (s, r=document) => r.querySelector(s);
const $$ = (s, r=document) => [...r.querySelectorAll(s)];

const state = {
  mode: 'tx',
  catalog: null,
  ws: null,
  wsReady: false,
  live: {
    psd: [], timeI: [], timeQ: [], constI: [], constQ: [], waterfallRow: null,
    metrics: {}, frames: 0, lastAt: 0,
  },
  trace: { mode: 'clear', max: null, min: null },
  txFamily: 'linear',
  tx: {
    waveform:'qpsk', sample_rate:1_000_000, duration:0.06, amplitude:1, power_dbm:0, center_hz:0, snr_db:30, seed:7,
    message:'VECTORRF LAB', pn_bits:4096, symbol_rate:50_000, rolloff:0.35, filter_span_symbols:10,
    fsk_deviation_hz:25_000, phase_offset_deg:0, message_freq_hz:5_000, modulation_index:0.6,
    fm_deviation_hz:25_000, pm_deviation_rad:1.0, sideband:'usb', tones_hz:[-80_000,45_000,130_000],
    chirp_start_hz:-100_000, chirp_stop_hz:100_000, ofdm_fft:64, ofdm_cp:16, ofdm_occupied:52,
  },
  spectrum: { span_hz:1_000_000, rbw_hz:2000, window:'hann', detector:'average', averaging:2, max_fft:32768 },
  radar: {
    mode:'sequence', segments:[{duration_us:25,start_hz:-100000,stop_hz:100000,amplitude:1,phase_deg:0},{duration_us:25,start_hz:0,stop_hz:0,amplitude:0,phase_deg:0}], sample_rate:1_000_000, duration_s:0.02, seed:7,
    continuous_waveform:'fmcw', center_offset_hz:0, sweep_bw_hz:200_000, sweep_period_s:250e-6,
    sweep_shape:'triangle', pmcw_chip_rate:250_000, pmcw_chips:127,
    stepped_start_hz:-100_000, stepped_stop_hz:100_000, stepped_points:16, stepped_dwell_s:20e-6,
    sequence:[
      {kind:'pulse',name:'LFM burst',repeat:8,pri_s:150e-6,off_time_s:100e-6,pri_jitter_fraction:0,pri_wobble_fraction:0,pri_wobble_period:16,scale:1,frequency_offset_hz:0,phase_offset_deg:0,
       pulse:{name:'LFM burst',waveform:'lfm',width_s:30e-6,amplitude:1,frequency_offset_hz:0,phase_offset_deg:0,envelope:'rect',sweep_bw_hz:250_000,nlfm_exponent:3,barker_length:13,polyphase_order:4,phase_code_chips:32,zc_root:5,custom_phase_deg:[0,180,0,0,180,0,180],costas_hops:7,stepped_steps:8,stepped_spacing_hz:20_000}},
      {kind:'off',name:'Off time',repeat:1,pri_s:100e-6,off_time_s:400e-6,pri_jitter_fraction:0,pri_wobble_fraction:0,pri_wobble_period:16,scale:1,frequency_offset_hz:0,phase_offset_deg:0,pulse:null},
    ],
  },
  display: { fps:30 },
  radarDisplay: { window_fraction:0.28, playback_speed:1.0 },
  radarSelected: 0,
  rx: {modulation:'qpsk',sample_rate:1_000_000,symbol_rate:50_000,rolloff:0.35,timing_phase:0,timing_recovery:'mueller-muller',timing_gain:0.30,manual_phase_deg:0,fsk_deviation_hz:25_000,fm_deviation_hz:25_000,use_costas:true,costas:{order:4,alpha:0.03,beta:0.0005},frame_enabled:false,frame_blocks:[],energy_detection:true,energy_threshold_dbm:-80,energy_window_symbols:1,preamble_correlation:true,preamble_threshold:.65},
  channel: {snr_db:20,freq_offset_hz:1200,phase_offset_deg:20,timing_offset_samples:0.15,iq_gain_db:0,iq_phase_deg:0,dc_i:0,dc_q:0},
  rxDisplay: {
    zero_span:'iq', error_trace:'evm', loop_trace:'frequency',
    sync_speed:1.0, sync_window:180, sync_pause:false,
  },
  rxData:null,
  blindData:null,
  radarData:null,
  timers: {poll:null,debounce:null,radarAnim:null,rxAnim:null,paint:null},
  busy: {rx:false,blind:false,radar:false},
  pending: {rx:false,blind:false,radar:false},
  refreshCounter:0,
};

const workspaces = [
  ['tx','Transmitter'], ['spectrum','Spectrum'], ['rx','Receiver'], ['blind','Classifier'], ['radar','Radar Studio'],
];

const signalFamilies = {
  fundamentals:['tone','multitone','awgn','chirp'],
  linear:['bpsk','qpsk','dbpsk','dqpsk','pi4-dqpsk','oqpsk','8psk','16qam','64qam','16apsk','32apsk'],
  cpfsk:['2fsk','4fsk','msk','gmsk'],
  multicarrier:['ofdm'],
  analog:['am','fm','pm','ssb','dsb-sc'],
};
const familyLabels = {fundamentals:'Basic',linear:'PSK / QAM / APSK',cpfsk:'FSK / MSK',multicarrier:'OFDM',analog:'Analog'};

function esc(s=''){return String(s).replace(/[&<>"']/g,m=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[m]))}
function fmt(v,unit=''){
  if(v===null||v===undefined||!Number.isFinite(Number(v))) return '—';
  v=Number(v); const a=Math.abs(v);
  if(a>=1e9)return `${(v/1e9).toFixed(3)} G${unit}`;
  if(a>=1e6)return `${(v/1e6).toFixed(3)} M${unit}`;
  if(a>=1e3)return `${(v/1e3).toFixed(3)} k${unit}`;
  if(a!==0&&a<1e-3)return `${v.toExponential(2)} ${unit}`.trim();
  return `${v.toFixed(a<10?3:a<100?2:1)} ${unit}`.trim();
}
function toast(msg){const t=$('#toast');t.textContent=msg;t.classList.add('show');clearTimeout(toast._t);toast._t=setTimeout(()=>t.classList.remove('show'),1800)}
function setStatus(msg){const el=$('#statusMessage');if(el)el.textContent=String(msg).toUpperCase()}
function setSourceStatus(source,fs){$('#statusSource').textContent=`SOURCE ${String(source).toUpperCase()}`;$('#statusFs').textContent=`FS ${fmt(fs,'S/s')}`}
function updateRate(){const now=performance.now();state.rateCount=(state.rateCount||0)+1;if(!state.rateAt)state.rateAt=now;if(now-state.rateAt>=1000){$('#statusRate').textContent=`DISPLAY ${(state.rateCount*1000/(now-state.rateAt)).toFixed(1)} Hz`;state.rateCount=0;state.rateAt=now;}}


async function api(path, options={}){const r=await fetch(path,options);if(!r.ok)throw new Error(await r.text());const ct=r.headers.get('content-type')||'';return ct.includes('json')?r.json():r.blob()}
function jsonPost(path,obj){return api(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(obj)})}
function setPath(obj,path,value){const parts=path.split('.');let cur=obj;for(let i=0;i<parts.length-1;i++){const k=/^\d+$/.test(parts[i])?Number(parts[i]):parts[i];cur=cur[k]}const last=/^\d+$/.test(parts.at(-1))?Number(parts.at(-1)):parts.at(-1);cur[last]=value}
function getPath(obj,path){return path.split('.').reduce((a,k)=>a?.[/^\d+$/.test(k)?Number(k):k],obj)}

function renderTabs(){
  $('#workspaceTabs').innerHTML=workspaces.map(([id,label])=>`<button class="workspace-tab ${state.mode===id?'active':''} ${id==='radar'?'workspace-separate':''}" data-mode="${id}">${label}</button>`).join('');
  $$('.workspace-tab').forEach(b=>b.onclick=()=>switchMode(b.dataset.mode));
}
function switchMode(mode){state.mode=mode;stopModeTimers();renderTabs();renderWorkspace();startMode()}

function field(label,path,value,opts={}) {
  let {type='number',step='any',min='',max='',placeholder='',clamp=false,clampMessage=''}=opts;
  let scale=1;
  if(type==='number' && path.startsWith('radar.') && /\.(width_s|pri_s|off_time_s|sweep_period_s|stepped_dwell_s)$/.test(path)){scale=1e6;label+=' (µs)';}
  if(type==='number' && path==='radar.duration_s'){scale=1e3;label+=' (ms)';}
  if(type==='number' && path==='radar.sample_rate')label+=' (S/s)';
  if(type==='number' && path.startsWith('radar.') && /_hz$/.test(path))label+=' (Hz)';
  if(scale!==1){value=Number(value)*scale;if(min!=='')min*=scale;if(max!=='')max*=scale;if(step!=='any')step*=scale;}
  const htmlType=type==='string'?'text':type;
  // The increment is a convenience for the custom stepper, not an HTML validity rule.
  // step="any" means a user may type every physically valid value instead of being
  // forced onto 100/1000/etc. lattices chosen only for arrow-button ergonomics.
  const increment=type==='number'&&step!=='any'&&Number.isFinite(Number(step))?Number(step):'';
  return `<div class="field"><label>${label}</label><input aria-label="${esc(label)}" data-bind="${path}" data-scale="${scale}" data-kind="${type}" ${increment!==''?`data-increment="${increment}"`:''} ${clamp?'data-clamp="1"':''} ${clampMessage?`data-clamp-message="${esc(clampMessage)}"`:''} type="${htmlType}" value="${esc(value??'')}" ${htmlType==='number'?'step="any" inputmode="decimal"':''} ${placeholder?`placeholder="${esc(placeholder)}"`:''} ${min!==''?`min="${min}"`:''} ${max!==''?`max="${max}"`:''}></div>`;
}

function numberListEditor(label,path,values=[],opts={}){
  const {min='',max='',increment='any',unit='',help='',empty='No explicit values — automatic/default behavior is active.',minLength=0}=opts;
  const rows=(values||[]).map((v,i)=>`<div class="structured-row"><span class="structured-index">${i+1}</span><input aria-label="${esc(label)} ${i+1}" type="number" inputmode="decimal" step="any" data-list-path="${path}" data-list-index="${i}" ${increment!=='any'?`data-increment="${increment}"`:''} value="${esc(v)}" ${min!==''?`min="${min}"`:''} ${max!==''?`max="${max}"`:''}><span class="structured-unit">${esc(unit)}</span><button type="button" class="icon-btn remove-value" data-list-remove="${path}" data-list-index="${i}" aria-label="Remove ${esc(label)} ${i+1}">×</button></div>`).join('');
  return `<div class="structured-editor" data-list-editor="${path}"><div class="structured-heading"><div><strong>${esc(label)}</strong>${help?`<small>${esc(help)}</small>`:''}</div><button type="button" class="btn small" data-list-add="${path}" data-list-default="${esc(opts.defaultValue??0)}">+ Add value</button></div><div class="structured-list">${rows||`<div class="structured-empty">${esc(empty)}</div>`}</div><input type="hidden" data-list-min-length="${minLength}"></div>`;
}
function bindListEditors(root,change,rerender){
  $$('[data-list-path]',root).forEach(el=>el.onchange=()=>{const a=getPath(state,el.dataset.listPath);const v=Number(el.value);if(!Number.isFinite(v)){el.setCustomValidity('Enter a finite number');el.reportValidity();return;}el.setCustomValidity('');a[Number(el.dataset.listIndex)]=v;change(el.dataset.listPath);});
  $$('[data-list-add]',root).forEach(b=>b.onclick=()=>{const a=getPath(state,b.dataset.listAdd);a.push(Number(b.dataset.listDefault)||0);change(b.dataset.listAdd);rerender?.();});
  $$('[data-list-remove]',root).forEach(b=>b.onclick=()=>{const editor=b.closest('[data-list-editor]'),minLength=Number($('[data-list-min-length]',editor)?.dataset.listMinLength||0),a=getPath(state,b.dataset.listRemove);if(a.length<=minLength)return toast(`Keep at least ${minLength} value${minLength===1?'':'s'}`);a.splice(Number(b.dataset.listIndex),1);change(b.dataset.listRemove);rerender?.();});
}

function sliderField(label,path,value,opts={}){
  const {step=.01,min=0,max=1,suffix=''}=opts;
  return `<div class="slider-field"><div><label>${label}</label><span class="slider-entry"><input aria-label="${esc(label)} exact value" data-bind="${path}" data-kind="number" data-increment="${step}" type="number" inputmode="decimal" step="any" value="${value}" min="${min}" max="${max}"><em>${esc(suffix)}</em></span></div><input class="slider-range" aria-label="${esc(label)} slider" data-bind="${path}" data-kind="range" type="range" value="${value}" step="${step}" min="${min}" max="${max}"></div>`;
}
function selectField(label,path,value,items){return `<div class="field"><label>${label}</label><select data-bind="${path}" data-kind="${typeof value==='number'?'number':'string'}">${items.map(x=>{const[v,l]=Array.isArray(x)?x:[x,x];return`<option value="${esc(v)}" ${String(v)===String(value)?'selected':''}>${esc(l)}</option>`}).join('')}</select></div>`}
function checkField(label,path,value){return `<label class="checkbox-row"><input type="checkbox" data-bind="${path}" data-kind="bool" ${value?'checked':''}><span>${label}</span></label>`}
function group(title,body,cls=''){return `<section class="control-group ${cls}"><div class="group-title">${title}</div>${body}</section>`}
function measure(label,id,val='—'){return `<div class="measure"><span>${label}</span><b id="${id}">${val}</b></div>`}
function plotPane(label,id,cls=''){return `<div class="plot-pane ${cls}"><div class="plot-label">${label}</div><div class="plot-value" id="${id}Value"></div><canvas id="${id}"></canvas></div>`}
function dock(title,body){return `<aside class="control-dock"><div class="dock-head"><strong>${title}</strong><div class="live-indicator"><i></i> LIVE</div></div><div class="dock-scroll">${body}</div></aside>`}
function layout(title,controls,measurements,plots,plotClass){return `<div class="instrument-layout">${dock(title,controls)}<section class="display-area"><div class="measurebar">${measurements}</div><div class="plot-grid ${plotClass}">${plots}</div></section></div>`}
function sourceReadout() {
  return `<div class="source-readout"><div><span>Source</span><b>TRANSMITTER · internal record</b></div><div><span>Waveform</span><b>${esc(state.tx.waveform.toUpperCase())}</b></div><div><span>Sample rate</span><b>${fmt(state.tx.sample_rate,'S/s')}</b></div><div><span>Revision</span><b>${state.sourceRevision||'connecting'}</b></div></div>`;
}

function signalChain(){
  const digital=signalFamilies.linear.includes(state.tx.waveform);
  const blocks=[
    ['SOURCE',['linear','cpfsk','multicarrier'].includes(state.txFamily)?state.tx.data_source.toUpperCase():state.tx.waveform.toUpperCase(),true],
    ['MODULATION',state.tx.waveform.toUpperCase(),true],
    ['SHAPING',digital?(state.tx.pulse_shape==='rrc'?`RRC β ${Number(state.tx.rolloff).toFixed(2)}`:'RECT'):'DIRECT',digital],
    ['RATE',fmt(state.tx.sample_rate,'S/s'),true],
    ['IMPAIR',state.tx.snr_db==null||state.tx.snr_db>=99.9?'CLEAN':`AWGN ${Number(state.tx.snr_db).toFixed(0)} dB`,state.tx.snr_db!=null&&state.tx.snr_db<99.9],
    ['OUTPUT','LIVE',true],
  ];
  return `<div class="signal-chain">${blocks.map(([k,v,on],i)=>`${i?'<span class="chain-arrow">›</span>':''}<div class="chain-block ${on?'on':''}"><span>${k}</span><b>${esc(v)}</b></div>`).join('')}</div>`;
}

function bindControls(root=$('#workspace'), onChange=scheduleCurrent){
  $$('[data-bind]',root).forEach(el=>{
    const handler=()=>{
      if(el.dataset.clamp==='1'&&el.type==='number'){
        const raw=Number(el.value);if(!Number.isFinite(raw)){el.setCustomValidity('Enter a finite number');el.reportValidity();return;}
        let bounded=raw;if(el.min!=='')bounded=Math.max(bounded,Number(el.min));if(el.max!=='')bounded=Math.min(bounded,Number(el.max));
        if(bounded!==raw){el.value=String(bounded);toast(el.dataset.clampMessage||`Value limited to ${bounded}`);}
        el.setCustomValidity('');
      }
      if(el.checkValidity && !el.checkValidity()){el.reportValidity();return;}
      let v;
      if(el.dataset.kind==='bool')v=el.checked;
      else if(el.dataset.kind==='string')v=el.value;
      else v=Number(el.value)/(Number(el.dataset.scale)||1);
      setPath(state,el.dataset.bind,v);
      $$('[data-bind]',root).forEach(peer=>{if(peer!==el&&peer.dataset.bind===el.dataset.bind){const scale=Number(peer.dataset.scale)||1;peer.value=Number(v)*scale;}});
      const out=$(`[data-output="${el.dataset.bind}"]`,root);if(out)out.textContent=String(v);
      onChange(el.dataset.bind,v,el);
    };
    const event=(el.tagName==='TEXTAREA'||el.tagName==='SELECT'||el.type==='checkbox'||el.type==='number'||el.type==='text')?'change':'input';
    el.addEventListener(event,handler);
  });
}

function renderWorkspace(){
  if(state.mode==='tx')renderTx();
  else if(state.mode==='radar')renderRadar();
  else if(state.mode==='rx')renderRx();
  else if(state.mode==='spectrum')renderSpectrum();
  else renderBlind();
}

function txSpecificControls(){
  const w=state.tx.waveform;let out='';
  if(['bpsk','qpsk','8psk','16qam','64qam'].includes(w)){
    out+=field('Symbol rate','tx.symbol_rate',state.tx.symbol_rate,{step:100,min:10});
    out+=field('RRC roll-off','tx.rolloff',state.tx.rolloff,{step:.01,min:0,max:1});
    out+=field('Filter span','tx.filter_span_symbols',state.tx.filter_span_symbols,{step:1,min:2,max:64});
    out+=field('PN bits','tx.pn_bits',state.tx.pn_bits,{step:32,min:32});
    out+=field('Message','tx.message',state.tx.message,{type:'string'});
  }else if(['2fsk','4fsk','msk'].includes(w)){
    out+=field('Symbol rate','tx.symbol_rate',state.tx.symbol_rate,{step:100,min:10});
    if(w!=='msk')out+=field('Deviation','tx.fsk_deviation_hz',state.tx.fsk_deviation_hz,{step:100,min:1});
    out+=field('PN bits','tx.pn_bits',state.tx.pn_bits,{step:32,min:32});
  }else if(w==='ofdm'){
    out+=field('FFT size','tx.ofdm_fft',state.tx.ofdm_fft,{step:1,min:16});
    out+=field('Cyclic prefix','tx.ofdm_cp',state.tx.ofdm_cp,{step:1,min:0});
    out+=field('Occupied carriers','tx.ofdm_occupied',state.tx.ofdm_occupied,{step:2,min:4});
    out+=field('PN bits','tx.pn_bits',state.tx.pn_bits,{step:32,min:32});
  }else if(['am','fm','pm','ssb','dsb-sc'].includes(w)){
    out+=field('Message tone','tx.message_freq_hz',state.tx.message_freq_hz,{step:100,min:1});
    if(w==='am')out+=field('AM index','tx.modulation_index',state.tx.modulation_index,{step:.01,min:0,max:1});
    if(w==='fm')out+=field('FM deviation','tx.fm_deviation_hz',state.tx.fm_deviation_hz,{step:100,min:0});
    if(w==='pm')out+=field('PM deviation rad','tx.pm_deviation_rad',state.tx.pm_deviation_rad,{step:.05,min:0});
    if(w==='ssb')out+=selectField('Sideband','tx.sideband',state.tx.sideband,[['usb','USB'],['lsb','LSB']]);
  }else if(w==='chirp'){
    out+=field('Start frequency','tx.chirp_start_hz',state.tx.chirp_start_hz,{step:100});
    out+=field('Stop frequency','tx.chirp_stop_hz',state.tx.chirp_stop_hz,{step:100});
  }else if(w==='multitone'){
    out+=numberListEditor('Tone frequencies','tx.tones_hz',state.tx.tones_hz,{increment:100,unit:'Hz',defaultValue:1000,minLength:1,help:'Add, remove, or type any finite tone frequency.'});
  }
  return out||`<div class="notice">This source has no additional modulation parameters.</div>`;
}


const frameModulations=['bpsk','qpsk','dbpsk','dqpsk','pi4-dqpsk','oqpsk','8psk','16qam','64qam','16apsk','32apsk','2fsk','4fsk','msk','gmsk','ofdm'];
const evmQualityModulations=new Set(['bpsk','qpsk','dbpsk','dqpsk','pi4-dqpsk','oqpsk','8psk','16qam','64qam','16apsk','32apsk','ofdm']);
function defaultFrameBlock(role='payload',mod=state.tx.waveform,symbolRate=state.tx.symbol_rate){
  if(!frameModulations.includes(mod))mod='qpsk';
  return {name:role==='preamble'?'Preamble':'Payload',role,modulation:mod,bit_count:role==='preamble'?32:256,bits:role==='preamble'?'10101010101010101101001110010001':'11010010',symbol_rate:Number(symbolRate)||50000,power_offset_db:0,repeat:1};
}
function frameBlockEditor(path,blocks){
  const rows=(blocks||[]).map((b,i)=>fold(`${i+1} · ${esc(b.name||b.role||'Block')}`,
    field('Name',`${path}.${i}.name`,b.name||`Block ${i+1}`,{type:'string'})+
    selectField('Role',`${path}.${i}.role`,b.role||'payload',[['preamble','Preamble / sync'],['payload','Payload source'],['bits','Exact bit pattern'],['pn','Random PN bits']])+
    selectField('Modulation',`${path}.${i}.modulation`,b.modulation||'qpsk',frameModulations)+
    field('Bits in block',`${path}.${i}.bit_count`,b.bit_count??128,{min:1,max:65536,step:1})+
    ((b.role==='preamble'||b.role==='bits')?field('Bit pattern',`${path}.${i}.bits`,b.bits||'1010',{type:'string'}):'')+
    field('Symbol rate (Bd)',`${path}.${i}.symbol_rate`,b.symbol_rate??state.tx.symbol_rate,{min:10,step:100})+
    field('Power offset (dB)',`${path}.${i}.power_offset_db`,b.power_offset_db??0,{min:-120,max:120,step:.5})+
    field('Repeat',`${path}.${i}.repeat`,b.repeat??1,{min:1,max:1024,step:1})+
    `<div class="block-actions"><button class="btn small" data-frame-up="${i}" data-frame-path="${path}">↑</button><button class="btn small" data-frame-down="${i}" data-frame-path="${path}">↓</button><button class="btn small" data-frame-remove="${i}" data-frame-path="${path}">Remove</button></div>`,i===0)).join('');
  return `${rows||'<p class="hint">No blocks yet.</p>'}<div class="block-actions"><button class="btn" data-frame-add="preamble" data-frame-path="${path}">+ Preamble</button><button class="btn" data-frame-add="payload" data-frame-path="${path}">+ Payload</button><button class="btn" data-frame-add="bits" data-frame-path="${path}">+ Bit block</button></div><p class="hint">Blocks are transmitted in order and may use different digital modulations, symbol rates, lengths and relative powers. The whole generic frame repeats to fill the capture.</p>`;
}
function bindFrameBlockButtons(root,rerender,onChange=scheduleCurrent){
  $$('[data-frame-add]',root).forEach(b=>b.onclick=()=>{const a=getPath(state,b.dataset.framePath);if(a.length>=64)return toast('Frame block limit is 64');a.push(defaultFrameBlock(b.dataset.frameAdd,b.dataset.framePath.startsWith('rx.')?state.rx.modulation:state.tx.waveform,b.dataset.framePath.startsWith('rx.')?state.rx.symbol_rate:state.tx.symbol_rate));rerender();onChange(b.dataset.framePath);});
  $$('[data-frame-remove]',root).forEach(b=>b.onclick=()=>{const a=getPath(state,b.dataset.framePath);a.splice(Number(b.dataset.frameRemove),1);rerender();onChange(b.dataset.framePath);});
  $$('[data-frame-up],[data-frame-down]',root).forEach(b=>b.onclick=()=>{const a=getPath(state,b.dataset.framePath),i=Number(b.dataset.frameUp??b.dataset.frameDown),d=b.hasAttribute('data-frame-up')?-1:1,j=i+d;if(j<0||j>=a.length)return;[a[i],a[j]]=[a[j],a[i]];rerender();onChange(b.dataset.framePath);});
}

function renderTx() {
  const familyTabs=selectField('Family','txFamily',state.txFamily,Object.entries(familyLabels));
  const signals=`<div class="signal-grid">${signalFamilies[state.txFamily].map(w=>`<button data-waveform="${w}" class="${w===state.tx.waveform?'active':''}">${w.toUpperCase()}</button>`).join('')}</div>`;
  const controls=group('Waveform',familyTabs+signals)+group('Output',field('Sample rate (S/s)','tx.sample_rate',state.tx.sample_rate,{min:2000,max:1e7,step:1000})+field('Transmit power (dBm)','tx.power_dbm',state.tx.power_dbm,{step:1})+field('Frequency offset (Hz)','tx.center_hz',state.tx.center_hz,{step:100})+field('Phase (°)','tx.phase_offset_deg',state.tx.phase_offset_deg,{step:1})+`<p class="hint">Transmit power is an absolute simulator calibration with no UI clamp. A bin-centered CW marker reads the configured dBm value; modulated signals distribute the same average power across spectrum bins.</p>`)+group('Routing',`<div class="routing">Transmitter → Spectrum<br>Transmitter → Receiver<br>Transmitter → Classifier</div><p class="hint">Advancing simulation captures with reproducible frame identities. Radar Studio has its own signal engine.</p>`);
  const digital=['linear','cpfsk','multicarrier'].includes(state.txFamily);
  let editor='';
  if(state.txPage==='data') {
    editor=digital?selectField('Payload source','tx.data_source',state.tx.data_source,[['pn','PN / PRBS'],['text','Text · UTF-8'],['bits','Raw bits']]):'<p class="hint">This waveform does not use a digital payload.</p>';
    if(digital && state.tx.data_source==='pn') editor+=selectField('PRBS order','tx.pn_order',state.tx.pn_order,[7,9,11,15].map(n=>[n,`PN${n} · period ${2**n-1}`]))+field('Repeat length (bits)','tx.pn_bits',state.tx.pn_bits,{min:32,max:1e7,step:1})+field('Seed','tx.seed',state.tx.seed,{step:1})+'<p class="hint">The nonzero PRBS state is derived from this seed. The pattern repeats at the requested length, capped at its natural period.</p>';
    if(digital && state.tx.data_source==='text') editor+=field('Message','tx.message',state.tx.message,{type:'string'})+'<p class="hint">UTF-8 bytes, most significant bit first. The message repeats to fill the record.</p>';
    if(digital && state.tx.data_source==='bits') editor+=field('Bit pattern','tx.custom_bits',state.tx.custom_bits,{type:'string'})+'<p class="hint">Only 0 and 1 are accepted. The pattern repeats to fill the record.</p>';
    if(digital){
      let framing=checkField('Enable frame builder','tx.frame_enabled',state.tx.frame_enabled);
      if(state.tx.frame_enabled){
        framing+=selectField('Frame model','tx.frame_mode',state.tx.frame_mode,[['simple','Simple · preamble + payload + CRC'],['blocks','Block frame · mixed modulation']]);
        if(state.tx.frame_mode==='simple')framing+=field('Preamble (bits)','tx.preamble_bits',state.tx.preamble_bits,{type:'string'})+field('Payload length (bits)','tx.payload_bits',state.tx.payload_bits,{min:8,max:65536,step:8})+checkField('Append CRC-16/CCITT-FALSE','tx.frame_crc',state.tx.frame_crc)+`<p class="hint">Preamble → payload → optional CRC-16/CCITT-FALSE over payload.</p>`;
        else framing+=frameBlockEditor('tx.frame_blocks',state.tx.frame_blocks);
      }
      editor+=fold('Frame builder',framing,state.tx.frame_enabled);
    }
  } else if(state.txPage==='modulation') {
    editor=txSpecificControls().replace(/<div class="field"><label>(RRC roll-off|Filter span)[\s\S]*?<\/div>/g,'').replace(/<div class="field"><label>PN bits[\s\S]*?<\/div>/g,'').replace(/<div class="field"><label>Message<\/label>[\s\S]*?<\/div>/g,'');
    if(state.txFamily==='linear' && !['bpsk','qpsk','8psk','16qam','64qam'].includes(state.tx.waveform)) editor=field('Symbol rate (Bd)','tx.symbol_rate',state.tx.symbol_rate,{min:10,step:100});
    if(['16apsk','32apsk'].includes(state.tx.waveform)) editor+=field('Ring 2 / ring 1 radius','tx.apsk_ratio2',state.tx.apsk_ratio2,{min:1.01,max:8,step:.01})+(state.tx.waveform==='32apsk'?field('Ring 3 / ring 1 radius','tx.apsk_ratio3',state.tx.apsk_ratio3,{min:1.01,max:12,step:.01}):'')+'<p class="hint">Natural bit labels; 16-APSK uses 4 + 12 points, 32-APSK uses 4 + 12 + 16. This is a configurable constellation, not a DVB framing implementation.</p>';
    if(state.tx.waveform==='gmsk') editor=field('Symbol rate (Bd)','tx.symbol_rate',state.tx.symbol_rate,{min:10,step:100})+sliderField('Gaussian BT','tx.gaussian_bt',state.tx.gaussian_bt,{min:.1,max:1,step:.01});
    if(['dbpsk','dqpsk','pi4-dqpsk','oqpsk'].includes(state.tx.waveform)) editor+=`<p class="hint">${{dbpsk:'Differential binary phase increments.',dqpsk:'Differential quadrature phase increments in multiples of π/2.','pi4-dqpsk':'Differential ±π/4 and ±3π/4 transitions; alternating constellation orientation.',oqpsk:'Q is delayed by half a symbol, including fractional delay at odd sample ratios.'}[state.tx.waveform]}</p>`;
  } else if(state.txPage==='shaping') {
    editor=state.txFamily==='linear'?selectField('Pulse shape','tx.pulse_shape',state.tx.pulse_shape,[['rrc','Root raised cosine'],['rect','Rectangular']])+(state.tx.pulse_shape==='rrc'?sliderField('Rolloff β','tx.rolloff',state.tx.rolloff,{min:0,max:1,step:.01})+field('Filter span (symbols)','tx.filter_span_symbols',state.tx.filter_span_symbols,{min:2,max:64,step:1}):''):'<p class="hint">Symbol-pulse filtering controls transitions and occupied bandwidth. PSK/QAM/APSK use rectangular or root-raised-cosine pulses; GMSK uses Gaussian frequency-pulse smoothing.</p>';
  } else if(state.txPage==='impairments') {
    editor=(state.tx.waveform==='awgn'?'':checkField('Add AWGN','txNoise',state.txNoise))+(state.txNoise&&state.tx.waveform!=='awgn'?sliderField('SNR (dB)','tx.snr_db',state.tx.snr_db??30,{min:-10,max:60,step:1}):'')+sliderField('I/Q gain imbalance (dB)','tx.iq_gain_db',state.tx.iq_gain_db,{min:-6,max:6,step:.1})+sliderField('Quadrature error (°)','tx.iq_phase_deg',state.tx.iq_phase_deg,{min:-30,max:30,step:1})+field('DC I','tx.dc_i',state.tx.dc_i,{min:-1,max:1,step:.01})+field('DC Q','tx.dc_q',state.tx.dc_q,{min:-1,max:1,step:.01});
    if(state.tx.waveform==='awgn')editor+=field('Noise bandwidth (Hz)','tx.noise_bandwidth_hz',state.tx.noise_bandwidth_hz??state.tx.sample_rate*.8,{min:10,step:1000});
  } else {
    editor=checkField('Burst gating','tx.burst_enabled',state.tx.burst_enabled)+(state.tx.burst_enabled?field('Period (s)','tx.burst_period_s',state.tx.burst_period_s,{min:.00001,step:.001})+sliderField('Duty fraction','tx.burst_duty',state.tx.burst_duty,{min:.01,max:1,step:.01}):'')+'<p class="hint">Repeating baseband gating for burst experiments. This does not add a protocol preamble or packet decoder.</p>';
  }
  const actualRate=state.tx.waveform==='ofdm'?state.tx.sample_rate/(state.tx.ofdm_fft+(state.tx.ofdm_design?.cp_lengths?.length?state.tx.ofdm_design.cp_lengths.reduce((a,b)=>a+b,0)/state.tx.ofdm_design.cp_lengths.length:state.tx.ofdm_cp)):state.tx.sample_rate/Math.max(2,Math.round(state.tx.sample_rate/state.tx.symbol_rate));
  const metrics=measure('Waveform','mWave',state.tx.waveform.toUpperCase())+measure('Actual symbol rate','txActual',digital?fmt(actualRate,'Bd'):'—')+measure('Transport','mStream','connecting');
  $('#workspace').innerHTML=layout('Transmitter',controls,metrics,`<div class="editor-surface"><div class="editor-heading"><span>TRANSMITTER SETUP</span><h1>${esc(state.tx.waveform.toUpperCase())}</h1><p>Configure the signal here. Inspect its behavior in Spectrum and Receiver.</p></div>${subnav([['data','Data'],['modulation','Modulation'],['shaping','Pulse shaping / filters'],['impairments','Impairments'],['burst','Burst']],state.txPage,'txpage')}<div class="parameter-editor">${editor}</div><div class="configuration-summary">${signalChain()}<div id="txSummary" class="hint">Waiting for transmitter record…</div></div></div>`,'transmitter');
  bindControls($('#workspace'),path=>{
    if(path==='txFamily'){state.tx.waveform=signalFamilies[state.txFamily][0];renderTx();scheduleCurrent('tx.waveform');return;}
    if(path==='txNoise'){state.tx.snr_db=state.txNoise?30:null;renderTx();scheduleCurrent('tx.snr_db');return;}
    if(path==='tx.frame_mode'&&state.tx.frame_mode==='blocks'&&!state.tx.frame_blocks.length){state.tx.frame_blocks=[defaultFrameBlock('preamble'),defaultFrameBlock('payload')];}
    if(['tx.normalize_output','tx.data_source','tx.pulse_shape','tx.burst_enabled','tx.frame_enabled','tx.frame_mode'].includes(path)||path.includes('.role'))renderTx();
    scheduleCurrent(path);
  });
  bindFrameBlockButtons($('#workspace'),renderTx,path=>scheduleCurrent(path));
  bindView([],v=>{state.txPage=v;renderTx();},'txpage');
  $$('[data-waveform]').forEach(b=>b.onclick=()=>{state.tx.waveform=b.dataset.waveform;renderTx();scheduleCurrent('tx.waveform');});
  bindListEditors($('#workspace'),path=>scheduleCurrent(path),renderTx);
  updateLiveMetrics();
}

function radarPulseEditorOriginal(item,idx){
  if(!item||item.kind==='off')return field('Off duration','radar.sequence.'+idx+'.off_time_s',item?.off_time_s||100e-6,{step:1e-6,min:1e-8})+field('Repeat','radar.sequence.'+idx+'.repeat',item?.repeat||1,{step:1,min:1});
  const p=item.pulse;let specific='';
  if(['lfm','nlfm','costas'].includes(p.waveform))specific+=field('Sweep bandwidth',`radar.sequence.${idx}.pulse.sweep_bw_hz`,p.sweep_bw_hz,{step:1000,min:1});
  if(p.waveform==='nlfm')specific+=field('NLFM exponent',`radar.sequence.${idx}.pulse.nlfm_exponent`,p.nlfm_exponent,{step:.1,min:1,max:9});
  if(p.waveform==='barker')specific+=selectField('Barker length',`radar.sequence.${idx}.pulse.barker_length`,p.barker_length,[2,3,4,5,7,11,13].map(x=>[x,String(x)]));
  if(['frank','p1','p2'].includes(p.waveform))specific+=field('Polyphase order',`radar.sequence.${idx}.pulse.polyphase_order`,p.polyphase_order,{step:1,min:2,max:32});
  if(['p3','p4','zadoff-chu'].includes(p.waveform))specific+=field('Code chips',`radar.sequence.${idx}.pulse.phase_code_chips`,p.phase_code_chips,{step:1,min:2});
  if(p.waveform==='zadoff-chu')specific+=field('ZC root',`radar.sequence.${idx}.pulse.zc_root`,p.zc_root,{step:1,min:1});
  if(p.waveform==='custom-phase')specific+=numberListEditor('Phase code',`radar.sequence.${idx}.pulse.custom_phase_deg`,p.custom_phase_deg,{increment:1,unit:'°',defaultValue:0,minLength:2,help:'One phase value per code chip.'});
  if(p.waveform==='costas')specific+=field('Costas hops',`radar.sequence.${idx}.pulse.costas_hops`,p.costas_hops,{step:1,min:3,max:64});
  if(p.waveform==='stepped-frequency')specific+=field('Frequency steps',`radar.sequence.${idx}.pulse.stepped_steps`,p.stepped_steps,{step:1,min:2})+field('Step spacing',`radar.sequence.${idx}.pulse.stepped_spacing_hz`,p.stepped_spacing_hz,{step:100,min:1});
  return selectField('Pulse waveform',`radar.sequence.${idx}.pulse.waveform`,p.waveform,['pulse','lfm','nlfm','barker','frank','p1','p2','p3','p4','zadoff-chu','custom-phase','costas','stepped-frequency'])+
    field('Pulse width',`radar.sequence.${idx}.pulse.width_s`,p.width_s,{step:1e-6,min:1e-8})+
    selectField('Envelope',`radar.sequence.${idx}.pulse.envelope`,p.envelope,['rect','gaussian','raised-cosine'])+
    field('Pulse amplitude',`radar.sequence.${idx}.pulse.amplitude`,p.amplitude,{step:.05,min:0,max:2})+
    field('Pulse freq offset',`radar.sequence.${idx}.pulse.frequency_offset_hz`,p.frequency_offset_hz,{step:100})+
    field('Pulse phase °',`radar.sequence.${idx}.pulse.phase_offset_deg`,p.phase_offset_deg,{step:1})+specific+
    field('Repeat',`radar.sequence.${idx}.repeat`,item.repeat,{step:1,min:1})+
    field('PRI',`radar.sequence.${idx}.pri_s`,item.pri_s,{step:1e-6,min:1e-8})+
    field('PRI jitter frac',`radar.sequence.${idx}.pri_jitter_fraction`,item.pri_jitter_fraction,{step:.01,min:0,max:.8})+
    field('PRI wobble frac',`radar.sequence.${idx}.pri_wobble_fraction`,item.pri_wobble_fraction,{step:.01,min:0,max:.8})+
    field('Wobble period',`radar.sequence.${idx}.pri_wobble_period`,item.pri_wobble_period,{step:1,min:2})+
    field('Item scale',`radar.sequence.${idx}.scale`,item.scale,{step:.05,min:0,max:2})+
    field('Item freq offset',`radar.sequence.${idx}.frequency_offset_hz`,item.frequency_offset_hz,{step:100})+
    field('Item phase °',`radar.sequence.${idx}.phase_offset_deg`,item.phase_offset_deg,{step:1});
}
function sequenceTable(){return `<div class="sequence-wrap"><div class="seq-toolbar"><button class="btn small" id="addPulse">+ Pulse</button><button class="btn small" id="addOff">+ Off</button><button class="btn small" id="seqUp">Up</button><button class="btn small" id="seqDown">Down</button><button class="btn small danger" id="seqDelete">Delete</button></div><table class="sequence-table"><thead><tr><th>#</th><th>Type</th><th>Name</th><th>Repeat</th></tr></thead><tbody>${state.radar.sequence.map((x,i)=>`<tr data-seq="${i}" class="${i===state.radarSelected?'selected':''}"><td>${i+1}</td><td>${x.kind}</td><td>${esc(x.name)}</td><td>${x.repeat}</td></tr>`).join('')}</tbody></table></div>`}
function renderRadarOriginal(){
  const modeTabs=`<div class="segmented"><button data-rmode="sequence" class="${state.radar.mode==='sequence'?'active':''}">Pulse sequence</button><button data-rmode="continuous" class="${state.radar.mode==='continuous'?'active':''}">Continuous wave</button></div>`;
  let editor='';
  if(state.radar.mode==='continuous'){
    editor=selectField('Waveform','radar.continuous_waveform',state.radar.continuous_waveform,[['cw','CW'],['fmcw','FMCW'],['pmcw','PMCW'],['stepped-cw','Stepped CW']])+field('Center offset','radar.center_offset_hz',state.radar.center_offset_hz,{step:100});
    if(state.radar.continuous_waveform==='fmcw')editor+=field('Sweep bandwidth','radar.sweep_bw_hz',state.radar.sweep_bw_hz,{step:1000,min:1})+field('Sweep period','radar.sweep_period_s',state.radar.sweep_period_s,{step:1e-6,min:1e-8})+selectField('Sweep shape','radar.sweep_shape',state.radar.sweep_shape,['up','down','triangle']);
    if(state.radar.continuous_waveform==='pmcw')editor+=field('Chip rate','radar.pmcw_chip_rate',state.radar.pmcw_chip_rate,{step:1000,min:10})+field('Code length','radar.pmcw_chips',state.radar.pmcw_chips,{step:1,min:2});
    if(state.radar.continuous_waveform==='stepped-cw')editor+=field('Start frequency','radar.stepped_start_hz',state.radar.stepped_start_hz,{step:100})+field('Stop frequency','radar.stepped_stop_hz',state.radar.stepped_stop_hz,{step:100})+field('Points','radar.stepped_points',state.radar.stepped_points,{step:1,min:2})+field('Dwell','radar.stepped_dwell_s',state.radar.stepped_dwell_s,{step:1e-6,min:1e-8});
  }else{
    const idx=Math.min(state.radarSelected,state.radar.sequence.length-1);state.radarSelected=Math.max(0,idx);const item=state.radar.sequence[state.radarSelected];
    editor=sequenceTable()+group('Selected sequence item',radarPulseEditor(item,state.radarSelected));
  }
  const controls=group('Radar mode',modeTabs)+
    group('Waveform record',field('Sample rate','radar.sample_rate',state.radar.sample_rate,{step:1000,min:1000,max:10_000_000})+field('Record length','radar.duration_s',state.radar.duration_s,{step:.005,min:.001,max:1}))+
    (state.radar.mode==='continuous'?group('Waveform',editor):editor)+
    group('Live display',sliderField('Window','radarDisplay.window_fraction',state.radarDisplay.window_fraction,{step:.01,min:.08,max:1})+sliderField('Playback','radarDisplay.playback_speed',state.radarDisplay.playback_speed,{step:.1,min:.2,max:4,suffix:'×'}));
  const metrics=measure('Mode','rMode',state.radar.mode)+measure('Active','rActive','—')+measure('Time','rTime','—')+measure('Peak','rPeak')+measure('Occupied BW','rBw')+measure('Analysis','rUpdate','starting');
  const plots=plotPane('Sequence / playback timeline','radarTimeline','radar-timeline')+plotPane('Envelope / pulse train','radarEnvelope','radar-envelope')+plotPane('Instantaneous frequency','radarFreq','radar-frequency')+plotPane('Phase / code','radarPhase','radar-phase')+plotPane('Spectrum','radarSpectrum','radar-spectrum')+plotPane('Pulse compression / autocorrelation','radarAuto','radar-autocorr')+plotPane('Delay–Doppler ambiguity','radarAmb','radar-ambiguity');
  $('#workspace').innerHTML=layout('Radar Studio',controls,metrics,plots,'radar');
  bindControls($('#workspace'),(path)=>{
    if(path.startsWith('radarDisplay.'))return;
    if(path.startsWith('rxDisplay.')){drawRxLive();drawRxAnalysis();return}
    if(path.endsWith('.waveform')||path==='radar.continuous_waveform'){renderRadar();scheduleCurrent(path)}else scheduleCurrent(path);
  });
  $$('[data-rmode]').forEach(b=>b.onclick=()=>{state.radar.mode=b.dataset.rmode;renderRadar();scheduleCurrent('radar.mode')});
  $$('[data-seq]').forEach(r=>r.onclick=()=>{state.radarSelected=Number(r.dataset.seq);renderRadar();drawRadarStatic()});
  bindListEditors($('#workspace'),path=>scheduleCurrent(path),renderRadar);
  if($('#addPulse'))$('#addPulse').onclick=()=>{state.radar.sequence.push(newPulseItem());state.radarSelected=state.radar.sequence.length-1;renderRadar();scheduleCurrent('radar.sequence')};
  if($('#addOff'))$('#addOff').onclick=()=>{state.radar.sequence.push({kind:'off',name:'Off time',repeat:1,pri_s:100e-6,off_time_s:100e-6,pri_jitter_fraction:0,pri_wobble_fraction:0,pri_wobble_period:16,scale:1,frequency_offset_hz:0,phase_offset_deg:0,pulse:null});state.radarSelected=state.radar.sequence.length-1;renderRadar();scheduleCurrent('radar.sequence')};
  if($('#seqDelete'))$('#seqDelete').onclick=()=>{if(state.radar.sequence.length<=1)return toast('Keep at least one sequence item');state.radar.sequence.splice(state.radarSelected,1);state.radarSelected=Math.max(0,state.radarSelected-1);renderRadar();scheduleCurrent('radar.sequence')};
  if($('#seqUp'))$('#seqUp').onclick=()=>moveSeq(-1);if($('#seqDown'))$('#seqDown').onclick=()=>moveSeq(1);
}
function newPulseItem(){return {kind:'pulse',name:`Pulse ${state.radar.sequence.filter(x=>x.kind==='pulse').length+1}`,repeat:1,pri_s:100e-6,off_time_s:100e-6,pri_jitter_fraction:0,pri_wobble_fraction:0,pri_wobble_period:16,scale:1,frequency_offset_hz:0,phase_offset_deg:0,pulse:{name:'Pulse',waveform:'lfm',width_s:20e-6,amplitude:1,frequency_offset_hz:0,phase_offset_deg:0,envelope:'rect',sweep_bw_hz:200_000,nlfm_exponent:3,barker_length:13,polyphase_order:4,phase_code_chips:32,zc_root:5,custom_phase_deg:[0,180,0,0,180,0,180],costas_hops:7,stepped_steps:8,stepped_spacing_hz:20_000}}}
function moveSeq(d){const i=state.radarSelected,j=i+d;if(j<0||j>=state.radar.sequence.length)return;[state.radar.sequence[i],state.radar.sequence[j]]=[state.radar.sequence[j],state.radar.sequence[i]];state.radarSelected=j;renderRadar();scheduleCurrent('radar.sequence')}

function spectrumRequestedSpan(){
  return Math.max(100,Math.min(10_000_000,Number(state.spectrum.span_hz??state.tx.sample_rate)||100));
}
function spectrumAcquisitionSpan(){return spectrumRequestedSpan();}
function spectrumDataBounds(){
  const f=state.live.freq;if(f?.length>=2)return {left:Number(f[0]),right:Number(f.at(-1))};
  const available=Math.min(spectrumRequestedSpan(),Number(state.tx.sample_rate)||spectrumRequestedSpan());return {left:-available/2,right:available/2};
}
function spectrumViewBounds(){
  const full=spectrumAcquisitionSpan(),minSpan=Math.max(20,Math.min(full,Number(state.live.metrics.rbw_actual_hz||state.spectrum.rbw_hz||1)*4));
  const span=Math.max(minSpan,Math.min(full,Number(state.specView.span_hz||full)));
  const half=full/2;let center=Number(state.specView.center_hz||0);center=Math.max(-half+span/2,Math.min(half-span/2,center));
  state.specView.center_hz=center;state.specView.span_hz=span>=full*.999?null:span;
  return {left:center-span/2,right:center+span/2,span,full};
}
function nearestSpectrumPoint(hz){
  const f=state.live.freq,p=state.live.psd;if(!f?.length||!p?.length)return null;
  let lo=0,hi=f.length-1;while(lo<hi){const m=(lo+hi)>>1;if(f[m]<hz)lo=m+1;else hi=m;}let k=lo;if(k>0&&Math.abs(f[k-1]-hz)<Math.abs(f[k]-hz))k--;
  return {hz:Number(f[k]),db:Number(p[k]),index:k};
}
function markerListHtml(){
  if(!state.markers.length)return '<p class="hint">Click the spectrum or add a marker. Up to eight markers are supported.</p>';
  return `<div class="memory-list">${state.markers.map((m,i)=>{const q=nearestSpectrumPoint(m.hz);return `<div><button class="btn small ${i===state.activeMarker?'selected':''}" data-marker-select="${i}">M${i+1}</button> <span>${fmt(q?.hz??m.hz,'Hz')} · ${fmt(q?.db,'dBm')}</span> <button class="btn small" data-marker-remove="${i}">×</button></div>`}).join('')}</div>`;
}
function setActiveMarker(hz){
  const data=spectrumDataBounds();hz=Math.max(data.left,Math.min(data.right,Number(hz)||0));
  if(!state.markers.length){state.markers.push({hz});state.activeMarker=0;}else{state.activeMarker=Math.max(0,Math.min(state.markers.length-1,state.activeMarker));state.markers[state.activeMarker].hz=hz;}
}
function addSpectrumMarker(hz=state.live.metrics.peak_hz??0){if(state.markers.length>=8)return toast('Marker limit is 8');state.markers.push({hz:Number(hz)||0});state.activeMarker=state.markers.length-1;}
function spectrumPeakList(){
  const y=state.live.psd||[],f=state.live.freq||[],peaks=[];for(let i=1;i<y.length-1;i++)if(y[i]>=y[i-1]&&y[i]>y[i+1])peaks.push({hz:f[i],db:y[i]});return peaks.sort((a,b)=>b.db-a.db);
}
function nextSpectrumPeak(){const peaks=spectrumPeakList();if(!peaks.length)return;const cur=state.markers[state.activeMarker]?.hz??Infinity;let at=peaks.findIndex(p=>Math.abs(p.hz-cur)<=Math.max(1,state.live.metrics.rbw_actual_hz||1));at=(at+1+peaks.length)%peaks.length;setActiveMarker(peaks[at].hz);}
function presetSpectrumDelta(reverse=false){const peak=Number(state.live.metrics.peak_hz)||0;state.markers=reverse?[{hz:peak},{hz:0}]:[{hz:0},{hz:peak}];state.activeMarker=1;state.specDeltaRef=0;}
function zoomSpectrum(factor,anchorHz=null){
  const b=spectrumViewBounds(),anchor=Number.isFinite(anchorHz)?anchorHz:(b.left+b.right)/2,newSpan=Math.max(Math.max(20,(state.live.metrics.rbw_actual_hz||1)*4),Math.min(b.full,b.span*factor));
  const ratio=newSpan/b.span,newCenter=anchor+(((b.left+b.right)/2)-anchor)*ratio;state.specView.span_hz=newSpan>=b.full*.999?null:newSpan;state.specView.center_hz=newCenter;spectrumViewBounds();drawSpectrumPage();
}
function panSpectrum(frac){const b=spectrumViewBounds();state.specView.center_hz+=(b.span*frac);spectrumViewBounds();drawSpectrumPage();}
function centerSpectrumView(hz){const full=spectrumAcquisitionSpan();if(!Number.isFinite(state.specView.span_hz)||state.specView.span_hz>=full*.999)state.specView.span_hz=full/2;state.specView.center_hz=Number(hz)||0;spectrumViewBounds();drawSpectrumPage();}
function resetSpectrumView(){state.specView={center_hz:0,span_hz:null};drawSpectrumPage();}

function renderSpectrum() {
  const page=state.specPage;
  let controls=group('Shared source',sourceReadout());
  if(page==='sweep') {
    const requested=spectrumRequestedSpan(),available=Math.min(requested,Number(state.tx.sample_rate)||requested);
    controls+=group('Frequency / resolution',field('Baseband span (Hz)','spectrum.span_hz',state.spectrum.span_hz??state.tx.sample_rate,{min:100,max:1e7,step:1000,clamp:true,clampMessage:'Spectrum span is limited to 10 MHz'})+`<p class="hint">Analyzer center is fixed at DC. The horizontal axis always follows the requested span up to 10 MHz. With the current ${fmt(state.tx.sample_rate,'S/s')} source, sampled data is available across ${fmt(available,'Hz')}; any wider portion is marked outside source Nyquist instead of being silently compressed. Current span: ${fmt(requested,'Hz')}.</p>`+field('Requested RBW (Hz)','spectrum.rbw_hz',state.spectrum.rbw_hz,{min:.1,step:100})+selectField('FFT window','spectrum.window',state.spectrum.window,['hann','blackmanharris','flattop','kaiser','rect'])+selectField('Detector','spectrum.detector',state.spectrum.detector,['average','peak','sample'])+field('FFT averages','spectrum.averaging',state.spectrum.averaging,{min:1,max:32,step:1}));
    controls+=fold('Traces',state.traces.map((t,i)=>selectField(`Trace ${i+1}`,'traces.'+i+'.mode',t.mode,[['off','Off'],['clear','Clear / write'],['max','Max hold'],['min','Min hold'],['average','Exponential average']])).join('')+field('Trace average count','specDisplay.average',state.specDisplay.average,{min:1,max:128,step:1})+`<button id="clearTrace" class="btn">Reset traces</button>`,true);
    controls+=fold('Amplitude / markers',checkField('Auto reference','specDisplay.auto',state.specDisplay.auto)+field('Reference (dBm)','specDisplay.ref',state.specDisplay.ref,{step:5})+field('Range (dB)','specDisplay.range',state.specDisplay.range,{min:10,max:300,step:10})+`<div class="block-actions"><button class="btn" id="addMarker">+ Marker</button><button class="btn" id="peakMarker">Marker → peak</button><button class="btn" id="nextPeak">Next peak</button><button class="btn" id="dcMarker">Marker → DC</button></div><div class="block-actions"><button class="btn" id="deltaM1">Δ to M1</button><button class="btn" id="dcPeakDelta">Δ DC → peak</button><button class="btn" id="peakDcDelta">Δ peak → DC</button></div><div class="block-actions"><button class="btn" id="markerToCenter">Marker → view center</button><button class="btn" id="centerToMarker">View center → marker</button><button class="btn" id="peakToCenter">Peak → view center</button></div>${markerListHtml()}<p class="hint">Click to move the active marker. Delta is active marker minus its reference marker. “View center” only pans the zoomed display; acquisition always stays fixed at baseband DC.</p>`,true);
    controls+=fold('View zoom / pan',`<div class="block-actions"><button class="btn" id="zoomIn">Zoom +</button><button class="btn" id="zoomOut">Zoom −</button><button class="btn" id="panLeft">← Pan</button><button class="btn" id="panRight">Pan →</button><button class="btn" id="zoomFull">Full span</button></div><p class="hint">Mouse wheel zooms around the pointer. Double-click restores the full acquired span. Zoom/pan is display-only; the analyzer remains centered at baseband DC.</p>`);
    controls+=group('Capture memories',`<button class="btn" id="captureTrace">Capture trace</button> <button class="btn" id="clearCaptures">Clear</button><div class="memory-list">${state.memories.map((m,i)=>`<div><span style="color:${m.color}">M${i+1} · ${esc(m.label)}</span> <button class="btn" data-memory-remove="${i}" aria-label="Remove capture ${i+1}">×</button></div>`).join('')||'<p class="hint">Keep up to four traces in memory. Each retains its frequency coordinates.</p>'}</div>`);
  } else if(page==='zero') {
    controls+=group('Measurement interval',field('Offset (s)','measurement.offset_s',state.measurement.offset_s,{min:0,step:.001})+field('Length (s)','measurement.duration_s',state.measurement.duration_s,{min:.00001,max:.06,step:.001})+'<p class="hint">AM, PM and FM are calculated at the original sample rate before reducing the plotted points. Phase and frequency at signal nulls are masked.</p>')+group('Capture memory',`<button class="btn" id="freezeZero">${state.zeroFrozen?'Resume live':'Hold capture'}</button>`);
  } else controls+=group('Signal quality',`<p class="hint">Linear digital and OFDM formats show decision-directed EVM. Basic, analog, FSK/MSK and mixed-frame signals show direct baseband I/Q, envelope, phase and frequency quality instead of inventing an EVM number.</p>`);
  const views=subnav([['sweep','Spectrum / waterfall'],['zero','Zero span'],['quality','Modulation quality']],page,'specpage');
  const evm=evmQualityModulations.has(state.tx.waveform)&&!(state.tx.frame_enabled&&state.tx.frame_mode==='blocks');
  const plots=page==='sweep'?plotPane('Spectrum · dBm','specMain')+plotPane('Waterfall · newest at bottom','specWaterfall'):page==='zero'?plotPane('I / Q · amplitude vs time','zeroIQ')+plotPane('AM · envelope vs time','zeroAM')+plotPane('PM · unwrapped phase (°)','zeroPM')+plotPane('FM · instantaneous frequency (Hz)','zeroFM'):evm?plotPane('Recovered constellation','qualityConst')+plotPane('EVM (%) vs symbol','qualityEvm')+plotPane('Magnitude error (%)','qualityMag')+plotPane('Phase error (°)','qualityPhase'):plotPane('I / Q trajectory','qualityConst')+plotPane('Envelope','qualityEvm')+plotPane('Instantaneous frequency (Hz)','qualityMag')+plotPane('Unwrapped phase (°)','qualityPhase');
  const metrics=page==='quality'?(evm?measure('EVM RMS','qEvm')+measure('EVM peak','qPeak')+measure('MER','qMer'):measure('RMS','qEvm')+measure('Peak','qPeak')+measure('Mean freq','qMer'))+measure('Method','qMethod',evm?'decision-directed':'direct waveform'):measure('Actual RBW','sRbw')+measure('Peak freq','sPeakF')+measure('Peak power','sPeakL')+measure('99% BW','sBw99')+measure('Marker / Δ','sMarker');
  $('#workspace').innerHTML=layout('Spectrum Analyzer',controls,metrics,views+`<div class="view-plots ${page==='sweep'?'sweep':'quad'}">${plots}</div>`,'with-views');
  bindControls($('#workspace'),path=>{if(path.startsWith('traces.')){resetTraces();drawSpectrumPage();}else if(path.startsWith('specDisplay.'))drawSpectrumPage();else if(path.startsWith('measurement.'))refreshMeasurements(true);else {if(path==='spectrum.span_hz')resetSpectrumView();scheduleCurrent(path);}});
  bindView([],v=>{state.specPage=v;renderSpectrum();startMode();},'specpage');
  if($('#clearTrace'))$('#clearTrace').onclick=()=>{resetTraces();drawSpectrumPage();};
  if($('#captureTrace'))$('#captureTrace').onclick=()=>{if(!state.live.psd.length)return toast('Waiting for a spectrum frame');if(state.memories.length===4)state.memories.shift();state.memories.push({y:Array.from(state.live.psd),x:Array.from(state.live.freq),color:['#c084fc','#f472b6','#93c5fd','#d1d5db'][state.memoryCount++%4],label:`${state.tx.waveform.toUpperCase()} · ${new Date().toLocaleTimeString()}`});renderSpectrum();drawSpectrumPage();};
  $$('[data-memory-remove]').forEach(b=>b.onclick=()=>{state.memories.splice(Number(b.dataset.memoryRemove),1);renderSpectrum();drawSpectrumPage();});
  if($('#clearCaptures'))$('#clearCaptures').onclick=()=>{state.memories=[];renderSpectrum();drawSpectrumPage();};
  $$('[data-marker-select]').forEach(b=>b.onclick=()=>{state.activeMarker=Number(b.dataset.markerSelect);drawSpectrumPage();renderSpectrum();});
  $$('[data-marker-remove]').forEach(b=>b.onclick=()=>{const i=Number(b.dataset.markerRemove);state.markers.splice(i,1);state.activeMarker=Math.max(0,Math.min(state.markers.length-1,state.activeMarker));if(state.specDeltaRef===i)state.specDeltaRef=null;else if(Number.isInteger(state.specDeltaRef)&&state.specDeltaRef>i)state.specDeltaRef--;renderSpectrum();drawSpectrumPage();});
  const spec=$('#specMain');if(spec){spec.onclick=e=>{const box=spec.getBoundingClientRect(),b=spectrumViewBounds();setActiveMarker(b.left+(e.clientX-box.left)/box.width*b.span);renderSpectrum();drawSpectrumPage();};spec.ondblclick=e=>{e.preventDefault();resetSpectrumView();};spec.addEventListener('wheel',e=>{e.preventDefault();const box=spec.getBoundingClientRect(),b=spectrumViewBounds(),hz=b.left+(e.clientX-box.left)/box.width*b.span;zoomSpectrum(e.deltaY<0?.72:1.38,hz);},{passive:false});}
  if($('#addMarker'))$('#addMarker').onclick=()=>{addSpectrumMarker();renderSpectrum();drawSpectrumPage();};
  if($('#peakMarker'))$('#peakMarker').onclick=()=>{setActiveMarker(state.live.metrics.peak_hz);renderSpectrum();drawSpectrumPage();};
  if($('#nextPeak'))$('#nextPeak').onclick=()=>{nextSpectrumPeak();renderSpectrum();drawSpectrumPage();};
  if($('#dcMarker'))$('#dcMarker').onclick=()=>{setActiveMarker(0);renderSpectrum();drawSpectrumPage();};
  if($('#deltaM1'))$('#deltaM1').onclick=()=>{state.specDeltaRef=state.markers.length>1&&state.activeMarker!==0?0:null;drawSpectrumPage();renderSpectrum();};
  if($('#dcPeakDelta'))$('#dcPeakDelta').onclick=()=>{presetSpectrumDelta(false);renderSpectrum();drawSpectrumPage();};
  if($('#peakDcDelta'))$('#peakDcDelta').onclick=()=>{presetSpectrumDelta(true);renderSpectrum();drawSpectrumPage();};
  if($('#markerToCenter'))$('#markerToCenter').onclick=()=>{const m=state.markers[state.activeMarker];if(m)centerSpectrumView(m.hz);renderSpectrum();};
  if($('#centerToMarker'))$('#centerToMarker').onclick=()=>{const b=spectrumViewBounds();setActiveMarker((b.left+b.right)/2);renderSpectrum();drawSpectrumPage();};
  if($('#peakToCenter'))$('#peakToCenter').onclick=()=>{centerSpectrumView(Number(state.live.metrics.peak_hz)||0);renderSpectrum();};
  if($('#zoomIn'))$('#zoomIn').onclick=()=>zoomSpectrum(.5);if($('#zoomOut'))$('#zoomOut').onclick=()=>zoomSpectrum(2);if($('#zoomFull'))$('#zoomFull').onclick=resetSpectrumView;if($('#panLeft'))$('#panLeft').onclick=()=>panSpectrum(-.25);if($('#panRight'))$('#panRight').onclick=()=>panSpectrum(.25);
  if($('#freezeZero'))$('#freezeZero').onclick=()=>{state.zeroFrozen=!state.zeroFrozen;renderSpectrum();drawMeasurements();};
  drawSpectrumPage();drawMeasurements();
}

function renderRx() {
  const page=state.rxPage;
  let controls=group('Shared source',sourceReadout())+group('Demodulator',checkField('Follow transmitter format','rxFollow',state.rxFollow)+(state.rxFollow?'':selectField('Modulation','rx.modulation',state.rx.modulation,Object.values(signalFamilies).flat().filter(x=>!signalFamilies.fundamentals.includes(x)))+field('Symbol rate (Bd)','rx.symbol_rate',state.rx.symbol_rate,{min:10,step:100})));
  if(!state.rxFollow) {
    const linear=['bpsk','qpsk','dbpsk','dqpsk','pi4-dqpsk','oqpsk','8psk','16qam','64qam','16apsk','32apsk'].includes(state.rx.modulation);
    let params=field('Tuner offset (Hz)','rx.tuner_hz',state.rx.tuner_hz,{step:100});
    if(linear)params+=selectField('Matched filter','rx.pulse_shape',state.rx.pulse_shape,[['rrc','Root raised cosine'],['rect','Rectangular']])+field('Rolloff','rx.rolloff',state.rx.rolloff,{min:0,max:1,step:.05})+field('Filter span (symbols)','rx.filter_span_symbols',state.rx.filter_span_symbols,{min:2,max:64,step:1});
    if(state.rx.modulation.includes('apsk'))params+=field('Middle / inner radius','rx.apsk_ratio2',state.rx.apsk_ratio2,{min:1.01,max:8,step:.05})+(state.rx.modulation==='32apsk'?field('Outer / inner radius','rx.apsk_ratio3',state.rx.apsk_ratio3,{min:1.01,max:12,step:.05}):'');
    if(['2fsk','4fsk'].includes(state.rx.modulation))params+=field('FSK deviation (Hz)','rx.fsk_deviation_hz',state.rx.fsk_deviation_hz,{min:1,step:100});
    if(state.rx.modulation==='fm')params+=field('FM deviation (Hz)','rx.fm_deviation_hz',state.rx.fm_deviation_hz,{min:1,step:100});
    if(state.rx.modulation==='ofdm')params+=field('FFT length','rx.ofdm_fft',state.rx.ofdm_fft,{min:16,max:4096,step:16})+field('Cyclic prefix','rx.ofdm_cp',state.rx.ofdm_cp,{min:0,max:1024,step:1})+field('Occupied carriers','rx.ofdm_occupied',state.rx.ofdm_occupied,{min:4,max:4094,step:2});
    controls+=fold('Demodulator parameters',params,true);
  }
  controls+=fold('Channel experiment',sliderField('SNR (dB)','channel.snr_db',state.channel.snr_db,{step:1,min:-10,max:50})+sliderField('Carrier offset (Hz)','channel.freq_offset_hz',state.channel.freq_offset_hz,{step:50,min:-10000,max:10000})+sliderField('Phase offset (°)','channel.phase_offset_deg',state.channel.phase_offset_deg,{step:1,min:-180,max:180})+sliderField('Timing offset (samples)','channel.timing_offset_samples',state.channel.timing_offset_samples,{step:.01,min:-.95,max:.95})+sliderField('Delayed echo gain','channel.echo_gain',state.channel.echo_gain,{step:.01,min:0,max:.9})+field('Echo delay (symbols)','channel.echo_delay_symbols',state.channel.echo_delay_symbols,{min:1,max:8,step:1}),page==='acquire');
  if(page==='framing')controls+=group('Frame acquisition',checkField('Enable frame acquisition','rx.frame_enabled',state.rx.frame_enabled)+checkField('Energy detector','rx.energy_detection',state.rx.energy_detection)+(state.rx.energy_detection?field('Energy threshold (dBm)','rx.energy_threshold_dbm',state.rx.energy_threshold_dbm,{step:1})+field('Energy window (symbols)','rx.energy_window_symbols',state.rx.energy_window_symbols,{min:.1,max:100,step:.1}):'')+checkField('Preamble correlation','rx.preamble_correlation',state.rx.preamble_correlation)+(state.rx.preamble_correlation?sliderField('Correlation threshold','rx.preamble_threshold',state.rx.preamble_threshold,{min:0,max:1,step:.01}):'')+'<p class="hint">The first passing preamble correlation defines frame start; energy detection is the fallback. Correlation is normalized, so its threshold is independent of absolute transmit power.</p>')+group('Expected frame',frameBlockEditor('rx.frame_blocks',state.rx.frame_blocks));
  if(page==='timing'||page==='acquire')controls+=(page==='acquire'?fold:group)('Timing recovery',selectField('Algorithm','rx.timing_recovery',state.rx.timing_recovery,[['manual','Manual sampler'],['mueller-muller','Mueller & Müller'],['gardner','Gardner · interpolated']])+(state.rx.timing_recovery==='manual'?'':sliderField('Loop gain','rx.timing_gain',state.rx.timing_gain,{min:.01,max:.8,step:.01}))+(state.rx.timing_recovery==='mueller-muller'?'':sliderField('Initial timing (symbols)','rx.timing_phase',state.rx.timing_phase,{min:-.5,max:.5,step:.01}))+`<p class="hint">${state.rx.timing_recovery==='gardner'?'Gardner uses midpoint samples without symbol decisions.':state.rx.timing_recovery==='mueller-muller'?'M&M is decision-directed; large carrier error can disturb timing acquisition.':'Manual mode is useful for comparing sampling phase and eye opening.'}</p>`);
  if(page==='carrier'||page==='acquire')controls+=(page==='acquire'?fold:group)('Carrier recovery',checkField('Coarse CFO acquisition','rx.coarse_cfo',state.rx.coarse_cfo)+selectField('Tracking loop','rx.carrier_algorithm',state.rx.carrier_algorithm,[['costas','Costas (BPSK/QPSK); DD otherwise'],['decision-directed','Decision-directed PLL'],['off','Bypass']])+sliderField('Loop α','rx.costas.alpha',state.rx.costas.alpha,{min:.001,max:.2,step:.001})+sliderField('Loop β','rx.costas.beta',state.rx.costas.beta,{min:.0001,max:.02,step:.0001})+sliderField('Residual phase (°)','rx.manual_phase_deg',state.rx.manual_phase_deg,{min:-180,max:180,step:1})+`<p class="hint">Coarse acquisition supports BPSK, QPSK, their unrotated differential variants and 8-PSK. Differential decoding bypasses the tracking loop. Quadrant ambiguity still requires framing or phase adjustment.</p>`);
  if(page==='equalizer')controls+=group('Equalizer',selectField('Algorithm','rx.equalizer',state.rx.equalizer,[['off','Bypass'],['cma','CMA · blind, PSK-oriented']])+field('Taps','rx.equalizer_taps',state.rx.equalizer_taps,{min:3,max:21,step:2})+sliderField('Adaptation gain','rx.equalizer_gain',state.rx.equalizer_gain,{min:.0001,max:.03,step:.0001})+'<p class="hint">CMA minimizes modulus error and is best suited to PSK. The separate channel estimate uses known transmitted symbols and assumes aligned timing; it is reference-aided.</p>');
  controls+=fold('I/Q channel imperfections',sliderField('I/Q gain (dB)','channel.iq_gain_db',state.channel.iq_gain_db,{min:-12,max:12,step:.1})+sliderField('Quadrature error (°)','channel.iq_phase_deg',state.channel.iq_phase_deg,{min:-45,max:45,step:1})+sliderField('DC I','channel.dc_i',state.channel.dc_i,{min:-1,max:1,step:.01})+sliderField('DC Q','channel.dc_q',state.channel.dc_q,{min:-1,max:1,step:.01}));
  controls+=fold('Replay / loop telemetry',sliderField('Playback speed','rxDisplay.sync_speed',state.rxDisplay.sync_speed,{min:.2,max:4,step:.1})+sliderField('Window (symbols)','rxDisplay.sync_window',state.rxDisplay.sync_window,{min:40,max:600,step:20})+checkField('Pause replay','rxDisplay.sync_pause',state.rxDisplay.sync_pause)+selectField('Trace','rxDisplay.loop_trace',state.rxDisplay.loop_trace,[['frequency','Carrier frequency'],['phase','Carrier phase'],['error','Carrier error'],['timing-error','Timing error'],['timing-mu','Timing phase μ']]));
  const views=subnav([['acquire','Acquisition'],['framing','Framing'],['timing','Timing'],['carrier','Carrier'],['equalizer','Equalizer / channel']],page,'rxpage');
  let plots,metrics;
  if(page==='framing'){
    plots=plotPane('Energy detector · calibrated dBm','rxFrameEnergy')+plotPane('Normalized preamble correlation','rxFrameCorr')+plotPane('Received I/Q · acquisition record','rxRaw')+`<div class="plot-pane result-cell"><div class="plot-label">Frame blocks / decode</div><div class="result-panel" id="rxFrameResult"></div></div>`;
    metrics=measure('Detected','rxFrameDetected')+measure('Frame start','rxFrameStart')+measure('Correlation peak','rxFramePeak')+measure('Energy crossing','rxFrameEnergyStart');
  }else{
    plots=plotPane(page==='equalizer'?'Before equalizer':'Before carrier recovery','rxRaw')+plotPane('Synchronization replay · actual symbol sequence','rxSync');
    plots+=page==='timing'?plotPane('Eye · matched-filter samples','rxEye')+plotPane('Timing detector error','rxTiming'):page==='equalizer'?plotPane('Equalizer adaptation error','rxEqError')+plotPane('Channel estimate · tap magnitude','rxChannel'):plotPane('Loop telemetry','rxLoop')+`<div class="plot-pane result-cell"><div class="plot-label">Decoded output / state</div><div class="result-panel"><p id="rxSupport" class="hint"></p>${subnav([['text','Text'],['hex','Hex'],['bits','Bits']],state.rxOutput,'rxoutput')}<pre id="rxText" class="decoded"></pre></div></div>`;
    metrics=measure('Demod','rxDemod',state.rx.modulation.toUpperCase())+measure('EVM RMS','rxEvm')+measure('MER','rxMer')+measure('Coarse CFO','rxCoarse')+measure('Fine CFO','rxCfo');
  }
  $('#workspace').innerHTML=layout('Receiver · synchronization lab',controls,metrics,views+`<div class="view-plots quad">${plots}</div>`,'with-views');
  bindControls($('#workspace'),path=>{if(path==='rxFollow'){state.rxRevision++;followTransmitter();renderRx();refreshRx(true);}else if(path.startsWith('rxDisplay.'))drawRxAnalysis();else {if(path==='rx.frame_enabled'||path.startsWith('rx.frame_blocks.'))state.rxFollow=false;if(['rx.modulation','rx.timing_recovery','rx.carrier_algorithm','rx.frame_enabled','rx.energy_detection','rx.preamble_correlation'].includes(path)||path.includes('.role'))renderRx();scheduleCurrent(path);}});
  bindFrameBlockButtons($('#workspace'),renderRx,path=>{state.rxFollow=false;state.rxRevision++;scheduleCurrent(path);});
  bindView([],v=>{state.rxPage=v;if(v==='timing')state.rxDisplay.loop_trace='timing-error';else state.rxDisplay.loop_trace='frequency';renderRx();drawRxAnalysis();},'rxpage');
  bindView([],v=>{state.rxOutput=v;renderRx();},'rxoutput');
  drawRxAnalysis();
}

function renderBlind() {
  const controls=group('Shared source',sourceReadout())+group('Observer',field('Observation cadence (ms)','blindDisplay.interval',state.blindDisplay.interval,{min:100,step:100})+sliderField('Report score threshold','blindDisplay.threshold',state.blindDisplay.threshold,{min:.2,max:.95,step:.05})+`<p class="hint">Scores measure agreement with heuristic rules; they are not calibrated probabilities. No transmitter modulation label is supplied to the classifier.</p>`)+fold('Supported hypotheses',`<p class="hint">BPSK, QPSK, 8-PSK, 16/64-QAM, FSK/MSK, OFDM-like, noise, tone and analog families. Offset, differential and APSK variants may resemble these classes. No protocol identification.</p><p class="hint">Unknown CFO, low SNR, bursts and mixed signals can make estimates unreliable. Compare evidence before accepting a label.</p>`,true)+group('Observation history',`<button class="btn" id="clearHistory">Clear history</button><p class="hint">Bounded to 80 changes in class or source revision; recurring observations update the same row.</p>`);
  const views=subnav([['overview','Observation'],['evidence','Evidence'],['history','History']],state.blindPage,'blindpage');
  let plots=state.blindPage==='overview'?plotPane('Observed spectrum','blindSpectrum')+`<div class="plot-pane"><div class="plot-label">Current hypothesis</div><div class="result-panel" id="blindResult"></div></div>`+plotPane('Observed waterfall','blindWaterfall')+plotPane('Amplitude distribution','blindHistogram'):state.blindPage==='evidence'?`<div class="plot-pane full"><div class="plot-label">Evidence / alternatives</div><div class="result-panel" id="blindEvidence"></div></div>`:`<div class="plot-pane full"><div class="plot-label">Observation history · internal session</div><div class="result-panel" id="blindHistory"></div></div>`;
  const metrics=measure('Hypothesis','bClass')+measure('Heuristic score','bConf')+measure('Symbol rate','bRs')+measure('SNR estimate','bSnr')+measure('99% BW','bBw');
  $('#workspace').innerHTML=layout('Classifier · signal observer',controls,metrics,views+`<div class="view-plots quad">${plots}</div>`,'with-views');
  bindControls($('#workspace'),path=>{if(path==='blindDisplay.interval')startMode();else drawBlindResult();});
  bindView([],v=>{state.blindPage=v;renderBlind();},'blindpage');
  $('#clearHistory').onclick=()=>{state.blindHistory=[];drawBlindResult();};drawBlindResult();
}

function scheduleCurrent(path='') {
  if(path.startsWith('tx.')){state.revision++;followTransmitter();}
  if(path.startsWith('rx.')||path.startsWith('channel.'))state.rxRevision++;
  if(path.startsWith('radar.'))state.radarRevision++;
  clearTimeout(state.timers.debounce);
  state.timers.debounce=setTimeout(()=>{
    if(path.startsWith('radarDisplay.'))return;
    if(path.startsWith('radar.')){refreshRadar();if(state.radarPage==='returns')refreshScene(true);return;}
    if(path.startsWith('tx.')||path.startsWith('spectrum.')||path.startsWith('display.'))sendWsUpdate();
    if(state.mode==='rx')refreshRx(true);
    if(state.mode==='blind')refreshBlind(true);
  },140);
}

function stopModeTimers(){
  if(state.timers.poll){clearInterval(state.timers.poll);state.timers.poll=null}
  if(state.timers.radarAnim){cancelAnimationFrame(state.timers.radarAnim);state.timers.radarAnim=null}
  if(state.timers.rxAnim){cancelAnimationFrame(state.timers.rxAnim);state.timers.rxAnim=null}
}
function startMode() {
  stopModeTimers();
  if(state.mode==='radar') {
    closeWs();setSourceStatus('Radar Studio',state.radar.sample_rate);
    if(!state.radarData||state.radarDataRevision!==state.radarRevision)refreshRadar();
    else drawRadarStatic();
    if(state.radarPage==='design')animateRadarPlayback();
    if(state.radarPage==='returns'){refreshScene(true);state.timers.poll=setInterval(()=>refreshScene(false),500);}
    return;
  }
  openWs();setSourceStatus(`Transmitter · ${state.tx.waveform}`,state.tx.sample_rate);
  if(state.mode==='rx'){followTransmitter();refreshRx(true);animateRxSync();state.timers.poll=setInterval(()=>refreshRx(false),2000);}
  if(state.mode==='blind'){refreshBlind(true);state.timers.poll=setInterval(()=>refreshBlind(false),Number(state.blindDisplay.interval));}
  if(state.mode==='spectrum'&&state.specPage!=='sweep'){refreshMeasurements(true);state.timers.poll=setInterval(()=>refreshMeasurements(false),800);}
  schedulePaint();
}

function wsPayload(){return {action:'update',tx:cleanTx(),spectrum:cleanSpectrum(),fps:state.display.fps}}
function cleanTx() {const x=structuredClone(state.tx);x.duration=.06;if(!state.txNoise||x.snr_db>=99.9)x.snr_db=null;return x;}

function cleanSpectrum(){const x=structuredClone(state.spectrum);delete x.center_hz;if(!Number.isFinite(x.span_hz)||x.span_hz<=0)x.span_hz=Math.min(10_000_000,state.tx.sample_rate);x.span_hz=Math.max(100,Math.min(10_000_000,x.span_hz));x.max_fft=Math.min(x.max_fft||32768,32768);return x}
function openWs() {
  if(state.mode==='radar'||document.hidden)return;
  if(state.ws && [WebSocket.OPEN,WebSocket.CONNECTING].includes(state.ws.readyState))return;
  clearTimeout(state.reconnectTimer);
  const proto=location.protocol==='https:'?'wss':'ws';const ws=new WebSocket(`${proto}://${location.host}/ws/live`);
  ws.binaryType='arraybuffer';state.ws=ws;state.wsReady=false;
  ws.onopen=()=>{if(state.ws!==ws)return;state.wsReady=true;state.reconnectAttempts=0;state.lastFrameAt=performance.now();ws.send(JSON.stringify({...wsPayload(),action:'start',acknowledge_frames:true,capture_id:(state.captureID??-1)+1}));setStatus('transmitter streaming');};
  ws.onmessage=e=>{if(state.ws!==ws)return;try {if(typeof e.data==='string')handleWsText(JSON.parse(e.data));else handleWsBinary(e.data);}catch(err){setStatus('stream decode error');}};
  ws.onerror=()=>setStatus('stream reconnecting');
  ws.onclose=()=>{if(state.ws!==ws)return;state.ws=null;state.wsReady=false;setStatus('stream disconnected · retrying');state.reconnectTimer=setTimeout(openWs,Math.min(8000,500*2**state.reconnectAttempts++));};
}

function sendWsUpdate() {
  if(state.ws?.readyState===WebSocket.OPEN){state.ws.send(JSON.stringify(wsPayload()));setSourceStatus(`Transmitter · ${state.tx.waveform}`,state.tx.sample_rate);}
  else openWs();
}

function closeWs() {
  clearTimeout(state.reconnectTimer);const old=state.ws;state.ws=null;state.wsReady=false;if(old)old.close();
}

function handleWsText(m) {
  if(m.type==='error'){setStatus('invalid configuration');toast(m.message);return;}
  if(m.type==='metrics'){state.live.metrics=m;if(Array.isArray(m.freq_hz)&&m.freq_hz.length)state.live.freq=m.freq_hz;updateLiveMetrics();}
  if(m.type==='meta'){state.sourceRevision=m.metadata?.source_revision;state.sourceMetadata=m.metadata;resetTraces();updateLiveMetrics();}
}

function handleWsBinary(buf) {
  if(buf.byteLength<16)return;const dv=new DataView(buf);
  if(dv.getUint32(0,true)!==0x31465256||dv.getUint32(12,true)!==buf.byteLength-16)return;
  const type=dv.getUint16(4,true);if(type!==1)return;
  const arr=new Float32Array(buf,16);state.live.waterfallRow=arr;state.live.psd=applyTrace(arr);state.live.frames++;state.lastFrameAt=performance.now();updateRate();schedulePaint();
}

function schedulePaint(){if(state.timers.paint)return;state.timers.paint=requestAnimationFrame(()=>{state.timers.paint=null;if(state.mode==='tx')drawTx();else if(state.mode==='spectrum')drawSpectrumPage();else if(state.mode==='rx')drawRxLive();else if(state.mode==='blind')drawBlindPlots()})}
function applyTrace(y) {
  const signature=JSON.stringify([state.sourceRevision,state.spectrum]);
  if(signature!==state.traceSignature){for(const t of state.traces)t.y=null;state.traceSignature=signature;}
  for(const t of state.traces) {
    if(t.mode==='off')continue;
    if(!t.y||t.y.length!==y.length||t.mode==='clear')t.y=y.slice();
    else for(let i=0;i<y.length;i++) {
      if(t.mode==='max')t.y[i]=Math.max(t.y[i],y[i]);
      if(t.mode==='min')t.y[i]=Math.min(t.y[i],y[i]);
      if(t.mode==='average') {const a=1/Math.max(1,state.specDisplay.average);t.y[i]=10*Math.log10((1-a)*10**(t.y[i]/10)+a*10**(y[i]/10));}
    }
  }
  return y;
}

function updateLiveMetrics() {
  const m=state.live.metrics;
  for(const [id,value,unit] of [['sRbw',m.rbw_actual_hz,'Hz'],['sPeakF',m.peak_hz,'Hz'],['sPeakL',m.peak_db,m.power_unit||'dBm'],['sBw99',m.bw99_hz,'Hz']])if($('#'+id))$('#'+id).textContent=fmt(value,unit);
  if($('#mStream'))$('#mStream').textContent=state.wsReady?`${Number(m.display_fps||8).toFixed(0)} Hz · active`:'connecting';
  if($('#txSummary'))$('#txSummary').textContent=`Record ${state.sourceRevision||'—'} · ${fmt(state.sourceMetadata?.duration,'s')} · engine ${fmt(m.compute_ms,'ms')} / spectrum frame`;
}

async function refreshRx(force=false) {
  if(state.mode!=='rx')return;
  if(state.busy.rx){if(force)state.pending.rx=true;return;}
  state.busy.rx=true;followTransmitter();const version=`${state.revision}:${state.rxRevision}`;
  try {const tx=cleanTx(),rx=structuredClone(state.rx);rx.sample_rate=tx.sample_rate;rx.use_costas=rx.carrier_algorithm!=='off';const d=await jsonPost('/api/rx/loopback',{tx,channel:state.channel,rx});if(version!==`${state.revision}:${state.rxRevision}`)return;
    // Keep acquisition playback running while replacing its source with this capture.
    state.rxData=d;if(state.mode==='rx'){drawRxAnalysis();setStatus('receiver · new capture '+d.tx_metadata.capture_id);}
  } catch(e){toast(shortError(e));setStatus('receiver configuration error');}finally{state.busy.rx=false;if(state.pending.rx){state.pending.rx=false;refreshRx(true);}}
}

async function refreshBlind(force=false) {
  if(state.mode!=='blind')return;if(state.busy.blind){if(force)state.pending.blind=true;return;}
  const version=state.revision;state.busy.blind=true;
  try {const started=performance.now(),d=await jsonPost('/api/blind/generate',cleanTx());if(version!==state.revision)return;d.elapsed_ms=performance.now()-started;state.blindData=d;
    const key=`${d.source_revision}/${d.modulation}`,last=state.blindHistory.at(-1),stamp=new Date().toLocaleTimeString();
    if(last?.key===key){last.last=stamp;last.count++;last.score=d.confidence;}else {state.blindHistory.push({key,first:stamp,last:stamp,count:1,label:d.modulation,score:d.confidence,revision:d.source_revision});if(state.blindHistory.length>80)state.blindHistory.shift();}
    if(state.mode==='blind'){drawBlindResult();setStatus('classifier observing');}
  } catch(e){toast(shortError(e));}finally{state.busy.blind=false;if(state.pending.blind){state.pending.blind=false;refreshBlind(true);}}
}

async function refreshRadar() {
  if(state.busy.radar){state.pending.radar=true;return;}state.busy.radar=true;const version=state.radarRevision;
  try {const t=performance.now(),d=await jsonPost('/api/radar/studio',state.radar);if(version!==state.radarRevision)return;state.radarData=d;state.radarDataRevision=version;if(state.mode==='radar'){drawRadarStatic();if($('#rUpdate'))$('#rUpdate').textContent=fmt(performance.now()-t,'ms');setStatus('radar waveform ready');}}
  catch(e){toast(shortError(e));setStatus('radar configuration error');}finally{state.busy.radar=false;if(state.pending.radar){state.pending.radar=false;refreshRadar();}}
}

function shortError(e){const s=String(e?.message||e);return s.length>180?s.slice(0,177)+'…':s}

function canvasCtx(canvas){
  if(!canvas)return null;const dpr=Math.min(1.5,window.devicePixelRatio||1);const r=canvas.getBoundingClientRect();const w=Math.max(10,Math.round(r.width*dpr)),h=Math.max(10,Math.round(r.height*dpr));
  if(canvas.width!==w||canvas.height!==h){canvas.width=w;canvas.height=h}const c=canvas.getContext('2d',{alpha:false});c.setTransform(dpr,0,0,dpr,0,0);return[c,r.width,r.height]
}
function grid(c,w,h){c.clearRect(0,0,w,h);c.fillStyle='#090b0d';c.fillRect(0,0,w,h);c.strokeStyle='#20262a';c.lineWidth=1;for(let i=1;i<8;i++){const x=i*w/8;c.beginPath();c.moveTo(x,0);c.lineTo(x,h);c.stroke()}for(let i=1;i<6;i++){const y=i*h/6;c.beginPath();c.moveTo(0,y);c.lineTo(w,y);c.stroke()}}
function bounds(y){let lo=Infinity,hi=-Infinity;for(const v of y){if(Number.isFinite(v)){if(v<lo)lo=v;if(v>hi)hi=v}}if(!Number.isFinite(lo)||!Number.isFinite(hi)){lo=-1;hi=1}return[lo,hi]}
function linePlot(canvas,y,opt={}) {
  if(!canvas)return;const z=canvasCtx(canvas);if(!z)return;const[c,w,h]=z;grid(c,w,h);if(!y||y.length<2)return;
  let[lo,hi]=bounds(y);if(opt.secondary?.length){const[b,t]=bounds(opt.secondary);lo=Math.min(lo,b);hi=Math.max(hi,t);}if(Number.isFinite(opt.threshold)){lo=Math.min(lo,opt.threshold);hi=Math.max(hi,opt.threshold);}lo=opt.min??lo;hi=opt.max??hi;if(hi<=lo){lo-=1;hi+=1;}
  const pad=22,plotH=Math.max(1,h-pad*2);
  const draw=(arr,color)=>{c.strokeStyle=color;c.lineWidth=1.15;c.beginPath();let active=false;const target=Math.max(200,Math.floor(w*1.5)),step=Math.max(1,Math.floor(arr.length/target));for(let i=0;i<arr.length;i+=step){const end=Math.min(arr.length,i+step);let minI=-1,maxI=-1,minV=Infinity,maxV=-Infinity;for(let j=i;j<end;j++){const v=arr[j];if(!Number.isFinite(v))continue;if(v<minV){minV=v;minI=j}if(v>maxV){maxV=v;maxI=j}}const pts=[];if(minI>=0&&maxI>=0){if(minI===maxI)pts.push([minI,minV]);else if(minI<maxI)pts.push([minI,minV],[maxI,maxV]);else pts.push([maxI,maxV],[minI,minV]);}for(const [j,v] of pts){const px=j*(w-1)/(arr.length-1),py=pad+(hi-v)/(hi-lo)*plotH;if(active)c.lineTo(px,py);else c.moveTo(px,py);active=true;}}c.stroke();};
  draw(y,'#d3b05c');if(opt.secondary)draw(opt.secondary,'#62a0bd');if(Number.isFinite(opt.threshold)){const py=pad+(hi-opt.threshold)/(hi-lo)*plotH;if(py>=pad&&py<=h-pad){c.strokeStyle='#d68b6b';c.setLineDash([5,4]);c.beginPath();c.moveTo(0,py);c.lineTo(w,py);c.stroke();c.setLineDash([]);}}c.fillStyle='#94a4ae';c.font='10px Consolas, monospace';c.fillText(fmt(hi),5,12);c.fillText(fmt(lo),5,h-25);
  const x=opt.x;const first=x?.[0]??0,last=x?.at(-1)??y.length-1;c.fillText(fmt(first,opt.xUnit||''),8,h-6);const end=fmt(last,opt.xUnit||'');c.fillText(end,Math.max(8,w-c.measureText(end).width-8),h-6);
  if(Number.isFinite(opt.cursor)){const px=Math.max(0,Math.min(1,opt.cursor))*w;c.strokeStyle='#f0d49b';c.beginPath();c.moveTo(px,0);c.lineTo(px,h);c.stroke();}
}

function idealConstellationOriginal(mod){
  if(mod==='bpsk')return [[-1,0],[1,0]];
  if(mod==='qpsk')return [[-1,-1],[-1,1],[1,-1],[1,1]].map(([i,q])=>[i/Math.SQRT2,q/Math.SQRT2]);
  if(mod==='8psk')return Array.from({length:8},(_,k)=>[Math.cos(2*Math.PI*k/8),Math.sin(2*Math.PI*k/8)]);
  if(mod==='16qam'||mod==='64qam'){const lv=mod==='16qam'?[-3,-1,1,3]:[-7,-5,-3,-1,1,3,5,7],norm=Math.sqrt((lv.reduce((s,x)=>s+x*x,0)/lv.length)*2);const out=[];for(const i of lv)for(const q of lv)out.push([i/norm,q/norm]);return out}
  return [];
}
function scatter(canvas,I,Q,opt={}){
  if(!canvas||!I?.length)return;let scale=opt.scale||Math.sqrt(I.reduce((s,v,i)=>s+v*v+(Q[i]||0)**2,0)/I.length);scale=Math.max(scale,1e-12);I=Array.from(I,v=>v/scale);Q=Array.from(Q,v=>v/scale);const z=canvasCtx(canvas);if(!z)return;const[c,w,h]=z;grid(c,w,h);let lim=.001;for(let i=0;i<I.length;i++)lim=Math.max(lim,Math.abs(I[i]),Math.abs(Q[i]||0));for(const p of opt.ideals||[])lim=Math.max(lim,Math.abs(p[0]),Math.abs(p[1]));lim*=1.15;c.strokeStyle='#394247';c.beginPath();c.moveTo(w/2,0);c.lineTo(w/2,h);c.moveTo(0,h/2);c.lineTo(w,h/2);c.stroke();
  const xy=(i,q)=>[(i/(2*lim)+.5)*w,(.5-q/(2*lim))*h];
  if(opt.ideals?.length){c.strokeStyle=opt.dimIdeals?'rgba(173,186,194,.3)':'rgba(240,212,155,.75)';c.lineWidth=1.2;for(const [i,q] of opt.ideals){const[x,y]=xy(i,q);c.beginPath();c.arc(x,y,4.2,0,2*Math.PI);c.stroke()}}
  c.fillStyle='rgba(226,169,59,.52)';const stride=Math.max(1,Math.floor(I.length/1400));for(let i=0;i<I.length;i+=stride){const[x,y]=xy(I[i],Q[i]||0);c.fillRect(x-1,y-1,2,2)}
}
const waterfallCache=new WeakMap();
function waterfall(canvas,latest,version=state.live.frames){
  if(!canvas||!latest?.length)return;const z=canvasCtx(canvas);if(!z)return;const[c,w,h]=z;const W=Math.min(512,latest.length),H=96;
  const signature=JSON.stringify([state.sourceRevision,state.spectrum,canvas.id==='specWaterfall'?state.specView:null]);
  let cache=waterfallCache.get(canvas)||state.waterfallRaster;
  if(cache?.signature!==signature)cache=null;
  if(!cache||cache.W!==W||cache.H!==H){const off=document.createElement('canvas');off.width=W;off.height=H;const oc=off.getContext('2d',{alpha:false});oc.fillStyle='#08101a';oc.fillRect(0,0,W,H);cache={off,oc,W,H,version:-1,signature};waterfallCache.set(canvas,cache);state.waterfallRaster=cache}
  if(cache.version!==version){
    const {off,oc}=cache;oc.drawImage(off,0,-1);const row=oc.createImageData(W,1);
    let wfLow=-120,wfHigh=-20,peak=-Infinity;for(const v of latest)if(Number.isFinite(v)&&v>peak)peak=v;if(canvas.id==='specWaterfall'){wfHigh=state.specDisplay.auto&&Number.isFinite(peak)?Math.ceil((peak+8)/10)*10:Number(state.specDisplay.ref);wfLow=wfHigh-Number(state.specDisplay.range);}for(let xx=0;xx<W;xx++){const v=latest[Math.min(latest.length-1,Math.floor(xx*latest.length/W))],p=xx*4;if(!Number.isFinite(v)){row.data[p]=5;row.data[p+1]=8;row.data[p+2]=10;row.data[p+3]=255;continue;}const q=Math.max(0,Math.min(1,(v-wfLow)/Math.max(1e-9,wfHigh-wfLow)));row.data[p]=Math.round(8+225*Math.max(0,(q-.55)/.45));row.data[p+1]=Math.round(15+160*q);row.data[p+2]=Math.round(28+95*(1-q));row.data[p+3]=255}
    oc.putImageData(row,0,H-1);cache.version=version;
  }
  c.imageSmoothingEnabled=false;c.drawImage(cache.off,0,0,w,h);
}
function eyePlot(canvas,traces){if(!canvas||!traces?.length)return;const z=canvasCtx(canvas);if(!z)return;const[c,w,h]=z;grid(c,w,h);let lim=.001;for(const t of traces)for(const v of t)lim=Math.max(lim,Math.abs(v));lim*=1.1;c.strokeStyle='rgba(226,169,59,.20)';c.lineWidth=1;for(const t of traces.slice(0,48)){c.beginPath();for(let i=0;i<t.length;i++){const x=i*(w-1)/(t.length-1||1),y=h-(t[i]+lim)/(2*lim)*h;i?c.lineTo(x,y):c.moveTo(x,y)}c.stroke()}}
function heatmap(canvas,flat,shape,opt={}) {
  if(!canvas||!flat?.length||!shape?.length)return;const[c,w,h]=canvasCtx(canvas);const[H,W]=shape;
  const off=document.createElement('canvas');off.width=W;off.height=H;const oc=off.getContext('2d',{alpha:false}),img=oc.createImageData(W,H);
  for(let p=0;p<flat.length;p++){const q=Math.max(0,Math.min(1,(flat[p]+80)/80));img.data[p*4]=Math.round(20+220*q);img.data[p*4+1]=Math.round(20+170*Math.max(0,q-.3));img.data[p*4+2]=Math.round(35+70*(1-q));img.data[p*4+3]=255;}
  oc.putImageData(img,0,0);c.imageSmoothingEnabled=false;c.drawImage(off,0,0,w,h);c.fillStyle='#e3e9ed';c.font='10px Consolas,monospace';
  if(opt.x){c.fillText(fmt(opt.x[0],opt.xUnit),6,h-6);const text=fmt(opt.x.at(-1),opt.xUnit);c.fillText(text,Math.max(6,w-c.measureText(text).width-6),h-6);}
  if(opt.y){c.fillText(fmt(opt.y[0],opt.yUnit),6,13);c.fillText(fmt(opt.y.at(-1),opt.yUnit),6,h-22);}
}

function drawTx() {updateLiveMetrics();}

function lowerBound(a,v){let l=0,r=a?.length||0;while(l<r){const m=(l+r)>>1;if(a[m]<v)l=m+1;else r=m;}return l;}
function drawSpectrumTrace(c,w,h,x,y,left,right,low,high,color){
  if(!x?.length||!y?.length)return;let a=Math.max(0,lowerBound(x,left)-1),b=Math.min(x.length,lowerBound(x,right)+1);if(b-a<2)return;
  const maxPoints=Math.max(160,Math.floor(w*1.35)),step=Math.max(1,Math.floor((b-a)/maxPoints));c.strokeStyle=color;c.lineWidth=1.15;c.beginPath();let active=false;
  for(let i=a;i<b;i+=step){let jEnd=Math.min(b,i+step),k=i;if(step>1){for(let j=i+1;j<jEnd;j++)if(y[j]>y[k])k=j;}const f=x[k],v=y[k];if(!Number.isFinite(f)||!Number.isFinite(v))continue;const px=(f-left)/(right-left)*w,py=20+(high-v)/(high-low)*(h-40);if(active)c.lineTo(px,py);else c.moveTo(px,py);active=true;}c.stroke();
}
function shadeSpectrumUnavailable(c,w,h,left,right){
  const d=spectrumDataBounds(),span=Math.max(1e-12,right-left);c.save();c.fillStyle='rgba(4,7,9,.72)';c.font='10px Consolas, monospace';c.textAlign='center';
  const block=(a,b)=>{a=Math.max(left,a);b=Math.min(right,b);if(b<=a)return;const x0=(a-left)/span*w,x1=(b-left)/span*w;c.fillRect(x0,0,x1-x0,h);if(x1-x0>115){c.fillStyle='rgba(137,151,160,.72)';c.fillText('OUTSIDE SOURCE NYQUIST',(x0+x1)/2,h/2);c.fillStyle='rgba(4,7,9,.72)';}};
  block(left,Math.min(right,d.left));block(Math.max(left,d.right),right);c.restore();
}
function spectrumWaterfallViewRow(row,points=512){
  const x=state.live.freq,b=spectrumViewBounds();if(!row?.length||!x?.length||row.length!==x.length)return row;
  const out=new Float32Array(points);out.fill(NaN);const first=Number(x[0]),last=Number(x.at(-1));
  for(let i=0;i<points;i++){const hz=b.left+(i+.5)/points*b.span;if(hz<first||hz>last)continue;let k=lowerBound(x,hz);if(k>=x.length)k=x.length-1;else if(k>0&&Math.abs(x[k-1]-hz)<Math.abs(x[k]-hz))k--;out[i]=row[k];}
  return out;
}
function drawSpectrumPage() {
  const canvas=$('#specMain');
  if(canvas&&state.live.psd.length&&state.live.freq.length) {
    const z=canvasCtx(canvas);if(!z)return;const[c,w,h]=z;grid(c,w,h);const b=spectrumViewBounds(),left=b.left,right=b.right;
    let visiblePeak=-Infinity,a=Math.max(0,lowerBound(state.live.freq,left)-1),e=Math.min(state.live.psd.length,lowerBound(state.live.freq,right)+1);for(let i=a;i<e;i++)if(Number.isFinite(state.live.psd[i]))visiblePeak=Math.max(visiblePeak,state.live.psd[i]);
    const high=state.specDisplay.auto&&Number.isFinite(visiblePeak)?Math.ceil((visiblePeak+8)/10)*10:Number(state.specDisplay.ref),low=high-Number(state.specDisplay.range);
    for(const t of state.traces)if(t.mode!=='off')drawSpectrumTrace(c,w,h,state.live.freq,t.y,left,right,low,high,t.color);
    for(const m of state.memories)drawSpectrumTrace(c,w,h,m.x,m.y,left,right,low,high,m.color);
    shadeSpectrumUnavailable(c,w,h,left,right);
    c.fillStyle='#a9b5bd';c.font='11px Consolas, monospace';c.fillText(fmt(left,'Hz'),8,h-7);const rtxt=fmt(right,'Hz');c.fillText(rtxt,Math.max(8,w-c.measureText(rtxt).width-8),h-7);c.fillText(`${fmt(high,'dBm')}`,8,15);c.fillText(fmt(low,'dBm'),8,h-24);
    state.markers.forEach((m,i)=>{if(m.hz<left||m.hz>right)return;const q=nearestSpectrumPoint(m.hz);if(!q)return;const x=(q.hz-left)/b.span*w;c.strokeStyle=i===state.activeMarker?'#ffffff':'#a8b3ba';c.lineWidth=i===state.activeMarker?1.4:1;c.beginPath();c.moveTo(x,0);c.lineTo(x,h);c.stroke();c.fillStyle=c.strokeStyle;c.fillText(`M${i+1}`,Math.min(w-28,Math.max(3,x+3)),30+(i%3)*12);});
    const active=state.markers[state.activeMarker],q=active?nearestSpectrumPoint(active.hz):null,ref=Number.isInteger(state.specDeltaRef)?state.markers[state.specDeltaRef]:null,qr=ref?nearestSpectrumPoint(ref.hz):null;if($('#sMarker'))$('#sMarker').textContent=q?(qr&&state.specDeltaRef!==state.activeMarker?`M${state.activeMarker+1} ${fmt(q.hz,'Hz')} / ${fmt(q.db,'dBm')} · Δ ${fmt(q.hz-qr.hz,'Hz')} / ${fmt(q.db-qr.db,'dB')}`:`M${state.activeMarker+1} ${fmt(q.hz,'Hz')} / ${fmt(q.db,'dBm')}`):'—';
    const legend=state.traces.filter(t=>t.mode!=='off').map(t=>`T${state.traces.indexOf(t)+1} ${t.mode}`).join('   ');c.fillStyle='#a9b5bd';c.fillText(legend,130,15);
  }
  const row=spectrumWaterfallViewRow(state.live.waterfallRow);waterfall($('#specWaterfall'),row);updateLiveMetrics();
}

function setPlotLabel(id,text){const c=$('#'+id);const l=c?.parentElement?.querySelector('.plot-label');if(l)l.textContent=text}
function drawRxLive() {}

function drawRxAnalysis() {
  const d=state.rxData;if(!d)return;
  if(state.rxPage==='framing'){drawFrameAnalysis();return;}
  const before=state.rxPage==='equalizer'&&d.telemetry.pre_equalizer_i?{i:d.telemetry.pre_equalizer_i,q:d.telemetry.pre_equalizer_q}:d.raw_symbols;
  scatter($('#rxRaw'),before?.i,before?.q,{ideals:idealConstellation(state.rx.modulation,state.rx),dimIdeals:true});eyePlot($('#rxEye'),d.telemetry.eye_i);
  linePlot($('#rxTiming'),d.telemetry.timing_error||[0,0]);linePlot($('#rxEqError'),d.telemetry.equalizer_error||[0,0]);linePlot($('#rxChannel'),d.telemetry.channel_estimate_taps||[0,0],{min:0});
  const m=d.metrics||{};for(const [id,v,u]of [['rxEvm',m.evm_rms_percent,'%'],['rxMer',m.mer_db,'dB'],['rxCoarse',d.telemetry.coarse_cfo_hz,'Hz'],['rxCfo',d.telemetry.costas_freq_hz?.at(-1),'Hz']])if($('#'+id))$('#'+id).textContent=fmt(v,u);
  if($('#rxDemod'))$('#rxDemod').textContent=state.rx.modulation.toUpperCase();
  if($('#rxText')){const output=state.rxOutput==='bits'?(d.bits||[]).join('').match(/.{1,64}/g)?.join('\n'):state.rxOutput==='hex'?d.hex:d.text;$('#rxText').textContent=output?.slice(0,1600)||'No decoded output for this waveform.';}
  if($('#rxSupport'))$('#rxSupport').textContent=`${state.tx.data_source==='text'?'Text payload; phase ambiguity or loss of lock may corrupt decoding.':'PN/raw bit source; decoded bytes need not form readable text.'} Capture ${d.tx_metadata?.capture_id??'—'} · t ${fmt(d.tx_metadata?.simulation_time_s,'s')} · ${d.tx_metadata?.source_revision||'—'}.`;
  if(state.rxPage==='equalizer')setPlotLabel('rxChannel',d.telemetry.channel_estimate_taps?'Channel estimate · reference-aided tap magnitude':'Channel estimate unavailable for this format');
  drawRxSyncFrame(state.rxReplay);
}

function drawFrameAnalysis(){
  const d=state.rxData,f=d?.frame;if(!d)return;
  const before=d.raw_symbols||d.symbols;scatter($('#rxRaw'),before?.i,before?.q,{ideals:null});
  if(!f){if($('#rxFrameResult'))$('#rxFrameResult').innerHTML='<p class="hint">Enable frame acquisition to run energy detection and preamble correlation.</p>';for(const id of ['rxFrameDetected','rxFrameStart','rxFramePeak','rxFrameEnergyStart'])if($('#'+id))$('#'+id).textContent='—';return;}
  linePlot($('#rxFrameEnergy'),f.energy?.dbm||[],{threshold:f.energy?.threshold_dbm});
  linePlot($('#rxFrameCorr'),f.correlation?.score||[],{min:0,max:1,threshold:f.correlation?.threshold});
  if($('#rxFrameDetected'))$('#rxFrameDetected').textContent=f.detected?'YES':'NO';
  if($('#rxFrameStart'))$('#rxFrameStart').textContent=f.start_sample==null?'—':`${f.start_sample} · ${fmt(f.start_time_s,'s')}`;
  if($('#rxFramePeak'))$('#rxFramePeak').textContent=f.correlation?.peak==null?'—':Number(f.correlation.peak).toFixed(3);
  if($('#rxFrameEnergyStart'))$('#rxFrameEnergyStart').textContent=f.energy?.first_crossing_sample==null?'—':`${f.energy.first_crossing_sample} samples`;
  const blocks=f.decoded_blocks||f.blocks||[],box=$('#rxFrameResult');if(box)box.innerHTML=blocks.length?`<table class="result-table"><thead><tr><th>Block</th><th>Role / modulation</th><th>Samples</th><th>Decode</th></tr></thead><tbody>${blocks.map((b,i)=>`<tr><td>${esc(b.name||`Block ${i+1}`)}</td><td>${esc((b.role||'')+' · '+(b.modulation||''))}</td><td>${b.start_sample??'—'}…${b.end_sample??'—'}</td><td>${esc(b.status||'located')}${b.bit_count!=null?` · ${b.bit_count} bits`:''}${b.metrics?.evm_rms_percent!=null?` · EVM ${fmt(b.metrics.evm_rms_percent,'%')}`:''}${b.error?` · ${esc(b.error)}`:''}</td></tr>`).join('')}</tbody></table>`:`<p class="hint">${f.detected?'Frame start found, but no expected blocks are configured.':'No frame start met the enabled detector thresholds.'}</p>`;
}

function drawRxSyncFrame(p){
  const d=state.rxData, I=d?.symbols?.i||[], Q=d?.symbols?.q||[];if(!I.length)return;
  const n=Math.min(I.length,Math.max(20,Math.round(state.rxDisplay.sync_window))),end=Math.max(1,Math.min(I.length,Math.floor((p%1)*I.length))),start=Math.max(0,end-n);
  scatter($('#rxSync'),I.slice(start,end),Q.slice(start,end),{ideals:idealConstellation(state.rx.modulation,state.rx),scale:d.symbols.scale});
  const v=$('#rxSyncValue');if(v){const frac=end/Math.max(1,I.length),f=d.telemetry?.costas_freq_hz||[];const k=Math.min(f.length-1,Math.max(0,end-1));v.textContent=state.rx.use_costas&&f.length?`symbol ${end} · ${fmt(f[k],'Hz')}`:`symbol ${end}`}
  const lm=state.rxDisplay.loop_trace;const loop=lm==='phase'?(d.telemetry?.costas_phase_deg||[]):lm==='error'?(d.telemetry?.costas_error||[]):lm==='timing-error'?(d.telemetry?.timing_error||[]):lm==='timing-mu'?(d.telemetry?.timing_mu||[]):(d.telemetry?.costas_freq_hz||[]);if(loop.length)linePlot($('#rxLoop'),loop.slice(0,I.length),{cursor:end/Math.max(1,I.length)});
}
function animateRxSync() {
  let previous=performance.now(),lastPaint=0;
  const loop=now=>{if(state.mode!=='rx')return;const dt=(now-previous)/1000;previous=now;if(!state.rxDisplay.sync_pause){const n=state.rxData?.symbols?.i?.length||1;state.rxReplay=(state.rxReplay+dt*state.rxDisplay.sync_speed/Math.max(2,n/700))%1;}if(now-lastPaint>=1000/30){lastPaint=now;drawRxSyncFrame(state.rxReplay);}state.timers.rxAnim=requestAnimationFrame(loop);};state.timers.rxAnim=requestAnimationFrame(loop);
}

function drawBlindPlots(){linePlot($('#blindSpectrum'),state.live.psd,{min:-140});waterfall($('#blindWaterfall'),state.live.waterfallRow)}
function drawBlindResult() {
  const d=state.blindData;if(!d)return;const reported=d.confidence>=state.blindDisplay.threshold?d.modulation:'uncertain';
  for(const [id,value]of [['bClass',reported.toUpperCase()],['bConf',`${(100*d.confidence).toFixed(0)} / 100`],['bRs',fmt(d.symbol_rate_baud,'Bd')],['bSnr',fmt(d.snr_db,'dB')],['bBw',fmt(d.bandwidth_99_hz,'Hz')]])if($('#'+id))$('#'+id).textContent=value;
  const features=`<table class="result-table"><tr><td>PAPR</td><td>${fmt(d.features?.papr_db,'dB')}</td></tr><tr><td>Envelope CV</td><td>${fmt(d.features?.envelope_cv)}</td></tr><tr><td>Clock peak</td><td>${fmt(d.features?.clock_peak_db,'dB')}</td></tr><tr><td>Analysis time</td><td>${fmt(d.elapsed_ms,'ms')}</td></tr></table>`;
  const notes=`<ul class="result-notes">${d.notes.map(n=>`<li>${esc(n)}</li>`).join('')}</ul>`;
  if($('#blindResult'))$('#blindResult').innerHTML=`<div class="result-primary"><strong>${esc(reported)}</strong><span>${esc(d.family)} · heuristic score ${(d.confidence*100).toFixed(0)}/100</span></div>${features}${notes}`;
  if($('#blindEvidence'))$('#blindEvidence').innerHTML=`<h3>Constellation hypotheses</h3>${d.candidates?.length?`<table class="result-table"><thead><tr><th>Candidate</th><th>Residual EVM</th></tr></thead><tbody>${d.candidates.map(c=>`<tr><td>${esc(c.modulation)}</td><td>${fmt(c.evm_percent,'%')}</td></tr>`).join('')}</tbody></table>`:'<p class="hint">A linear digital constellation fit was not used for this observation.</p>'}${features}${notes}<p class="hint">Candidate fits use bounded rate, phase and RRC hypotheses. Additional signal classes require their own detectors.</p>`;
  if($('#blindHistory'))$('#blindHistory').innerHTML=`<table class="history-table"><thead><tr><th>First</th><th>Last</th><th>Hypothesis</th><th>Score</th><th>Count</th><th>Source revision</th></tr></thead><tbody>${state.blindHistory.slice().reverse().map(h=>`<tr><td>${h.first}</td><td>${h.last}</td><td>${esc(h.label)}</td><td>${(h.score*100).toFixed(0)}</td><td>${h.count}</td><td>${h.revision}</td></tr>`).join('')}</tbody></table>`;
  linePlot($('#blindHistogram'),d.features?.amplitude_histogram,{x:d.features?.amplitude_bins,min:0});drawBlindPlots();
}

function drawRadarStatic() {
  const d=state.radarData;if(!d)return;
  if($('#rUpdate'))$('#rUpdate').textContent='cached';
  linePlot($('#radarSpectrum'),d.psd_db,{min:-140,x:d.freq_hz,xUnit:'Hz'});linePlot($('#radarAuto'),d.autocorr_db,{min:-80,max:1,x:d.autocorr_delay_s,xUnit:'s'});
  heatmap($('#radarAmb'),d.ambiguity_db,d.ambiguity_shape,{x:d.ambiguity_delay_s,y:d.ambiguity_doppler_hz,xUnit:'s',yUnit:'Hz'});
  for(const[id,v,u]of [['rPeak',d.metrics.peak_hz,'Hz'],['rBw',d.metrics.occupied_bw_hz,'Hz']])if($('#'+id))$('#'+id).textContent=fmt(v,u);
  if($('#rMode'))$('#rMode').textContent=(d.metadata.waveform||state.radar.mode).toUpperCase();
  if($('#radarDesignNotes'))$('#radarDesignNotes').innerHTML=`<h3>Waveform properties</h3><table class="result-table"><tr><td>99% occupied bandwidth</td><td>${fmt(d.metrics.occupied_bw_hz,'Hz')}</td></tr><tr><td>Crest factor</td><td>${fmt(d.metrics.crest_factor_db,'dB')}</td></tr><tr><td>Actual RBW</td><td>${fmt(d.metrics.rbw_hz,'Hz')}</td></tr></table><p class="hint">Autocorrelation shows zero-Doppler compression. Ambiguity shows sensitivity to delay and Doppler for the reference pulse. These are design diagnostics, not measured target returns.</p>`;
  drawRadarDynamic(state.radarReplay);drawRadarTimeline(state.radarReplay);
}

function circularSlice(a,start,n){if(!a?.length)return[];const out=new Array(Math.min(n,a.length));for(let i=0;i<out.length;i++)out[i]=a[(start+i)%a.length];return out}
function drawRadarDynamic(p){
  const d=state.radarData;if(!d?.envelope?.length)return;
  const N=d.envelope.length,n=Math.max(16,Math.min(N,Math.round(N*state.radarDisplay.window_fraction))),start=Math.floor((p%1)*N);
  const x=Array.from({length:Math.min(N,n)},(_,i)=>i*(d.time_s[1]||1/state.radar.sample_rate));
  linePlot($('#radarEnvelope'),circularSlice(d.envelope,start,n),{min:0,x,xUnit:'s'});
  const trace=state.radarDisplay.trace||'frequency';
  linePlot($('#radarFreq'),circularSlice(trace==='phase'?d.phase_deg:trace==='iq'?d.i:d.inst_freq_hz,start,n),{x,xUnit:'s',secondary:trace==='iq'?circularSlice(d.q,start,n):undefined});
  setPlotLabel('radarFreq',trace==='phase'?'Phase / code (°)':trace==='iq'?'I / Q · waveform playback':'Instantaneous frequency (Hz)');
  drawRadarTimeline(p);
}

function drawRadarTimeline(p){
  const c0=$('#radarTimeline'),d=state.radarData;if(!c0||!d)return;const z=canvasCtx(c0);if(!z)return;const[c,w,h]=z;grid(c,w,h);const timeline=d.metadata?.timeline||[],logical=Number(d.metadata?.logical_sequence_s||state.radar.duration_s||1),t=(p%1)*Number(state.radar.duration_s||logical),tl=logical>0?t%logical:t;
  c.fillStyle='#1a2024';c.fillRect(0,30,w,Math.max(12,h-50));
  if(timeline.length){const mono=getComputedStyle(document.documentElement).getPropertyValue('--mono');for(let i=0;i<timeline.length;i++){const e=timeline[i],start=Number(e.start_s||0),dur=Number(e.pri_s||e.duration_s||e.width_s||0),x=start/logical*w,ww=Math.max(1,dur/logical*w);c.fillStyle=e.kind==='pulse'?(i%2?'#69572f':'#7b6433'):'#20272c';c.fillRect(x,34,ww,Math.max(8,h-58));if(ww>42){c.fillStyle='#c6cdd1';c.font=`9px ${mono}`;c.fillText(String(e.name||e.kind).slice(0,16),x+4,49)}}}
  else{c.fillStyle='#66562f';c.fillRect(0,34,w,Math.max(8,h-58))}
  const x=(logical>0?tl/logical:0)*w;c.strokeStyle='#f0d49b';c.lineWidth=1.3;c.beginPath();c.moveTo(x,24);c.lineTo(x,h);c.stroke();
  let active=null;for(const e of timeline){const dur=Number(e.pri_s||e.duration_s||e.width_s||0);if(tl>=Number(e.start_s||0)&&tl<Number(e.start_s||0)+dur){active=e;break}}$('#rActive')&&($('#rActive').textContent=(active?.name||d.metadata?.waveform||state.radar.mode).toUpperCase());$('#rTime')&&($('#rTime').textContent=fmt(tl,'s'));const v=$('#radarTimelineValue');if(v)v.textContent=`${fmt(tl,'s')} / ${fmt(logical,'s')}`;
}
function animateRadarPlayback() {
  let prior=performance.now(),last=0;const loop=now=>{if(state.mode!=='radar'||state.radarPage!=='design')return;state.radarReplay=(state.radarReplay+(now-prior)/1000*state.radarDisplay.playback_speed)%1;prior=now;if(now-last>=1000/30){last=now;drawRadarDynamic(state.radarReplay);}state.timers.radarAnim=requestAnimationFrame(loop);};state.timers.radarAnim=requestAnimationFrame(loop);
}

function subnav(items,current,attribute='view') {
  return `<div class="subnav">${items.map(([key,label])=>`<button data-${attribute}="${key}" class="${key===current?'active':''}">${label}</button>`).join('')}</div>`;
}

function fold(title,body,open=false) {return `<details class="control-group" ${open?'open':''}><summary>${title}</summary>${body}</details>`;}

function bindView(items,callback,attribute='view') { $$(`[data-${attribute}]`).forEach(b=>b.onclick=()=>callback(b.dataset[attribute])); }

function followTransmitter() {
  if(!state.rxFollow)return;
  const allowed=Object.values(signalFamilies).flat();
  if(allowed.includes(state.tx.waveform))state.rx.modulation=state.tx.waveform;
  state.rx.tuner_hz=state.tx.center_hz;
  if(state.tx.data_source!=='text'&&state.rxOutput==='text')state.rxOutput='bits';
  for(const key of ['sample_rate','symbol_rate','rolloff','filter_span_symbols','pulse_shape','apsk_ratio2','apsk_ratio3','ofdm_fft','ofdm_cp','ofdm_occupied','fsk_deviation_hz','fm_deviation_hz'])state.rx[key]=state.tx[key];
  state.rx.frame_enabled=!!state.tx.frame_enabled;
  if(state.tx.frame_enabled){
    if(state.tx.frame_mode==='blocks')state.rx.frame_blocks=structuredClone(state.tx.frame_blocks||[]);
    else if(frameModulations.includes(state.tx.waveform))state.rx.frame_blocks=[
      {name:'Preamble',role:'preamble',modulation:state.tx.waveform,bit_count:Math.max(1,(state.tx.preamble_bits||'1').length),bits:state.tx.preamble_bits||'1',symbol_rate:state.tx.symbol_rate,power_offset_db:0,repeat:1},
      {name:'Payload',role:'payload',modulation:state.tx.waveform,bit_count:Number(state.tx.payload_bits||256)+(state.tx.frame_crc?16:0),bits:'1',symbol_rate:state.tx.symbol_rate,power_offset_db:0,repeat:1}
    ];
  }
}

function radarPulseEditor(item,index) {
  let html=radarPulseEditorOriginal(item,index);
  if(item?.kind==='off')return html;
  const timing=html.indexOf('<div class="field"><label>Repeat');
  const more=html.indexOf('<div class="field"><label>PRI jitter');
  const title=field('Item name',`radar.sequence.${index}.name`,item.name,{type:'string'});
  return title+html.slice(0,timing)+fold('Repetition / timing',html.slice(timing,more),true)+fold('Agility / item offsets',html.slice(more));
}

function renderSegmentEditor() {
  const tabs=$('.dock-scroll .segmented');
  if(tabs){const b=document.createElement('button');b.textContent='Segments';b.className=state.radar.mode==='segments'?'active':'';b.onclick=()=>{state.radar.mode='segments';renderRadar();scheduleCurrent('radar.mode');};tabs.append(b);tabs.classList.add('cols3');}
  if(state.radar.mode!=='segments')return;
  const content=group('Segmented waveform',`<p class="hint">Ordered constant-amplitude segments with linear frequency sweeps and phase jumps. Phase accumulates through the sequence; the record repeats at its boundary. This is a bounded segment editor, not arbitrary IQ import.</p>`)+state.radar.segments.map((v,i)=>fold(`Segment ${i+1}`,field('Duration (µs)',`radar.segments.${i}.duration_us`,v.duration_us,{min:.01,step:1})+field('Start frequency',`radar.segments.${i}.start_hz`,v.start_hz,{step:1000})+field('Stop frequency',`radar.segments.${i}.stop_hz`,v.stop_hz,{step:1000})+field('Amplitude',`radar.segments.${i}.amplitude`,v.amplitude,{min:0,max:2,step:.1})+field('Phase jump (°)',`radar.segments.${i}.phase_deg`,v.phase_deg,{step:1})+`<button class="btn small" data-segment-remove="${i}">Remove</button>`,i===0)).join('')+group('Sequence',`<button id="addSegment" class="btn">Add segment</button>`);
  const scroll=$('.dock-scroll'),groups=[...scroll.children];groups.slice(2).forEach(el=>el.remove());scroll.insertAdjacentHTML('beforeend',content);
  bindControls(scroll,path=>scheduleCurrent(path));
  $('#addSegment').onclick=()=>{if(state.radar.segments.length>=128)return;state.radar.segments.push({duration_us:25,start_hz:0,stop_hz:0,amplitude:1,phase_deg:0});renderRadar();scheduleCurrent('radar.segments');};
  $$('[data-segment-remove]').forEach(b=>b.onclick=()=>{if(state.radar.segments.length===1)return;state.radar.segments.splice(Number(b.dataset.segmentRemove),1);renderRadar();scheduleCurrent('radar.segments');});
}

function renderRadar() {
  renderRadarOriginal();
  renderSegmentEditor();
  const gridEl=$('.plot-grid.radar');gridEl.className='plot-grid with-views';
  let plots='';
  if(state.radarPage==='design')plots=plotPane('Sequence · cached waveform replay','radarTimeline','full')+plotPane('Envelope · playback window','radarEnvelope')+plotPane('Instantaneous frequency (Hz)','radarFreq');
  if(state.radarPage==='diagnostics')plots=plotPane('Spectrum · full waveform record','radarSpectrum')+plotPane('Autocorrelation · first active pulse','radarAuto')+plotPane('Delay–Doppler ambiguity · first active pulse','radarAmb')+`<div class="plot-pane"><div class="plot-label">Design interpretation</div><div class="result-panel" id="radarDesignNotes"></div></div>`;
  if(state.radarPage==='returns')plots=plotPane('Receive envelope · current coherent burst','radarReceive')+plotPane('Matched-filter range profile','radarRange')+plotPane('Range–Doppler · range horizontal, velocity vertical','radarRD')+`<div class="plot-pane"><div class="plot-label">Simulation / measurement</div><div class="result-panel" id="radarReturns"></div></div>`;
  gridEl.innerHTML=subnav([['design','Waveform / sequence'],['diagnostics','Waveform diagnostics'],['returns','Targets / returns']],state.radarPage,'radarpage')+`<div class="view-plots ${state.radarPage==='design'?'radar-design':'quad'}">${plots}</div>`;
  if(state.radarPage==='returns') {
    $('.dock-scroll').innerHTML=group('Coherent burst experiment',`<p class="hint">Test the selected pulse at its nominal, fixed PRI. Sequence jitter and mode scheduling are not applied to this return experiment.</p>`+field('Carrier (Hz)','scene.carrier_hz',state.scene.carrier_hz,{min:1e6,max:1e11,step:1e8})+field('Pulses / burst','scene.pulses',state.scene.pulses,{min:2,max:256,step:1})+sliderField('Noise amplitude (dB)','scene.noise_db',state.scene.noise_db,{min:-80,max:0,step:1})+checkField('Pause target motion','scenePaused',state.scenePaused)+`<button id="restartScene" class="btn">Reset simulation time</button>`)+state.scene.targets.map((t,i)=>fold(esc(t.name),field('Name',`scene.targets.${i}.name`,t.name,{type:'string'})+field('Initial range (m)',`scene.targets.${i}.range_m`,t.range_m,{min:0,step:100})+field('Radial velocity (m/s)',`scene.targets.${i}.velocity_mps`,t.velocity_mps,{min:-3000,max:3000,step:5})+sliderField('Echo gain (dB)',`scene.targets.${i}.gain_db`,t.gain_db,{min:-80,max:0,step:1})+`<button class="btn small" data-target-remove="${i}">Remove target</button>`,true)).join('')+group('Targets',`<button class="btn" id="addTarget">Add target</button><p class="hint">Positive velocity is receding. Targets have relative echo gain, not a calibrated radar cross section. A single receive channel cannot determine azimuth or elevation.</p>`);
    bindControls($('.dock-scroll'),()=>refreshScene(true));
    $('#restartScene').onclick=()=>{state.sceneElapsed=0;state.sceneClock=performance.now();refreshScene(true);};
    $('#addTarget').onclick=()=>{if(state.scene.targets.length>=8)return;state.scene.targets.push({name:`Target ${state.scene.targets.length+1}`,range_m:6500,velocity_mps:-20,gain_db:-12});renderRadar();refreshScene(true);};
    $$('[data-target-remove]').forEach(b=>b.onclick=()=>{state.scene.targets.splice(Number(b.dataset.targetRemove),1);renderRadar();refreshScene(true);});
  } else if(state.radarPage==='diagnostics') {
    $('.dock-scroll').innerHTML=group('Waveform diagnostics',`<p class="hint">These are properties of the designed waveform. They update after edits; they do not change just because the playback cursor moves.</p><p class="hint">The ambiguity surface is a bounded, sampled visualization. Narrow sidelobes may require a higher-rate analysis than this display provides.</p>`)+group('Current design',`<p>${esc(state.radar.mode==='sequence'?'Pulse sequence':state.radar.mode==='segments'?'Segmented waveform':state.radar.continuous_waveform.toUpperCase())}</p><p class="hint">${fmt(state.radar.sample_rate,'S/s')} · ${fmt(state.radar.duration_s,'s')} record</p>`);
  }
  bindView([],v=>{state.radarPage=v;renderRadar();startMode();},'radarpage');
  if(state.radarPage==='returns')$('.measurebar').innerHTML=measure('Simulation time','sceneTime')+measure('Strongest range','sceneRange')+measure('Radial velocity','sceneVelocity')+measure('Burst','scenePulses',`${state.scene.pulses} pulses`);
  if(state.radarPage==='design'){const scroll=$('.dock-scroll');scroll.insertAdjacentHTML('beforeend',group('Playback trace',selectField('Secondary view','radarDisplay.trace',state.radarDisplay.trace||'frequency',[['frequency','Instantaneous frequency'],['phase','Phase / code'],['iq','I / Q']])));bindControls(scroll.lastElementChild,()=>drawRadarDynamic(state.radarReplay));}
  drawRadarStatic();drawScene();
}

function resetTraces() {for(const t of state.traces)t.y=null;state.traceSignature='';}

async function refreshMeasurements(force=false) {
  if(state.zeroFrozen && state.specPage==='zero')return;
  if(state.busy.measurement){if(force)state.pending.measurement=true;return;}
  const version=state.revision, params=JSON.stringify(state.measurement);
  state.busy.measurement=true;
  try {const d=await jsonPost('/api/measurements',{tx:cleanTx(),...state.measurement,quality:state.specPage==='quality'});if(version!==state.revision||params!==JSON.stringify(state.measurement))return;state.measurementData=d;if(state.mode==='spectrum')drawMeasurements();}
  catch(e){toast(shortError(e));}finally{state.busy.measurement=false;if(state.pending.measurement){state.pending.measurement=false;refreshMeasurements(true);}}
}

function drawMeasurements() {
  const d=state.measurementData;if(!d)return;
  const opts={x:d.time_s,xUnit:'s'};
  linePlot($('#zeroIQ'),d.i,{...opts,secondary:d.q});linePlot($('#zeroAM'),d.amplitude,{...opts,min:0});linePlot($('#zeroPM'),d.phase_deg.map((v,i)=>d.valid[i]?v:NaN),opts);linePlot($('#zeroFM'),d.frequency_hz.map((v,i)=>d.valid[i]?v:NaN),opts);
  const q=d.quality;if(!q){if(state.specPage==='quality')setStatus('Quality view unavailable for this waveform');return;}
  if(q.kind==='basic'){
    scatter($('#qualityConst'),q.symbols?.i,q.symbols?.q,{ideals:null});
    linePlot($('#qualityEvm'),q.telemetry?.amplitude||[],{min:0});linePlot($('#qualityMag'),q.telemetry?.frequency_hz||[]);linePlot($('#qualityPhase'),q.telemetry?.phase_deg||[]);
    if($('#qEvm'))$('#qEvm').textContent=fmt(q.metrics?.rms);
    if($('#qPeak'))$('#qPeak').textContent=fmt(q.metrics?.peak);
    if($('#qMer'))$('#qMer').textContent=fmt(q.metrics?.mean_frequency_hz,'Hz');
    if($('#qMethod'))$('#qMethod').textContent=q.method||'direct waveform';
  }else{
    scatter($('#qualityConst'),q.symbols?.i,q.symbols?.q,{ideals:idealConstellation(state.tx.waveform)});
    linePlot($('#qualityEvm'),q.telemetry?.evm_time_percent||[],{min:0});linePlot($('#qualityMag'),q.telemetry?.magnitude_error_percent||[]);linePlot($('#qualityPhase'),q.telemetry?.phase_error_deg||[]);
    for(const [id,key,unit]of [['qEvm','evm_rms_percent','%'],['qPeak','evm_peak_percent','%'],['qMer','mer_db','dB']])if($('#'+id))$('#'+id).textContent=fmt(q.metrics?.[key],unit);
    if($('#qMethod'))$('#qMethod').textContent=q.method||'decision-directed';
  }
}

async function refreshScene(force=false) {
  if(state.mode!=='radar'||state.radarPage!=='returns')return;if(state.scenePaused&&!force)return;
  if(state.busy.scene){if(force)state.pending.scene=true;return;}
  const now=performance.now();if(!state.scenePaused)state.sceneElapsed+=Math.min(1,(now-state.sceneClock)/1000);state.sceneClock=now;
  const signature=JSON.stringify([state.radar,state.scene,state.radarSelected]);state.busy.scene=true;
  try {const d=await jsonPost('/api/radar/scene',{radar:state.radar,...state.scene,pulse_index:state.radarSelected,elapsed_s:state.sceneElapsed});if(signature!==JSON.stringify([state.radar,state.scene,state.radarSelected]))return;state.sceneData=d;drawScene();}
  catch(e){toast(shortError(e));}finally{state.busy.scene=false;if(state.pending.scene){state.pending.scene=false;refreshScene(true);}}
}

function drawScene() {
  const d=state.sceneData,box=$('#radarReturns');if(!d||!box)return;
  if($('#scenePulses'))$('#scenePulses').textContent=`${state.scene.pulses} pulses`;
  if(!d.available){box.textContent=d.reason;return;}
  for(const[id,v,u]of [['sceneTime',d.time_s,'s'],['sceneRange',d.peak_range_m,'m'],['sceneVelocity',d.peak_velocity_mps,'m/s']])if($('#'+id))$('#'+id).textContent=fmt(v,u);
  linePlot($('#radarReceive'),d.receive_envelope,{min:0,x:d.receive_time_s,xUnit:'s'});linePlot($('#radarRange'),d.range_db,{min:-80,max:5,x:d.range_m,xUnit:'m'});
  heatmap($('#radarRD'),d.rd_db,d.rd_shape,{x:d.range_m,y:d.velocity_mps,xUnit:'m',yUnit:'m/s'});
  box.innerHTML=`<h3>t = ${d.time_s.toFixed(1)} s</h3><table class="result-table"><tr><td>Strongest range bin</td><td>${fmt(d.peak_range_m,'m')}</td></tr><tr><td>Strongest velocity bin</td><td>${fmt(d.peak_velocity_mps,'m/s')}</td></tr><tr><td>Range interval</td><td>0…${fmt(d.unambiguous_range_m,'m')}</td></tr><tr><td>Velocity interval</td><td>±${fmt(d.unambiguous_velocity_mps,'m/s')}</td></tr><tr><td>Velocity bin width</td><td>${fmt(d.velocity_bin_mps,'m/s')}</td></tr></table><p class="hint">Strongest-bin readings, not a CFAR detector. Slow velocities may share a Doppler bin.</p><h4>Simulation truth</h4>${d.truth.map(t=>`<div class="hint">${esc(t.name)} · ${fmt(t.range_m,'m')} · ${fmt(t.velocity_mps,'m/s')}</div>`).join('')}${d.warnings.map(w=>`<p class="error-text">${esc(w)}</p>`).join('')}`;
}

function idealConstellation(mod,format=state.tx) {
  mod={dbpsk:'bpsk',dqpsk:'qpsk','pi4-dqpsk':'qpsk',oqpsk:'qpsk'}[mod]||mod;
  if(['16apsk','32apsk'].includes(mod)){const counts=mod==='16apsk'?[4,12]:[4,12,16],radii=[1,format.apsk_ratio2,format.apsk_ratio3],points=[];let power=0;
    counts.forEach((n,j)=>{for(let i=0;i<n;i++){const phase=2*Math.PI*i/n+Math.PI/n;points.push([radii[j]*Math.cos(phase),radii[j]*Math.sin(phase)]);power+=radii[j]**2;}});const norm=Math.sqrt(power/points.length);return points.map(([i,q])=>[i/norm,q/norm]);}
  return idealConstellationOriginal(mod);
}

window.addEventListener('resize',()=>{if(state.mode==='tx')drawTx();else if(state.mode==='spectrum')drawSpectrumPage();else if(state.mode==='rx')drawRxAnalysis();else if(state.mode==='radar')drawRadarStatic();else{drawBlindResult();drawBlindPlots()}});

async function init(){
  renderTabs();
  try{state.catalog=await api('/api/catalog');const h=await api('/api/health');$('#engineLed').classList.add('ok');$('#engineText').textContent=`engine ${h.version}`;setStatus('ready')}
  catch(e){$('#engineLed').classList.add('err');$('#engineText').textContent='backend unavailable';setStatus('backend unavailable')}
  renderWorkspace();startMode();
}

Object.assign(state, {
  rxOutput:'text',
  txPage:'data', specPage:'sweep', rxPage:'acquire', blindPage:'overview', radarPage:'design',
  txNoise:true, rxFollow:true, revision:0, rxRevision:0, radarRevision:0, radarDataRevision:-1,
  sourceRevision:null, sourceMetadata:null, reconnectAttempts:0, reconnectTimer:null, lastFrameAt:0,
  traces:[{mode:'clear',color:'#e2b553',y:null},{mode:'off',color:'#79b9df',y:null},{mode:'off',color:'#ba91de',y:null},{mode:'off',color:'#8bcaa3',y:null}],
  memories:[], memoryCount:0, specDisplay:{ref:10,range:110,average:16,auto:true}, markers:[], activeMarker:0, specDeltaRef:null, specView:{center_hz:0,span_hz:null},
  measurement:{offset_s:0,duration_s:.002}, measurementData:null, zeroFrozen:false,
  blindDisplay:{interval:2000,threshold:.5}, blindHistory:[],
  rxReplay:0, radarReplay:0,
  scene:{carrier_hz:3e9,pulses:32,noise_db:-35,targets:[{name:'Target 1',range_m:7500,velocity_mps:35,gain_db:-6}]},
  sceneElapsed:0,sceneClock:performance.now(),scenePaused:false,sceneData:null,
});
Object.assign(state.tx,{data_source:'text',custom_bits:'11010010',pn_order:15,pulse_shape:'rrc',frame_enabled:false,frame_mode:'simple',frame_blocks:[],preamble_bits:'10101010101010101101001110010001',payload_bits:256,frame_crc:false,gaussian_bt:.3,apsk_ratio2:2.85,apsk_ratio3:5.27,burst_enabled:false,burst_period_s:.01,burst_duty:.5,noise_bandwidth_hz:null,iq_gain_db:0,iq_phase_deg:0,dc_i:0,dc_q:0});
Object.assign(state.rx,{coarse_cfo:false,carrier_algorithm:'costas',equalizer:'off',equalizer_gain:.002,equalizer_taps:7,filter_span_symbols:10,pulse_shape:'rrc',apsk_ratio2:2.85,apsk_ratio3:5.27,ofdm_fft:64,ofdm_cp:16,ofdm_occupied:52});
Object.assign(state.channel,{echo_gain:0,echo_delay_symbols:1});
state.live.freq=[];
setInterval(()=>{if(state.mode!=='radar'&&!document.hidden&&state.wsReady&&performance.now()-state.lastFrameAt>5000){closeWs();openWs();}},2000);
document.addEventListener('visibilitychange',()=>{if(document.hidden){stopModeTimers();closeWs();}else startMode();});
