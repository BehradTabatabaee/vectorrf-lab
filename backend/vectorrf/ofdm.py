"""Editable frequency/time resource grids with variable CP, pilots and WOLA."""
from typing import Literal
import numpy as np
from scipy import signal
from pydantic import BaseModel,Field,model_validator,ConfigDict
from .modem import map_bits,bits_per_symbol,slice_symbols,evm_rms

class GridCell(BaseModel):
    symbol:int=Field(ge=0,le=127)
    carrier:int
    kind:Literal['null','pilot','data','value']='value'
    i:float=1
    q:float=0
    modulation:Literal['bpsk','qpsk','8psk','16qam','64qam']='qpsk'
    gain:float=Field(default=1,ge=0,le=10)

class OFDMDesign(BaseModel):
    model_config=ConfigDict(allow_inf_nan=False)
    enabled:bool=False
    symbols:int=Field(default=8,ge=1,le=128)
    modulation:Literal['bpsk','qpsk','8psk','16qam','64qam']='qpsk'
    guard_left:int=Field(default=6,ge=0,le=2048)
    guard_right:int=Field(default=5,ge=0,le=2048)
    dc_null:bool=True
    active_carriers:list[int]=Field(default_factory=list,max_length=4096)
    pilot_carriers:list[int]=Field(default_factory=lambda:[-21,-7,7,21],max_length=4096)
    pilot_i:float=1
    pilot_q:float=0
    pilot_alternating:bool=True
    cp_lengths:list[int]=Field(default_factory=list,max_length=128)
    window_samples:int=Field(default=0,ge=0,le=1024)
    window_kind:Literal['raised-cosine','linear','custom']='raised-cosine'
    window_ramp:list[float]=Field(default_factory=list,max_length=1024)
    output_filter:Literal['off','lowpass','custom']='off'
    filter_cutoff_hz:float=Field(default=400000,gt=0)
    filter_length:int=Field(default=33,ge=3,le=2047)
    filter_window:Literal['hann','hamming','blackman']='hann'
    filter_i:list[float]=Field(default_factory=lambda:[1],min_length=1,max_length=2048)
    filter_q:list[float]=Field(default_factory=list,max_length=2048)
    cells:list[GridCell]=Field(default_factory=list,max_length=8192)
    equalizer:Literal['off','pilot-ls']='pilot-ls'

def layout(nfft,cp,occupied,design):
    d=design;carriers=np.arange(-(nfft//2),nfft-nfft//2)
    if d.enabled:
        if d.guard_left+d.guard_right>=nfft:raise ValueError('Guard bands consume the FFT')
        active=np.array(d.active_carriers if d.active_carriers else carriers[d.guard_left:nfft-d.guard_right or None].tolist(),int)
        if d.dc_null:active=active[active!=0]
        pilots=np.array(d.pilot_carriers,int)
        cpv=d.cp_lengths or [cp]
        if min(cpv)<0 or max(cpv)>=nfft:raise ValueError('Every cyclic prefix must be between zero and FFT length - 1')
        if d.window_samples>min(cpv):raise ValueError('Overlap window must fit inside the shortest cyclic prefix')
        if any(k not in active for k in pilots):raise ValueError('Pilot indices must be in the active carrier allocation')
        if any(c.symbol>=d.symbols or c.carrier not in carriers for c in d.cells):raise ValueError('Resource cell is outside the grid')
    else:
        occupied=min(occupied,nfft-2);occupied-=occupied%2
        # Legacy ordering preserved for receiver compatibility.
        active=np.r_[np.arange(1,occupied//2+1),np.arange(-occupied//2,0)]
        pilots=np.array([],int);cpv=[cp]
    if not len(active) or len(set(active))!=len(active) or any(k not in carriers for k in active):raise ValueError('Active carrier allocation is empty, duplicated or outside FFT bins')
    if len(set(pilots))!=len(pilots):raise ValueError('Pilot carriers must be unique')
    return carriers,active,pilots,cpv

def symbol_position(sample,nfft,cpv):
    lengths=np.array([nfft+c for c in cpv],dtype=np.int64)
    cycle=int(lengths.sum());cycles,local=divmod(int(sample),cycle)
    ends=np.cumsum(lengths);index=int(np.searchsorted(ends,local,side='right'))
    start=0 if index==0 else int(ends[index-1])
    return cycles*len(cpv)+index,int(local-start)

def resource_grid(nfft,cp,occupied,design,bit_source,blocks,block_offset=0):
    carriers,active,pilots,cpv=layout(nfft,cp,occupied,design)
    mod=design.modulation if design.enabled else 'qpsk';k=bits_per_symbol(mod)
    data=[c for c in active if c not in pilots]
    bits=bit_source(blocks*max(1,len(data))*k+blocks*64)
    vals=map_bits(bits,mod)
    grid=np.zeros((blocks,nfft),complex);roles=np.zeros((blocks,nfft),int)
    data=np.asarray(data,int);rows=np.arange(blocks);symbols=rows+block_offset
    if len(data):
        grid[:,data%nfft]=vals[:blocks*len(data)].reshape(blocks,len(data));roles[:,data%nfft]=1
    if len(pilots):
        grid[:,pilots%nfft]=(design.pilot_i+1j*design.pilot_q)*((-1.)**symbols[:,None] if design.pilot_alternating else 1)
        roles[:,pilots%nfft]=2
    if design.enabled:
        for cell in design.cells:
            selected=rows[symbols%design.symbols==cell.symbol];c=cell.carrier%nfft
            if cell.kind=='data':
                kb=bits_per_symbol(cell.modulation)
                bb=np.take(bits,(np.arange(kb)[None,:]+(selected[:,None]*nfft+cell.carrier)*kb)%len(bits))
                grid[selected,c]=map_bits(bb.ravel(),cell.modulation)*cell.gain
            else:grid[selected,c]=0 if cell.kind=='null' else (cell.i+1j*cell.q)*cell.gain
            roles[selected,c]={'null':0,'data':1,'pilot':2,'value':3}[cell.kind]
    return grid,roles,bits,cpv

def synthesize(cfg,n,bit_source,start_s=0):
    d=cfg.ofdm_design
    if not d.enabled or d.output_filter=='off':return _synthesize_raw(cfg,n,bit_source,start_s)
    if d.output_filter=='lowpass':
        if d.filter_cutoff_hz>=cfg.sample_rate/2:raise ValueError('OFDM filter cutoff must be below sample rate / 2')
        h=signal.firwin(d.filter_length,d.filter_cutoff_hz,fs=cfg.sample_rate,window=d.filter_window)
    else:
        if d.filter_q and len(d.filter_q)!=len(d.filter_i):raise ValueError('Complex filter I and Q taps must match')
        h=np.asarray(d.filter_i)+1j*np.asarray(d.filter_q or [0]*len(d.filter_i))
    # Render actual preceding samples so output-filter memory crosses capture boundaries.
    pad=len(h)-1;x,bits,m=_synthesize_raw(cfg,n+pad,bit_source,start_s-pad/cfg.sample_rate)
    x=signal.fftconvolve(x,h,'full')[pad:pad+n]
    m.update(output_filter=d.output_filter,filter_taps=len(h),filter_nominal_delay_samples=(len(h)-1)/2)
    return x.astype(np.complex64),bits,m

def _synthesize_raw(cfg,n,bit_source,start_s=0):
    d=cfg.ofdm_design
    _,_,_,cpv=layout(cfg.ofdm_fft,cfg.ofdm_cp,cfg.ofdm_occupied,d)
    block_offset,trim=symbol_position(round(start_s*cfg.sample_rate),cfg.ofdm_fft,cpv)
    if d.enabled and d.window_samples:
        block_offset-=1;trim+=cfg.ofdm_fft+cpv[block_offset%len(cpv)]
    blocks=int(np.ceil((n+trim)/(cfg.ofdm_fft+min(cpv))))+1
    if blocks*cfg.ofdm_fft>4_000_000:raise ValueError('OFDM grid exceeds 4 million cells')
    _,active,pilots,_=layout(cfg.ofdm_fft,cfg.ofdm_cp,cfg.ofdm_occupied,d)
    offset_bits=block_offset*(len(active)-len(pilots))*bits_per_symbol(d.modulation if d.enabled else 'qpsk')
    grid,roles,bits,cpv=resource_grid(cfg.ofdm_fft,cfg.ofdm_cp,cfg.ofdm_occupied,d,lambda count:bit_source(count,offset_bits),blocks,block_offset)
    td=np.fft.ifft(grid,axis=1)*np.sqrt(cfg.ofdm_fft)
    w=d.window_samples if d.enabled else 0
    lengths=[cfg.ofdm_fft+cpv[(i+block_offset)%len(cpv)] for i in range(blocks)]
    out=np.zeros(sum(lengths)+w,complex);cursor=0
    for i,row in enumerate(td):
        cp=cpv[(i+block_offset)%len(cpv)];z=np.r_[row[-cp:] if cp else [],row,row[:w]]
        if w:
            ramp=np.sin(np.pi*(np.arange(w)+.5)/(2*w))**2 if d.window_kind=='raised-cosine' else (np.arange(w)+.5)/w
            if d.window_kind=='custom':
                ramp=np.asarray(d.window_ramp)
                if len(ramp)!=w or np.any((ramp<0)|(ramp>1)):raise ValueError('Custom WOLA ramp needs one value in [0,1] per overlap sample')
            z[:w]*=ramp;z[-w:]*=1-ramp
        out[cursor:cursor+len(z)]+=z;cursor+=lengths[i]
    m={'modulation':'ofdm','fft_size':cfg.ofdm_fft,'subcarrier_spacing_hz':cfg.sample_rate/cfg.ofdm_fft,
       'symbol_offset':block_offset,'sample_trim':trim,'cp_lengths':cpv,'occupied_subcarriers':int(np.count_nonzero(roles[0])),
       'grid_shape':[min(blocks,d.symbols),cfg.ofdm_fft],
       'grid_roles':np.fft.fftshift(roles[:d.symbols],axes=1).ravel().tolist(),
       'grid_i':np.fft.fftshift(grid[:d.symbols],axes=1).real.ravel().tolist(),
       'grid_q':np.fft.fftshift(grid[:d.symbols],axes=1).imag.ravel().tolist()}
    return out[trim:trim+n].astype(np.complex64),bits,m

def receive(x,fs,nfft,cp,occupied,d,coarse_cfo=False,start_s=0):
    carriers,active,pilots,cpv=layout(nfft,cp,occupied,d);rows=[];cursor=0;b=0
    block_offset,trim=symbol_position(round(start_s*fs),nfft,cpv)
    if trim:
        cursor=nfft+cpv[block_offset%len(cpv)]-trim;block_offset+=1
    cfo=0.
    if coarse_cfo:
        correlation=0j;at=cursor;index=block_offset;skip=d.window_samples if d.enabled else 0
        while at+nfft+cpv[index%len(cpv)]<=len(x) and index-block_offset<2048:
            length=cpv[index%len(cpv)]
            if length>skip:correlation+=np.vdot(x[at+skip:at+length],x[at+nfft+skip:at+nfft+length])
            at+=nfft+length;index+=1
        cfo=float(np.angle(correlation)*fs/(2*np.pi*nfft))
        x=x*np.exp(-2j*np.pi*cfo*np.arange(len(x))/fs)
    while cursor+nfft+cpv[(b+block_offset)%len(cpv)]<=len(x) and b<2048:
        count=cpv[(b+block_offset)%len(cpv)];rows.append(np.fft.fft(x[cursor+count:cursor+count+nfft])/np.sqrt(nfft));cursor+=count+nfft;b+=1
    if not rows:return np.array([],complex),np.array([],np.uint8),{},{}
    grid=np.array(rows);channels=[];data=[];formats=[];calibrations=[];mod=d.modulation if d.enabled else 'qpsk'
    format_names=['bpsk','qpsk','8psk','16qam','64qam'];mapping_cache={}
    for b,row in enumerate(grid):
        symbol=b+block_offset
        key=(symbol%d.symbols,symbol%2)
        if key not in mapping_cache:
            known={int(c):(d.pilot_i+1j*d.pilot_q)*((-1)**symbol if d.pilot_alternating else 1) for c in pilots}
            mapping={int(c):(mod,1.) for c in active if c not in pilots}
            if d.enabled:
                for cell in d.cells:
                    if cell.symbol==symbol%d.symbols:
                        mapping.pop(cell.carrier,None);known.pop(cell.carrier,None)
                        if cell.kind in ('pilot','value'):known[cell.carrier]=(cell.i+1j*cell.q)*cell.gain
                        elif cell.kind=='data' and cell.gain>0:mapping[cell.carrier]=(cell.modulation,cell.gain)
            cs=np.array(list(mapping),int)
            gains=np.array([v[1] for v in mapping.values()])
            fmts=np.array([format_names.index(v[0]) for v in mapping.values()],np.int8)
            mapping_cache[key]=(known,cs,gains,fmts)
        known,cs,gains,fmts=mapping_cache[key]
        h=np.ones(nfft,complex);calibrated=False
        if d.enabled and d.equalizer=='pilot-ls' and known:
            keys=sorted(k for k,v in known.items() if abs(v)>1e-10)
            if keys:
                calibrated=True
                hp=np.array([row[k%nfft]/known[k] for k in keys]);h_order=np.interp(carriers,keys,hp.real)+1j*np.interp(carriers,keys,hp.imag);h[carriers%nfft]=h_order
                row=row/np.where(abs(h)>1e-8,h,1)
        channels.append(h)
        data.append(row[cs%nfft]/gains);formats.append(fmts)
        calibrations.append(np.full(len(cs),calibrated,bool))
    syms=np.concatenate(data);formats=np.concatenate(formats);calibrations=np.concatenate(calibrations)
    decisions=np.zeros_like(syms);z=syms.copy();bit_matrix=np.zeros((len(syms),6),np.uint8);bit_lengths=np.zeros(len(syms),np.int8)
    for code in np.unique(formats):
        indices=np.flatnonzero(formats==code);format=format_names[code]
        calibrated=bool(np.all(calibrations[indices]))
        chosen,decoded=slice_symbols(syms[indices],format,normalize=not calibrated)
        decisions[indices]=chosen
        if not calibrated:z[indices]/=max(float(np.sqrt(np.mean(abs(syms[indices])**2))),1e-15)
        width=bits_per_symbol(format);bit_matrix[indices,:width]=decoded.reshape(len(indices),width);bit_lengths[indices]=width
    bits=bit_matrix[np.arange(6)[None,:]<bit_lengths[:,None]]
    errors=z-decisions;ref_rms=max(float(np.sqrt(np.mean(abs(decisions)**2))),1e-15)
    evm=100*abs(errors)/ref_rms
    metrics={'evm_rms_percent':float(np.sqrt(np.mean(evm**2))),'evm_peak_percent':float(np.max(evm,initial=0)),
             'mer_db':float(-20*np.log10(max(np.sqrt(np.mean(evm**2))/100,1e-15)))} if len(syms) else {}
    telemetry={'ofdm_cfo_hz':cfo,'evm_time_percent':evm[:2000].tolist(),
               'magnitude_error_percent':(100*(abs(z)-abs(decisions))/ref_rms)[:2000].tolist(),
               'phase_error_deg':np.rad2deg(np.angle(z*decisions.conj()))[:2000].tolist(),
               'ofdm_channel_magnitude':np.abs(np.fft.fftshift(np.mean(channels,axis=0))).tolist(),
               'ofdm_carriers':carriers.tolist(),'ofdm_symbols':len(rows),'ofdm_method':'Known symbol boundaries; per-symbol pilot LS with frequency interpolation'}
    return syms,bits,metrics,telemetry
