"""T5 phase 1: dump every acf_fft / lombscargle spectral peak (P0, power, local-continuum estimates) for
each dev_medium star in both coverage modes (raw-power order, all peaks). Resumable, atomic per-star pickles.
Usage: python reports/2026-11-11_t5-short-period-recovery/dump_peaks.py runs/t5_peaks   (~35 min at 4 workers)"""
import sys, os, warnings, pickle, time as _t
sys.path[:0]=['claude_files','benchmarks']
import determinism; determinism.pin_blas_threads()
import numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor
OUT=sys.argv[1]
WS=(0.05,0.1,0.2,0.3)
def cont(freq,power,fpk,w):
    lf=np.log10(freq); o=np.argsort(lf); lf=lf[o]; pw=power[o]
    lo=np.searchsorted(lf,np.log10(fpk)-w); hi=np.searchsorted(lf,np.log10(fpk)+w)
    return np.array([np.median(pw[a:b]) if b>a else np.nan for a,b in zip(lo,hi)])
def peaks(gs,key_f,key_p):
    if not gs: return None
    info=gs[0].info
    if key_f=='period':
        per,pw=info['periodogram']; freq=1/per
    else:
        freq,pw=info['fft_freq'],info['fft_power']
    P=np.array([g.P0 for g in gs]); pk=np.array([g.strength for g in gs])
    fpk=1/P
    # raw power of each peak
    praw=np.interp(fpk, np.sort(freq), pw[np.argsort(freq)])
    d=dict(P0=P,strength=pk,power=praw)
    for w in WS: d[f'cont{w}']=cont(freq,pw,fpk,w)
    return d
def one(job):
    sid,mode,path,sectors=job
    fn=os.path.join(OUT,f'{sid}_{mode}.pkl')
    if os.path.exists(fn): return
    warnings.simplefilter('ignore')
    from preprocessing import load_smarts_fits
    from acf_utils import compute_acf, STAGE1_MAX_LAG_FRAC
    import guesses as G
    kw={} if mode=='fy' else {'sectors':sectors}
    pre=load_smarts_fits(path,**kw); t,f=pre.time,pre.flux
    lags,acf=compute_acf(t,f); l1,a1=compute_acf(t,f,max_lag_frac=STAGE1_MAX_LAG_FRAC)
    rec=dict(sid=sid,mode=mode,N=len(t),B=t[-1]-t[0],lag1=l1[-1]-l1[0],lag2=lags[-1]-lags[0])
    for view,(L,A) in (('short',(l1,a1)),('long',(lags,acf))):
        t0=_t.perf_counter(); g=G.guess_acf_fft(t,f,L,A,n_guesses=100000, rank_by="power"); rec[f'acf_fft_{view}_s']=_t.perf_counter()-t0
        rec[f'acf_fft_{view}']=peaks(g,'fft',None)
    t0=_t.perf_counter(); g=G.guess_lombscargle(t,f,l1,a1,n_guesses=100000, rank_by="power"); rec['ls_s']=_t.perf_counter()-t0
    rec['lombscargle']=peaks(g,'period',None)
    tmp=fn+'.tmp'; pickle.dump(rec,open(tmp,'wb')); os.replace(tmp,fn)
if __name__=='__main__':
    man=pd.read_csv('datasets/dev_medium.csv')
    jobs=[]
    for _,r in man.iterrows():
        sec=[int(s) for s in str(r.sectors).split('-')]
        for mode in ('fy','asg'):
            jobs.append((r.star_id,mode,'data/smarts_dev/'+r.path,sec))
    with ProcessPoolExecutor(4,initializer=determinism.init_worker) as ex:
        for i,_ in enumerate(ex.map(one,jobs,chunksize=4)):
            if i%200==0: print(i,flush=True)
