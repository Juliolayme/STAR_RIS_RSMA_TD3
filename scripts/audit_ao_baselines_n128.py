from __future__ import annotations
import argparse,json,math,time
from pathlib import Path
import numpy as np,pandas as pd
from scipy import stats
from star_ris_rsma.action import DecodedAction,decode_action,encode_action,wrap_phase
from star_ris_rsma.baselines.analytical_ris import analytical_action
from star_ris_rsma.baselines.ao_sca import solve as solve_legacy,_block_gradient as legacy_grad,_proximal_surrogate_maximizer as legacy_prox
from star_ris_rsma.baselines.common import merit,physical_slices,project_physical,state_from_action,state_from_vector
from star_ris_rsma.config import ExperimentConfig
from star_ris_rsma.env import StarRisRsmaEnv
from star_ris_rsma.physics import effective_channels
from star_ris_rsma.scenario_bank import generate_bank

CHECK='f4c80269e5fb3cf553900b2e82f235af875b7c2e33b4ff71ec5d85cc25eb2b4e'
METHODS=('legacy','claude','pairwise','hybrid','grid_corrected')

def cfg_bank(repo):
 c=ExperimentConfig.from_yaml(repo/'configs/v3/constrained_action_n128.yaml'); b=generate_bank(c,1000,33001,'test')
 assert b.checksum()==CHECK; return c,b

def bounds(c):
 s=physical_slices(c.n_users,c.n_ris); lo=np.full(s['theta_r'].stop,-np.inf); hi=np.full_like(lo,np.inf)
 lo[s['powers']]=0; hi[s['powers']]=c.p_max; lo[s['common']]=0; hi[s['common']]=1; lo[s['beta']]=0; hi[s['beta']]=1; return lo,hi

def ambient_merit(env,x):
 c=env.config;s=physical_slices(c.n_users,c.n_ris)
 a=DecodedAction(np.maximum(x[s['powers']],0),np.maximum(x[s['common']],0),np.clip(x[s['beta']],0,1),wrap_phase(x[s['theta_t']]),wrap_phase(x[s['theta_r']]))
 return merit(env.evaluate_decoded_action(a))

def ambient_grad(env,x,idx,eps,lo,hi):
 g=np.zeros_like(x);ev=0;base=None
 for j in idx:
  up=min(x[j]+eps,hi[j]);dn=max(x[j]-eps,lo[j])
  if up-dn<1e-15: continue
  au=up-x[j]<1e-12; al=x[j]-dn<1e-12
  if au or al:
   if base is None: base=ambient_merit(env,x);ev+=1
   z=x.copy()
   if au: z[j]=dn; g[j]=(base-ambient_merit(env,z))/(x[j]-dn)
   else: z[j]=up; g[j]=(ambient_merit(env,z)-base)/(up-x[j])
   ev+=1
  else:
   p=x.copy();m=x.copy();p[j]=up;m[j]=dn;g[j]=(ambient_merit(env,p)-ambient_merit(env,m))/(up-dn);ev+=2
 return g,ev

def structured(env):
 a=analytical_action(env);c=env.config;h=effective_channels(env.channel,a.beta_t,a.theta_t,a.theta_r);k=int(np.argmax(np.abs(h)**2))
 ratio=2**(c.n_users*c.qos_min)-1;pc=float(np.clip(ratio/(1+ratio),.05,.95));p=np.zeros(c.n_users+1);p[0]=pc*c.p_max;p[1+k]=(1-pc)*c.p_max
 return a.copy_with(powers=p)

def prox_step(env,current,g,idx,rho0=1.,growth=2.,backs=16,gain_tol=-1e-10):
 ev=0;rho=rho0
 if np.linalg.norm(g[idx])<1e-12:return current,ev,0
 for _ in range(backs):
  v=current.vector.copy();v[idx]=current.vector[idx]+g[idx]/rho;v=project_physical(v,env.config.n_users,env.config.n_ris,env.config.p_max);p=state_from_vector(env,v);ev+=1
  d=p.vector-current.vector;tg=p.score-current.score;sg=float(g@d-.5*rho*np.dot(d,d))
  if tg>=gain_tol and sg>=-1e-10:return p,ev,int(np.linalg.norm(d)>1e-12)
  rho*=growth
 return current,ev,0

def claude_from(env,start,max_iter=20,tol=1e-4,eps=1e-3):
 c=env.config;lo,hi=bounds(c);s=physical_slices(c.n_users,c.n_ris);blocks=[np.arange(s['powers'].start,s['common'].stop),np.arange(s['beta'].start,s['theta_r'].stop)]
 cur=state_from_action(env,start);hist=[cur.score];ev=1;ac=0
 for _ in range(max_iter):
  old=cur.score
  for idx in blocks:
   g,e=ambient_grad(env,cur.vector,idx,eps,lo,hi);ev+=e;cur,e,a=prox_step(env,cur,g,idx);ev+=e;ac+=a
  hist.append(cur.score)
  if abs(cur.score-old)/max(1,abs(old))<tol:break
 return cur,hist,ev,ac

def solve_claude(env,seed=0,max_iter=20):
 del seed;best=None;tot=0;scores=[]
 for name,start in [('analytical_ris',analytical_action(env)),('structured_qos_split',structured(env))]:
  cur,h,ev,ac=claude_from(env,start,max_iter=max_iter);tot+=ev;k=(bool(cur.metrics['all_qos']),float(cur.metrics['sum_rate']));scores.append((name,k[1]))
  if best is None or k>best[0]:best=(k,cur,h,ac,name)
 _,cur,h,ac,name=best;m=dict(cur.metrics);m.update(solver='claude_ambient_fd_multistart',iterations=len(h)-1,evaluations=tot,accepted_steps=ac,initialization=name,restart_scores=scores)
 return encode_action(cur.action,env.config.p_max,env.config.action_parameterization),m

def pair_ascent(env,cur,sl,total,probe=1e-3,max_steps=12,line_points=12,tol=1e-8):
 ev=ac=0;st,sp=sl.start,sl.stop;d=sp-st
 for _ in range(max_steps):
  vals=cur.vector[sl].copy();best=None
  for i in range(d):
   av=float(vals[i]);delta=min(probe*total,av)
   if delta<=1e-14:continue
   for j in range(d):
    if i==j:continue
    v=cur.vector.copy();v[st+i]-=delta;v[st+j]+=delta;c=state_from_vector(env,v);ev+=1;gain=c.score-cur.score;slope=gain/delta
    if best is None or slope>best[0]:best=(slope,i,j,delta,gain)
  if best is None or best[0]<=tol:break
  _,i,j,pd,pg=best;mx=float(cur.vector[st+i]);alphas={pd,mx}
  if mx>pd*(1+1e-12):alphas|={float(x) for x in np.geomspace(pd,mx,num=line_points)}
  bc=None;bg=-np.inf
  for a in sorted(alphas):
   a=min(a,mx);v=cur.vector.copy();v[st+i]-=a;v[st+j]+=a;c=state_from_vector(env,v);ev+=1;g=c.score-cur.score
   if g>bg:bc,bg=c,g
  if bc is None or bg<=tol:
   if pg<=tol:break
   v=cur.vector.copy();v[st+i]-=pd;v[st+j]+=pd;bc=state_from_vector(env,v);ev+=1;bg=bc.score-cur.score
  if bg<=tol:break
  cur=bc;ac+=1
 return cur,ev,ac

def gap(env,cur,sl,total,eps=1e-4):
 best=0.;ev=0;st,sp=sl.start,sl.stop;vals=cur.vector[sl]
 for i in range(sp-st):
  de=min(eps*total,float(vals[i]));
  if de<=1e-14:continue
  for j in range(sp-st):
   if i==j:continue
   v=cur.vector.copy();v[st+i]-=de;v[st+j]+=de;c=state_from_vector(env,v);ev+=1;best=max(best,float(c.score-cur.score))
 return best,ev

def solve_pairwise(env,seed=0,max_iter=40):
 del seed;c=env.config;s=physical_slices(c.n_users,c.n_ris);ridx=np.arange(s['beta'].start,s['theta_r'].stop);cur=state_from_action(env,analytical_action(env));hist=[cur.score];ev=1;ac=0
 for _ in range(max_iter):
  old=cur.score
  cur,e,a=pair_ascent(env,cur,s['powers'],c.p_max);ev+=e;ac+=a;cur,e,a=pair_ascent(env,cur,s['common'],1.);ev+=e;ac+=a
  g,e=legacy_grad(env,cur.vector,ridx,1e-3);ev+=e
  if np.linalg.norm(g[ridx])>=1e-12:
   rho=1.
   for _ in range(16):
    v=legacy_prox(cur.vector,g,ridx,rho,env);p=state_from_vector(env,v);ev+=1;d=p.vector-cur.vector;tg=p.score-cur.score;sg=float(g@d-.5*rho*np.dot(d,d))
    if tg>=-1e-10 and sg>=-1e-10:cur=p;ac+=int(np.linalg.norm(d)>1e-12);break
    rho*=2
  hist.append(cur.score)
  if abs(cur.score-old)/max(1,abs(old))<1e-4:break
 pg,e=gap(env,cur,s['powers'],c.p_max);ev+=e;cg,e=gap(env,cur,s['common'],1.);ev+=e;m=dict(cur.metrics);m.update(solver='pairwise_simplex_from_scratch',iterations=len(hist)-1,evaluations=ev,accepted_steps=ac,initialization='analytical_ris',power_stationarity_gap=pg,common_stationarity_gap=cg)
 return encode_action(cur.action,c.p_max,c.action_parameterization),m

def hybrid_from(env,start,max_iter=40):
 c=env.config;s=physical_slices(c.n_users,c.n_ris);lo,hi=bounds(c);ridx=np.arange(s['beta'].start,s['theta_r'].stop);cur=state_from_action(env,start);hist=[cur.score];ev=1;ac=0
 for _ in range(max_iter):
  old=cur.score;cur,e,a=pair_ascent(env,cur,s['powers'],c.p_max);ev+=e;ac+=a;cur,e,a=pair_ascent(env,cur,s['common'],1.);ev+=e;ac+=a
  g,e=ambient_grad(env,cur.vector,ridx,1e-3,lo,hi);ev+=e;cur,e,a=prox_step(env,cur,g,ridx,gain_tol=1e-8);ev+=e;ac+=a;hist.append(cur.score)
  if abs(cur.score-old)/max(1,abs(old))<1e-4:break
 pg,e=gap(env,cur,s['powers'],c.p_max);ev+=e;cg,e=gap(env,cur,s['common'],1.);ev+=e;return cur,hist,ev,ac,pg,cg

def solve_hybrid(env,seed=0,max_iter=40):
 del seed;best=None;tot=0
 for name,start in [('analytical_ris',analytical_action(env)),('structured_qos_split',structured(env))]:
  cur,h,ev,ac,pg,cg=hybrid_from(env,start,max_iter);tot+=ev;k=(bool(cur.metrics['all_qos']),float(cur.metrics['sum_rate']))
  if best is None or k>best[0]:best=(k,cur,h,ac,name,pg,cg)
 _,cur,h,ac,name,pg,cg=best;m=dict(cur.metrics);m.update(solver='hybrid_pairwise_bounded_ris_multistart',iterations=len(h)-1,evaluations=tot,accepted_steps=ac,initialization=name,power_stationarity_gap=pg,common_stationarity_gap=cg)
 return encode_action(cur.action,env.config.p_max,env.config.action_parameterization),m

def redist(v,i,sel,total):
 r=np.asarray(v,float).copy();sel=float(np.clip(sel,0,total));o=np.arange(r.size)!=i;rem=total-sel;prev=float(r[o].sum());r[i]=sel
 if np.any(o):r[o]=rem/int(o.sum()) if prev<=1e-12 else r[o]*rem/prev
 return r

def grid_run(env,reverse=False,rounds=2):
 cur=state_from_action(env,analytical_action(env));c=env.config;s=physical_slices(c.n_users,c.n_ris);ag=np.r_[0.,np.linspace(.05,.80,7)];pg=ag*c.p_max;bg=np.linspace(.05,.95,5);th=np.linspace(-np.pi,np.pi,8,endpoint=False);hist=[cur.score];ev=1;ac=0
 for _ in range(rounds):
  old=cur.score
  for sl,gr,total,steps in [(s['powers'],pg,c.p_max,c.n_users+1),(s['common'],ag,1.,c.n_users)]:
   for _ in range(steps):
    best=cur;base=cur.vector[sl].copy()
    for i in range(sl.stop-sl.start):
     for val in gr:
      v=cur.vector.copy();v[sl]=redist(base,i,val,total);x=state_from_vector(env,v);ev+=1
      if x.score>best.score+1e-12:best=x
    if best.score<=cur.score+1e-12:break
    cur=best;ac+=1
  for block,gr in [('beta',bg),('theta_t',th),('theta_r',th)]:
   ids=list(range(s[block].start,s[block].stop));ids=ids[::-1] if reverse else ids
   for j in ids:
    best=cur
    for val in gr:
     v=cur.vector.copy();v[j]=val;x=state_from_vector(env,v);ev+=1
     if x.score>best.score+1e-12:best=x
    if best.score>cur.score+1e-12:cur=best;ac+=1
  hist.append(cur.score)
  if abs(cur.score-old)/max(1,abs(old))<1e-4:break
 return cur,hist,ev,ac

def solve_grid(env,seed=0,rounds=2):
 del seed;cs=[];tot=0
 for rev in (False,True):
  cur,h,ev,ac=grid_run(env,rev,rounds);tot+=ev;cs.append((cur,h,ac,rev))
 cur,h,ac,rev=max(cs,key=lambda z:(bool(z[0].metrics['all_qos']),float(z[0].metrics['sum_rate'])));m=dict(cur.metrics);m.update(solver='grid_corrected_best_simplex_bidirectional_ris',iterations=len(h)-1,evaluations=tot,accepted_steps=ac,initialization='analytical_ris',selected_ris_sweep='reverse' if rev else 'forward')
 return encode_action(cur.action,env.config.p_max,env.config.action_parameterization),m

def solver(method):
 return {'legacy':(solve_legacy,{'max_iter':20}),'claude':(solve_claude,{'max_iter':20}),'pairwise':(solve_pairwise,{'max_iter':40}),'hybrid':(solve_hybrid,{'max_iter':40}),'grid_corrected':(solve_grid,{'rounds':2})}[method]

def run(repo,method,start,end,out):
 c,b=cfg_bank(repo);fn,kw=solver(method);rows=[]
 for sc in range(start,end):
  env=StarRisRsmaEnv(c,seed=sc);env.reset(channel=b.channel(sc));t=time.perf_counter();raw,m=fn(env,seed=sc,**kw);dt=time.perf_counter()-t;a=decode_action(raw,c.n_users,c.n_ris,c.p_max,c.action_parameterization);p=np.asarray(a.powers);eta=np.asarray(a.common_fractions)
  rows.append(dict(method=method,n_ris=128,scenario=sc,bank_checksum=CHECK,sum_rate=float(m['sum_rate']),reward=float(m['reward']),all_qos=bool(m['all_qos']),qos_fraction=float(m['qos_fraction']),violation=float(m['violation']),elapsed_s=dt,iterations=int(m.get('iterations',0)),evaluations=int(m.get('evaluations',0)),accepted_steps=int(m.get('accepted_steps',0)),initialization=str(m.get('initialization','')),pc=p[0],p1=p[1],p2=p[2],p3=p[3],p4=p[4],eta1=eta[0],eta2=eta[1],eta3=eta[2],eta4=eta[3],power_stationarity_gap=float(m.get('power_stationarity_gap',np.nan)),common_stationarity_gap=float(m.get('common_stationarity_gap',np.nan)),selected_ris_sweep=str(m.get('selected_ris_sweep',''))))
  if (sc-start+1)%10==0:print(method,sc+1,end,rows[-1]['sum_rate'],dt,flush=True)
 out.parent.mkdir(parents=True,exist_ok=True);pd.DataFrame(rows).to_csv(out,index=False)

def ci(x):
 x=np.asarray(x,float);n=len(x);mu=float(x.mean());sd=float(x.std(ddof=1));h=float(stats.t.ppf(.975,n-1)*sd/math.sqrt(n));return mu,sd,mu-h,mu+h

def summarize(inp,out):
 df=pd.concat([pd.read_csv(p) for p in sorted(inp.rglob('*.csv'))],ignore_index=True)
 if df.duplicated(['method','scenario']).any():raise RuntimeError('duplicates')
 for m in METHODS:
  g=df[df.method==m];assert len(g)==1000 and set(g.scenario)==set(range(1000)) and set(g.bank_checksum)=={CHECK}
 rows=[]
 for m in METHODS:
  g=df[df.method==m].sort_values('scenario');mu,sd,lo,hi=ci(g.sum_rate);priv=g[['p1','p2','p3','p4']].to_numpy(float)
  rows.append(dict(method=m,scenario_count=1000,sum_rate_mean=mu,sum_rate_std=sd,ci95_low=lo,ci95_high=hi,min=float(g.sum_rate.min()),max=float(g.sum_rate.max()),all_qos_count=int(g.all_qos.astype(bool).sum()),pc_mean=float(g.pc.mean()),common_only_count=int((np.max(priv,axis=1)<1e-9).sum()),elapsed_s_mean=float(g.elapsed_s.mean()),evaluations_mean=float(g.evaluations.mean()),iterations_mean=float(g.iterations.mean()),power_gap_max=float(g.power_stationarity_gap.max(skipna=True)) if g.power_stationarity_gap.notna().any() else np.nan,common_gap_max=float(g.common_stationarity_gap.max(skipna=True)) if g.common_stationarity_gap.notna().any() else np.nan))
 summ=pd.DataFrame(rows);leg=df[df.method=='legacy'].sort_values('scenario').set_index('scenario');pairs=[]
 for m in METHODS[1:]:
  g=df[df.method==m].sort_values('scenario').set_index('scenario');d=g.sum_rate.to_numpy()-leg.sum_rate.to_numpy();tt=stats.ttest_rel(g.sum_rate,leg.sum_rate);ww=stats.wilcoxon(d,zero_method='wilcox',method='auto')
  pairs.append(dict(method=m,mean_delta_vs_legacy=float(d.mean()),wins=int((d>1e-9).sum()),ties=int((abs(d)<=1e-9).sum()),losses=int((d<-1e-9).sum()),paired_t_p=float(tt.pvalue),wilcoxon_p=float(ww.pvalue)))
 head=[]
 for i,a in enumerate(METHODS[1:]):
  A=df[df.method==a].sort_values('scenario').set_index('scenario')
  for b in METHODS[i+2:]:
   B=df[df.method==b].sort_values('scenario').set_index('scenario');d=A.sum_rate.to_numpy()-B.sum_rate.to_numpy();head.append(dict(method_a=a,method_b=b,mean_a_minus_b=float(d.mean()),a_wins=int((d>1e-9).sum()),ties=int((abs(d)<=1e-9).sum()),b_wins=int((d<-1e-9).sum())))
 out.mkdir(parents=True,exist_ok=True);df.sort_values(['method','scenario']).to_csv(out/'AO_BASELINE_N128_1000_ALL.csv',index=False);summ.to_csv(out/'AO_BASELINE_N128_1000_SUMMARY.csv',index=False);pd.DataFrame(pairs).to_csv(out/'AO_BASELINE_N128_1000_PAIRED_VS_LEGACY.csv',index=False);pd.DataFrame(head).to_csv(out/'AO_BASELINE_N128_1000_HEAD_TO_HEAD.csv',index=False);(out/'AO_BASELINE_N128_1000_REPORT.json').write_text(json.dumps(dict(bank_checksum=CHECK,summary=summ.to_dict('records'),paired_vs_legacy=pairs,head_to_head=head),indent=2,allow_nan=True));print(summ.to_string(index=False))

def main():
 p=argparse.ArgumentParser();sp=p.add_subparsers(dest='cmd',required=True);r=sp.add_parser('run');r.add_argument('--repo',type=Path,default=Path('.'));r.add_argument('--method',choices=METHODS,required=True);r.add_argument('--start',type=int,required=True);r.add_argument('--end',type=int,required=True);r.add_argument('--output',type=Path,required=True);s=sp.add_parser('summarize');s.add_argument('--input-dir',type=Path,required=True);s.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();run(a.repo,a.method,a.start,a.end,a.output) if a.cmd=='run' else summarize(a.input_dir,a.output_dir)
if __name__=='__main__':main()
