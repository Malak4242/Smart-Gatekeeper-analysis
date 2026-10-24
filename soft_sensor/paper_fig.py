"""Predicted vs observed COD (a) and BOD direct (b), Random Forest, final 15% chronological test set (Figure 7)."""
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

import sys
p=sys.argv[1]   # usage: python paper_fig.py path/to/ActualData_Subsample.xlsx
d=pd.read_excel(p); d.columns=[" ".join(c.split()) for c in d.columns]
d=d.sort_values('Timestamp',kind='stable').reset_index(drop=True)
F=['F-PH','Temperature (C)','DO (mg/L)','Flow (L/s)','Flow-Recycle (L/s)','Flow-Aeration (L/s)','SRT (days)']
X=d[F].values; c=d['F-COD (mg/L)'].values; b=d['F-BOD (mg/L)'].values
n=len(d); s=int(n*.85); Xtr,Xte=X[:s],X[s:]
def tune(Xa,y):
    est=RandomForestRegressor(n_estimators=300,random_state=42,n_jobs=1)
    g=GridSearchCV(est,{"max_depth":[8,None],"min_samples_leaf":[1,5]},cv=TimeSeriesSplit(5),scoring="neg_root_mean_squared_error",n_jobs=-1).fit(Xa,y)
    return g.best_estimator_
def oof(est,Xa,y):
    o=np.zeros(len(y))
    for a,bb in KFold(5).split(Xa): o[bb]=clone(est).fit(Xa[a],y[a]).predict(Xa[bb])
    return o
def sc(y,p): return dict(R2=round(r2_score(y,p),3),RMSE=round(float(np.sqrt(mean_squared_error(y,p))),2),MAE=round(mean_absolute_error(y,p),2))
ec=tune(Xtr,c[:s]); pc=ec.predict(Xte); print('COD RF',sc(c[s:],pc))
ea=tune(Xtr,b[:s]); pa=ea.predict(Xte); print('BOD RF direct',sc(b[s:],pa))
Xb_tr=np.c_[Xtr,oof(ec,Xtr,c[:s])]; Xb_te=np.c_[Xte,pc]
eb=tune(Xb_tr,b[:s]); pb=eb.predict(Xb_te); print('BOD RF two-stage',sc(b[s:],pb))
bt=b[s:]; print('test BOD mean %.1f SD %.1f min %.1f max %.1f | share >1000: %.2f'%(bt.mean(),bt.std(),bt.min(),bt.max(),(bt>1000).mean()))
print('train BOD mean %.1f SD %.1f'%(b[:s].mean(),b[:s].std()))
print('RMSE/mean test BOD: %.1f%%'%(100*np.sqrt(mean_squared_error(bt,pa))/bt.mean()))
big=bt>1000; print('RMSE where BOD>1000: %.1f | where <=1000: %.1f'%(np.sqrt(mean_squared_error(bt[big],pa[big])),np.sqrt(mean_squared_error(bt[~big],pa[~big]))))
print('mean bias (pred-obs): all %.1f | BOD>1000 %.1f'%((pa-bt).mean(),(pa[big]-bt[big]).mean()))

ink='#222222'; grey='#8a8a8a'; blue='#1f5fae'
plt.rcParams.update({'font.size':9,'axes.edgecolor':grey,'axes.labelcolor':ink,'xtick.color':ink,'ytick.color':ink,'font.family':'DejaVu Sans'})
fig,ax=plt.subplots(1,2,figsize=(7.0,3.0),dpi=300)
for a,(yo,yp,ttl,m) in zip(ax,[(c[s:],pc,'(a) COD, Random Forest',sc(c[s:],pc)),(bt,pa,'(b) BOD, Random Forest (direct)',sc(bt,pa))]):
    lo=min(yo.min(),yp.min()); hi=max(yo.max(),yp.max()); pad=(hi-lo)*.04
    a.scatter(yo,yp,s=7,color=blue,alpha=.35,linewidths=0)
    a.plot([lo-pad,hi+pad],[lo-pad,hi+pad],color=grey,lw=1,ls='--')
    a.set_xlim(lo-pad,hi+pad); a.set_ylim(lo-pad,hi+pad); a.set_aspect('equal','box')
    a.set_xlabel('Observed (mg/L)'); a.set_ylabel('Predicted (mg/L)'); a.set_title(ttl,fontsize=9.5,loc='left',color=ink)
    a.text(0.04,0.96,f"R² = {m['R2']:.3f}\nRMSE = {m['RMSE']:.2f} mg/L",transform=a.transAxes,va='top',fontsize=8.5,color=ink)
    a.grid(color='#e6e6e6',lw=.6); a.set_axisbelow(True)
    for sp in ('top','right'): a.spines[sp].set_visible(False)
fig.tight_layout()
fig.savefig('fig_pred_vs_obs.png',dpi=300)
fig.savefig('fig_pred_vs_obs.pdf')
print('saved')
