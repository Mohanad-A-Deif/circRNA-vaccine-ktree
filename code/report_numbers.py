"""Collect every number quoted in the manuscript into results/numbers.txt."""
import numpy as np, pandas as pd, json
from scipy.stats import wilcoxon
def ci(x):
    x=np.asarray(x,float); m=x.mean(); h=2.093*x.std(ddof=1)/np.sqrt(len(x)); return m,h
out=[]
def P(s): out.append(s)
d=pd.read_csv('results/main_comparison.csv')
cols=['V','doses','resp30','prog20','logE24','g3_per_boost','any_g3','viol_rate','p95_risk','mean_interval']
P('== MAIN (mean ± CI half-width)')
for p,g in d.groupby('policy'):
    P(p+' | '+' | '.join(f'{c}={ci(g[c])[0]:.3f}±{ci(g[c])[1]:.3f}' for c in cols))
pp=pd.read_csv('results/main_paired.csv')
P('== PAIRED dV (k-tree minus policy)')
for p,g in pp.groupby('policy'):
    m,h=ci(g.dV_mean); P(f'{p}: {m:.3f} [{m-h:.3f},{m+h:.3f}] wins={int((g.dV_mean>0).sum())}/20 p={wilcoxon(g.dV_mean).pvalue:.2e}')
for f in ['ablation']:
    a=pd.read_csv(f'results/{f}.csv'); P('== '+f)
    full=a[a.policy=='Full model'].set_index('seed').V
    for p,g in a.groupby('policy'):
        g=g.set_index('seed'); diff=(full-g.V)
        pv=wilcoxon(diff).pvalue if (diff!=0).any() else 1.0
        P(p+' | '+' | '.join(f'{c}={ci(g[c])[0]:.3f}±{ci(g[c])[1]:.3f}' for c in ['V','doses','resp30','any_g3','g3_per_boost','viol_rate','fit_sec'])+f' | n_params={g.n_params.mean():.0f} | full-minus={diff.mean():.3f} p={pv:.2e}')
open('results/numbers.txt','w').write('\n'.join(out)); print('\n'.join(out))
