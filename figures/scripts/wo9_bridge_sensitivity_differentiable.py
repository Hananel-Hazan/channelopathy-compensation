"""Differentiable side of the model-bridge sensitivity test.

Scales one conductance at a time (gNa, gK, gleak by 0.25x to 4x) in the differentiable model
(experiments/wo5/wo5_lib.py) and reports the summed spike count over its 35-level current
ladder, plus the wild-type and variant (sodium-activation shift of 6.1 mV) baselines.  Pair with
wo9_bridge_sensitivity_neuron.py; the two are combined into
figures/wo9_model_bridge_sensitivity.json.

Writes experiments/wo9/sens_diff.json.  Needs a CUDA GPU.
"""
import os, sys, json, numpy as np, torch
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "experiments", "wo5"))
import wo5_lib as W
DEV="cuda"
I1=torch.tensor(np.arange(W.N_CUR)*W.I_STEP+W.I_LOW, dtype=torch.float32, device=DEV)
def total(fNa,fK,fl,shift=0.0):
    g=[torch.full((W.N_CUR,), W.WT[0]*fNa, device=DEV),
       torch.full((W.N_CUR,), W.WT[1]*fK,  device=DEV),
       torch.full((W.N_CUR,), W.WT[2]*fl,  device=DEV)]
    st=W.simulate_stats_fast(g[0],g[1],g[2],I1,shift=shift,device=DEV,steps_per_call=20)
    return int(st["spikes"].sum().item())
F=[0.25,0.5,0.75,1.0,1.5,2.0,4.0]
out={"baseline_WT":total(1,1,1),"baseline_variant_shift6.1":total(1,1,1,shift=6.1)}
for ax,idx in (("Na",0),("K",1),("leak",2)):
    out[ax]={}
    for f in F:
        m=[1.0,1.0,1.0]; m[idx]=f
        out[ax][str(f)]=total(*m)
json.dump(out,open(os.path.join(REPO, "experiments", "wo9", "sens_diff.json"),"w"),indent=1)
print(json.dumps(out))
