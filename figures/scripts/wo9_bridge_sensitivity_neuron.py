"""NEURON side of the model-bridge sensitivity test.

Scales one conductance at a time (gnatbar, gkfbar, gl by 0.25x to 4x) in the archived
wild-type NEURON mechanism (ichanWT2005) and reports the summed spike count over the 35-level
current ladder (20-360 pA in 10 pA steps), plus the wild-type and R859C (ichanR859C1)
baselines.  Pair with wo9_bridge_sensitivity_differentiable.py; the two are combined into
figures/wo9_model_bridge_sensitivity.json.

Needs NEURON, and the same two workarounds documented in the header of
figures/scripts/wo9_task11_neuron_verification.py (strip the VERBATIM no-ops from copies of the
.mod files, and LD_PRELOAD the system libstdc++).  Run from a directory holding the compiled
mechanisms; writes sens_nrn.json there.
"""
import numpy as np, json
from neuron import h
h.load_file("stdrun.hoc")
def detector(V):
    v=np.atleast_2d(V).copy(); v[v<-10]=-10; v[v>-10]=10
    return int((np.diff(v,axis=1)!=0).sum()/2)
def total(mech,fNa,fK,fl):
    tot=0
    for I in range(20,361,10):
        s=h.Section(); s.nseg,s.L,s.diam,s.Ra,s.cm=1,25,25,210,1
        s.insert(mech)
        for seg in s:
            setattr(seg,"gnatbar_"+mech,0.2*fNa); setattr(seg,"gkfbar_"+mech,0.06*fK)
            setattr(seg,"gl_"+mech,0.0005*fl);    setattr(seg,"el_"+mech,-60)
        s.enat,s.ekf=50,-80
        ic=h.IClamp(s(0.5)); ic.delay,ic.dur,ic.amp=50,200,I/1000.0
        vec=h.Vector().record(s(0.5)._ref_v)
        h.secondorder=2; h.dt=0.01; h.finitialize(-60); h.continuerun(300)
        tot+=detector(np.array(vec)[::10])
    return tot
F=[0.25,0.5,0.75,1.0,1.5,2.0,4.0]
out={"baseline_WT": total("ichanWT2005",1,1,1), "baseline_MT": total("ichanR859C1",1,1,1)}
for ax,idx in (("Na",0),("K",1),("leak",2)):
    out[ax]={}
    for f in F:
        m=[1.0,1.0,1.0]; m[idx]=f
        out[ax][str(f)]=total("ichanWT2005",*m)
json.dump(out,open("sens_nrn.json","w"),indent=1); print(json.dumps(out))
