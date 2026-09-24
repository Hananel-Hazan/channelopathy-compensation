"""Figure 14 -- dynamic time warping (DTW) schematic, drawn as a vector PDF.

Two synthetic two-spike voltage traces (a reference and a delayed variant) are
aligned with a plain DTW implementation, and the computed warping path is drawn
between them, so the correspondences shown are the ones the algorithm makes.

Writes figures/dtw_warping_schematic.pdf.
"""
import os
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# embed TrueType rather than Type 3 fonts, so text in the PDF stays selectable
# and copyable when the figure is included in a document
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

WT  = '#0072B2'   # reference / wild type
VAR = '#D55E00'   # variant / mutant

def spike(t, t0, amp=95.0, rise=0.45, fall=1.5, base=-68.0):
    """Synthetic action potential starting at t0 (ms): a difference of
    exponentials (time constants in ms) scaled to peak `amp` mV above `base` mV."""
    x = (t - t0)
    y = np.where(x >= 0, amp*(np.exp(-x/fall) - np.exp(-x/rise)), 0.0)
    y = y / max(y.max(), 1e-9) * amp
    return base + y

t = np.linspace(0, 30, 600)
a = spike(t, 6.0) + spike(t, 15.0)*0.0
a = spike(t, 6.0) + (spike(t, 16.0) + 68.0)      # reference: two spikes
b = spike(t, 9.5) + (spike(t, 21.5) + 68.0)      # variant: same shape, later

def dtw_path(x, y):
    """Squared-difference DTW between 1-D arrays x and y; returns the optimal
    warping path as a list of (index in x, index in y) pairs."""
    n, m = len(x), len(y)
    D = np.full((n+1, m+1), np.inf); D[0,0] = 0.0
    C = (x[:,None] - y[None,:])**2
    for i in range(1, n+1):
        for j in range(1, m+1):
            D[i,j] = C[i-1,j-1] + min(D[i-1,j], D[i,j-1], D[i-1,j-1])
    i, j, path = n, m, []
    while i > 0 and j > 0:
        path.append((i-1, j-1))
        k = np.argmin([D[i-1,j-1], D[i-1,j], D[i,j-1]])
        if k == 0: i, j = i-1, j-1
        elif k == 1: i -= 1
        else: j -= 1
    return path[::-1]

step = 6                                          # decimate for the DTW grid
ia, ib = t[::step], t[::step]
path = dtw_path(a[::step], b[::step])

OFF = 130.0                                       # vertical offset for legibility
fig, ax = plt.subplots(figsize=(6.6, 3.4))
for (p, q) in path[::7]:
    ax.plot([ia[p], ib[q]], [a[::step][p] + OFF, b[::step][q]],
            color='0.70', lw=0.45, ls=(0,(2.0,2.6)), zorder=1)
pk_a = int(np.argmax(a[::step])); pk_b = int(np.argmax(b[::step]))
for (p, q) in path:
    if p == pk_a and q == pk_b:
        ax.plot([ia[p], ib[q]], [a[::step][p] + OFF, b[::step][q]],
                color='0.25', lw=1.2, ls=(0,(3,2)), zorder=2)
        ax.annotate('matched peaks', xy=((ia[p]+ib[q])/2, OFF/2 - 10),
                    fontsize=8, color='0.25', ha='center')
        break
ax.plot(t, a + OFF, color=WT,  lw=1.9, zorder=3, label='reference (wild type)')
ax.plot(t, b,       color=VAR, lw=1.9, zorder=3, label='variant (mutant)')

ax.set_xlabel('time (ms)')
ax.set_ylabel('membrane potential (mV, traces offset)')
ax.set_xlim(0, 30)
ax.set_yticks([-68+OFF, 27+OFF, -68, 27])
ax.set_yticklabels(['-68', '+27', '-68', '+27'])
ax.legend(loc='upper right', frameon=False, fontsize=8.5)
ax.spines[['top','right']].set_visible(False)
ax.grid(axis='x', color='0.9', lw=0.6)
ax.set_axisbelow(True)
ax.tick_params(labelsize=8.5)
ax.xaxis.label.set_size(9.5); ax.yaxis.label.set_size(9.5)
fig.tight_layout()
fig.savefig(os.path.join(REPO, 'figures', 'dtw_warping_schematic.pdf'), bbox_inches='tight')
print("wrote", os.path.join(REPO, 'figures', 'dtw_warping_schematic.pdf'), "| DTW path length:", len(path))
