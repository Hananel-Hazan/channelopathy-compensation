"""Differentiable single-compartment Hodgkin-Huxley neuron in PyTorch.

An independent implementation of the Hodgkin and Huxley (1952) membrane
equations, written for this repository. The membrane potential and the three
gating variables are advanced by forward Euler under a constant injected
current. The sodium and potassium maximal conductances are leaf tensors that
require gradients, so a loss computed on a simulated voltage trace can be
back-propagated to them.

Units: voltage in mV, time in ms, conductance in mS/cm^2, current in uA/cm^2
and capacitance in uF/cm^2. The rate constants use the convention in which the
resting potential sits near -60 mV.

The model
---------
Membrane equation, with the three ionic currents

    C dV/dt = I - I_Na - I_K - I_leak
    I_Na    = g_Na m^3 h (V - E_Na)
    I_K     = g_K  n^4   (V - E_K)
    I_leak  = g_l        (V - E_leak)

with C = 1, E_Na = 50, E_K = -77 and E_leak = -54. Each gate x in {m, h, n}
relaxes according to

    dx/dt = alpha_x(V) (1 - x) - beta_x(V) x

with the rates, in 1/ms,

    alpha_n = 0.01 (V + 50) / (1 - exp(-(V + 50) / 10))
    beta_n  = 0.125 exp(-(V + 60) / 80)
    alpha_m = 0.1 (V + 35) / (1 - exp(-(V + 35) / 10))
    beta_m  = 4 exp(-0.0556 (V + 60))
    alpha_h = 0.07 exp(-0.05 (V + 60))
    beta_h  = 1 / (1 + exp(-0.1 (V + 30)))

Each gate starts at its steady state alpha_x / (alpha_x + beta_x) for the
initial voltage.

Integration
-----------
One call to `HH.forward` is one forward-Euler step of length dt. The currents
and the gate updates are computed from the state at the start of the step, and
the voltage is updated last. The results of the published runs depend on the
exact order of the tensor operations in `forward`, so that order must not be
changed.

Batches
-------
Every argument of `HH` may be an array instead of a number. Each element is
then an independent cell, and all of them are stepped together.

How other scripts use this file
-------------------------------
Other scripts in this repository read this file as text. They take the class
body between the line `class HH()` and the line `def plot(`, or they execute
everything above the module-level assignment of `running_time`. Keep those
three markers, in this order, and keep them out of every docstring. Running the file directly performs the example fit in `main()`.

Reference
---------
Hodgkin AL, Huxley AF. A quantitative description of membrane current and its
application to conduction and excitation in nerve. J Physiol.
1952;117(4):500-544.
doi:10.1113/jphysiol.1952.sp004764
"""
import numpy as np
import torch


def pearsonr(a, b):
    """Pearson correlation coefficient of two 1-D tensors, kept differentiable.

    Both series are centred on their own means; the result is the dot product
    of the centred series divided by the product of their Euclidean norms.
    """
    centre_a = torch.mean(a)
    centre_b = torch.mean(b)
    dev_a = a.sub(centre_a)
    dev_b = b.sub(centre_b)
    covariance_sum = dev_a.dot(dev_b)
    scale = torch.norm(dev_a, 2) * torch.norm(dev_b, 2)
    return covariance_sum / scale


class HH():
    """One isopotential Hodgkin-Huxley cell, or a batch of them.

    v is the initial membrane potential; g_Na, g_K and g_l are the maximal
    conductances; dt is the Euler step; device is the torch device. `Var_g_Na`
    and `Var_g_K` require gradients and the leak `Var_g_l` does not.
    `optimizer` (ASGD over the three conductances) is created here, so a new
    cell has a new optimizer. `const_I` is the injected current, 7 by
    default; a caller may replace it before stepping.

    After each call to `forward`, `v`, `m`, `h` and `n` hold the new state and
    `i_sodium`, `i_potassium` and `i_leak` the ionic currents of the step just
    taken. Every attribute is a tensor of the shape of the initial voltage.
    """

    def __init__(self, v, g_Na=120.0, g_K=36.0, g_l=0.03, dt=0.05, device='cpu'):
        self.device = device

        def as_float(value, grad=False):
            return torch.tensor(value, requires_grad=grad, dtype=torch.float, device=device)

        # conductances (mS/cm^2): the fitted parameters
        self.Var_g_Na = as_float(g_Na, grad=True)
        self.Var_g_K = as_float(g_K, grad=True)
        self.Var_g_l = as_float(g_l)

        # injected current, capacitance and reversal potentials
        self.const_I = as_float(7)
        self.capacitance = as_float(1.0)
        self.e_sodium = as_float(50)
        self.e_potassium = as_float(-77)
        self.e_leak = as_float(-54)

        self.v = as_float(v)
        self.dt = as_float(dt)
        self.i_sodium = 0
        self.i_potassium = 0
        self.i_leak = 0

        # gating variables at their steady state for the initial voltage
        v0 = self.v.clone()
        self.m = self.alpha_m(v0) / (self.alpha_m(v0) + self.beta_m(v0))
        self.h = self.alpha_h(v0) / (self.alpha_h(v0) + self.beta_h(v0))
        self.n = self.alpha_n(v0) / (self.alpha_n(v0) + self.beta_n(v0))

        # per-conductance learning rates
        self.parameters = [
            {'params': self.Var_g_l, 'lr': 0.01},
            {'params': self.Var_g_K, 'lr': 0.5},
            {'params': self.Var_g_Na, 'lr': 1.5},
        ]
        self.optimizer = torch.optim.ASGD(self.parameters)

    # Opening (alpha) and closing (beta) rates, in 1/ms, of the potassium
    # activation n, the sodium activation m and the sodium inactivation h.
    # alpha_n and alpha_m are the closed-form expressions, evaluated as written.
    # The argument v is a membrane potential in mV.
    def alpha_n(self, v):
        return 0.01 * (v + 50) / (1 - torch.exp(-(v + 50) / 10))

    def beta_n(self, v):
        return 0.125 * torch.exp(-(v + 60) / 80)

    def alpha_m(self, v):
        return 0.1 * (v + 35) / (1 - torch.exp(-(v + 35) / 10))

    def beta_m(self, v):
        return 4.0 * torch.exp(-0.0556 * (v + 60))

    def alpha_h(self, v):
        return 0.07 * torch.exp(-0.05 * (v + 60))

    def beta_h(self, v):
        return 1 / (1 + torch.exp(-0.1 * (v + 30)))

    def forward(self):
        """Advance the cell by one Euler step of `dt`.

        The ionic currents and the gate updates both use the state at the start
        of the step; the voltage is updated last. The order of the tensor
        operations here fixes the floating-point results and the gradients, so
        it is part of the model's definition.
        """
        u = self.v.clone()
        sodium = self.Var_g_Na.clone() * self.h * self.m ** 3
        potassium = self.Var_g_K.clone() * self.n ** 4
        self.i_sodium = sodium * (u - self.e_sodium)
        self.i_potassium = potassium * (u - self.e_potassium)
        self.i_leak = self.Var_g_l.clone() * (u - self.e_leak)

        self.m = self.m + self.dt * ((self.alpha_m(u) * (1 - self.m)) - self.beta_m(u) * self.m)
        self.n = self.n + self.dt * ((self.alpha_n(u) * (1 - self.n)) - self.beta_n(u) * self.n)
        self.h = self.h + self.dt * ((self.alpha_h(u) * (1 - self.h)) - self.beta_h(u) * self.h)

        self.v = u + self.dt * ((1 / self.capacitance)
                                * (self.const_I - (self.i_sodium + self.i_potassium + self.i_leak)))


def plot(target, fitted, epoch, conductances, dt, path_prefix='fit'):
    """Save one snapshot of a fit: voltage and gates against time, and the
    (V, n) phase plane, for the target trace and the current fitted trace.

    `target` and `fitted` are (v, m, h, n) tuples of 1-D tensors, and
    `conductances` is (g_Na, g_K, g_l).
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    time_ms = np.arange(len(target[0])) * dt
    fig, (ax_v, ax_gates, ax_phase) = plt.subplots(3, 1, figsize=(10, 12))
    for trace, style, who in ((target, '-', 'target'), (fitted, '--', 'fitted')):
        v, m, h, n = (x.detach().cpu().numpy() for x in trace)
        ax_v.plot(time_ms, v, 'k' + style, label=who)
        for gate, colour, name in ((m, 'tab:green', 'm'), (h, 'tab:red', 'h'), (n, 'tab:blue', 'n')):
            ax_gates.plot(time_ms, gate, style, color=colour, label=f'{name}, {who}')
        ax_phase.plot(v, n, 'k' + style, label=who)
    ax_v.set(xlabel='time (ms)', ylabel='membrane potential (mV)')
    ax_gates.set(xlabel='time (ms)', ylabel='gating variable')
    ax_phase.set(xlabel='membrane potential (mV)', ylabel='n')
    for ax in (ax_v, ax_gates, ax_phase):
        ax.legend(loc='upper right', fontsize='small')
    g_na, g_k, g_l = conductances
    fig.suptitle(f'epoch {epoch}: g_Na {g_na:.3f}, g_K {g_k:.3f}, g_l {g_l:.3f}')
    fig.tight_layout()
    fig.savefig(f'{path_prefix}_{epoch:05d}.png')
    plt.close(fig)


running_time = 1500


def record(cell, steps):
    """Step `cell` `steps` times and return its (v, m, h, n) histories."""
    history = [torch.zeros(steps) for _ in range(4)]
    for k in range(steps):
        cell.forward()
        for store, value in zip(history, (cell.v, cell.m, cell.h, cell.n)):
            store[k] = value
    return tuple(history)


def main(epochs=5000, snapshot_every=10):
    """Example fit. A target trace is simulated at the textbook conductances
    (120, 36, 0.03); a second cell starting from (15, 22, 0.15) is then fitted
    to it by maximising the Pearson correlation of the two voltage traces.

    The cell, and with it the optimizer, is rebuilt from the current
    conductances at the start of every epoch, so the optimizer takes one step
    from a fresh state each time.
    """
    with torch.no_grad():
        target = record(HH(v=-60.0, g_Na=120, g_K=36, g_l=0.03), running_time)

    cell = HH(v=-60.0, g_Na=15.0, g_K=22, g_l=0.15)
    for epoch in range(epochs):
        cell = HH(v=-60.0,
                  g_Na=cell.Var_g_Na.data.cpu().numpy(),
                  g_K=cell.Var_g_K.data.cpu().numpy(),
                  g_l=cell.Var_g_l.data.cpu().numpy())
        fitted = record(cell, running_time)

        r = pearsonr(fitted[0], target[0])
        loss = 1 - r if r > 0 else 1 * (1 + r)
        loss.backward(retain_graph=True)
        cell.optimizer.step()
        print(f'epoch {epoch}: loss {loss.item():.4f}  '
              f'g_Na {cell.Var_g_Na.item():.4f} (grad {cell.Var_g_Na.grad.item():.4f})  '
              f'g_K {cell.Var_g_K.item():.4f} (grad {cell.Var_g_K.grad.item():.4f})  '
              f'g_l {cell.Var_g_l.item():.4f}')
        cell.optimizer.zero_grad()

        if epoch % snapshot_every == 0:
            plot(target, fitted, epoch,
                 (cell.Var_g_Na.item(), cell.Var_g_K.item(), cell.Var_g_l.item()),
                 cell.dt.item())


if __name__ == '__main__':
    main()
