# -*- coding: utf-8 -*-
"""
EEG Wavefield Viewer — Live Frequency Pairs (controls below)
- Four sliders: Band A low/high, Band B low/high
- Toggle: A only  <->  A + B (Huygens superposition)
- Phase / Power(A) / Huygens modes
- All controls sit UNDER the scalp so visuals stay clean.

Usage:
  python EEG_Wavefield_viewer_V4_pairs_ui_below.py -f your.edf --band 3 10 --mode huygens --fps 30
"""

import argparse, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from scipy.signal import butter, filtfilt, hilbert, detrend, iirnotch
from scipy.interpolate import griddata

# ----------------------------
# Electrode layout (10-20/10-10 approx.)
# ----------------------------
COORDS = {
    "Fp1":(-0.5,1.0),"Fpz":(0.0,1.05),"Fp2":(0.5,1.0),
    "AF7":(-0.7,0.9),"AF3":(-0.35,0.92),"AFz":(0.0,0.95),
    "AF4":(0.35,0.92),"AF8":(0.7,0.9),
    "F7":(-0.85,0.75),"F5":(-0.55,0.78),"F3":(-0.3,0.8),"F1":(-0.1,0.82),
    "Fz":(0.0,0.85),"F2":(0.1,0.82),"F4":(0.3,0.8),"F6":(0.55,0.78),"F8":(0.85,0.75),
    "Ft7":(-0.98,0.58),"Ft8":(0.98,0.58),
    "FC5":(-0.6,0.55),"FC3":(-0.35,0.58),"FC1":(-0.15,0.6),"FCz":(0.0,0.62),
    "FC2":(0.15,0.6),"FC4":(0.35,0.58),"FC6":(0.6,0.55),
    "T7":(-1.05,0.40),"C5":(-0.65,0.38),"C3":(-0.35,0.40),"C1":(-0.15,0.42),
    "Cz":(0.0,0.45),"C2":(0.15,0.42),"C4":(0.35,0.40),"C6":(0.65,0.38),"T8":(1.05,0.40),
    "T9":(-1.12,0.35),"T10":(1.12,0.35),
    "CP5":(-0.6,0.25),"CP3":(-0.35,0.28),"CP1":(-0.15,0.30),"CPz":(0.0,0.32),
    "CP2":(0.15,0.30),"CP4":(0.35,0.28),"CP6":(0.6,0.25),
    "P7":(-0.9,0.08),"P5":(-0.6,0.10),"P3":(-0.35,0.12),"P1":(-0.15,0.13),
    "Pz":(0.0,0.14),"P2":(0.15,0.13),"P4":(0.35,0.12),"P6":(0.6,0.10),"P8":(0.9,0.08),
    "Po7":(-0.8,-0.02),"Po3":(-0.4,0.0),"Poz":(0.0,0.0),"Po4":(0.4,0.0),"Po8":(0.8,-0.02),
    "O1":(-0.35,-0.25),"Oz":(0.0,-0.27),"O2":(0.35,-0.25)
}
ALIASES = {"FT7":"Ft7","FT8":"Ft8","PO7":"Po7","PO3":"Po3","PO4":"Po4","PO8":"Po8",
           "T7.":"T7","T8.":"T8","T9.":"T9","T10.":"T10","FCz":"FCz","Fcz":"FCz",
           "Pz.":"Pz","Fz.":"Fz","Cz.":"Cz","Oz.":"Oz"}

def _map_label(lbl):
    k = ALIASES.get(lbl.strip(), lbl.strip())
    return k.replace('..','').replace('.','')

# ----------------------------
# IO
# ----------------------------
def load_csv(path, fs=None):
    df = pd.read_csv(path)
    cols = [c for c in df.columns if c.lower() not in ("time","t","sample","index")]
    data = df[cols].to_numpy(float)
    ch = [_map_label(c) for c in cols]
    return data, ch, fs

def load_edf(path):
    try:
        import pyedflib
    except Exception:
        print("pyedflib not installed; `pip install pyedflib` or use CSV.")
        sys.exit(1)
    f = pyedflib.EdfReader(path)
    n = f.signals_in_file
    ch = [_map_label(f.getLabel(i)) for i in range(n)]
    sig = np.vstack([f.readSignal(i) for i in range(n)]).T
    fs = float(f.getSampleFrequency(0))
    f.close()
    return sig, ch, fs

# ----------------------------
# DSP helpers
# ----------------------------
def bandpass(X, fs, lo, hi, order=4):
    lo = max(0.1, float(lo))
    hi = min(float(hi), fs/2 - 1e-3)
    if hi <= lo + 1e-6:
        hi = lo + 0.1
    b,a = butter(order, [lo/(fs/2), hi/(fs/2)], btype='band')
    return filtfilt(b,a,X,axis=0)

def notch(X, fs, f0=50.0, Q=30.0):
    try:
        b,a = iirnotch(f0/(fs/2), Q);  return filtfilt(b,a,X,axis=0)
    except Exception:
        return X

def analytic(X):
    Z = hilbert(X,axis=0);  A = np.abs(Z);  P = np.angle(Z)
    return Z,A,P

# ----------------------------
# Grid & field
# ----------------------------
def make_grid(n=256, margin=0.15):
    x = np.linspace(-1.1, 1.1, n)
    y = np.linspace(-0.9, 1.15, n)
    Xg,Yg = np.meshgrid(x,y)
    mask = ((Xg/1.1)**2 + ((Yg-0.15)/1.0)**2) <= (1.0 - margin)
    return Xg,Yg,mask

def huygens(ch_names, amps, phases, Xg, Yg, k=40.0, eps=1e-2):
    F = np.zeros_like(Xg,dtype=np.complex128)
    for a,ph,nm in zip(amps,phases,ch_names):
        if nm not in COORDS: continue
        sx,sy = COORDS[nm]
        r = np.hypot(Xg - sx, Yg - sy)
        F += (a/np.sqrt(r+eps))*np.exp(1j*(k*r + ph))
    return F

# ----------------------------
# Viewer
# ----------------------------
class WavefieldViewer:
    def __init__(self, data, ch, fs, band=(4,12), notch_hz=0.0,
                 mode="phase", show_vectors=False,
                 fps=30, grid=256, phase_c=1.0, dpi=150):

        self.fs = float(fs)
        self.mode = mode
        self.show_vectors = show_vectors
        self.grid_n = grid
        self.phase_c = phase_c
        self.fps = int(fps)
        self.hop = max(1, int(self.fs/self.fps))

        # Keep mappable channels
        keep = [i for i,n in enumerate(ch) if n in COORDS]
        self.ch = [ch[i] for i in keep]
        X0 = detrend(data[:,keep], axis=0, type='constant')
        if notch_hz>0: X0 = notch(X0, self.fs, notch_hz)
        self.X_raw = X0

        # Two bands
        self.bandA = tuple(band)
        self.bandB = (max(0.1, band[1]+5), band[1]+15)
        self.use_pair = True  # A+B by default

        # First filter/analytic
        self._refilter_all()

        # Grid
        self.Xg, self.Yg, self.mask = make_grid(n=self.grid_n)

        # State/UI
        self.playing = False
        self.index = 0
        self.timer = None
        self.quiv = None

        # Figure/UI
        self._build_figure(dpi)

    # ---- Filtering for both bands
    def _refilter_all(self):
        # Band A
        XA = bandpass(self.X_raw, self.fs, self.bandA[0], self.bandA[1])
        _, self.AA, self.PA = analytic(XA)
        # Band B
        XB = bandpass(self.X_raw, self.fs, self.bandB[0], self.bandB[1])
        _, self.AB, self.PB = analytic(XB)

        # Power (A)
        self.powerA = self.AA**2

        self.n_samples = self.AA.shape[0]
        self.duration = self.n_samples/self.fs

        # Robust norms
        self.amp_lo_A = np.percentile(self.AA, 5);  self.amp_hi_A = np.percentile(self.AA, 95)
        self.amp_lo_B = np.percentile(self.AB, 5);  self.amp_hi_B = np.percentile(self.AB, 95)
        self.pow_lo_A = np.percentile(self.powerA, 5); self.pow_hi_A = np.percentile(self.powerA, 95)

        # Wave numbers for Huygens
        fA = 0.5*(self.bandA[0]+self.bandA[1])
        fB = 0.5*(self.bandB[0]+self.bandB[1])
        self.kA = 2*np.pi*max(0.1,fA)/max(1e-6, self.phase_c)
        self.kB = 2*np.pi*max(0.1,fB)/max(1e-6, self.phase_c)

    # --- UI ---
    def _build_figure(self, dpi):
        # Taller bottom margin to host controls
        self.fig = plt.figure(figsize=(11.2,8.8), dpi=dpi)
        gs = self.fig.add_gridspec(12, 7, left=0.05, right=0.98, top=0.95, bottom=0.18, hspace=0.6, wspace=0.35)
        ax = self.fig.add_subplot(gs[:10, :7])
        self.ax = ax
        ax.set_xticks([]); ax.set_yticks([])
        self.title = ax.set_title(self._title_text(0.0), fontsize=12)

        # Scalp outline
        scalp = plt.Circle((0,0.15), 1.0, edgecolor=(0.65,0.66,0.74), facecolor='none', lw=1.0)
        ax.add_patch(scalp)

        # Layers
        self.im_phase = ax.imshow(np.zeros((256,256)), origin='lower',
                                  extent=[-1.1,1.1,-0.9,1.15],
                                  cmap='twilight', vmin=-np.pi, vmax=np.pi, animated=True)
        self.im_amp = ax.imshow(np.zeros((256,256)), origin='lower', alpha=0.65,
                                extent=[-1.1,1.1,-0.9,1.15],
                                cmap='magma', vmin=0, vmax=1, animated=True)

        # Electrodes
        xs,ys = zip(*[COORDS[n] for n in self.ch])
        ax.scatter(xs, ys, s=8, c='white', alpha=0.6, linewidths=0, zorder=3)

        # ---------- Controls BELOW the scalp ----------
        # Frequency sliders (Band A & B) — placed in two rows above the time bar
        ax_Alo = self.fig.add_axes([0.10, 0.12, 0.30, 0.03])
        ax_Ahi = self.fig.add_axes([0.10, 0.08, 0.30, 0.03])
        self.s_Alo = Slider(ax_Alo, "A low", 0.1, self.fs/2-1, valinit=self.bandA[0])
        self.s_Ahi = Slider(ax_Ahi, "A high", 0.5, self.fs/2-0.5, valinit=self.bandA[1])

        ax_Blo = self.fig.add_axes([0.55, 0.12, 0.30, 0.03])
        ax_Bhi = self.fig.add_axes([0.55, 0.08, 0.30, 0.03])
        self.s_Blo = Slider(ax_Blo, "B low", 0.1, self.fs/2-1, valinit=self.bandB[0])
        self.s_Bhi = Slider(ax_Bhi, "B high", 0.5, self.fs/2-0.5, valinit=self.bandB[1])

        for s in (self.s_Alo, self.s_Ahi, self.s_Blo, self.s_Bhi):
            s.on_changed(self._on_band_sliders)

        # Pair toggle button (right of the B sliders)
        ax_pair = self.fig.add_axes([0.88, 0.095, 0.08, 0.05])
        self.btn_pair = Button(ax_pair, "A + B")
        self.btn_pair.on_clicked(self._toggle_pair)

        # Time slider and transport buttons at the very bottom
        axt = self.fig.add_axes([0.08, 0.03, 0.60, 0.03])
        self.s_time = Slider(axt, "Time (s)", 0.0, self.duration, valinit=0.0)
        self.s_time.on_changed(self._on_slider)

        axp = self.fig.add_axes([0.70, 0.026, 0.08, 0.037])
        axb1= self.fig.add_axes([0.08,  0.005, 0.06, 0.027])
        axb5= self.fig.add_axes([0.145, 0.005, 0.06, 0.027])
        axf1= self.fig.add_axes([0.78,  0.026, 0.08, 0.037])
        axf5= self.fig.add_axes([0.86,  0.026, 0.08, 0.037])
        self.btn_play = Button(axp, "Play"); self.btn_play.on_clicked(lambda _ : self.toggle_play())
        Button(axb1,"⟲ 1s").on_clicked(lambda _ : self.seek_seconds(-1))
        Button(axb5,"⟲ 5s").on_clicked(lambda _ : self.seek_seconds(-5))
        Button(axf1,"1s ⟳").on_clicked(lambda _ : self.seek_seconds(+1))
        Button(axf5,"5s ⟳").on_clicked(lambda _ : self.seek_seconds(+5))

        # Key bindings & timer
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)
        self.timer = self.fig.canvas.new_timer(interval=int(1000/self.fps))
        self.timer.add_callback(self._tick)

        # First draw
        self._draw_index(self.index)

    def _toggle_pair(self, _):
        self.use_pair = not self.use_pair
        self.btn_pair.label.set_text("A + B" if self.use_pair else "A only")
        self._draw_index(self.index)

    # --- rendering helpers ---
    def _title_text(self, t):
        def rng(b): return f"{b[0]:.1f}-{b[1]:.1f} Hz"
        if self.mode == "huygens":
            pair = "A+B" if self.use_pair else "A"
            return f"EEG Wavefield — Huygens ({pair}) | A:{rng(self.bandA)}  B:{rng(self.bandB)} | t={t:0.2f}s"
        elif self.mode == "power":
            return f"EEG Wavefield — Band Power (A) | {self.bandA[0]:.1f}–{self.bandA[1]:.1f} Hz | t={t:0.2f}s"
        else:
            return f"EEG Wavefield — Phase Interpolation (A) | {self.bandA[0]:.1f}–{self.bandA[1]:.1f} Hz | t={t:0.2f}s"

    def _norm(self, a, lo, hi):
        return np.clip((a - lo)/(hi - lo + 1e-9), 0, 1)

    def _draw_index(self, idx):
        idx = int(np.clip(idx, 0, self.n_samples-1))
        self.index = idx
        t = self.index/self.fs

        # Pick band A for phase/power defaults
        ampsA = self._norm(self.AA[idx,:], self.amp_lo_A, self.amp_hi_A)
        phA   = self.PA[idx,:]

        if self.mode == "phase":
            R = griddata([COORDS[n] for n in self.ch], np.cos(phA), (self.Xg, self.Yg), method='cubic')
            I = griddata([COORDS[n] for n in self.ch], np.sin(phA), (self.Xg, self.Yg), method='cubic')
            phase_grid = np.arctan2(I, R)
            amp_grid   = griddata([COORDS[n] for n in self.ch], ampsA, (self.Xg, self.Yg), method='cubic')

        elif self.mode == "huygens":
            F = huygens(self.ch, ampsA, phA, self.Xg, self.Yg, k=self.kA)
            if self.use_pair:
                ampsB = self._norm(self.AB[idx,:], self.amp_lo_B, self.amp_hi_B)
                phB   = self.PB[idx,:]
                FB = huygens(self.ch, ampsB, phB, self.Xg, self.Yg, k=self.kB)
                F = F + FB
            phase_grid = np.angle(F)
            amp_grid   = np.abs(F)
            amp_grid  /= (np.nanpercentile(amp_grid[self.mask], 99) + 1e-9)

        else:  # power mode uses band A
            band_power_norm = self._norm(self.powerA[idx,:], self.pow_lo_A, self.pow_hi_A)
            power_grid = griddata([COORDS[n] for n in self.ch], band_power_norm, (self.Xg, self.Yg), method='cubic')
            phase_grid = power_grid
            amp_grid   = power_grid * 0.5
            self.im_phase.set_cmap('hot'); self.im_phase.set_clim(0,1); self.im_amp.set_alpha(0.30)

        if self.mode != "power":
            self.im_phase.set_cmap('twilight'); self.im_phase.set_clim(-np.pi, np.pi); self.im_amp.set_alpha(0.65)

        Pm = np.where(self.mask, phase_grid, np.nan)
        Am = np.where(self.mask, amp_grid,   np.nan)
        self.im_phase.set_data(Pm)
        self.im_amp.set_data(Am)

        if self.show_vectors:
            if getattr(self, "quiv", None) is not None: self.quiv.remove()
            gy, gx = np.gradient(Pm)
            mag = np.hypot(gx, gy)
            step = max(1, Pm.shape[0]//20)
            xs = self.Xg[::step,::step]; ys = self.Yg[::step,::step]
            vx = -gx[::step,::step]/(mag[::step,::step]+1e-6)
            vy = -gy[::step,::step]/(mag[::step,::step]+1e-6)
            self.quiv = self.ax.quiver(xs, ys, vx, vy, color=(0.95,0.95,1,0.65), scale=30, width=0.003)

        self.title.set_text(self._title_text(t))
        if abs(self.s_time.val - t) > (0.25/self.fs):
            self.s_time.set_val(t)
        self.fig.canvas.draw_idle()

    # --- playback / control ---
    def _tick(self):
        if not self.playing: return
        self._draw_index(self.index + self.hop)

    def toggle_play(self):
        self.playing = not self.playing
        self.btn_play.label.set_text("Pause" if self.playing else "Play")
        if self.playing: self.timer.start()
        else:            self.timer.stop()

    def seek_seconds(self, secs):
        if self.playing and abs(secs)<1e-6:
            self.toggle_play()
        self._draw_index(self.index + int(secs*self.fs))

    def _on_slider(self, val):
        self._draw_index(int(val*self.fs))

    def _on_band_sliders(self, _val):
        # Keep ordering and small gap
        Alo, Ahi = sorted([self.s_Alo.val, self.s_Ahi.val])
        Blo, Bhi = sorted([self.s_Blo.val, self.s_Bhi.val])
        if Ahi - Alo < 0.2: Ahi = Alo + 0.2
        if Bhi - Blo < 0.2: Bhi = Blo + 0.2
        self.bandA = (Alo, Ahi)
        self.bandB = (Blo, Bhi)
        cur_t = self.index / self.fs
        self._refilter_all()
        self._draw_index(int(cur_t * self.fs))

    def _on_key(self, ev):
        k = (ev.key or "").lower()
        if k == ' ':
            self.toggle_play()
        elif k in ('right',):
            self.seek_seconds(+1.0)
        elif k in ('left',):
            self.seek_seconds(-1.0)
        elif k == 'shift+right':
            self.seek_seconds(+5.0)
        elif k == 'shift+left':
            self.seek_seconds(-5.0)
        elif k == '.':
            self._draw_index(self.index + self.hop)
        elif k == ',':
            self._draw_index(self.index - self.hop)
        elif k == 'home':
            self._draw_index(0)
        elif k == 'end':
            self._draw_index(self.n_samples-1)
        elif k == 'v':
            self.show_vectors = not self.show_vectors
            if not self.show_vectors and getattr(self, "quiv", None) is not None:
                self.quiv.remove(); self.quiv = None
            self._draw_index(self.index)
        elif k == 'm':
            modes = ["phase", "power", "huygens"]
            i = modes.index(self.mode) if self.mode in modes else 0
            self.mode = modes[(i + 1) % len(modes)]
            print(f"Switched to {self.mode} mode")
            self._draw_index(self.index)
        elif k == 's':
            out = f"wavefield_{self.mode}_{int(self.index/self.fs*1000):06d}ms.png"
            self.fig.savefig(out, dpi=self.fig.dpi)
            print(f"Saved {out}")

    def show(self):
        plt.show()

# ----------------------------
# CLI
# ----------------------------
def main():
    p = argparse.ArgumentParser(description="Interactive EEG wavefield viewer with live frequency-pair sliders (controls below)")
    p.add_argument("-f","--file", required=True, help="CSV (header=channels) or EDF")
    p.add_argument("--fs", type=float, default=None, help="Sample rate (CSV only)")
    p.add_argument("--band", nargs=2, type=float, default=[3,10], help="Initial Band A Hz (e.g. 3 10)")
    p.add_argument("--notch", type=float, default=0.0, help="Notch frequency (50 or 60; 0=off)")
    p.add_argument("--mode", choices=["phase","huygens","power"], default="huygens")
    p.add_argument("--vectors", action="store_true", help="Show phase-gradient flow vectors")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--grid", type=int, default=256, help="Grid resolution (pixels per side)")
    p.add_argument("--c", type=float, default=1.0, help="Phase-speed scale for Huygens (k=2π f / c)")
    p.add_argument("--dpi", type=int, default=150, help="Figure DPI")
    args = p.parse_args()

    path = args.file.lower()
    if path.endswith(".edf"):
        data, ch, fs = load_edf(args.file)
    else:
        data, ch, fs0 = load_csv(args.file, args.fs)
        fs = args.fs or fs0 or 256.0

    viewer = WavefieldViewer(
        data, ch, fs,
        band=tuple(args.band),
        notch_hz=args.notch,
        mode=args.mode,
        show_vectors=args.vectors,
        fps=args.fps,
        grid=args.grid,
        phase_c=args.c,
        dpi=args.dpi
    )

    print("\nControls:")
    print("  Sliders: A low/high, B low/high (live re-filter)")
    print("  Button:  A only  ↔  A + B (Huygens superposition)")
    print("  Space/M/V/S & arrows: playback and navigation\n")
    viewer.show()

if __name__ == "__main__":
    main()
