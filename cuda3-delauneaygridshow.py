# -*- coding: utf-8 -*-
"""
EEG Wavefield Viewer — CUDA Moiré (local Δk→λ overlay + 3D stack + Delaunay Grid)

New features:
  • Grid toggle button to show/hide Delaunay triangulation
  • Shows the electrode lattice structure that creates the triangular/polygonal interference patterns
  • Helps understand how the electrode geometry influences the wave reconstruction

Usage:
  python cuda3_with_grid.py -f your.edf --band 3 10 --mode huygens --fps 30
"""

import argparse, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from scipy.signal import butter, filtfilt, hilbert, detrend, iirnotch
from scipy.interpolate import griddata
from scipy.spatial import Delaunay

# --- Optional CUDA / CuPy integration (for 3D stack + some array math) ---
try:
    import cupy as cp
    from cupyx.scipy.signal import hilbert as cupy_hilbert
    CUPY_AVAILABLE = True
    print("✅ CuPy found. GPU acceleration enabled for 3D stack + gradients.")
except Exception:
    CUPY_AVAILABLE = False
    cp = None
    cupy_hilbert = None
    print("⚠️ CuPy not found. Running on CPU; install a CUDA toolkit + cupy-cudaXXx for speed.")

# ----------------------------
# Electrode layout (10-20/10-10 approx.)
# ----------------------------
COORDS = {
    "Fp1":(-0.5,1.0),"Fpz":(0.0,1.05),"Fp2":(0.5,1.0),
    "AF7":(-0.7,0.9),"AF3":(-0.35,0.92),"AFz":(0.0,0.95),"AF4":(0.35,0.92),"AF8":(0.7,0.9),
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
# IO + DSP helpers
# ----------------------------
def load_csv(path, fs=None):
    df = pd.read_csv(path)
    cols = [c for c in df.columns if c.lower() not in ("time","t","sample","index")]
    return df[cols].to_numpy(float), [_map_label(c) for c in cols], fs

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

def analytic_cpu(X):
    Z = hilbert(X,axis=0)
    return np.abs(Z), np.angle(Z)

def analytic_gpu(X):
    # X is numpy; move to GPU, Hilbert on GPU, back to numpy
    Zg = cp.asarray(X)
    Zg = cupy_hilbert(Zg, axis=0)
    A = cp.asnumpy(cp.abs(Zg))
    P = cp.asnumpy(cp.angle(Zg))
    return A, P

# ----------------------------
# Grid + Huygens
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
# 3D Frequency Stack (GPU-accelerated)
# ----------------------------
class MoireStack3D:
    def __init__(self, elec_xy: dict[str, tuple[float, float]], rbf_sigma=0.30):
        self.elec_xy = elec_xy
        self.rbf_sigma = float(rbf_sigma)
        self._cache = {}  # basis cache

    @staticmethod
    def _bp(sig, fs, lo, hi, order=4):
        from scipy.signal import butter, filtfilt
        ny = 0.5 * fs
        lo = max(0.1, float(lo))
        hi = min(float(hi), fs * 0.49)
        b, a = butter(order, [lo/ny, hi/ny], btype='band')
        return filtfilt(b, a, sig, axis=-1)

    # ---------- BASIS CACHE (CPU/GPU) ----------
    def _prepare_basis_cpu(self, labels, grid):
        import numpy as np
        pos = np.array([self.elec_xy[ch] for ch in labels if ch in self.elec_xy], dtype=np.float32)
        if pos.size == 0:
            raise ValueError("No channels match electrode map.")
        lin = np.linspace(-1.0, 1.0, grid, dtype=np.float32)
        xx, yy = np.meshgrid(lin, lin)                        # (G,G)
        mask = (xx*xx + yy*yy) <= 1.0
        # g will be built per call on CPU (smaller jobs)
        return pos, xx, yy, mask

    def _prepare_basis_gpu(self, labels, grid):
        if not CUPY_AVAILABLE:
            return None
        import numpy as np, cupy as cp
        pos = np.array([self.elec_xy[ch] for ch in labels if ch in self.elec_xy], dtype=np.float32)
        if pos.size == 0:
            raise ValueError("No channels match electrode map.")
        key = (tuple(np.round(pos.ravel(), 3)), grid, round(self.rbf_sigma, 3))
        if key in self._cache:
            return self._cache[key]
        # build once on GPU
        pos_g = cp.asarray(pos, dtype=cp.float32)             # (N,2)
        lin = cp.linspace(-1.0, 1.0, grid, dtype=cp.float32)
        xx, yy = cp.meshgrid(lin, lin)                        # (G,G)
        gx = xx[None] - pos_g[:, 0, None, None]               # (N,G,G)
        gy = yy[None] - pos_g[:, 1, None, None]
        s2 = (self.rbf_sigma ** 2)
        g = cp.exp(-0.5*(gx*gx + gy*gy)/s2, dtype=cp.float32) + 1e-8  # (N,G,G)
        mask = (xx*xx + yy*yy) <= 1.0
        self._cache[key] = (pos_g, xx, yy, g, mask)
        return self._cache[key]

    # ---------- MAP MAKERS ----------
    def band_map_cpu(self, labels, data, fs, lo, hi, grid=256, win_sec=1.0, center_idx=None):
        import numpy as np
        # indices & order
        use_idx = [i for i, ch in enumerate(labels) if ch in self.elec_xy]
        pos, xx, yy, mask = self._prepare_basis_cpu(labels, grid)
        sig = np.asarray(data)[use_idx, :]                     # (N,T)

        # window
        T = sig.shape[1]; W = min(max(1, int(round(win_sec * fs))), T)
        t0 = max(0, min(int(center_idx) - W//2, T-W)) if center_idx is not None else (T - W)//2

        # filter + analytic (CPU)
        from scipy.signal import hilbert
        bp = self._bp(sig, fs, lo, hi)[:, t0:t0+W]             # (N,W)
        an = hilbert(bp, axis=1)
        amp = np.abs(an); ph = np.angle(an)
        w = amp / (np.std(amp, axis=1, keepdims=True) + 1e-8)
        ic = (w * np.cos(ph)).mean(axis=1).astype(np.float32)  # (N,)

        # RBF weights (CPU) — compute once per call
        gx = xx[None] - pos[:, 0, None, None]                  # (N,G,G)
        gy = yy[None] - pos[:, 1, None, None]
        s2 = (self.rbf_sigma ** 2)
        g = np.exp(-0.5*(gx*gx + gy*gy)/s2, dtype=np.float32) + 1e-8
        num = np.tensordot(ic, g, axes=(0,0))                  # (G,G)
        den = g.sum(axis=0)                                    # (G,G)
        field = (num / den)
        return np.where(mask, field, np.nan)

    def band_map_cuda(self, labels, data, fs, lo, hi, grid=256, win_sec=1.0, center_idx=None):
        if not CUPY_AVAILABLE:
            return self.band_map_cpu(labels, data, fs, lo, hi, grid, win_sec, center_idx)
        import numpy as np, cupy as cp
        use_idx = [i for i, ch in enumerate(labels) if ch in self.elec_xy]
        pos_g, xx, yy, g, mask = self._prepare_basis_gpu(labels, grid)  # cached!
        sig = np.asarray(data)[use_idx, :]                     # (N,T) on CPU

        # window
        T = sig.shape[1]; W = min(max(1, int(round(win_sec * fs))), T)
        t0 = max(0, min(int(center_idx) - W//2, T-W)) if center_idx is not None else (T - W)//2

        # filter on CPU (IIR) + analytic on GPU
        bp = self._bp(sig, fs, lo, hi)[:, t0:t0+W]             # (N,W)
        an = cupy_hilbert(cp.asarray(bp, dtype=cp.float32), axis=1)
        amp = cp.abs(an); ph = cp.angle(an)
        w = amp / (cp.std(amp, axis=1, keepdims=True) + 1e-8)
        ic = (w * cp.cos(ph)).mean(axis=1).astype(cp.float32)  # (N,)

        # weighted sum using tensordot over N
        num = cp.tensordot(ic, g, axes=(0,0))                  # (G,G)
        den = g.sum(axis=0)                                    # (G,G)
        field = (num / den)
        return np.where(cp.asnumpy(mask), cp.asnumpy(field), np.nan)

    # ---------- STACK BUILDER & VIEWER ----------
    def build_stack(self, labels, data, fs, fstart, fend, bw, step, grid=256, win_sec=1.0, center_idx=None):
        maps, bands = [], []
        f = float(fstart)
        while f < fend + 1e-9:
            lo, hi = f, min(f+bw, fend)
            print(f"  Slice {lo:.1f}-{hi:.1f} Hz...")
            fmap = (self.band_map_cuda if CUPY_AVAILABLE else self.band_map_cpu)(
                labels, data, fs, lo, hi, grid, win_sec, center_idx
            )
            maps.append(fmap); bands.append((lo, hi)); f += step
        return np.array(bands, float), np.stack(maps, axis=0)

    def show(self, bands, stack, title="Frequency—Cell Moiré (3D)"):
        import numpy as np, matplotlib.pyplot as plt
        from matplotlib.widgets import Slider, Button
        nb, H, W = stack.shape
        fig = plt.figure(figsize=(12, 7))
        gs  = fig.add_gridspec(3, 2, height_ratios=[20,1,1], left=0.06, right=0.98, top=0.94, bottom=0.16, wspace=0.22)
        ax2 = fig.add_subplot(gs[0,0])
        ax3 = fig.add_subplot(gs[0,1], projection='3d')

        # Left: current slice
        def _norm(a):
            mn, mx = np.nanmin(a), np.nanmax(a)
            return (a - mn)/((mx-mn)+1e-9)
        im = ax2.imshow(_norm(stack[0]), cmap="twilight", origin="lower")
        lo,hi=bands[0]
        ax2.set_title(f"{title} — Slice 1/{nb}  {lo:.1f}-{hi:.1f} Hz")
        ax2.axis('off')

        # Right: cylinder (downsampled for speed)
        ds = max(4, H//64)  # stride
        X, Y = np.meshgrid(np.arange(W)[::ds], np.arange(H)[::ds])
        ax3.clear()
        for k in range(nb):
            Z = np.ones_like(X) * k
            img = _norm(stack[k][::ds, ::ds])
            rgba = plt.cm.twilight(img); rgba[~np.isfinite(img)] = (0,0,0,0)
            ax3.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors=rgba, shade=False, linewidth=0, antialiased=False)
        ax3.set_title("3D Frequency Stack (Z = band slice)")
        ax3.set_zlabel("slice")
        ax3.set_xticks([]); ax3.set_yticks([])
        ax3.view_init(elev=25, azim=-60)

        # Controls (non-blocking timer)
        s = Slider(fig.add_subplot(gs[1,0]), "Slice", 1, nb, valinit=1, valstep=1)
        b = Button(fig.add_subplot(gs[2,0]), "▶ Play")
        state = {"i": 0, "playing": False}
        timer = fig.canvas.new_timer(interval=70)
        def tick():
            if not state["playing"]: return
            state["i"] = (state["i"] + 1) % nb
            s.set_val(state["i"] + 1)  # triggers on_slide
        timer.add_callback(tick)

        def on_slide(_):
            i = int(s.val) - 1
            im.set_data(_norm(stack[i]))
            lo, hi = bands[i]
            ax2.set_title(f"{title} — Slice {i+1}/{nb}  {lo:.1f}-{hi:.1f} Hz")
            fig.canvas.draw_idle()

        def on_play(_):
            state["playing"] = not state["playing"]
            b.label.set_text("⸼ Pause" if state["playing"] else "▶ Play")
            if state["playing"]: timer.start()
            else:                timer.stop()

        s.on_changed(on_slide)
        b.on_clicked(on_play)
        plt.show(block=False)  # no event-loop warning

    def run(
        self,
        labels,
        data,
        fs,
        fstart,
        fend,
        bw,
        step,
        grid=256,
        win_sec=1.0,
        center_idx=None,
        title="Moiré 3D"
    ):
        """Build the band stack and open the viewer."""
        print(f"Building 3D Moiré stack... (Using {'GPU' if CUPY_AVAILABLE else 'CPU'})")
        bands, stack = self.build_stack(
            labels=labels,
            data=data,
            fs=fs,
            fstart=fstart,
            fend=fend,
            bw=bw,
            step=step,
            grid=grid,
            win_sec=win_sec,
            center_idx=center_idx,
        )
        print("...done. Launching viewer.")
        self.show(bands, stack, title=title)


# ----------------------------
# Main Viewer
# ----------------------------
class WavefieldViewer:
    def __init__(self, data, ch, fs, band=(4,12), notch_hz=0.0,
                 mode="huygens", show_vectors=False, fps=30, grid=256, phase_c=1.0, dpi=150):
        self.fs = float(fs)
        self.mode = mode
        self.show_vectors = show_vectors
        self.grid_n = grid
        self.phase_c = phase_c
        self.fps = int(fps)
        self.hop = max(1, int(self.fs/self.fps))
        self._moire = MoireStack3D(COORDS)

        # channels present in coord map
        keep = [i for i,n in enumerate(ch) if n in COORDS]
        self.ch = [ch[i] for i in keep]
        X0 = detrend(data[:,keep], axis=0, type='constant')
        if notch_hz>0:
            X0 = notch(X0, self.fs, notch_hz)
        self.X_raw = X0

        self.bandA = tuple(band)
        self.bandB = (max(0.1, band[1]+5), band[1]+15)
        self.use_pair = True

        self._refilter_all()

        self.Xg, self.Yg, self.mask = make_grid(n=self.grid_n)
        self.playing = False; self.index = 0; self.timer = None; self.quiv = None
        self.overlay_on = False; self.overlay_alpha = 0.45; self.overlay_im = None
        
        # Grid overlay variables
        self.grid_on = False
        self.grid_lines = []

        self._build_figure(dpi)

    # --- filtering + analytic ---
    def _refilter_all(self):
        XA = bandpass(self.X_raw, self.fs, self.bandA[0], self.bandA[1])
        XB = bandpass(self.X_raw, self.fs, self.bandB[0], self.bandB[1])
        if CUPY_AVAILABLE:
            self.AA, self.PA = analytic_gpu(XA)
            self.AB, self.PB = analytic_gpu(XB)
        else:
            self.AA, self.PA = analytic_cpu(XA)
            self.AB, self.PB = analytic_cpu(XB)
        self.powerA = self.AA**2
        self.n_samples = self.AA.shape[0]; self.duration = self.n_samples/self.fs
        self.amp_lo_A, self.amp_hi_A = np.percentile(self.AA, [5, 95])
        self.amp_lo_B, self.amp_hi_B = np.percentile(self.AB, [5, 95])
        self.pow_lo_A, self.pow_hi_A = np.percentile(self.powerA, [5, 95])
        fA = 0.5*(self.bandA[0]+self.bandA[1]); fB = 0.5*(self.bandB[0]+self.bandB[1])
        self.kA = 2*np.pi*max(0.1,fA)/max(1e-6, self.phase_c)
        self.kB = 2*np.pi*max(0.1,fB)/max(1e-6, self.phase_c)

    # --- Grid overlay methods ---
    def _draw_grid_overlay(self):
        # remove old lines
        for ln in self.grid_lines:
            ln.remove()
        self.grid_lines.clear()

        pts = np.array([COORDS[n] for n in self.ch], dtype=float)
        if len(pts) < 3:  # need at least a triangle
            return

        tri = Delaunay(pts)
        # draw all triangle edges
        for a,b,c in tri.simplices:
            for i,j in [(a,b),(b,c),(c,a)]:
                x = [pts[i,0], pts[j,0]]
                y = [pts[i,1], pts[j,1]]
                ln, = self.ax.plot(x, y, lw=0.8, color='white', alpha=0.6, zorder=4)
                self.grid_lines.append(ln)

        # re-draw figure
        self.fig.canvas.draw_idle()

    def _draw_voronoi_overlay(self):
        from scipy.spatial import Voronoi
        # remove old lines
        for ln in self.grid_lines:
            ln.remove()
        self.grid_lines.clear()

        pts = np.array([COORDS[n] for n in self.ch], dtype=float)
        if len(pts) < 3:
            return

        try:
            vor = Voronoi(pts)
            # Draw Voronoi cell edges
            for pointidx, simplex in zip(vor.ridge_points, vor.ridge_vertices):
                simplex = np.asarray(simplex)
                if np.all(simplex >= 0):  # finite ridge
                    ln, = self.ax.plot(vor.vertices[simplex, 0], vor.vertices[simplex, 1], 
                                     lw=0.8, color='orange', alpha=0.6, zorder=4)
                    self.grid_lines.append(ln)
        except Exception:
            # Fallback to Delaunay if Voronoi fails
            self._draw_grid_overlay()
            return

        self.fig.canvas.draw_idle()

    def _toggle_grid(self, _):
        self.grid_on = (self.grid_on + 1) % 3  # 0=off, 1=delaunay, 2=voronoi
        if self.grid_on == 0:
            self.btn_grid.label.set_text("Grid")
            for ln in self.grid_lines:
                ln.remove()
            self.grid_lines.clear()
        elif self.grid_on == 1:
            self.btn_grid.label.set_text("Delaunay")
            self._draw_grid_overlay()
        else:  # self.grid_on == 2
            self.btn_grid.label.set_text("Voronoi")
            self._draw_voronoi_overlay()
        self.fig.canvas.draw_idle()

    # --- UI ---
    def _build_figure(self, dpi):
        self.fig = plt.figure(figsize=(11.6,8.9), dpi=dpi)
        gs = self.fig.add_gridspec(12, 7, left=0.05, right=0.98, top=0.95, bottom=0.18)
        ax = self.fig.add_subplot(gs[:10, :7]); self.ax = ax
        ax.set_xticks([]); ax.set_yticks([])
        self.title = ax.set_title(self._title_text(0.0), fontsize=12)

        scalp = plt.Circle((0,0.15), 1.0, edgecolor=(0.65,0.66,0.74), facecolor='none', lw=1.0)
        ax.add_patch(scalp)

        self.im_phase = ax.imshow(np.zeros((256,256)), origin='lower',
                                  extent=[-1.1,1.1,-0.9,1.15], cmap='twilight', vmin=-np.pi, vmax=np.pi, animated=True)
        self.im_amp = ax.imshow(np.zeros((256,256)), origin='lower', alpha=0.65,
                                extent=[-1.1,1.1,-0.9,1.15], cmap='magma', vmin=0, vmax=1, animated=True)

        xs,ys = zip(*[COORDS[n] for n in self.ch]); ax.scatter(xs, ys, s=8, c='white', alpha=0.6, lw=0, zorder=3)

        # Controls below
        self.s_Alo = Slider(self.fig.add_axes([0.10, 0.12, 0.30, 0.03]), "A low", 0.1, self.fs/2-1, valinit=self.bandA[0])
        self.s_Ahi = Slider(self.fig.add_axes([0.10, 0.08, 0.30, 0.03]), "A high", 0.5, self.fs/2-0.5, valinit=self.bandA[1])
        self.s_Blo = Slider(self.fig.add_axes([0.55, 0.12, 0.30, 0.03]), "B low", 0.1, self.fs/2-1, valinit=self.bandB[0])
        self.s_Bhi = Slider(self.fig.add_axes([0.55, 0.08, 0.30, 0.03]), "B high", 0.5, self.fs/2-0.5, valinit=self.bandB[1])
        for s in (self.s_Alo, self.s_Ahi, self.s_Blo, self.s_Bhi): s.on_changed(self._on_band_sliders)

        self.btn_pair = Button(self.fig.add_axes([0.88, 0.095, 0.08, 0.05]), "A + B")
        self.btn_pair.on_clicked(self._toggle_pair)

        # Δk→λ overlay controls
        self.btn_overlay = Button(self.fig.add_axes([0.88, 0.055, 0.08, 0.035]), "Δk→λ")
        self.btn_overlay.on_clicked(self._toggle_overlay)
        self.s_alpha = Slider(self.fig.add_axes([0.55, 0.05, 0.30, 0.02]), "Overlay α", 0.05, 0.90, valinit=self.overlay_alpha)
        self.s_alpha.on_changed(self._on_alpha)

        # Grid button
        self.btn_grid = Button(self.fig.add_axes([0.88, 0.015, 0.08, 0.035]), "Grid")
        self.btn_grid.on_clicked(self._toggle_grid)

        # Time + transport
        self.s_time = Slider(self.fig.add_axes([0.08, 0.03, 0.55, 0.03]), "Time (s)", 0.0, self.duration, valinit=0.0)
        self.s_time.on_changed(self._on_slider)
        self.btn_play = Button(self.fig.add_axes([0.72, 0.026, 0.05, 0.037]), "Play")
        self.btn_play.on_clicked(lambda _ : self.toggle_play())
        Button(self.fig.add_axes([0.64, 0.026, 0.04, 0.037]), "« 5s").on_clicked(lambda _ : self.seek_seconds(-5))
        Button(self.fig.add_axes([0.68, 0.026, 0.04, 0.037]), "« 1s").on_clicked(lambda _ : self.seek_seconds(-1))
        Button(self.fig.add_axes([0.77, 0.026, 0.04, 0.037]), "1s »").on_clicked(lambda _ : self.seek_seconds(+1))
        Button(self.fig.add_axes([0.81, 0.026, 0.04, 0.037]), "5s »").on_clicked(lambda _ : self.seek_seconds(+5))
        Button(self.fig.add_axes([0.88, 0.026, 0.08, 0.037]), "3D Stack").on_clicked(self._launch_moire)

        self.fig.canvas.mpl_connect('key_press_event', self._on_key)
        self.timer = self.fig.canvas.new_timer(interval=int(1000/self.fps)); self.timer.add_callback(self._tick)
        self._draw_index(self.index)

    # --- overlay helpers ---
    def _compute_phase_grid(self, phases_t):
        # go via cos/sin + arctan2 to avoid branch cuts
        pts = [COORDS[n] for n in self.ch]
        R = griddata(pts, np.cos(phases_t), (self.Xg, self.Yg), method='cubic')
        I = griddata(pts, np.sin(phases_t), (self.Xg, self.Yg), method='cubic')
        return np.arctan2(I, R)

    def _local_lambda_from_phase(self, phiA, phiB):
        # λ_local = 2π / |∇φ_A - ∇φ_B|
        if CUPY_AVAILABLE:
            PhiA = cp.asarray(phiA); PhiB = cp.asarray(phiB)
            gyA, gxA = cp.gradient(PhiA); gyB, gxB = cp.gradient(PhiB)
            dkg = cp.hypot((gxA-gxB), (gyA-gyB)) + 1e-9
            lam = (2.0*np.pi) / cp.asnumpy(dkg)
        else:
            gyA, gxA = np.gradient(phiA); gyB, gxB = np.gradient(phiB)
            dkg = np.hypot((gxA-gxB), (gyA-gyB)) + 1e-9
            lam = (2.0*np.pi) / dkg
        lam = np.where(self.mask, lam, np.nan)
        # robust normalize for coloring (large λ = coarse; small λ = fine)
        finite = np.isfinite(lam)
        if not np.any(finite): return lam, (0,1)
        p2, p98 = np.nanpercentile(lam[finite], [2, 98])
        return lam, (p2, p98)

    # --- drawing ---
    def _draw_index(self, idx):
        self.index = idx = int(np.clip(idx, 0, self.n_samples-1))
        t = self.index/self.fs

        ampsA = np.clip((self.AA[idx,:]-self.amp_lo_A)/(self.amp_hi_A-self.amp_lo_A+1e-9), 0, 1)
        phA_e = self.PA[idx,:]

        if self.mode == "phase":
            phase_grid = self._compute_phase_grid(phA_e)
            amp_grid   = griddata([COORDS[n] for n in self.ch], ampsA, (self.Xg, self.Yg), method='cubic')

        elif self.mode == "huygens":
            F = huygens(self.ch, ampsA, phA_e, self.Xg, self.Yg, k=self.kA)
            if self.use_pair:
                ampsB = np.clip((self.AB[idx,:]-self.amp_lo_B)/(self.amp_hi_B-self.amp_lo_B+1e-9), 0, 1)
                phB_e = self.PB[idx,:]
                F += huygens(self.ch, ampsB, phB_e, self.Xg, self.Yg, k=self.kB)
            phase_grid = np.angle(F)
            amp_grid   = np.abs(F); amp_grid /= (np.nanpercentile(amp_grid[self.mask], 99) + 1e-9)
        else:  # power(A)
            pwr = np.clip((self.powerA[idx,:]-self.pow_lo_A)/(self.pow_hi_A-self.pow_lo_A+1e-9), 0, 1)
            phase_grid = griddata([COORDS[n] for n in self.ch], pwr, (self.Xg, self.Yg), method='cubic')
            amp_grid   = phase_grid * 0.5
            self.im_phase.set_cmap('hot'); self.im_phase.set_clim(0,1); self.im_amp.set_alpha(0.30)

        if self.mode != "power":
            self.im_phase.set_cmap('twilight'); self.im_phase.set_clim(-np.pi, np.pi); self.im_amp.set_alpha(0.65)

        self.im_phase.set_data(np.where(self.mask, phase_grid, np.nan))
        self.im_amp.set_data(np.where(self.mask, amp_grid,   np.nan))

        # Optional Δk→λ overlay (needs both bands)
        if self.overlay_on and self.mode == "huygens" and self.use_pair:
            phiA = self._compute_phase_grid(self.PA[idx,:])
            phiB = self._compute_phase_grid(self.PB[idx,:])
            lam_map, (lo,hi) = self._local_lambda_from_phase(phiA, phiB)
            if self.overlay_im is None:
                self.overlay_im = self.ax.imshow(lam_map, origin="lower",
                                                 extent=[-1.1,1.1,-0.9,1.15],
                                                 cmap="viridis", vmin=lo, vmax=hi, alpha=self.overlay_alpha)
            else:
                self.overlay_im.set_data(lam_map); self.overlay_im.set_clim(lo,hi); self.overlay_im.set_alpha(self.overlay_alpha)
        else:
            if self.overlay_im is not None:
                self.overlay_im.remove(); self.overlay_im = None

        self.title.set_text(self._title_text(t))
        if abs(self.s_time.val - t) > (0.25/self.fs):
            self.s_time.set_val(t)
        self.fig.canvas.draw_idle()

    # --- title with Δf and λ prediction ---
    def _title_text(self, t):
        rng = lambda b: f"{b[0]:.1f}-{b[1]:.1f} Hz"
        base = f"EEG Wavefield — {'Huygens' if self.mode=='huygens' else ('Band Power' if self.mode=='power' else 'Phase')}"
        if self.mode == "huygens":
            pair = "(A+B)" if self.use_pair else "(A)"
            fA = 0.5*(self.bandA[0]+self.bandA[1]); fB = 0.5*(self.bandB[0]+self.bandB[1])
            df = abs(fA - fB)
            lam = (self.phase_c / max(1e-6, df)) if self.use_pair else None
            cells = (2.0 / lam) if lam else None
            info = f"  Δf={df:.2f} Hz → λ≈{lam:.2f} (≈{cells:.1f} cells)" if lam else ""
            return f"{base} {pair} | A:{rng(self.bandA)}  B:{rng(self.bandB)}{info} | t={t:0.2f}s"
        else:
            return f"{base} (A) | {rng(self.bandA)} | t={t:0.2f}s"

    # --- callbacks / controls ---
    def _toggle_pair(self, _):
        self.use_pair = not self.use_pair
        self.btn_pair.label.set_text("A + B" if self.use_pair else "A only")
        self._draw_index(self.index)

    def _toggle_overlay(self, _):
        self.overlay_on = not self.overlay_on
        self.btn_overlay.label.set_text("Δk→λ ✓" if self.overlay_on else "Δk→λ")
        self._draw_index(self.index)

    def _on_alpha(self, _val):
        self.overlay_alpha = float(self.s_alpha.val)
        if self.overlay_im is not None:
            self.overlay_im.set_alpha(self.overlay_alpha)
            self.fig.canvas.draw_idle()

    def _tick(self):
        if self.playing:
            self._draw_index(self.index + self.hop)

    def toggle_play(self):
        self.playing = not self.playing
        self.btn_play.label.set_text("Pause" if self.playing else "Play")
        if self.playing: self.timer.start()
        else:            self.timer.stop()

    def seek_seconds(self, secs):
        self._draw_index(self.index + int(secs*self.fs))

    def _on_slider(self, val):
        self._draw_index(int(val*self.fs))

    def _on_band_sliders(self, _):
        Alo, Ahi = sorted([self.s_Alo.val, self.s_Ahi.val]); Blo, Bhi = sorted([self.s_Blo.val, self.s_Bhi.val])
        if Ahi - Alo < 0.2: Ahi = Alo + 0.2
        if Bhi - Blo < 0.2: Bhi = Blo + 0.2
        self.bandA = (Alo, Ahi); self.bandB = (Blo, Bhi)
        cur = self.index
        self._refilter_all()
        self._draw_index(cur)

    def _launch_moire(self, _=None):
        fstart = min(self.s_Alo.val, self.s_Blo.val)
        fend   = max(self.s_Ahi.val, self.s_Bhi.val)
        # keep 3D light: 256 or your main grid, whichever is smaller
        stack_grid = int(min(256, self.grid_n))
        self._moire.run(
            labels=self.ch, data=self.X_raw.T, fs=self.fs,
            fstart=fstart, fend=fend, bw=2.0, step=2.0,
            grid=stack_grid, win_sec=0.75, center_idx=self.index,
            title="Moiré 3D (GPU)" if CUPY_AVAILABLE else "Moiré 3D (CPU)"
        )

    def _on_key(self, ev):
        k = (ev.key or "").lower()
        if k == ' ':
            self.toggle_play()
        elif k == '3':
            self._launch_moire()
        elif k == 'l':
            self._toggle_overlay(None)
        elif k == 'g':
            self._toggle_grid(None)
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

    def show(self):
        plt.show()

# ----------------------------
# CLI
# ----------------------------
def main():
    p = argparse.ArgumentParser(description="Interactive EEG wavefield viewer with CUDA-accelerated moiré tools and Delaunay grid overlay")
    p.add_argument("-f","--file", required=True, help="CSV (header=channels) or EDF")
    p.add_argument("--fs", type=float, default=None, help="Sample rate (CSV only)")
    p.add_argument("--band", nargs=2, type=float, default=[3,10], help="Initial Band A Hz (e.g. 3 10)")
    p.add_argument("--notch", type=float, default=0.0, help="Notch frequency (50 or 60; 0=off)")
    p.add_argument("--mode", choices=["phase","huygens","power"], default="huygens")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--grid", type=int, default=256, help="Grid resolution (pixels per side)")
    p.add_argument("--c", type=float, default=1.0, help="Phase-speed scale for Huygens (k=2π f / c)")
    p.add_argument("--dpi", type=int, default=150, help="Figure DPI")
    args = p.parse_args()

    if args.file.lower().endswith(".edf"):
        data, ch, fs = load_edf(args.file)
    else:
        data, ch, fs0 = load_csv(args.file, args.fs)
        fs = args.fs or fs0 or 256.0

    viewer = WavefieldViewer(
        data, ch, fs,
        band=tuple(args.band),
        notch_hz=args.notch,
        mode=args.mode,
        fps=args.fps,
        grid=args.grid,
        phase_c=args.c,
        dpi=args.dpi
    )
    print("\nKeyboard:")
    print("  Space Play/Pause   Arrow ←/→ step   Shift+Arrows ±5s")
    print("  L toggle Δk→λ overlay   G toggle Grid overlay   3 open 3D Stack")
    viewer.show()

if __name__ == "__main__":
    main()