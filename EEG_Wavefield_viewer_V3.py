# eeg_wavefield_viewer_ui_enhanced.py
import argparse, sys, math, time, os
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
# Preprocessing
# ----------------------------
def bandpass(X, fs, lo, hi, order=4):
    b,a = butter(order, [lo/(fs/2), hi/(fs/2)], btype='band')
    return filtfilt(b,a,X,axis=0)

def notch(X, fs, f0=50.0, Q=30.0):
    try:
        b,a = iirnotch(f0/(fs/2), Q);  return filtfilt(b,a,X,axis=0)
    except Exception:
        return X

def laplacian(X, ch_names):
    out = X.copy()
    coords = np.array([COORDS[c] for c in ch_names if c in COORDS])
    used = [i for i,c in enumerate(ch_names) if c in COORDS]
    for idx,i in enumerate(used):
        xi, yi = COORDS[ch_names[i]]
        d = np.linalg.norm(coords - np.array([xi,yi]), axis=1)
        nbr_idx = [used[j] for j in np.where((d>1e-6) & (d<0.5))[0]]
        if len(nbr_idx)>=3:
            out[:,i] = X[:,i] - np.mean(X[:,nbr_idx],axis=1)
    return out

def analytic(X):
    Z = hilbert(X,axis=0);  A = np.abs(Z);  P = np.angle(Z)
    return Z,A,P

def compute_alpha_power(X, fs, alpha_band=(8,12), window_len=1.0):
    """Compute alpha power using sliding window approach similar to MNE"""
    # Apply bandpass filter for alpha band
    X_alpha = bandpass(X, fs, alpha_band[0], alpha_band[1])
    
    # Compute instantaneous power using Hilbert transform
    analytic_signal = hilbert(X_alpha, axis=0)
    power = np.abs(analytic_signal) ** 2
    
    # Apply smoothing window (similar to MNE's approach)
    window_samples = int(window_len * fs)
    if window_samples > 1:
        # Simple moving average smoothing
        kernel = np.ones(window_samples) / window_samples
        power_smooth = np.zeros_like(power)
        for ch in range(power.shape[1]):
            power_smooth[:, ch] = np.convolve(power[:, ch], kernel, mode='same')
        return power_smooth
    return power

def compute_interference_bands_power(X, fs, base_freq, combination_freqs, window_len=1.0):
    """
    Compute power at specific frequency combinations that create interference patterns.
    
    Parameters:
    -----------
    X : array, shape (n_times, n_channels)
        EEG data
    fs : float
        Sampling frequency
    base_freq : tuple
        Base frequency band (low, high) in Hz
    combination_freqs : list of tuples
        List of frequency bands that create interference with base_freq
    window_len : float
        Smoothing window length in seconds
    
    Returns:
    --------
    power_bands : dict
        Dictionary with power for each frequency combination
    """
    power_bands = {}
    
    # Compute power for base frequency
    base_name = f"Base_{base_freq[0]}-{base_freq[1]}Hz"
    X_base = bandpass(X, fs, base_freq[0], base_freq[1])
    power_bands[base_name] = np.abs(hilbert(X_base, axis=0)) ** 2
    
    # Compute power for each combination frequency
    for i, comb_freq in enumerate(combination_freqs):
        comb_name = f"Comb{i+1}_{comb_freq[0]}-{comb_freq[1]}Hz"
        X_comb = bandpass(X, fs, comb_freq[0], comb_freq[1])
        power_bands[comb_name] = np.abs(hilbert(X_comb, axis=0)) ** 2
    
    # Compute interference patterns (sum and difference frequencies)
    interference_bands = {}
    for i, comb_freq in enumerate(combination_freqs):
        # Beat frequency (difference)
        beat_center = abs(np.mean(base_freq) - np.mean(comb_freq))
        if beat_center > 0.5:  # Only if beat frequency is meaningful
            beat_bw = min(2.0, beat_center/2)  # Adaptive bandwidth
            beat_low = max(0.1, beat_center - beat_bw)
            beat_high = beat_center + beat_bw
            if beat_high < fs/2:  # Below Nyquist
                beat_name = f"Beat_{beat_low:.1f}-{beat_high:.1f}Hz"
                X_beat = bandpass(X, fs, beat_low, beat_high)
                interference_bands[beat_name] = np.abs(hilbert(X_beat, axis=0)) ** 2
        
        # Sum frequency (harmonics)
        sum_center = np.mean(base_freq) + np.mean(comb_freq)
        if sum_center < fs/2 - 2:  # Below Nyquist with margin
            sum_bw = min(3.0, sum_center/4)
            sum_low = sum_center - sum_bw
            sum_high = sum_center + sum_bw
            sum_name = f"Sum_{sum_low:.1f}-{sum_high:.1f}Hz"
            X_sum = bandpass(X, fs, sum_low, sum_high)
            interference_bands[sum_name] = np.abs(hilbert(X_sum, axis=0)) ** 2
    
    # Apply smoothing to all bands
    window_samples = int(window_len * fs)
    if window_samples > 1:
        kernel = np.ones(window_samples) / window_samples
        for name, power in {**power_bands, **interference_bands}.items():
            power_smooth = np.zeros_like(power)
            for ch in range(power.shape[1]):
                power_smooth[:, ch] = np.convolve(power[:, ch], kernel, mode='same')
            if name in power_bands:
                power_bands[name] = power_smooth
            else:
                interference_bands[name] = power_smooth
    
    return {**power_bands, **interference_bands}

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
                 use_laplacian=False, mode="phase", show_vectors=False,
                 fps=30, grid=256, phase_c=1.0, dpi=150):
        self.fs = float(fs)
        self.mode = mode
        self.show_vectors = show_vectors
        self.grid_n = grid
        self.phase_c = phase_c
        self.fps = int(fps)
        self.hop = max(1, int(self.fs/self.fps))
        self.band = band

        # keep mappable channels
        keep = [i for i,n in enumerate(ch) if n in COORDS]
        self.ch = [ch[i] for i in keep]
        X = detrend(data[:,keep], axis=0, type='constant')
        if notch_hz>0: X = notch(X, self.fs, notch_hz)
        X = bandpass(X, self.fs, band[0], band[1])
        if use_laplacian: X = laplacian(X, self.ch)

        # Store both filtered and original data for different analyses
        self.X_filtered = X
        self.X_original = detrend(data[:,keep], axis=0, type='constant')
        if notch_hz>0: self.X_original = notch(self.X_original, self.fs, notch_hz)
        
        # analytic
        _, self.A, self.P = analytic(X)
        
        # Compute power at the selected frequency band (whatever was specified with --band)
        self.band_power = self.A ** 2  # Power from the already filtered data
        
        # Remove interference computation that's no longer needed
        # The original approach is kept simpler: just power at the chosen band
        
        self.n_samples = self.A.shape[0]
        self.duration = self.n_samples/self.fs

        # amplitude display normalization (robust)
        self.amp_lo = np.percentile(self.A, 5)
        self.amp_hi = np.percentile(self.A, 95)
        
        # power normalization for the selected band
        self.power_lo = np.percentile(self.band_power, 5)
        self.power_hi = np.percentile(self.band_power, 95)

        # grid
        self.Xg, self.Yg, self.mask = make_grid(n=self.grid_n)
        self.k = 2*np.pi*(0.5*(band[0]+band[1]))/max(1e-6, self.phase_c)

        # state
        self.playing = False
        self.index = 0
        self.timer = None
        self.quiv = None

        # figure/UI
        self._build_figure(dpi, band)

    # --- UI ---
    def _build_figure(self, dpi, band):
        self.fig = plt.figure(figsize=(9.5,8.3), dpi=dpi)
        gs = self.fig.add_gridspec(12, 6, left=0.05, right=0.98, top=0.96, bottom=0.09, hspace=0.6, wspace=0.3)
        ax = self.fig.add_subplot(gs[:10, :6])
        self.ax = ax
        ax.set_xticks([]); ax.set_yticks([])
        self.title = ax.set_title(self._title_text(0.0, band), fontsize=12)

        # scalp outline
        scalp = plt.Circle((0,0.15), 1.0, edgecolor=(0.65,0.66,0.74), facecolor='none', lw=1.0)
        ax.add_patch(scalp)

        # images (phase + amplitude overlay)
        self.im_phase = ax.imshow(np.zeros_like(self.Xg), origin='lower',
                                  extent=[self.Xg.min(), self.Xg.max(), self.Yg.min(), self.Yg.max()],
                                  cmap='twilight', vmin=-np.pi, vmax=np.pi, animated=True)
        self.im_amp = ax.imshow(np.zeros_like(self.Xg), origin='lower', alpha=0.65,
                                extent=[self.Xg.min(), self.Xg.max(), self.Yg.min(), self.Yg.max()],
                                cmap='magma', vmin=0, vmax=1, animated=True)

        # electrode dots (to help you orient)
        xs,ys = [],[]
        for nm in self.ch:
            x,y = COORDS[nm]
            xs.append(x); ys.append(y)
        ax.scatter(xs, ys, s=8, c='white', alpha=0.6, linewidths=0, zorder=3)

        # time slider
        axt = self.fig.add_axes([0.08, 0.04, 0.72, 0.03])
        self.s_time = Slider(axt, "Time (s)", 0.0, self.duration, valinit=0.0)
        self.s_time.on_changed(self._on_slider)

        # buttons
        axp = self.fig.add_axes([0.82, 0.035, 0.06, 0.04])
        axb1 = self.fig.add_axes([0.08, 0.008, 0.06, 0.025])
        axb2 = self.fig.add_axes([0.145,0.008, 0.06, 0.025])
        axf1 = self.fig.add_axes([0.725,0.008, 0.06, 0.025])
        axf2 = self.fig.add_axes([0.79, 0.008, 0.06, 0.025])

        self.btn_play = Button(axp, "Play")
        self.btn_back1 = Button(axb1, "⟲ 1s")
        self.btn_back5 = Button(axb2, "⟲ 5s")
        self.btn_fwd1  = Button(axf1, "1s ⟳")
        self.btn_fwd5  = Button(axf2, "5s ⟳")

        self.btn_play.on_clicked(lambda _ : self.toggle_play())
        self.btn_back1.on_clicked(lambda _ : self.seek_seconds(-1))
        self.btn_back5.on_clicked(lambda _ : self.seek_seconds(-5))
        self.btn_fwd1.on_clicked(lambda _ : self.seek_seconds(+1))
        self.btn_fwd5.on_clicked(lambda _ : self.seek_seconds(+5))

        # key bindings
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)

        # timer for playback
        self.timer = self.fig.canvas.new_timer(interval=int(1000/self.fps))
        self.timer.add_callback(self._tick)

        # first draw
        self._draw_index(self.index)

    def _title_text(self, t, band):
        mode_names = {
            "phase": "Phase Interpolation", 
            "huygens": "Huygens Interference", 
            "power": "Band Power"
        }
        mode_display = mode_names.get(self.mode, self.mode)
        return f"EEG Wavefield — {mode_display} | {band[0]}–{band[1]} Hz | t={t:0.2f}s"

    # --- rendering ---
    def _amp_norm(self, a):
        return np.clip((a - self.amp_lo)/(self.amp_hi - self.amp_lo + 1e-9), 0, 1)
    
    def _alpha_norm(self, a):
        return np.clip((a - self.power_lo)/(self.power_hi - self.power_lo + 1e-9), 0, 1)

    def _draw_index(self, idx):
        idx = int(np.clip(idx, 0, self.n_samples-1))
        self.index = idx
        t = self.index/self.fs

        amps = self._amp_norm(self.A[idx,:])
        ph   = self.P[idx,:]
        band_power_norm = self._alpha_norm(self.band_power[idx,:])

        if self.mode == "phase":
            R = griddata([COORDS[n] for n in self.ch], np.cos(ph), (self.Xg, self.Yg), method='cubic')
            I = griddata([COORDS[n] for n in self.ch], np.sin(ph), (self.Xg, self.Yg), method='cubic')
            phase_grid = np.arctan2(I, R)
            amp_grid   = griddata([COORDS[n] for n in self.ch], amps, (self.Xg, self.Yg), method='cubic')
            
        elif self.mode == "huygens":
            F = huygens(self.ch, amps, ph, self.Xg, self.Yg, k=self.k)
            phase_grid = np.angle(F)
            amp_grid   = np.abs(F)
            # robust norm for display
            amp_grid /= (np.nanpercentile(amp_grid[self.mask], 99) + 1e-9)
            
        elif self.mode == "power":
            # For power mode, show power at the selected frequency band
            power_grid = griddata([COORDS[n] for n in self.ch], band_power_norm, (self.Xg, self.Yg), method='cubic')
            # Use power for both phase and amplitude layers, but with different representations
            phase_grid = power_grid  # Show power in the phase layer
            amp_grid = power_grid * 0.5  # Dimmer overlay for depth

        Pm = np.where(self.mask, phase_grid, np.nan)
        Am = np.where(self.mask, amp_grid,   np.nan)

        # Adjust color schemes based on mode
        if self.mode == "power":
            # For power mode, use a more appropriate colormap
            self.im_phase.set_cmap('hot')
            self.im_phase.set_clim(vmin=0, vmax=1)
            self.im_amp.set_alpha(0.3)  # Make overlay more subtle
        else:
            # Reset to default phase visualization
            self.im_phase.set_cmap('twilight')
            self.im_phase.set_clim(vmin=-np.pi, vmax=np.pi)
            self.im_amp.set_alpha(0.65)

        self.im_phase.set_data(Pm)
        self.im_amp.set_data(Am)

        if self.show_vectors:
            if self.quiv is not None: self.quiv.remove()
            gy, gx = np.gradient(Pm)
            mag = np.hypot(gx, gy)
            step = max(1, Pm.shape[0]//20)
            xs = self.Xg[::step,::step]; ys = self.Yg[::step,::step]
            vx = -gx[::step,::step]/(mag[::step,::step]+1e-6)
            vy = -gy[::step,::step]/(mag[::step,::step]+1e-6)
            self.quiv = self.ax.quiver(xs, ys, vx, vy, color=(0.95,0.95,1,0.65), scale=30, width=0.003)

        self.title.set_text(self._title_text(t, self.band))
        # Prevent slider recursion: only set if value differs notably
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
        self.toggle_play() if self.playing and abs(secs)<1e-6 else None
        self._draw_index(self.index + int(secs*self.fs))

    def _on_slider(self, val):
        self._draw_index(int(val*self.fs))

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
            if not self.show_vectors and self.quiv is not None:
                self.quiv.remove(); self.quiv = None
            self._draw_index(self.index)
        elif k == 'm':
            # Cycle through three modes: phase, power, huygens
            modes = ["phase", "power", "huygens"]
            current_idx = modes.index(self.mode) if self.mode in modes else 0
            self.mode = modes[(current_idx + 1) % len(modes)]
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
    p = argparse.ArgumentParser(description="Interactive EEG wavefield viewer with phase, Huygens, and MNE alpha power modes")
    p.add_argument("-f","--file", required=True, help="CSV (header=channels) or EDF")
    p.add_argument("--fs", type=float, default=None, help="Sample rate (CSV only)")
    p.add_argument("--band", nargs=2, type=float, default=[4,12], help="Bandpass Hz (e.g. 4 12)")
    p.add_argument("--notch", type=float, default=0.0, help="Notch frequency (50 or 60; 0=off)")
    p.add_argument("--laplacian", action="store_true", help="Apply surface Laplacian")
    p.add_argument("--mode", choices=["phase","huygens","power"], default="phase")
    p.add_argument("--vectors", action="store_true", help="Show phase-gradient flow vectors")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--grid", type=int, default=256, help="Grid resolution (pixels per side)")
    p.add_argument("--c", type=float, default=1.0, help="Phase-speed scale for Huygens (k=2π f_c / c)")
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
        use_laplacian=args.laplacian,
        mode=args.mode,
        show_vectors=args.vectors,
        fps=args.fps,
        grid=args.grid,
        phase_c=args.c,
        dpi=args.dpi
    )
    
    print("\nControls:")
    print("  Space: Play/Pause")
    print("  M: Cycle through modes (Phase → Power → Huygens)")
    print("  V: Toggle phase gradient vectors")
    print("  S: Save current frame")
    print("  Arrow keys: Navigate time")
    
    viewer.show()

if __name__ == "__main__":
    main()