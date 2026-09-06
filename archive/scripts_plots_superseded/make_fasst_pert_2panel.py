"""Side-by-side (a)/(b) panel version of temp_pert.pdf + flds_pert.pdf.

(a) FASST response to a +2 K ocean temperature perturbation
    pert = memo_pop_standalone_gx1v7_diagfw_iceemu_pert2K   (+2 K top-100m IC kick)
    ctrl = memo_pop_standalone_gx1v7_diagfw_iceloop         (FASST production run)
(b) FASST response to a -2 W/m^2 downwelling-LW forcing perturbation
    pert = memo_pop_standalone_gx1v7_diagfw_iceemu_m2_1yr   (--flds_forcing_wm2 -2.0)
    ctrl = memo_pop_standalone_gx1v7_diagfw_iceemu1yr       (emu-aice control)

Data logic is unchanged from make_fasst_pert2K_plot.py / make_fasst_flds_m2_plot.py.
Theme (rcParams, colors, panel-label style, suptitle, annotation style) is
copied exactly from make_fasst_neurips_2panel.py -- including its structural
rule that each axes carries only ONE title object, "(a)"/"(b)" at loc="left"
(the descriptive subject of each panel goes in the suptitle + an in-panel
label instead of a second, competing axes title, which is what caused the
previous version's title collision).
"""
import glob, re
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

# ------------------------------------------------------------- styling ----
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5,
    "axes.titlesize": 9.5, "axes.labelsize": 9.5,
    "legend.fontsize": 8.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.linewidth": 0, "axes.grid": True, "axes.axisbelow": True,
    "grid.color": "#ececec", "grid.linewidth": 0.6,
    "xtick.color": "#6b7177", "ytick.color": "#6b7177",
    "text.color": "#1a1a1a", "axes.labelcolor": "#3d3d3d",
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.spines.left": False, "axes.spines.bottom": False,
    "xtick.major.size": 0, "ytick.major.size": 0,
})
C_SIM, MUTE = "#D2691E", "#6b7177"
LW_SIM = 2.0
ANNOT_FS = 8.0
PANEL_LABEL_FS = 11.0
SUBJECT_FS = 9.5

S = "/glade/derecho/scratch/praggarwal"
OUT = "output/fasst_pub/pert_2panel.pdf"
MON_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def ym(f):
    m = re.search(r"\.(\d{4})-(\d{2})\.nc$", f)
    return int(m.group(1)), int(m.group(2))


def gmsst_from_files(files):
    out = {}
    for f in files:
        y, mo = ym(f)
        ds = xr.open_dataset(f)
        sst = ds["TEMP"].isel(time=0, z_t=0) if "TEMP" in ds else ds["SST"].isel(time=0)
        w = np.nan_to_num(ds["TAREA"].values.astype(np.float64))
        s = sst.values.astype(np.float64)
        ok = np.isfinite(s)
        out[(y, mo)] = float(np.nansum(np.where(ok, s, 0) * w) / (w * ok).sum())
        ds.close()
    return out


def gmsst_series(rundir, years):
    files = sorted(glob.glob(f"{rundir}/*.pop.h.[0-9][0-9][0-9][0-9]-[0-9][0-9].nc"))
    files = [f for f in files if ym(f)[0] in years]
    return gmsst_from_files(files)


# Reference panel proportions (make_fasst_neurips_2panel.py, figsize 7.6x5.0,
# 2 stacked rows, top=0.88/bottom=0.14/left=0.13/right=0.97/hspace=0.32):
#   axes width  = (0.97-0.13)*7.6              = 6.384 in
#   axes height = (0.88-0.14)*5.0 / (2+0.32)   = 1.595 in   (per row, w/ hspace)
#   box_aspect (height/width)                  = 1.595/6.384 = 0.25
# Reproduced here with an EXACT inches-based layout (fig.add_axes, not
# subplots_adjust + set_box_aspect) so each panel gets that identical
# width:height box with no left-over centering whitespace, arranged side by
# side instead of stacked.
REF_BOX_ASPECT = 0.25
LEFT_IN, RIGHT_IN, WSPACE_IN = 0.55, 0.12, 0.38
TOP_IN, BOTTOM_IN = 0.62, 0.42

WFIG = 13.0
panel_w_in = (WFIG - LEFT_IN - RIGHT_IN - WSPACE_IN) / 2.0
panel_h_in = panel_w_in * REF_BOX_ASPECT
HFIG = TOP_IN + panel_h_in + BOTTOM_IN

fig = plt.figure(figsize=(WFIG, HFIG))
left_a = LEFT_IN / WFIG
w_frac = panel_w_in / WFIG
left_b = (LEFT_IN + panel_w_in + WSPACE_IN) / WFIG
bottom = BOTTOM_IN / HFIG
h_frac = panel_h_in / HFIG
axa = fig.add_axes([left_a, bottom, w_frac, h_frac])
axb = fig.add_axes([left_b, bottom, w_frac, h_frac])
axes_top_frac = bottom + h_frac
subject_y = axes_top_frac + 0.14 / HFIG   # ~0.14in gap above the axes box
suptitle_y = axes_top_frac + 0.46 / HFIG  # ~0.46in gap above the axes box
fig.suptitle("FASST Response to Perturbation Experiments",
             fontsize=13, fontweight="bold", y=suptitle_y)

# ============================================================ panel (a) ====
PERT_A = f"{S}/memo_pop_standalone_gx1v7_diagfw_iceemu_pert2K/run"
CTRL_A = f"{S}/memo_pop_standalone_gx1v7_diagfw_iceloop/run"
YEARS_A = range(1980, 1985)

ctrl_a = gmsst_series(CTRL_A, YEARS_A)
pert_a = gmsst_series(PERT_A, YEARS_A)

keys_a = sorted(k for k in pert_a if k in ctrl_a)
missing_expected = [(y, m) for y in YEARS_A for m in range(1, 13)]
missing = [k for k in missing_expected if k not in dict.fromkeys(keys_a)]
if missing:
    print(f"(a) NOTE: {len(missing)} months missing from the overlap: {missing[:5]}...")

anom_a = np.array([pert_a[k] - ctrl_a[k] for k in keys_a])
y0 = keys_a[0][0]
x_a = np.array([(y - y0) + (m - 0.5) / 12.0 for (y, m) in keys_a])

ann = {}
for y in YEARS_A:
    vals = [pert_a[(y, m)] - ctrl_a[(y, m)] for m in range(1, 13) if (y, m) in pert_a and (y, m) in ctrl_a]
    if len(vals) == 12:
        ann[y] = np.mean(vals)
ann_years = sorted(ann)
print(f"(a) {'year':>6} {'model_yr':>9} {'anom(K)':>9}")
for i, y in enumerate(ann_years):
    print(f"(a) {y:>6} {i:>9} {ann[y]:>+9.3f}")
if len(ann_years) >= 2:
    a0, a1 = ann[ann_years[0]], ann[ann_years[-1]]
    print(f"(a) decay: {a0:+.2f} -> {a1:+.2f} K "
          f"({100*(1 - a1/a0):.0f}% reduction, years 0-{len(ann_years)-1})")

axa.set_title("(a)", loc="left", fontweight="bold", fontsize=PANEL_LABEL_FS)
axa.axhline(0, color="#c7c7c7", lw=0.8, zorder=0)
axa.plot(x_a, anom_a, color=C_SIM, lw=LW_SIM, solid_capstyle="round",
         marker="o", ms=2.4, mfc=C_SIM, mec="none", zorder=3)
axa.set_ylabel(r"$\Delta$SST (K)")
axa.set_xlabel("Model year")
axa.set_xlim(x_a.min(), x_a.max())
axa.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1))

if len(ann_years) >= 2:
    a0, a1 = ann[ann_years[0]], ann[ann_years[-1]]
    axa.text(0.97, 0.94,
             f"{a0:+.2f} K (yr 0) to {a1:+.2f} K (yr {len(ann_years)-1})\n"
             f"{100*(1 - a1/a0):.0f}% reduction",
             transform=axa.transAxes, fontsize=ANNOT_FS, color=MUTE, ha="right", va="top")

# ============================================================ panel (b) ====
PERT_B = f"{S}/memo_pop_standalone_gx1v7_diagfw_iceemu_m2_1yr/run"
CTRL_B = f"{S}/memo_pop_standalone_gx1v7_diagfw_iceemu1yr/run"
YEAR_B = 1980

ctrl_b_full = gmsst_series(CTRL_B, [YEAR_B])
pert_b_full = gmsst_series(PERT_B, [YEAR_B])
ctrl_b = {mo: v for (y, mo), v in ctrl_b_full.items()}
pert_b = {mo: v for (y, mo), v in pert_b_full.items()}

months = [m for m in range(1, 13) if m in ctrl_b and m in pert_b]
if len(months) != 12:
    missing = sorted(set(range(1, 13)) - set(months))
    raise SystemExit(f"(b) missing months: {missing}")

anom_b = np.array([pert_b[m] - ctrl_b[m] for m in months])
peak_i = int(np.argmin(anom_b))

print(f"(b) {'mo':>4} {'ctrl':>10} {'pert':>10} {'anom(K)':>10}")
for m, a in zip(months, anom_b):
    print(f"(b) {MON_LABELS[m-1]:>4} {ctrl_b[m]:10.4f} {pert_b[m]:10.4f} {a:10.4f}")
print(f"(b) peak: {MON_LABELS[peak_i]} {anom_b[peak_i]:+.3f} K; year-end (Dec): {anom_b[-1]:+.3f} K")

x_b = np.array([1.0 + (m - 0.5) / 12.0 for m in months])

axb.set_title("(b)", loc="left", fontweight="bold", fontsize=PANEL_LABEL_FS)
axb.axhline(0, color="#c7c7c7", lw=0.8, zorder=0)
axb.plot(x_b, anom_b, color=C_SIM, lw=LW_SIM, solid_capstyle="round",
         marker="o", ms=3.0, mfc=C_SIM, mec="none", zorder=3)
axb.set_ylabel(r"$\Delta$SST (K)")
axb.set_xlabel("Model year")
axb.set_xlim(1.0, 2.0)
axb.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1 / 6))
axb.set_xticks(1.0 + np.arange(12) / 12.0 + (0.5 / 12.0))
axb.set_xticklabels(MON_LABELS)

pct_dec_dec = (ctrl_b[months[-1]] - pert_b[months[-1]]) / ctrl_b[months[-1]] * 100.0
axb.text(0.97, 0.94,
         f"{anom_b[0]:+.2f} K ({MON_LABELS[0]}) to {anom_b[-1]:+.2f} K ({MON_LABELS[-1]})\n"
         f"peak {anom_b[peak_i]:+.2f} K ({MON_LABELS[peak_i]})\n"
         f"{pct_dec_dec:.2f}% SST decrease by year-end",
         transform=axb.transAxes, fontsize=ANNOT_FS, color=MUTE, ha="right", va="top")

# per-panel subject line, placed in FIGURE coordinates between the suptitle
# and the "(a)"/"(b)" axes titles -- kept off the axes entirely so it never
# competes with the single per-axes title (the collision in the previous
# version) or overlaps the plotted data near the top of panel (a)
fig.canvas.draw()
subjects = {"a": "+2 K Ocean Temperature Perturbation",
            "b": r"$\mathbf{-2}$ W m$^{\mathbf{-2}}$ Downwelling-LW Forcing"}
for ax, key in ((axa, "a"), (axb, "b")):
    bbox = ax.get_position()
    fig.text((bbox.x0 + bbox.x1) / 2.0, subject_y, subjects[key],
              fontsize=SUBJECT_FS, fontweight="bold", color="#3d3d3d",
              ha="center", va="bottom")

fig.savefig(OUT)
fig.savefig(OUT.replace(".pdf", ".png"), dpi=200)
print(f"\nwrote {OUT}")
