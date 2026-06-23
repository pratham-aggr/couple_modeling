#!/usr/bin/env python
"""Hardening checks for the prob (p5ge0uvh) vs point (vfcg1m61) comparison.

CHECK 2 : CRPS estimator = unbiased/fair (Gneiting); member-size sweep 20/40/60 on heat fluxes.
CHECK 3 : point model fully deterministic; ensemble-mean stability 20 vs 40 members.
CHECK 4 : regional breakdown at SH convective failure cells (lat<-55, lon 200-220E ~140-160W).
CHECK 5 : FSDS_J calibration failure characterization (truth distribution vs symmetric spread).
"""
import sys, json
import numpy as np
import torch
sys.path.insert(0, "/glade/u/home/praggarwal/couple")
from train_unet import UNet, Normalizer, enable_mc_dropout

CACHE     = "/glade/work/praggarwal/couple_cache_mem24h"
ROOT      = "/glade/u/home/praggarwal/couple"
DIR_PROB  = f"{ROOT}/output/output_unet_mem24h_dsst_temporal_radprecip_ucast"
DIR_POINT = f"{ROOT}/output/output_unet_mem24h_dsst_temporal_radprecip"
VARS      = ["TAUX","TAUY","SHFLX","LHFLX","QFLX","FSDS_J","FLDS_J","PRECT"]
HEAT_IDX  = {"SHFLX":2,"LHFLX":3,"FLDS_J":6}
N_OUT     = 8
MMAX      = 60
BATCH     = 8
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"

LAT = np.linspace(-90, 90, 192)
LON = np.linspace(0, 358.75, 288)

def load_model(d, dropout):
    m = UNet(n_in=6, n_out=N_OUT, base=64, dropout=dropout).to(DEVICE)
    sd = torch.load(f"{d}/best_model.pt", map_location=DEVICE, weights_only=True)
    m.load_state_dict(sd); m.eval(); return m

def build_inputs(idx):
    X=np.load(f"{CACHE}/X.npy",mmap_mode="r"); Y=np.load(f"{CACHE}/Y.npy",mmap_mode="r")
    Yr=np.load(f"{CACHE}/Y_rad.npy",mmap_mode="r"); Yp=np.load(f"{CACHE}/Y_precip.npy",mmap_mode="r")
    M=np.load(f"{CACHE}/mask.npy",mmap_mode="r")
    xs,ys_,mk=[],[],[]
    for i in idx:
        xi=X[i][[0,1,2,3,4]]; dsst=((X[i][0]-X[i][3])/86400.0)[None]
        xs.append(np.concatenate([xi,dsst],0).astype(np.float32))
        ys_.append(np.concatenate([Y[i],Yr[i],Yp[i]],0).astype(np.float32))
        mk.append((M[i]>0.5).astype(np.float32))
    return np.stack(xs),np.stack(ys_),np.stack(mk)

def fair_crps(ens, y, M):
    """Unbiased CRPS for first M members. ens:(Mmax,B,C,H,W) y:(B,C,H,W) -> (B,C,H,W)."""
    e = ens[:M]
    skill = (e - y.unsqueeze(0)).abs().mean(0)
    es,_ = torch.sort(e, dim=0)
    k = torch.arange(1, M+1, device=e.device).view(-1,1,1,1,1).float()
    pair = (2*(2*k - M - 1) * es).sum(0)                 # = sum_ij|xi-xj|
    return skill - pair/(2.0*M*(M-1))

def main():
    print(f"device={DEVICE}")
    idx = np.load(f"{DIR_PROB}/test_indices.npy")
    np_, npnt = Normalizer.load(f"{DIR_PROB}/normalizer.npz"), Normalizer.load(f"{DIR_POINT}/normalizer.npz")
    mprob = load_model(DIR_PROB, 0.1); mpoint = load_model(DIR_POINT, 0.0)

    # CHECK 3a: confirm point model has no active stochastic layers
    stoch = [type(m).__name__ for m in mpoint.modules() if isinstance(m,(torch.nn.Dropout,torch.nn.Dropout2d)) and m.training]
    drop_layers = [type(m).__name__ for m in mpoint.modules() if isinstance(m,(torch.nn.Dropout,torch.nn.Dropout2d))]
    print(f"[CHECK3] point model: training={mpoint.training}, active-dropout-layers={stoch}, dropout-modules={drop_layers}")

    X,Yt,Mk = build_inputs(idx)
    def nrm(a,n): return (a-n.x_mean[None,:,None,None])/(n.x_std[None,:,None,None]+1e-8)
    Xp=torch.tensor(nrm(X,np_),dtype=torch.float32); Xq=torch.tensor(nrm(X,npnt),dtype=torch.float32)
    ysp=torch.tensor(np_.y_std,device=DEVICE).view(1,-1,1,1); ymp=torch.tensor(np_.y_mean,device=DEVICE).view(1,-1,1,1)
    ysq=torch.tensor(npnt.y_std,device=DEVICE).view(1,-1,1,1); ymq=torch.tensor(npnt.y_mean,device=DEVICE).view(1,-1,1,1)
    Y=torch.tensor(Yt,device=DEVICE)

    # region mask (per cell, broadcast to all samples)
    latg,long_ = np.meshgrid(LAT, LON, indexing='ij')         # (H,W)
    region = ((latg < -55) & (long_>=200) & (long_<=220)).astype(np.float32)
    Rg = torch.tensor(region, device=DEVICE)                  # (H,W)
    print(f"[CHECK4] region cells (grid): {int(region.sum())}  (lat<-55 & lon 200-220E)")

    N=len(idx); Msizes=[20,40,60]
    # accumulators
    crps_heat={m:{v:0.0 for v in HEAT_IDX} for m in Msizes}; wheat=0.0
    # ensemble-mean stability
    meanshift_num=np.zeros(N_OUT); meanshift_den=np.zeros(N_OUT)
    # global + regional full stats
    acc={r:{k:np.zeros(N_OUT) for k in ['crps_p','crps_q','se_p','se_q','ae_p','ae_q','cov90','cov50','w']} for r in ['glob','reg']}
    for r in acc: acc[r]['w']=0.0
    # FSDS truth + spread distributions (ocean cells, subsample)
    fsds_truth=[]; fsds_spread=[]; fsds_zerofrac_num=0.0; fsds_zerofrac_den=0.0
    region_sample=None

    with torch.no_grad():
        for b0 in range(0,N,BATCH):
            b1=min(b0+BATCH,N)
            xbp=Xp[b0:b1].to(DEVICE); xbq=Xq[b0:b1].to(DEVICE); y=Y[b0:b1]; mk=Mk[b0:b1]
            mkb=torch.tensor(mk,device=DEVICE).unsqueeze(1)         # (B,1,H,W)
            enable_mc_dropout(mprob)
            ens=torch.stack([mprob(xbp)*ysp+ymp for _ in range(MMAX)],0)  # (60,B,8,H,W)
            mprob.eval()
            pt=mpoint(xbq)*ysq+ymq

            wB=mkb.sum().item()
            def ms(t): return (t*mkb).sum(dim=(0,2,3)).cpu().numpy()

            # CHECK2 heat-flux CRPS at member sizes
            wheat+=wB
            for M in Msizes:
                c=fair_crps(ens,y,M)
                for v,vi in HEAT_IDX.items():
                    crps_heat[M][v]+=float((c[:,vi:vi+1]*mkb).sum())
            # CHECK3b ensemble-mean stability 20 vs 40
            m20=ens[:20].mean(0); m40=ens[:40].mean(0)
            meanshift_num+=ms((m20-m40).abs()); meanshift_den+=ms(m40.abs())

            # full stats global + regional, using 60-member ensemble
            crps60=fair_crps(ens,y,60); mean60=ens.mean(0)
            crps_q=(pt-y).abs()
            q05=torch.quantile(ens,0.05,0); q95=torch.quantile(ens,0.95,0)
            q25=torch.quantile(ens,0.25,0); q75=torch.quantile(ens,0.75,0)
            in90=((y>=q05)&(y<=q95)).float(); in50=((y>=q25)&(y<=q75)).float()
            for r,rmask in [('glob',mkb),('reg',mkb*Rg.view(1,1,*Rg.shape))]:
                def msr(t): return (t*rmask).sum(dim=(0,2,3)).cpu().numpy()
                acc[r]['crps_p']+=msr(crps60); acc[r]['crps_q']+=msr(crps_q)
                acc[r]['se_p']+=msr((mean60-y)**2); acc[r]['se_q']+=msr((pt-y)**2)
                acc[r]['ae_p']+=msr((mean60-y).abs()); acc[r]['ae_q']+=msr((pt-y).abs())
                acc[r]['cov90']+=msr(in90); acc[r]['cov50']+=msr(in50)
                acc[r]['w']+=float(rmask.sum())

            # FSDS distribution (var idx5): truth + std over ocean cells (subsample 1/200)
            fi=5; stdf=ens[:,:,fi].std(0)                  # (B,H,W)
            mb=(mkb.squeeze(1)>0.5)
            tv=y[:,fi][mb].cpu().numpy(); sv=stdf[mb].cpu().numpy()
            sel=slice(None,None,200)
            fsds_truth.append(tv[sel]); fsds_spread.append(sv[sel])
            fsds_zerofrac_num+=float((tv<1e3).sum()); fsds_zerofrac_den+=float(tv.size)

            # grab one regional ocean cell (SHFLX) for hand-check
            if region_sample is None:
                rm=(mkb.squeeze(1)*Rg.view(1,*Rg.shape))>0.5    # (B,H,W)
                nz=torch.nonzero(rm)
                if len(nz):
                    bb,yy,xx=nz[len(nz)//2].tolist()
                    region_sample={"var":"SHFLX","global_row":int(idx[b0+bb]),"cell_yx":[yy,xx],
                        "lat":float(LAT[yy]),"lon":float(LON[xx]),
                        "truth":float(y[bb,2,yy,xx]),"point_pred":float(pt[bb,2,yy,xx]),
                        "ens_mean":float(ens[:,bb,2,yy,xx].mean()),"ens_std":float(ens[:,bb,2,yy,xx].std()),
                        "ens60":[float(ens[k,bb,2,yy,xx]) for k in range(60)]}
            if b0%(BATCH*40)==0: print(f"  {b1}/{N}",flush=True)

    out={}
    print("\n"+"="*70+"\nCHECK 2 — CRPS estimator + member-size sweep (heat fluxes, global)\n"+"="*70)
    print("Estimator: UNBIASED/FAIR (Gneiting):  mean_i|x_i-y| - 1/(2M(M-1)) sum_ij|x_i-x_j|")
    print("Point model CRPS == MAE (degenerate spike).")
    out['crps_sweep']={}
    for v,vi in HEAT_IDX.items():
        cq=acc['glob']['crps_q'][vi]/acc['glob']['w']
        print(f"\n[{v}]  point CRPS(=MAE)={cq:.5g}")
        out['crps_sweep'][v]={'point':cq}
        for M in Msizes:
            cp=crps_heat[M][v]/wheat; adv=100*(1-cp/cq)
            print(f"   M={M:2d}: prob CRPS={cp:.5g}   advantage vs point={adv:5.1f}%")
            out['crps_sweep'][v][f'M{M}']=cp
    print("\n"+"="*70+"\nCHECK 3 — ensemble-mean stability (20 vs 40 members)\n"+"="*70)
    rel=meanshift_num/(meanshift_den+1e-30)
    for i,v in enumerate(VARS): print(f"  {v:8s}: |mean20-mean40|/|mean40| = {100*rel[i]:.3f}%")
    out['meanshift_pct']=(100*rel).tolist()

    print("\n"+"="*70+"\nCHECK 4 — regional (failure cells) vs global\n"+"="*70)
    out['regional']={}
    for r in ['glob','reg']:
        w=acc[r]['w']
        print(f"\n--- {r}  (ocean-cell-samples={w:.0f}) ---")
        print(f"{'var':8s}{'CRPSprob':>12s}{'CRPSpt':>12s}{'advcrps%':>9s}{'RMSEp':>11s}{'RMSEq':>11s}{'cov90':>7s}{'cov50':>7s}")
        out['regional'][r]={}
        for i,v in enumerate(VARS):
            cp=acc[r]['crps_p'][i]/w; cq=acc[r]['crps_q'][i]/w
            rp=np.sqrt(acc[r]['se_p'][i]/w); rq=np.sqrt(acc[r]['se_q'][i]/w)
            c90=100*acc[r]['cov90'][i]/w; c50=100*acc[r]['cov50'][i]/w
            print(f"{v:8s}{cp:12.4g}{cq:12.4g}{100*(1-cp/cq):9.1f}{rp:11.4g}{rq:11.4g}{c90:7.1f}{c50:7.1f}")
            out['regional'][r][v]={'crps_p':cp,'crps_q':cq,'rmse_p':rp,'rmse_q':rq,'cov90':c90,'cov50':c50}

    print("\n"+"="*70+"\nCHECK 5 — FSDS_J calibration failure characterization\n"+"="*70)
    ft=np.concatenate(fsds_truth); fsp=np.concatenate(fsds_spread)
    zf=fsds_zerofrac_num/fsds_zerofrac_den
    print(f"FSDS_J truth (ocean): n={ft.size}  min={ft.min():.4g} max={ft.max():.4g} mean={ft.mean():.4g}")
    print(f"  fraction near-zero (<1e3, i.e. night/no-sun): {100*zf:.1f}%")
    print(f"  truth percentiles 1/10/50/90/99: "+"/".join(f"{np.percentile(ft,p):.4g}" for p in [1,10,50,90,99]))
    print(f"  skewness(truth) = {((ft-ft.mean())**3).mean()/(ft.std()**3+1e-30):.3f}  (0=symmetric)")
    print(f"  ensemble spread is SYMMETRIC (gaussian-ish dropout) but truth bounded at 0 -> mismatch")
    print(f"  mean spread where truth<1e3 (night): {fsp[ft<1e3].mean():.4g}   where truth>0: {fsp[ft>=1e3].mean():.4g}")
    out['fsds']={'zero_frac':zf,'skew':float(((ft-ft.mean())**3).mean()/(ft.std()**3+1e-30)),
                 'min':float(ft.min()),'max':float(ft.max()),'mean':float(ft.mean())}

    print("\n"+"="*70+"\nRegional sample cell (hand-check)\n"+"="*70)
    rs=region_sample
    if rs:
        s=np.array(rs['ens60'])
        print(f"  {rs['var']} @ lat={rs['lat']:.1f} lon={rs['lon']:.1f} (row {rs['global_row']}, cell {rs['cell_yx']})")
        print(f"  truth={rs['truth']:.5g}  point={rs['point_pred']:.5g}  ens_mean={rs['ens_mean']:.5g}  ens_std={rs['ens_std']:.5g}")
        print(f"  ens60 min/5/50/95/max = {s.min():.5g}/{np.percentile(s,5):.5g}/{np.median(s):.5g}/{np.percentile(s,95):.5g}/{s.max():.5g}")
        print(f"  truth-in-90%-band: {np.percentile(s,5)<=rs['truth']<=np.percentile(s,95)}")
        print(f"  all 60 samples: {np.round(s,1).tolist()}")
        out['region_sample']=rs

    with open(f"{ROOT}/output/verify_prob_vs_point.json","w") as f: json.dump(out,f,indent=2)
    print(f"\nwrote {ROOT}/output/verify_prob_vs_point.json")

if __name__=="__main__": main()
