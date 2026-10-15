"""Evaluate frozen substrate-connected X/Y filters with independent X/Y plane waves."""
from pathlib import Path
import argparse,sys,json,subprocess,hashlib,time,platform
import numpy as np

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def evaluate(args,key,wl):
    import torch
    from types import SimpleNamespace
    root=Path(args.package).resolve()
    engine=Path(args.engine_x if key=='X' else args.engine_y).resolve()
    sys.path.insert(0,str(engine/'src'))
    import run_tio2_xypol_filter_voxel3d as r
    torch.set_num_threads(6)
    r.eigmod.EigConfig.backend='cpu';r.eigmod.EigConfig.cpu_workers=6
    r.eigmod.WarmStart.enabled=False
    config=json.loads((root/'config.json').read_text())
    source=root/'structures'/f'{key}_binary.npz'
    expected=config['structure_sha256'][key]
    assert sha(source)==expected,'Structure hash mismatch'
    cfg=SimpleNamespace(**config['solver'])
    stack,pupil,rays,_=r.build_stack(cfg)
    class Capture(r.FastPupilForward):
        def _wells(self,out,g):
            n=self.n_orders;ox,oy=self.orders
            exy=out*g.out_phase[:,:,None]
            hxy=g.vo.left(exy)
            ez=(g.ky[:,:,None]*hxy[:,:n]-g.kx[:,:,None]*hxy[:,n:])/g.eps_out[:,None,None]
            comp=torch.stack((exy[:,:n],exy[:,n:],ez),1).reshape(out.shape[0],3,2*ox+1,2*oy+1,2)
            self.field=torch.einsum('bxi,bcijp,byj->bcxyp',g.phase_x,comp,g.phase_y).detach().cpu().numpy()[0]
            return super()._wells(out,g)
    a=np.load(source)['stack'];assert a.shape==(100,400,400)
    density=torch.as_tensor(a.astype(float),dtype=torch.float64,device='cuda')
    f=Capture(stack=stack,pupil_spec=pupil,wavelengths_nm=[wl],rays=rays,device=torch.device('cuda'),chunk=1,dtype=torch.complex128,verbose_guard=False)
    f.checkpoint_finish=True
    print(f'START {key} {wl} nm',flush=True);start=time.time()
    with torch.no_grad(): tabs=r.polarized_tabs(f,density[None],tags=('connected_fields',),grad=[False])
    t=tabs[0,:,0,:].sum(0).cpu().numpy();e=f.field
    assert e.shape==(3,128,128,2) and np.isfinite(e).all() and np.isfinite(t).all()
    ref=root/'reference_checks'/f'{key}_{wl}nm.json'
    validation=None
    if ref.exists():
        old=json.loads(ref.read_text());delta=np.abs(t-[old['tx'],old['ty']])
        assert np.allclose(t,[old['tx'],old['ty']],rtol=1e-5,atol=1e-6),(t,old)
        validation={'passed':True,'max_abs_delta':float(delta.max()),'reference_detector_samples':44}
    coords=(np.arange(128)+.5)*cfg.period_um/128
    fields=root/'fields';fields.mkdir(exist_ok=True)
    np.savez_compressed(fields/f'{key}_{wl}nm.npz',E=e,total_E2=(abs(e)**2).sum(0),transmission=t,
        x_um=coords,y_um=coords,wavelength_nm=wl,input_polarizations=np.array(['X','Y']),
        field_components=np.array(['Ex','Ey','Ez']),structure_sha256=expected,
        output_propagation_um=stack.detector_z_um,source_E_amplitude=1.0)
    pass_t=float(t[0 if key=='X' else 1]);block_t=float(t[1 if key=='X' else 0])
    energy=(abs(e)**2).sum(0)
    result=dict(filter=key,wavelength_nm=wl,tx=float(t[0]),ty=float(t[1]),pass_transmission=pass_t,
        block_transmission=block_t,transmission_difference=pass_t-block_t,
        contrast=(pass_t-block_t)/(pass_t+block_t),extinction_ratio_db=10*np.log10(pass_t/block_t),
        insertion_loss_db=-10*np.log10(pass_t),leakage_rejection_db=-10*np.log10(block_t),
        target_met=bool(pass_t>.8 and block_t<.1),structure_sha256=expected,
        field_stats={pol:dict(mean_E2=float(energy[:,:,i].mean()),max_E2=float(energy[:,:,i].max()),
            min_E2=float(energy[:,:,i].min()),spatial_cv=float(energy[:,:,i].std()/energy[:,:,i].mean())) for i,pol in enumerate(('X','Y'))},
        reference_check=validation,elapsed_s=time.time()-start)
    (fields/f'{key}_{wl}nm.json').write_text(json.dumps(result,indent=2))
    print(f'DONE {key} {wl} TX={t[0]:.8f} TY={t[1]:.8f}',flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--package',required=True);p.add_argument('--engine-x',required=True);p.add_argument('--engine-y',required=True)
    p.add_argument('--case',choices=['X','Y']);p.add_argument('--wavelength',type=int)
    args=p.parse_args()
    if args.case:
        assert args.wavelength in range(400,701,50);evaluate(args,args.case,args.wavelength);return
    import torch,importlib.metadata as md
    root=Path(args.package);(root/'fields').mkdir(exist_ok=True)
    environment=dict(python=platform.python_version(),gpu=torch.cuda.get_device_name(),torch=torch.__version__,cuda=torch.version.cuda,
        packages={n:md.version(n) for n in ('numpy','scipy','torcwa','matplotlib')},evaluation_script_sha256=sha(__file__))
    (root/'environment.json').write_text(json.dumps(environment,indent=2))
    for wl in range(400,701,50):
        for key in ('X','Y'):
            subprocess.run([sys.executable,'-X','utf8','-u',__file__,'--package',str(root),'--engine-x',args.engine_x,'--engine-y',args.engine_y,'--case',key,'--wavelength',str(wl)],check=True)
    (root/'COMPLETE.json').write_text(json.dumps(dict(completed=True,wavelengths_nm=list(range(400,701,50)),filters=['X','Y'],independent_input_polarizations=['X','Y'],scope='RCWA order 10, monochromatic plane waves, not FDTD or an experiment'),indent=2))
if __name__=='__main__':main()
