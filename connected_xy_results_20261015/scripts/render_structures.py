"""Render only the two frozen connected structures. Requires scikit-image."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage.measure import marching_cubes
ROOT=Path(__file__).resolve().parents[1];out=ROOT/'figures';out.mkdir(exist_ok=True)
fig=plt.figure(figsize=(14,6),layout='constrained')
for i,key in enumerate(('X','Y')):
    a=np.load(ROOT/'structures'/f'{key}_binary.npz')['stack'].astype(bool)
    vol=np.pad(a[::-1].transpose(1,2,0).astype(float),1)
    v,f,_,_=marching_cubes(vol,.5,spacing=(.01,.01,.01),step_size=3);v-=.01
    tri=v[f];normal=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);normal/=np.maximum(np.linalg.norm(normal,axis=1,keepdims=True),1e-15)
    light=np.array([-.4,-.5,.75]);light/=np.linalg.norm(light)
    shade=.35+.65*np.maximum(normal@light,0);colors=shade[:,None]*np.array(to_rgb('#ba781e' if key=='X' else '#257fad'))[None]
    ax=fig.add_subplot(1,2,i+1,projection='3d');ax.add_collection3d(Poly3DCollection(tri,linewidths=0,edgecolors='none',facecolors=colors,rasterized=True))
    ax.set(xlim=(0,4),ylim=(0,4),zlim=(0,1),xlabel='x (um)',ylabel='y (um)',zlabel='z (um)',title=f'{key}-pass: connected to substrate, no enclosed voids')
    ax.set_box_aspect((4,4,1));ax.view_init(30,-55)
    sf,axes=plt.subplots(2,3,figsize=(12,7),layout='constrained')
    for j,k in enumerate((0,49,99)):
        axes[0,j].imshow(a[k].T,origin='lower',extent=(0,4,0,4),cmap='gray_r',interpolation='nearest',vmin=0,vmax=1)
        axes[0,j].set(title=f'XY z={(99.5-k)*.01:.3f} um',xlabel='x (um)',ylabel='y (um)')
    for ax,data,title,ext in [(axes[1,0],a[::-1,:,200],'XZ y=2.005 um',(0,4,0,1)),(axes[1,1],a[::-1,200,:],'YZ x=2.005 um',(0,4,0,1)),(axes[1,2],a.sum(0).T*.01,'Total TiO2 thickness (um)',(0,4,0,4))]:
        im=ax.imshow(data,origin='lower',extent=ext,cmap='viridis' if ax==axes[1,2] else 'gray_r',vmin=0,vmax=1,interpolation='nearest');ax.set_title(title)
    sf.colorbar(im,ax=axes[1,2]);sf.suptitle(f'{key}-pass connected structure, original 10 nm grid, black = TiO2')
    sf.savefig(out/f'{key}_sections.png',dpi=140);plt.close(sf)
fig.suptitle('Current connected structures only | 10 nm voxels, 100 layers | display surface sampled at 30 nm')
fig.savefig(out/'connected_structures.png',dpi=150);plt.close(fig)
print('Structure figures complete')
