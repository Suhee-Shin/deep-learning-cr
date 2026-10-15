"""Build offline HTML, Korean GitHub README, metrics CSV and scientific figures."""
from pathlib import Path
import json,csv,hashlib,html
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
plt.rcParams.update({'font.size':10,'axes.unicode_minus':False})

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def main():
    figures=ROOT/'figures';figures.mkdir(exist_ok=True)
    metrics=[];fieldstats=[]
    for wl in range(400,701,50):
        cases=[]
        for key in ('X','Y'):
            f=ROOT/'fields'/f'{key}_{wl}nm.npz';j=f.with_suffix('.json')
            if not f.exists() or not j.exists():continue
            d=read(j);a=np.load(f);cases.append((key,a));metrics.append(d)
            for pol,stats in d['field_stats'].items():fieldstats.append(dict(filter=key,wavelength_nm=wl,input_polarization=pol,**stats))
            e=a['E']
            for mode in ('real','amplitude','phase'):
                dest=figures/f'{key}_{wl}nm_{mode}.png'
                if dest.exists():continue
                v={'real':e.real,'amplitude':abs(e),'phase':np.angle(e)}[mode]
                if mode=='phase':v=np.where(abs(e)>max(abs(e).max()*1e-3,1e-12),v,np.nan)
                limit=float(np.nanmax(abs(v))) if mode!='phase' else np.pi
                fig,axes=plt.subplots(3,2,figsize=(9,10),layout='constrained')
                for c in range(3):
                    for q in range(2):
                        ax=axes[c,q];im=ax.imshow(v[c,:,:,q].T,origin='lower',extent=(0,4,0,4),interpolation='nearest',cmap='viridis' if mode=='amplitude' else ('twilight' if mode=='phase' else 'RdBu_r'),vmin=0 if mode=='amplitude' else -limit,vmax=limit)
                        ax.set(title=f'{("Ex","Ey","Ez")[c]}, {("X","Y")[q]} input',xlabel='x (um)',ylabel='y (um)')
                fig.colorbar(im,ax=axes.ravel().tolist(),shrink=.8,label='radians' if mode=='phase' else 'normalized E amplitude')
                fig.suptitle(f'{key}-pass connected structure, {wl} nm: {mode}\nMonitor: 1.35 um after exit in SiO2; incident E amplitude = 1')
                fig.savefig(dest,dpi=125);plt.close(fig)
        dest=figures/f'{wl}nm_intensity.png'
        if len(cases)==2 and not dest.exists():
            fig,axes=plt.subplots(2,2,figsize=(10,9),layout='constrained');vmax=max(float(a['total_E2'].max()) for _,a in cases)
            for row,(key,a) in enumerate(cases):
                for q in range(2):
                    ax=axes[row,q];im=ax.imshow(a['total_E2'][:,:,q].T,origin='lower',extent=(0,4,0,4),vmin=0,vmax=vmax,cmap='magma',interpolation='nearest')
                    ax.set(title=f'{key}-pass, {("X","Y")[q]} input, T={100*a["transmission"][q]:.3f}%',xlabel='x (um)',ylabel='y (um)')
            fig.colorbar(im,ax=axes.ravel().tolist(),shrink=.85,label='normalized |E|^2 (not W/m^2)')
            fig.suptitle(f'{wl} nm, substrate-connected structures\n1.35 um after exit in SiO2; common color scale for all four panels')
            fig.savefig(dest,dpi=140);plt.close(fig)
    if not metrics:return
    metrics.sort(key=lambda d:(d['filter'],d['wavelength_nm']))
    columns=['filter','wavelength_nm','tx','ty','pass_transmission','block_transmission','transmission_difference','contrast','extinction_ratio_db','insertion_loss_db','leakage_rejection_db','target_met','structure_sha256']
    for name,rows,keys in [('metrics.csv',metrics,columns),('field_statistics.csv',fieldstats,list(fieldstats[0]))]:
        with (ROOT/name).open('w',newline='',encoding='utf-8-sig') as f:
            writer=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    (ROOT/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    aggregate=[]
    for key in ('X','Y'):
        rows=[d for d in metrics if d['filter']==key]
        if rows:aggregate.append(dict(filter=key,sample_count=len(rows),min_pass=min(d['pass_transmission'] for d in rows),max_block=max(d['block_transmission'] for d in rows),mean_pass=float(np.mean([d['pass_transmission'] for d in rows])),mean_block=float(np.mean([d['block_transmission'] for d in rows])),min_er_db=min(d['extinction_ratio_db'] for d in rows),max_il_db=max(d['insertion_loss_db'] for d in rows),all_samples_meet_target=all(d['target_met'] for d in rows)))
    (ROOT/'summary.json').write_text(json.dumps(aggregate,indent=2))
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for key,color in [('X','#ba781e'),('Y','#257fad')]:
        rows=[d for d in metrics if d['filter']==key];x=[d['wavelength_nm'] for d in rows]
        axes[0,0].plot(x,[100*d['pass_transmission'] for d in rows],'o-',c=color,label=key+' pass')
        axes[0,0].plot(x,[100*d['block_transmission'] for d in rows],'s--',c=color,label=key+' block')
        for ax,name in [(axes[0,1],'extinction_ratio_db'),(axes[1,0],'insertion_loss_db'),(axes[1,1],'contrast')]:ax.plot(x,[d[name] for d in rows],'o-',c=color,label=key+'-pass')
    for ax,title in zip(axes.flat,['Transmittance (%)','Extinction ratio (dB)','Insertion loss (dB)','Polarization contrast']):ax.set(xlabel='Wavelength (nm)',ylabel=title);ax.legend();ax.grid(alpha=.2)
    fig.suptitle('Connected / open-void structures only; RCWA order 10; discrete wavelength samples')
    fig.savefig(figures/'metrics.png',dpi=150);plt.close(fig)
    lines=['# 기판 연결 X-pass / Y-pass 최신 구조와 광학 결과','',
        '> 대상은 연결성·열린 공극 처리를 완료한 두 구조뿐입니다. 이전 원본과 100 nm 공정 변환 후보의 성능·필드는 포함하지 않습니다.',
        '',f'평가 완료: **{len(metrics)}/14개 구조·파장 조합**, 각 조합마다 X·Y 선형 편광을 독립 입사시켰습니다.',
        '', '## 구조와 계산 조건','',
        '- 구조: 4 × 4 × 1 µm, 10 nm 정육면체 voxel, 100층, TiO₂/공기 및 SiO₂ 기판.',
        '- X 구조는 원본 best step 222, Y 구조는 step 274에서 연결성 조건을 적용한 고정 스냅샷입니다.',
        '- 두 구조 모두 고체 연결 성분 1개, 기판 연결, 밀폐 공극 0개입니다. 면 접촉(6-neighbor) 기준이며, 모든 공극은 윗면으로 연결됩니다.',
        '- 400, 450, 500, 550, 600, 650, 700 nm의 **이산 파장**을 계산했습니다. 연속 스펙트럼 검증은 아닙니다.',
        '- 실제 계산 입력: 수직 입사 단색 평면파, 입사 전기장 진폭 1, X 또는 Y 선형 편광. 두 편광은 동시 입사가 아닌 독립 계산입니다.',
        '- GPU RCWA order 10, complex128입니다. FDTD 시간영역 계산이나 실험 측정이 아닙니다.',
        '- 모니터: 필터 출구 이후 **SiO₂ 안으로 1.35 µm**, 4 × 4 µm 면의 128 × 128점.',
        '- `detector_gap_um=0.35`가 코드에서 총 높이 1 µm와 합쳐져 전파 거리 1.35 µm로 사용됩니다. 350 nm 관측면으로 해석하면 안 됩니다.',
        '- 형상 연결성은 검증했으나 최소 배출 통로 폭, 실제 현상 속도, 수축·기계 안정성과 차수 수렴은 미검증입니다.',
        '', '## 구조 파일','',
        '| 필터 | 원본 배열 | 연결성 검사 |','|---|---|---|',
        '| X-pass | [X_binary.npz](structures/X_binary.npz) | [X_topology.json](structures/X_topology.json) |',
        '| Y-pass | [Y_binary.npz](structures/Y_binary.npz) | [Y_topology.json](structures/Y_topology.json) |',
        '', '배열 순서는 `[상부→기판 z, x, y]`, 크기는 `(100,400,400)`입니다. 1은 TiO₂, 0은 공기입니다.',
        '', '![최신 연결 구조](figures/connected_structures.png)',
        '', '[X 원본 격자 단면](figures/X_sections.png) · [Y 원본 격자 단면](figures/Y_sections.png)',
        '', '## 성능 지표','',
        '[전체 성능 CSV](metrics.csv) · [전체 성능 JSON](metrics.json) · [집계 JSON](summary.json) · [필드 통계 CSV](field_statistics.csv)',
        '', '| 필터 | λ (nm) | TX (%) | TY (%) | 소광비 (dB) | 삽입손실 (dB) | 대비도 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for d in metrics:lines.append(f'| {d["filter"]} | {d["wavelength_nm"]} | {100*d["tx"]:.4f} | {100*d["ty"]:.4f} | {d["extinction_ratio_db"]:.4f} | {d["insertion_loss_db"]:.4f} | {d["contrast"]:.4f} |')
    lines+=['','### 전체 평가 파장 중 최악 조건','',
        '| 필터 | 최소 통과율 (%) | 최대 억제 편광 누설 (%) | 최소 소광비 (dB) |','|---|---:|---:|---:|']
    for d in aggregate:lines.append(f'| {d["filter"]} | {100*d["min_pass"]:.4f} | {100*d["max_block"]:.4f} | {d["min_er_db"]:.4f} |')
    lines+=['','**450/550/650 nm 중간 파장에서 선택성이 크게 약해질 수 있습니다. 400/500/600/700 nm만으로 대역 전체 성능을 판단하면 안 됩니다. 현재 두 필터 모두 전 평가 파장의 목표를 달성하지 못했습니다.**',
        '','![파장별 성능 지표](figures/metrics.png)','',
        '정의: X-pass의 Tpass=TX, Tblock=TY이며 Y-pass는 반대입니다. 소광비=10log₁₀(Tpass/Tblock), 삽입손실=−10log₁₀(Tpass), 누설 억제량=−10log₁₀(Tblock), 대비도=(Tpass−Tblock)/(Tpass+Tblock). **TX/TY는 입사 편광별 전체 투과율**이며 출력 분석기를 통과한 특정 편광 성분의 파워가 아닙니다.',
        '', '목표는 Tpass>80%, Tblock<10%입니다. 목표 충족 여부를 각 파장 JSON에 저장합니다. 평균값이 대역 전체의 성능을 보장하지 않습니다.',
        '', '## 파장별 모니터 필드','',
        '|E|²는 정규화 전기장 세기이며 W/m² 또는 전력 투과율이 아닙니다. 각 파장의 네 패널은 같은 색상 범위를 사용하고, 다른 파장 사이에는 색상 최대값이 달라집니다. 위상 그림은 해당 복소 E의 진폭이 전체 최대의 0.1% 이하인 픽셀을 숨깁니다.',
        '', 'NPZ의 `E`는 complex128, 배열 `(Ex/Ey/Ez, x, y, 입사 X/Y)`입니다. 크기 `(3,128,128,2)`. `total_E2`, `transmission`, 좌표, 파장, 관측면과 구조 SHA256도 저장합니다. Re(E)는 phasor의 실수부이며 시간 애니메이션이 아닙니다.']
    for wl in range(400,701,50):
        if not (figures/f'{wl}nm_intensity.png').exists():continue
        lines += ['',f'### {wl} nm','',f'![{wl} nm 정규화 전기장 세기](figures/{wl}nm_intensity.png)','',
            '| 필터 | 복소 원본 데이터 | Re(E) | 성분별 진폭 | 성분별 위상 |','|---|---|---|---|---|']
        for key in ('X','Y'):lines.append(f'| {key} | [NPZ](fields/{key}_{wl}nm.npz) / [지표](fields/{key}_{wl}nm.json) | [Ex/Ey/Ez](figures/{key}_{wl}nm_real.png) | [진폭](figures/{key}_{wl}nm_amplitude.png) | [위상](figures/{key}_{wl}nm_phase.png) |')
    lines+=['','## 재현 및 파일 검증','',
        '1. 이 폴더를 내려받고 `index.html`을 브라우저로 열면 로컬에서 결과를 볼 수 있습니다. GitHub에서는 이 README의 그림과 표를 바로 볼 수 있습니다.',
        '2. NumPy/SciPy가 있는 Python에서 `python scripts/verify.py`로 파일 해시, 배열과 연결성, 14개 계산 결과를 검사합니다.',
        '3. GPU 재계산은 [REPRODUCE.md](REPRODUCE.md)를 따릅니다. 다른 사용자의 GPU 작업과 메모리를 먼저 확인하고 순차 실행하세요.',
        '4. `python scripts/report.py`는 저장된 NPZ/JSON에서 그림과 이 보고서를 다시 만듭니다.',
        '', '출처와 무결성: [계산 설정](config.json), [실행 환경](environment.json), [엔진 고정 버전](engine_sources.json), [파일 해시](manifest.json). 기존 연결 구조의 네 파장 평가와 재계산 투과율을 비교한 검증은 각 결과의 `reference_check`에 있습니다.',
        '', '범위: 현재 구조에서 얻은 결과를 보존한 스냅샷입니다. 새로운 최적화 이력, 실제 제작 결과, 컬러 라우터 결합 결과 또는 LN 마이크로링 결과는 포함하지 않습니다.']
    (ROOT/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    # Standalone HTML uses local relative assets and embedded summary, with no CDN.
    headers=['필터','λ nm','TX %','TY %','ER dB','IL dB','대비도']
    rows=''.join('<tr>'+''.join(f'<td>{v}</td>' for v in [d['filter'],d['wavelength_nm'],f'{100*d["tx"]:.4f}',f'{100*d["ty"]:.4f}',f'{d["extinction_ratio_db"]:.4f}',f'{d["insertion_loss_db"]:.4f}',f'{d["contrast"]:.4f}'])+'</tr>' for d in metrics)
    panels=''
    for wl in range(400,701,50):
        if not (figures/f'{wl}nm_intensity.png').exists():continue
        panels+=f'<section id="w{wl}"><h2>{wl} nm</h2><img src="figures/{wl}nm_intensity.png">'
        for key in ('X','Y'):
            panels+=f'<h3>{key}-pass</h3><a href="fields/{key}_{wl}nm.npz">복소 E 원본 NPZ</a> / <a href="fields/{key}_{wl}nm.json">성능 JSON</a>'
            for mode,title in [('real','Ex/Ey/Ez 실수부'),('amplitude','Ex/Ey/Ez 진폭'),('phase','Ex/Ey/Ez 위상')]:panels+=f'<details><summary>{title}</summary><img loading="lazy" src="figures/{key}_{wl}nm_{mode}.png"></details>'
        panels+='</section>'
    body=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>기판 연결 X/Y 필터 결과</title><style>body{{font:16px/1.7 sans-serif;max-width:1180px;margin:30px auto;padding:20px;color:#193143}}img{{max-width:100%}}table{{border-collapse:collapse}}td,th{{padding:8px 20px;border-bottom:1px solid #ccc}}section{{margin-top:40px;border-top:1px solid #ccc}}summary{{cursor:pointer;padding:10px;background:#edf2f7}}nav a{{margin-right:20px}}</style><h1>기판 연결 X/Y 필터의 파장별 성능과 필드</h1><p>{len(metrics)}/14개 구조·파장 평가. 10 nm voxel, 100층, 4 × 4 × 1 µm.</p><p>독립 X/Y 단색 평면파, RCWA order 10. 필터 출구 이후 SiO₂ 1.35 µm의 정규화 복소 E. FDTD 시간영역 또는 실험 측정 결과가 아닙니다. 연속 스펙트럼 및 회절 차수 수렴은 미검증입니다.</p><p>TX/TY는 입사 편광별 전체 투과율입니다. |E|²는 W/m²가 아닙니다. 자세한 설정·수식·파일 형식은 README.md에 있습니다.</p><img src="figures/connected_structures.png"><h2>성능</h2><a href="metrics.csv">성능 CSV</a> / <a href="field_statistics.csv">필드 통계 CSV</a><table><tr>{''.join('<th>'+h+'</th>' for h in headers)}</tr>{rows}</table><img src="figures/metrics.png"><nav>{''.join(f'<a href="#w{w}">{w} nm</a>' for w in range(400,701,50))}</nav>{panels}</html>'''
    (ROOT/'index.html').write_text(body,encoding='utf-8')
    print('Reported',len(metrics),'cases')
if __name__=='__main__':main()
