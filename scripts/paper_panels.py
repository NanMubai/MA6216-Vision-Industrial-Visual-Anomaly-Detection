#!/usr/bin/env python3
"""Render final-size qualitative panels from fixed predictions, on CPU only.

Test labels are used only for these explicitly post-hoc illustrations. No model
is instantiated, no inference is run, and metrics.json validation thresholds are
read without modification. Original data, saved predictions and v1 are untouched.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import patheffects
from matplotlib.cm import ScalarMappable
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
import numpy as np

from paper_figures import (DOUBLE, ROOT, STYLE, output_root, record, save)

sys.path.insert(0, str(ROOT / 'src'))
from ad.data import MVTecDataset
from ad.visualize import _load_image
from analysis_failure_panels import CASES, RESULTS, RUNS, _load

MVTEC = Path('/home/eva/code/ma6216/industrial-visual-anomaly-detection/data/mvtec_ad')
NORM = TwoSlopeNorm(vmin=.4, vcenter=1., vmax=1.6)
CMAP = matplotlib.colormaps['RdBu_r'].with_extremes(bad='#777777')
GT_COLOR = '#F0E442'
PANEL_OUTPUTS = ('fig_failures.pdf','fig_app_failures_1.pdf','fig_app_failures_2.pdf',
                 'fig_app_qual_1.pdf','fig_app_qual_2.pdf','fig_app_qual_3.pdf')
SELECTION_RULE = (
    'Post-hoc illustration selection only, using PatchCore scores and test labels: '
    '(1) anomalous image closest to the median PatchCore image score; '
    '(2) lowest-scoring anomaly; (3) largest absolute PaDiM-PatchCore score/threshold '
    'difference among unpicked anomalies; (4) highest PatchCore score/threshold normal. '
    'Ties use lowest index. If an even-sized anomaly set brackets the median, choose '
    'the equidistant lower-index image. No fitted parameter or threshold is selected.'
)


def select_examples(pd_scores, pc_scores, labels, pd_thr, pc_thr) -> list[int]:
    """Deterministic, test-informed illustration selection; no model selection."""
    pd_scores, pc_scores = np.asarray(pd_scores, dtype=float), np.asarray(pc_scores, dtype=float)
    labels = np.asarray(labels)
    assert pd_scores.shape == pc_scores.shape == labels.shape
    assert np.isfinite(pd_scores).all() and np.isfinite(pc_scores).all()
    assert pd_thr > 0 and pc_thr > 0
    anomalies, normals = np.flatnonzero(labels == 1).tolist(), np.flatnonzero(labels == 0).tolist()
    if len(anomalies) < 3 or not normals:
        raise ValueError('illustrations need at least three anomalies and one normal')
    median = float(np.median(pc_scores[anomalies]))
    picked = [min(anomalies, key=lambda i: (abs(pc_scores[i]-median), i))]
    picked.append(min(anomalies, key=lambda i: (pc_scores[i], i)))
    assert picked[0] != picked[1], 'median and lowest anomaly coincide; selection is ambiguous'
    picked.append(min((i for i in anomalies if i not in picked),
                      key=lambda i: (-abs(pd_scores[i]/pd_thr-pc_scores[i]/pc_thr),i)))
    picked.append(min(normals, key=lambda i: (-pc_scores[i]/pc_thr,i)))
    return picked


def decision(label: int, detected: bool) -> str:
    return {(1,True):'TP',(1,False):'FN',(0,True):'FP',(0,False):'TN'}[label,detected]


class FixedPredictions:
    def __init__(self):
        self.runs, self.datasets = {}, {}

    def category(self, cat):
        if cat not in self.runs:
            pair = [_load(run, cat) for run,_ in RUNS]
            assert pair[0]['keys'] == pair[1]['keys'], f'test ordering differs: {cat}'
            assert np.array_equal(pair[0]['labels'],pair[1]['labels'])
            data=pair[0]['data']
            assert data['resize']==256 and data['crop']==224
            assert pair[0]['maps'].shape[1:] == pair[1]['maps'].shape[1:] == (224,224)
            assert all(r['image_thr']>0 and r['pixel_thr']>0 for r in pair)
            ds=MVTecDataset(cat,'test',root=MVTEC,resize=256,crop=224)
            ds_keys=[f'{d}/{p.name}' for p,_,_,d in ds.samples]
            assert ds_keys == pair[0]['keys'], f'dataset ordering differs: {cat}'
            self.runs[cat], self.datasets[cat] = pair, ds
        return self.runs[cat]

    def case(self, cat, index, letter, case_id=None):
        pair=self.category(cat)
        sample=self.datasets[cat][index]
        assert Path(sample['path']).is_file(), sample['path']
        image=_load_image(sample['path'],256,224,(224,224))
        gt=sample['mask'].squeeze().numpy().astype(bool)
        label=int(sample['label'])
        assert label == int(pair[0]['labels'][index])
        assert np.array_equal(gt,pair[0]['masks'][index]), f'saved/dataset GT mismatch: {cat}/{index}'
        results=[]
        for r,(_,method) in zip(pair,RUNS):
            score_ratio=float(r['scores'][index])/r['image_thr']
            ratio=r['maps'][index].astype(np.float32)/r['pixel_thr']
            assert np.isfinite(ratio).all()
            results.append(dict(method=method,ratio=ratio,score_ratio=score_ratio,
                                decision=decision(label,score_ratio>=1.),
                                image_threshold=r['image_thr'],pixel_threshold=r['pixel_thr']))
        return dict(category=cat,index=int(index),key=pair[0]['keys'][index],letter=letter,
                    id=case_id,image=image,gt=gt,label=label,results=results,path=sample['path'],
                    mask_path=str(self.datasets[cat].samples[index][1]) if self.datasets[cat].samples[index][1] else None)


def gt_contour(ax, mask):
    if mask.any():
        # Pad the crop so a mask touching the boundary still has a visible outline.
        contour=ax.contour(np.arange(-1,225),np.arange(-1,225),np.pad(mask.astype(float),1),levels=[.5],colors=[GT_COLOR],
                           linewidths=.8,linestyles='--',zorder=5)
        contour.set_path_effects([patheffects.withStroke(linewidth=1.6,foreground='black')])


def pred_contour(ax, mask):
    if mask.any():
        ax.contour(np.arange(-1,225),np.arange(-1,225),np.pad(mask.astype(float),1),levels=[.5],colors=['black'],
                   linewidths=.8,linestyles='-',zorder=4)


def case_label(case):
    defect,file=case['key'].split('/')
    return f"({case['letter']}) {case['category']} / {defect} {Path(file).stem}"


def render(output, name, cases, categories=None):
    """Two cases per row, each comprising three native 224-pixel image tiles."""
    qualitative=categories is not None
    height=7.0 if qualitative else 5.35
    fig=plt.figure(figsize=(DOUBLE,height),layout='constrained')
    if qualitative:
        ratios=[.16,1.2,1.2,.16,1.2,1.2,.12,.12]
        case_rows=(1,2,4,5)
    else:
        ratios=[1.2,1.2,1.2,.12,.12]
        case_rows=(0,1,2)
    grid=fig.add_gridspec(len(ratios),2,height_ratios=ratios,hspace=.02,wspace=.06)
    if qualitative:
        for row,cat in zip((0,3),categories):
            ax=fig.add_subplot(grid[row,:]);ax.axis('off')
            ax.text(0,.5,cat.title(),ha='left',va='center',fontsize=8,fontweight='bold')
    for i,case in enumerate(cases):
        row, col=case_rows[i//2],i%2
        sub=grid[row,col].subgridspec(4,3,height_ratios=(.15,.12,1,.15),hspace=.01,wspace=.015)
        title=fig.add_subplot(sub[0,:]); title.axis('off')
        title.text(0,.1,case_label(case),fontsize=6.5,ha='left',va='bottom')
        gray=np.asarray(case['image'],dtype=float) @ np.array([.2126,.7152,.0722])
        for j in range(3):
            header=fig.add_subplot(sub[1,j]); header.axis('off')
            if i//2==0:
                header.text(.5,.5,('Image + GT','PaDiM','PatchCore')[j],fontsize=6.5,ha='center',va='center',transform=header.transAxes)
                if j>0:
                    color,marker=('#0072B2','o') if j==1 else ('#D55E00','s')
                    header.plot(.10,.5,marker=marker,color=color,markersize=3,transform=header.transAxes)
            ax=fig.add_subplot(sub[2,j])
            if j==0:
                ax.imshow(case['image'],interpolation='nearest')
            else:
                result=case['results'][j-1]
                ax.imshow(gray,cmap='gray',vmin=0,vmax=255,interpolation='nearest')
                ax.imshow(result['ratio'],norm=NORM,cmap=CMAP,alpha=.6,interpolation='nearest')
                pred_contour(ax,result['ratio']>=1.)
                label=fig.add_subplot(sub[3,j]); label.axis('off')
                label.text(.5,.9,f"score/thr {result['score_ratio']:.2f} · {result['decision']}",
                           fontsize=6,ha='center',va='top')
            gt_contour(ax,case['gt'])
            ax.set(xlim=(-.5,223.5),ylim=(223.5,-.5),xticks=[],yticks=[])
            for spine in ax.spines.values(): spine.set(linewidth=.4,color='0.4')
    cax=fig.add_subplot(grid[-2,:])
    bar=fig.colorbar(ScalarMappable(norm=NORM,cmap=CMAP),cax=cax,orientation='horizontal',extend='both')
    bar.set_ticks((.4,1.,1.6),labels=('0.4','1 (threshold)','1.6'))
    bar.ax.tick_params(labelsize=6)
    bar.set_label('anomaly map / val. pixel threshold',fontsize=6.5,labelpad=2)
    lax=fig.add_subplot(grid[-1,:]); lax.axis('off')
    gt_line=Line2D([],[],color=GT_COLOR,linestyle='--',linewidth=.8,
                   path_effects=[patheffects.withStroke(linewidth=1.6,foreground='black')],label='GT (dashed + black halo)')
    pred_line=Line2D([],[],color='black',linewidth=.8,label='Prediction (solid)')
    lax.legend(handles=[gt_line,pred_line],loc='lower center',bbox_to_anchor=(.5,-.9),ncol=2,fontsize=6.5)
    save(fig,output/name,generator='scripts/paper_panels.py')
    assert (output/name).stat().st_size <= 2_500_000, (name,(output/name).stat().st_size)
    picks=[]
    for c in cases:
        picks.append({k:c[k] for k in ('id','letter','category','index','key','path','mask_path','label')})
        picks[-1]['methods']={r['method']:{k:r[k] for k in ('score_ratio','decision','image_threshold','pixel_threshold')}
                              for r in c['results']}
    sources=[]
    for cat in dict.fromkeys(c['category'] for c in cases):
        for run,_ in RUNS:
            sources.extend(str(RESULTS/run/cat/f) for f in ('maps.npz','scores.npz','metrics.json'))
    sources.extend(c['path'] for c in cases)
    sources.extend(c['mask_path'] for c in cases if c['mask_path'])
    record(output,name,sources,
           'RGB input crop; greyscale base for maps; resize 256 -> crop 224 using ad.visualize._load_image; '
           'GT from MVTecDataset; ratio=stored map / unchanged validation pixel threshold; '
           'TwoSlopeNorm(0.4,1.0,1.6), RdBu_r, alpha=0.6, saturated ends shown by colourbar extensions; '
           'solid black predicted contour at ratio>=1; dashed yellow GT with black halo; '
           'native 224x224 input arrays, final-size vector labels, 300 dpi embedded rasters.',
           {'illustrative_cases':len(cases),'tiles':3*len(cases)},
           SELECTION_RULE if qualitative else 'Prescribed curated case IDs from analysis_failure_panels.CASES',
           generator='scripts/paper_panels.py',width_inches=DOUBLE,height_inches=height,
           picks=picks,native_tile_pixels=[224,224],dpi=300,
           normalization={'vmin':.4,'vcenter':1.,'vmax':1.6,'cmap':'RdBu_r','alpha':.6},
           contours={'prediction':'black, solid, 0.8pt','GT':'#F0E442, dashed, 0.8pt, black 1.6pt halo'},
           interpretation='Post-hoc illustrations only; cases are not a representative performance sample.')


def generate(output):
    output=output_root(output)
    store=FixedPredictions()
    with plt.style.context(STYLE):
        for name,ids in zip(PANEL_OUTPUTS[:3],((1,4,5,7,8,12),tuple(range(1,7)),tuple(range(7,13)))):
            cases=[]
            for i,cid in enumerate(ids):
                _,cat,key,_=CASES[cid-1]
                index=store.category(cat)[0]['keys'].index(key)
                cases.append(store.case(cat,index,chr(97+i),f'{cid:02d}'))
            render(output,name,cases)
            print(name)
        for number,cats in enumerate((('bottle','capsule'),('screw','carpet'),('tile','leather')),1):
            cases=[]
            for cat in cats:
                pd,pc=store.category(cat)
                picks=select_examples(pd['scores'],pc['scores'],pc['labels'],pd['image_thr'],pc['image_thr'])
                for index in picks:
                    cases.append(store.case(cat,index,chr(97+len(cases))))
            render(output,f'fig_app_qual_{number}.pdf',cases,categories=cats)
            print(f'fig_app_qual_{number}.pdf')
    names=(*PANEL_OUTPUTS,'fig_main_dumbbell.pdf','fig_defect_area.pdf','fig_regionsize.pdf','fig_tradeoff.pdf')
    total=sum((output/name).stat().st_size for name in names if (output/name).exists())
    assert total <= 15_000_000,total
    print(f'New outputs: {total/1_000_000:.2f} MB; v1 and legacy JPEGs unchanged.')


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output-dir',type=Path,default=ROOT/'paper/generated/figures')
    generate(ap.parse_args().output_dir)
