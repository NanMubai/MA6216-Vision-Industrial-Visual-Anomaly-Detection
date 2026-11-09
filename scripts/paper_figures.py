#!/usr/bin/env python3
"""Publication charts from committed evaluation reports; never fit a model.

The v1 archive and legacy bars/JPEGs are immutable. Only the four named chart
PDFs and their provenance records are written at the requested output root.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter

from paper_tables import CATEGORIES, ROOT, RUNS, VARIANTS, csv_rows, load_metrics, variant_value

STYLE = ROOT / 'paper/style/ieee.mplstyle'
SINGLE, DOUBLE = 3.5, 7.16
METHODS = {'PaDiM': ('#0072B2', 'o'), 'PatchCore': ('#D55E00', 's')}
FAMILIES = {'baseline': ('#000000', '*'), 'dinov2': ('#009E73', 'D'),
            'multiscale': ('#CC79A7', '^')}
PER_REGION = 'reports/figures/failure_cases/tables/per_region.csv'
POOLED = 'reports/figures/failure_cases/tables/region_detection_pooled.csv'
MAIN_SOURCE = 'reports/tables/metrics_long.csv'
EXPECTED_BINS = ('<1 cell', '1-3 cells', '3-10 cells', '10-30 cells', '30-100 cells', '>=100 cells')
CHARTS = ('fig_main_dumbbell.pdf', 'fig_defect_area.pdf', 'fig_regionsize.pdf', 'fig_tradeoff.pdf')
TRADEOFF_VARIANTS = ('Base', 'MsMaxCtrl', 'DinoB', 'DinoS', 'WrnDino', 'DinoNoLN',
                     'MsLOneTwoThree', 'MsLTwoThreeFour', 'PyrMean')


def output_root(output: Path) -> Path:
    output = output.resolve()
    if 'v1' in output.parts:
        raise ValueError('the v1 archives are immutable')
    output.mkdir(parents=True, exist_ok=True)
    return output


def save(fig, path: Path, generator='scripts/paper_figures.py'):
    output_root(path.parent)
    fig.savefig(path, dpi=300, bbox_inches=None, facecolor='white', transparent=False,
                metadata={'Creator': generator, 'CreationDate': None})
    plt.close(fig)


def record(output: Path, name: str, sources: list[str], transformation: str, n,
           selection_rule=None, generator='scripts/paper_figures.py', **extra):
    """Merge only top-level outputs, so charts and panels can run separately."""
    path = output / 'MANIFEST.json'
    manifest = json.loads(path.read_text()) if path.exists() else {
        'schema_version': 1, 'style': 'paper/style/ieee.mplstyle', 'minimum_text_pt': 6,
        'style_basis': 'scientific-visualization publication.mplstyle (copied values)',
        'method_encoding': {k: {'color': v[0], 'marker': v[1]} for k, v in METHODS.items()},
        'outputs': {},
    }
    assert '/' not in name and name.endswith('.pdf')
    manifest['outputs'][name] = dict(sources=sources, transformation=transformation, n=n,
        selection_rule=selection_rule, generator=generator, **extra)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')


def method_handles():
    return [Line2D([], [], color=c, marker=m, linestyle='none', markersize=4,
                   markerfacecolor=c if method == 'PaDiM' else 'white', label=method)
            for method, (c, m) in METHODS.items()]


def main_dumbbell(output, data):
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE, 2.1), layout='constrained', sharey=True)
    y = np.arange(6)
    for ax, key, title in zip(axes, ('image_auroc', 'image_f1', 'aupro'),
                             ('Image AUROC', 'Image F1 (val. threshold)', 'AU-PRO')):
        vals = [[data['baseline'].get(run, key, (cat,)) for cat in CATEGORIES] for run in RUNS.values()]
        ax.hlines(y, vals[0], vals[1], color='0.65', linewidth=0.7, zorder=2)
        for i, (method, (color, marker)) in enumerate(METHODS.items()):
            ax.plot(vals[i], y, linestyle='none', marker=marker, markersize=4,
                    color=color, markerfacecolor=color if i == 0 else 'none',
                    markeredgewidth=0.85, zorder=3+i, clip_on=False)
        ax.set(title=title, yticks=y, yticklabels=CATEGORIES, ylim=(5.55, -0.55))
        if key == 'image_f1':
            ax.set(xlim=(0, 1), xticks=(0, .25, .5, .75, 1))
        else:
            ax.set(xlim=(.75, 1), xticks=(.75, .875, 1), xticklabels=('0.75', '0.875', '1.00'))
        ax.grid(axis='x', color='0.9', linewidth=.5)
        ax.axhline(2.5, color='0.65', linewidth=.5)
        ax.spines['left'].set_visible(False)
        ax.tick_params(axis='y', length=0, pad=3)
    axes[0].text(-.48, .75, 'Objects', rotation=90, transform=axes[0].transAxes,
                 ha='center', va='center', fontsize=6)
    axes[0].text(-.48, .25, 'Textures', rotation=90, transform=axes[0].transAxes,
                 ha='center', va='center', fontsize=6)
    fig.legend(handles=method_handles(), loc='outside upper center', ncol=2)
    save(fig, output / CHARTS[0])
    record(output, CHARTS[0], [MAIN_SOURCE],
           'Per-category point estimates on a common row; hollow PatchCore squares retain visible ties; no rescaling of metric values.',
           {'categories': 6, 'methods': 2}, width_inches=DOUBLE, height_inches=2.1,
           uncertainty='Single-seed estimates; no between-run uncertainty is available.')


def defect_area(output):
    rows = [r for r in csv_rows(PER_REGION) if r['method'] == 'PatchCore']
    vals = [[float(r['area_cells']) for r in rows if r['category'] == cat] for cat in CATEGORIES]
    assert all(v > 0 for group in vals for v in group), 'log area requires positive regions'
    fig, ax = plt.subplots(figsize=(SINGLE, 2.4), layout='constrained')
    rng = np.random.default_rng(0)
    for i, group in enumerate(vals):
        ax.scatter(group, i + rng.uniform(-.16, .16, len(group)), s=3, color='0.25',
                   alpha=.35, linewidths=0, zorder=2)
        ax.text(1.015, i, f'n = {len(group)}', transform=ax.get_yaxis_transform(),
                ha='left', va='center', fontsize=6)
    ax.boxplot(vals, positions=np.arange(6), orientation='horizontal', widths=.48,
               showfliers=False, whis=1.5, patch_artist=False,
               boxprops={'color': '0.15', 'linewidth': .7}, medianprops={'color':'black', 'linewidth':1},
               whiskerprops={'color':'0.3', 'linewidth':.7}, capprops={'color':'0.3', 'linewidth':.7})
    for edge in (3, 10, 30, 100):
        ax.axvline(edge, color='0.9', linewidth=.5, zorder=0)
    ax.axvline(1, color='0.15', linestyle='--', linewidth=.8, zorder=1,
               label='1 feature cell (0.128% of crop)')
    ax.axhline(2.5, color='0.65', linewidth=.5)
    ax.set(xscale='log', xlim=(min(.09, min(map(min, vals))/1.2), 1300), ylim=(5.55, -.55),
           yticks=np.arange(6), yticklabels=CATEGORIES,
           xlabel='GT region area (stride-8 feature cells, log scale)')
    ax.set_xticks((.1,1,10,100,1000), labels=('0.1','1','10','100','1000'))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.legend(loc='lower center', bbox_to_anchor=(.5, 1.01), fontsize=6.5)
    save(fig, output / CHARTS[1])
    record(output, CHARTS[1], [PER_REGION],
           'PatchCore rows only (each GT region once); area_cells in the 224 crop; unfilled boxes (1.5 IQR whiskers) plus every region with vertical U(-0.16,0.16) jitter, seed 0; log10 x axis.',
           {'regions': len(rows), 'by_category': dict(zip(CATEGORIES,map(len,vals)))},
           'method == PatchCore; all rows; no exclusions', width_inches=SINGLE, height_inches=2.4,
           unit='8-connected GT region inside the 224 crop', reference_cell_pixels=64,
           reference_cell_crop_percent=100*64/224**2, jitter_seed=0)


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if not 0 <= k <= n or n <= 0:
        raise ValueError('Wilson interval requires 0 <= k <= n and n > 0')
    p, z2 = k/n, z*z
    center = (p + z2/(2*n)) / (1 + z2/n)
    half = z * math.sqrt(p*(1-p)/n + z2/(4*n*n)) / (1 + z2/n)
    return max(0., center-half), min(1., center+half)


def region_counts():
    pooled, regions = csv_rows(POOLED), csv_rows(PER_REGION)
    assert tuple(r['size_bin'] for r in pooled) == EXPECTED_BINS
    counts = {}
    for method in METHODS:
        counts[method] = []
        for row in pooled:
            selected = [r for r in regions if r['method'] == method and r['size_bin'] == row['size_bin']]
            k, n = sum(r['region_detected'] == 'True' for r in selected), len(selected)
            assert n == int(row['n'])
            assert abs(k/n-float(row[method])) <= .00005 + 1e-12, (method,row,k,n)
            counts[method].append((k,n))
    return pooled, counts


def region_size(output):
    pooled, counts = region_counts()
    fig, ax = plt.subplots(figsize=(SINGLE, 2.3), layout='constrained')
    x = np.arange(len(pooled))
    for i, (method, (color, marker)) in enumerate(METHODS.items()):
        vals = np.array([k/n for k,n in counts[method]])
        ci = np.array([wilson(k,n) for k,n in counts[method]])
        xx = x + (-.12 if i == 0 else .12)
        ax.plot(xx, vals, color=color, linewidth=.7, alpha=.5, linestyle='-' if i == 0 else '--')
        ax.errorbar(xx, vals, yerr=[vals-ci[:,0],ci[:,1]-vals], color=color, marker=marker,
                     linestyle='none', markersize=3.5, capsize=2, linewidth=.8, label=method)
    labels = [r['size_bin'].removesuffix(' cells').removesuffix(' cell').replace('>=','≥').replace('-','–')
              + '\n' + f"n={r['n']}" for r in pooled]
    ax.set(xticks=x, xticklabels=labels, xlim=(-.5,5.5), ylim=(0,1.05),
           xlabel='GT region size (stride-8 feature cells)', ylabel='Region detection rate')
    ax.grid(axis='y', color='0.9', linewidth=.5)
    ax.legend(loc='lower right', fontsize=7)
    save(fig, output / CHARTS[2])
    record(output, CHARTS[2], [PER_REGION, POOLED],
           'Exact region_detected/n per method and bin; 95% Wilson score intervals, z=1.959964; horizontal dodge ±0.12; lines are ordinal guides only.',
           {'regions_per_method': sum(int(r['n']) for r in pooled),
            'by_bin': {r['size_bin']:int(r['n']) for r in pooled}},
           'All regions; prescribed bins ordered as in pooled CSV', width_inches=SINGLE, height_inches=2.3,
           counts={m:[{'detected':k,'n':n} for k,n in v] for m,v in counts.items()},
           uncertainty='Nominal region-level Wilson intervals; regions within an image may be dependent (not an image-cluster interval).')


def tradeoff(output, data):
    fig, axes = plt.subplots(2,1,figsize=(SINGLE,3.65),layout='constrained',sharex=True)
    vs = [next(v for v in VARIANTS if v.macro == name) for name in TRADEOFF_VARIANTS]
    labels = {'Base':'Baseline','MsMaxCtrl':'Max ctrl','DinoB':'ViT-B','DinoS':'ViT-S',
              'WrnDino':'WRN+ViT-B','DinoNoLN':'No-LN†','MsLOneTwoThree':'L1+2+3',
              'MsLTwoThreeFour':'L2+3+4','PyrMean':'Pyramid'}
    offsets = [
        {'Base':(-4,-16),'MsMaxCtrl':(8,3),'DinoB':(5,2),'DinoS':(-12,-14),
         'WrnDino':(5,-4),'DinoNoLN':(6,0),'MsLOneTwoThree':(-30,-12),
         'MsLTwoThreeFour':(6,2),'PyrMean':(-36,6)},
        {'Base':(-5,8),'MsMaxCtrl':(7,-12),'DinoB':(5,-1),'DinoS':(-20,-12),
         'WrnDino':(5,-2),'DinoNoLN':(5,-10),'MsLOneTwoThree':(-32,5),'PyrMean':(-36,7)}]
    for panel, (ax, metric, title, lim) in enumerate(zip(axes,('ImgAuroc','Aupro'),
              ('(a) Image AUROC','(b) AU-PRO'),((.915,1.004),(.895,.958)))):
        baseline=variant_value(vs[0],metric,data)
        ax.axhline(baseline,color='0.55',linestyle=':',linewidth=.7)
        for v in vs:
            xx, yy=variant_value(v,'Infer',data),variant_value(v,metric,data)
            family='baseline' if v.macro in ('Base','MsMaxCtrl') else v.family
            color, marker=FAMILIES[family]
            if v.macro=='MsMaxCtrl': marker='+'
            outside=yy < lim[0]
            if outside:
                ax.plot(xx,lim[0],marker=marker,color=color,markersize=4,clip_on=False)
                ax.annotate(f'▼ {yy:.3f}  L2+3+4',(xx,lim[0]),xytext=(4,5),
                            textcoords='offset points',fontsize=6,ha='left',va='bottom')
            else:
                ax.plot(xx,yy,marker=marker,color=color,linestyle='none',markersize=4,
                         markerfacecolor='white' if v.posthoc else color,
                         markeredgewidth=.8,zorder=4,clip_on=False)
                ax.annotate(labels[v.macro],(xx,yy),xytext=offsets[panel][v.macro],
                             textcoords='offset points',fontsize=6,
                             arrowprops={'arrowstyle':'-','color':'0.55','lw':.4},zorder=5)
        ax.set(ylim=lim,title=title,xscale='log',xlim=(2,36))
        ax.set_xscale('log',base=2)
        ax.set_xticks((2,4,8,16,32),labels=('2','4','8','16','32'))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.grid(axis='x',color='0.92',linewidth=.5)
    axes[1].set_xlabel('Inference (ms/image, log base 2)')
    handles=[Line2D([],[],color=c,marker=m,linestyle='none',markersize=4,label=label)
             for (c,m),label in zip(FAMILIES.values(),('Baseline / control','DINOv2','Multi-scale'))]
    handles.append(Line2D([],[],color=FAMILIES['dinov2'][0],marker='D',markerfacecolor='white',
                          linestyle='none',markersize=4,label='hollow = post-hoc'))
    fig.legend(handles=handles,loc='outside lower center',ncol=2,fontsize=6.5)
    save(fig,output/CHARTS[3])
    record(output,CHARTS[3],[MAIN_SOURCE,'reports/bonus/dinov2_tables/metrics_long.csv',
                            'reports/bonus/multiscale_tables/metrics_long.csv'],
           'Six-category arithmetic means, identical to bn* macros; model variants only; log2 latency; baseline reference lines; out-of-range L2+3+4 AU-PRO explicitly marked on the lower spine.',
           {'variants':len(vs),'categories_per_variant':6},
           'Prescribed nine model/control variants; no post-processing points; z-mean ≈ mean, so only pyramid mean is drawn.',
           width_inches=SINGLE,height_inches=3.65,variants={v.macro:{m:variant_value(v,m,data) for m in ('Infer','ImgAuroc','Aupro')} for v in vs},
           uncertainty='Single-seed category means; no repeated-run uncertainty available.')


def generate(output):
    output=output_root(output)
    with plt.style.context(STYLE):
        data=load_metrics()
        main_dumbbell(output,data)
        defect_area(output)
        region_size(output)
        tradeoff(output,data)
    print(f'Wrote {len(CHARTS)} chart PDFs to {output}; legacy figures and v1 unchanged.')


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output-dir',type=Path,default=ROOT/'paper/generated/figures')
    generate(ap.parse_args().output_dir)
