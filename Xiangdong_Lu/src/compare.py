"""Join frozen AoA outputs to the user-confirmed floor coordinates and plot them."""
import argparse
import csv
import hashlib
import json
from math import atan2, degrees
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

POINTS = {'A': (.8, .6), 'C': (.8, -1.2), 'D': (2.4, -1.6), 'E': (1.64, -2)}
TX = (0., 0.)
RX = (3.65, 0.)
METHODS = {'G3': 'G3 · AoA-only SAGE', 'J6_product': 'Six-state spectral product'}
BRANCHES = {
    'raw': ('Raw CSI', '#616978'), 'background': ('Run 1 subtraction', '#007b82'),
    'kong_phase_hypothesis': ('Subtraction + phase diagnostic', '#b34c29'),
}


def load(path):
    with path.open(encoding='utf-8-sig', newline='') as fp:
        return list(csv.DictReader(fp))


def save(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', encoding='utf-8-sig', newline='') as fp:
        w = csv.DictWriter(fp, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def geometry():
    return {point: degrees(atan2(y-RX[1], RX[0]-x)) for point, (x, y) in POINTS.items()}


def plot(spectra, controls, geom, out):
    groups = {}
    for row in spectra:
        if row['mapping'] == 'cov' and row['run'] in ('run2', 'run3'):
            key = tuple(row[k] for k in ('point','method','run','branch'))
            groups.setdefault(key, []).append(row)
    ratios = {(r['point'], r['method']): float(r['target_empty_ratio']) for r in controls
              if r['mapping'] == 'cov' and r['branch'] == 'background'}
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.labelcolor': '#223044', 'text.color': '#223044'})
    for control_plot in (False, True):
        fig, axes = plt.subplots(4, 2, figsize=(11, 12), sharex=True, sharey=True)
        fig.subplots_adjust(left=.115, right=.97, top=.828, bottom=.12, wspace=.17, hspace=.32)
        title = 'Run 1 subtraction · target and independent empty-room control' if control_plot else 'Static WiCAL AoA · all three fixed processing branches'
        subtitle = ('Run 2 (solid) and Run 3 (dashed), shared scale within each panel' if control_plot else
                    'Run 2 · primary Intel mask3 decoding · each curve scaled to its own peak')
        fig.text(.08, .964, title, fontsize=18, weight='bold')
        fig.text(.08, .926, subtitle, fontsize=11, color='#49576b')
        for i, point in enumerate('ACDE'):
            for j, (method, label) in enumerate(METHODS.items()):
                ax = axes[i,j]
                ax.set_title(f'{point}   {label}', loc='left', fontsize=11, weight='bold', pad=8)
                ax.grid(color='#dce3eb', linewidth=.7, zorder=0)
                ax.axvline(geom[point], linestyle=':', color='#263240', linewidth=1.2, zorder=1)
                ax.set(xlim=(-90,90), ylim=(-.03,1.1), xticks=np.arange(-90,91,30), yticks=[0,.25,.5,.75,1])
                if j == 0:
                    ax.set_ylabel('Scaled beam power' if control_plot else 'Scaled spectrum')
                if i == 3:
                    ax.set_xlabel('AoA grid (degrees)')
                if control_plot:
                    branch = 'background'
                    rows_by_run = {r: groups[(point,method,r,branch)] for r in ('run2','run3')}
                    field = 'unnormalized_score_profile'
                    scale = max(float(r[field]) for rows in rows_by_run.values() for r in rows)
                    for run, rows in rows_by_run.items():
                        color = '#007b82' if run == 'run2' else '#a2643a'
                        angles = np.array([float(r['angle_deg']) for r in rows])
                        powers = np.array([float(r[field]) for r in rows])/scale
                        ax.plot(angles, powers, color=color, linestyle='-' if run=='run2' else '--', lw=1.7)
                    ax.text(.02, .05, f'Run 2 / Run 3 peak power = {ratios[(point,method)]:.2f}×',
                            transform=ax.transAxes, fontsize=9, bbox={'facecolor':'white','edgecolor':'none','alpha':.90})
                else:
                    for branch, (_, color) in BRANCHES.items():
                        rows = groups[(point,method,'run2',branch)]
                        angles = np.array([float(r['angle_deg']) for r in rows])
                        powers = np.array([float(r['power']) for r in rows])
                        chosen = next(k for k,r in enumerate(rows) if r['selected']=='True')
                        ax.plot(angles, powers/powers.max(), color=color, lw=1.6)
                        ax.plot(angles[chosen], 1, 'o', color=color, markersize=5, clip_on=False)
        legend = [Line2D([],[],color='#007b82',label='Run 2 · target'),
                  Line2D([],[],color='#a2643a',ls='--',label='Run 3 · empty control')] if control_plot else [
                  Line2D([],[],color=color,label=name) for name,color in BRANCHES.values()]
        legend.append(Line2D([],[],ls=':',color='#263240',label='Geometric bearing · assumed sign'))
        fig.legend(handles=legend, loc='upper left', bbox_to_anchor=(.075,.906), ncol=len(legend), frameon=False)
        note = ('Six-state control power is the mean of the six unnormalized state profiles. '
                'It is a strength measure independent of the product peak location.' if control_plot else
                'Dots show the fixed GT-blind peaks. Geometry was overlaid after fitting. A/C/D/E subset; B excluded after inspection.')
        fig.text(.08,.04,note,fontsize=8.5,color='#49576b')
        stem = 'empty_control_comparison' if control_plot else 'aoa_spectra_comparison'
        fig.savefig(out/f'{stem}.png', dpi=170, facecolor='white')
        plt.close(fig)
