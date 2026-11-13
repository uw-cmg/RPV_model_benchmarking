#!/usr/bin/env python3
"""Figures from unchanged full-fit coefficients and nine-coefficient CV."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT/'analysis_outputs/atr2_cf_validation'
FULL_METHOD = 'Full fit: unchanged code coefficients'
CV_METHOD = 'Repeated five-fold CV (pooled)'


def parity(ax,points,stats,quantity,title,cv=False,predicted_on_y=True):
    if quantity=='DSY':
        x=points.predicted_dsy_MPa.to_numpy();y=points.measured_dsy_MPa.to_numpy()
        label=r'$\Delta\sigma_y$';unit='MPa'
    else:
        x=points.predicted_TTS_degC.to_numpy();y=points.measured_TTS_degC.to_numpy()
        label='TTS';unit='°C'
    if predicted_on_y:
        x,y=y,x
    limit=np.ceil(max(x.max(),y.max())*1.05/50)*50
    ax.plot([0,limit],[0,limit],':',color='#333333',linewidth=1.7,label='1:1')
    ax.scatter(x,y,s=39 if cv else 46,marker='D',facecolor='#C51B1D',
        edgecolor='#7F0000',linewidth=.45,alpha=.45 if cv else .85,zorder=4,
        label='50 alloys × 5 out-of-fold predictions' if cv else '50 alloys')
    ax.set(xlim=(0,limit),ylim=(0,limit),aspect='equal')
    ax.set_xlabel(f'{"Measured" if predicted_on_y else "Predicted"} {label} ({unit})',fontsize=14)
    ax.set_ylabel(f'{"Predicted" if predicted_on_y else "Measured"} {label} ({unit})',fontsize=14)
    ax.set_title(title,fontsize=16,pad=10,fontweight='normal')
    ax.tick_params(labelsize=12);ax.grid(color='#D7D7D7',alpha=.65,linewidth=.7)
    ax.legend(loc='upper left',fontsize=10,framealpha=.96)
    text=(f"ME = {stats.ME:+.2f} {unit}\nMAE = {stats.MAE:.2f} {unit}\n"
        f"RMSE = {stats.RMSE:.2f} {unit}\nResidual SD = {stats.residual_SD:.2f} {unit}")
    ax.text(.97,.04,text,transform=ax.transAxes,ha='right',va='bottom',
        fontsize=11,linespacing=1.45,bbox=dict(facecolor='white',edgecolor='#aaa',
        alpha=.96,boxstyle='round,pad=.4'),zorder=5)

def main(output=None):
    global OUTPUT
    if output is not None:
        OUTPUT = Path(output)
    full = pd.read_csv(OUTPUT/'fullfit_predictions.csv')
    oof = pd.read_csv(OUTPUT/'cv_out_of_fold_predictions.csv')
    stats = pd.read_csv(OUTPUT/'error_statistics.csv').set_index(['method','quantity'])
    bars = pd.read_csv(OUTPUT/'figure2_bar_values.csv')
    fig,(left,right) = plt.subplots(1,2,figsize=(13.5,6.8),constrained_layout=True)
    parity(left,full,stats.loc[(FULL_METHOD,'TTS')],'TTS','Full-fit TTS predictions')
    group_x = np.array([0,1.1]); width=.32
    for quantity,label,color,offset,unit in [
        ('DSY',r'$\Delta\sigma_y$ (MPa)','#0072B2',-width/2,'MPa'),
        ('TTS','TTS (°C)','#009E73',width/2,'°C')]:
        sub = bars[bars.quantity.eq(quantity)].set_index('method')
        values = [float(sub.loc['Full fit','MAE']),float(sub.loc['5-fold CV (5 repeats)','MAE'])]
        sd = float(sub.loc['5-fold CV (5 repeats)','SD_across_25_splits'])
        rectangles = right.bar(group_x+offset,values,width=width,color=color,label=label,
            edgecolor='#333333',linewidth=.7,zorder=3)
        right.errorbar(group_x[1]+offset,values[1],yerr=sd,fmt='none',ecolor='black',
            elinewidth=1.4,capsize=6,capthick=1.4,zorder=4)
        for i,(rectangle,v) in enumerate(zip(rectangles,values)):
            text = f'{v:.2f} {unit}' if i==0 else f'{v:.2f} ± {sd:.2f}\n{unit}'
            right.text(rectangle.get_x()+rectangle.get_width()/2,v+.45+(sd if i else 0),
                text,ha='center',va='bottom',fontsize=11)
    right.set_xticks(group_x,['Full fit','5-fold CV\n(5 repeats)'],fontsize=13)
    right.set_ylabel(r'MAE ($\Delta\sigma_y$: MPa; TTS: °C)',fontsize=14)
    right.set_title('Full fit and cross-validation MAE',fontsize=16,pad=10,fontweight='normal')
    right.set(ylim=(0,25),xlim=(-.55,1.65)); right.tick_params(axis='y',labelsize=12)
    right.legend(loc='upper left',fontsize=12,title='CV error bars: ±1 SD (25 splits)',title_fontsize=10.5)
    right.grid(axis='y',color='#D7D7D7',linewidth=.7,alpha=.6,zorder=0)
    right.set_axisbelow(True); right.set_box_aspect(1)
    for ext in ['png','svg']:
        fig.savefig(OUTPUT/f'figure2_fixed_fullfit_nine_coefficient_cv.{ext}',dpi=300,bbox_inches='tight')
    plt.close(fig)

    fig,axes = plt.subplots(2,2,figsize=(13.5,13.4),constrained_layout=True)
    for row,(points,method,title,cv) in enumerate([
        (full,FULL_METHOD,'Full fit',False),(oof,CV_METHOD,'5 × 5-fold CV',True)]):
        for col,quantity in enumerate(['DSY','TTS']):
            parity(axes[row,col],points,stats.loc[(method,quantity)],quantity,
                title+(': hardening' if quantity=='DSY' else ': converted TTS'),cv)
    for ext in ['png','svg']:
        fig.savefig(OUTPUT/f'dsy_and_tts_fixed_fullfit_and_cv_parity.{ext}',dpi=300,bbox_inches='tight')
    plt.close(fig)
    for points,method,prefix,cv in [(full,FULL_METHOD,'fullfit',False),(oof,CV_METHOD,'cv',True)]:
        fig,axes = plt.subplots(1,2,figsize=(13.5,6.8),constrained_layout=True)
        for ax,quantity in zip(axes,['DSY','TTS']):
            title = ('Full fit' if not cv else '5 × 5-fold CV')+(': hardening' if quantity=='DSY' else ': converted TTS')
            parity(ax,points,stats.loc[(method,quantity)],quantity,title,cv)
        fig.savefig(OUTPUT/f'{prefix}_dsy_and_tts_parity.png',dpi=300,bbox_inches='tight')
        plt.close(fig)
    print(OUTPUT/'figure2_fixed_fullfit_nine_coefficient_cv.png')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT, help='Directory containing the validation CSV outputs.')
    main(parser.parse_args().output)
