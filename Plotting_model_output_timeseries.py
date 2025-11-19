import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

dir_path = 'XXX'
dir_model_outputs = os.path.join(dir_path,'ISM_Output_Timeseries')
dir_plots = os.path.join(dir_path,'Plots/')

#####################################################################################
### FUNCTIONS ###
#####################################################################################

def read_fort22(file):
    headers=['time', 'weirun', 'ro18', 'sealev', 'dtanta', 'dtants', 'dtantj',
           'dtseas', 'rco2', 'ecc', 'obl', 'prec', 'facice', 'facorb', 'facco2',
           'toti(km3)', 'totig(km3)', 'totif(km3)', 'tota(km2)', 'totag(km2)',
           'totaf(km2)', 'h(m)', 'eofe(m)', 'eofw(m)', 'eof(m)', 'esle(m)',
           'eslw(m)', 'esl(m)']
    with open(file) as f22:
        contents = f22.read()
        if 'esl(m)' in contents: # check if headers in file or not
            df = pd.read_csv(file, sep='\s+', skiprows=2, comment='M')
        else:
            df = pd.read_csv(file, sep='\s+',  comment='M', header=None)
            df.columns = headers
    return df

def plot_fort22(df, savename, var):
    fig, ax = plt.subplots()
    ax.plot(-1*df['time']/1000000, df[var])
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_xlabel("Ma")
    ax.set_xlim(3.4,2.9)
    ax.set_ylabel(var)
    ax.set_title(savename)
    ax.legend()
    plt.savefig(dir_plots + savename + '.png', transparent=False, dpi=1200)

def plot_fort22_bespoke_annotations(filename_fort22_comp, labels, var, savename, title, doannotate):

    colour_range = mpl.cm.rainbow(np.linspace(0,1,len(filename_fort22_comp)))
    colour_range = colour_range[::-1]

    fig, ax = plt.subplots(figsize=(10,5))
    i = 0
    for i in range(len(filename_fort22_comp)):
        file = os.path.join(dir_model_outputs,filename_fort22_comp[i])
        df = read_fort22(file)
        time = -1*df['time']/1000000
        ax.plot(time, df[var], color=colour_range[i], label=labels[i])
        i += 1
    if doannotate == True:
        ax.axvspan(3.325, 3.300, alpha=0.3, color='gray')
        ax.axvspan(3.215, 3.190, alpha=0.3, color='gray')
        ax.axvspan(3.085, 3.060, alpha=0.3, color='gray')
        ax.axvspan(2.970, 2.945, alpha=0.3, color='gray')
    ax.set_title(title)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_xlabel("Ma")
    ax.set_xlim(3.4,2.9)
    ax.set_ylabel('Equivalent Sea Level (m)')
    ax.set_ylim(-5,25)
    ax.legend()

    plt.savefig(dir_plots + savename + '.png', transparent=False, dpi=1200)

#####################################################################################
### PLOTTING A SINGLE FILE ###
#####################################################################################

filename_fort22 = 'ISM_Output_fort.22_Rae_Off'
vars = ['rco2', 'esle(m)']

file = os.path.join(dir_model_outputs,filename_fort22)
f22 = read_fort22(file)
for var in vars:
    savename = filename_fort22 + '_' + var
    plot_fort22(f22, savename, var)

#####################################################################################
### PLOT MULTIPLE FILES TOGETHER ESLE ###
#####################################################################################

filename_fort22_comp = ['ISM_Output_fort.22_Rae_Off',
                        'ISM_Output_fort.22_Rae_Low',
                        'ISM_Output_fort.22_Rae_MedLo',
                        'ISM_Output_fort.22_Rae_MedHi',
                        'ISM_Output_fort.22_Rae_Max']
labels = ['Off','Low','MedLo','MedHi','Max']
var = 'esle(m)'
savename = 'comp_f22_2' + 'Rae_' + var
title = 'Equivalent Sea Level from EAIS (m)'
plot_fort22_bespoke_annotations(filename_fort22_comp, labels, var, savename, title, True)
