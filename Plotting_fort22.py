import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

dir_path = 'XXX'
dir_model_outputs = os.path.join(dir_path,'Model_Output/')
dir_plots = os.path.join(dir_path,'Plots/F22/')
dir_probstack = os.path.join(dir_path,'Datasets/Probstack_Ahn17')
filename_probstack = 'Prob_stack.txt' #https://github.com/seonminahn/HMM-Stack
dir_co2 = os.path.join(dir_path,'Datasets/Co2')

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
    ax.plot(-1*df['time']/1000000, df[var])#, label="40km Resolution (OC2 Medhi)")
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_xlabel("Ma")
    ax.set_xlim(3.4,2.9)
    ax.set_ylabel(var)
    #ax.set_ylabel('Equivalent Sea Level (m)')
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

filename_fort22 = 'fort.22_1-31c'
vars = ['facice', 'facorb', 'rco2', 'facco2', 'esle(m)']

file = os.path.join(dir_model_outputs,filename_fort22)
f22 = read_fort22(file)
for var in vars:
    savename = filename_fort22 + '_' + var
    plot_fort22(f22, savename, var)

#####################################################################################
### PLOT MULTIPLE FILES TOGETHER ESLE ###
#####################################################################################

filename_fort22_comp = ['fort.22_1-30c','fort.22_1-33c','fort.22_1-32c','fort.22_1-31c','fort.22_1-34c']
labels = ['Off','Low','MedLo','MedHi','Max']
var = 'esle(m)'
savename = 'comp_f22_2' + 'Rae_t1-4_' + var
title = 'Equivalent Sea Level from EAIS (m)'
plot_fort22_bespoke_annotations(filename_fort22_comp, labels, var, savename, title, True)

filename_fort22_comp = ['fort.22_1-50c_Off','fort.22_1-51c_Low','fort.22_1-52c_MedLo','fort.22_1-53c_MedHi','fort.22_1-54c_Max']
labels = ['Off','Low','MedLo','MedHi', 'Max']
var = 'esle(m)'
savename = 'comp_f22_' + 'Lis_t1-4_' + var
title = 'Equivalent Sea Level from EAIS (m) for Lisiecki Co2'
plot_fort22_bespoke_annotations(filename_fort22_comp, labels, var, savename, title, False)

#####################################################################################
### PLOT PROBSTACK ###
#####################################################################################

file = os.path.join(dir_probstack,filename_probstack)
df_probstack = pd.read_csv(file, sep='   ', header=None)
df_probstack.columns = ['Age','d18O','sd_d18O','upper95','lower95']
df_probstack['Age_Ma'] = df_probstack['Age']/1000

fig, ax = plt.subplots(figsize=(12,4))
ax.plot(df_probstack['Age_Ma'], df_probstack['d18O'], color='k', label='Prob-stack')
plt.fill_between(df_probstack['Age_Ma'],df_probstack['lower95'],df_probstack['upper95'], label='Lower to Upper bound of 95 $\%$ interval')
ax.axvspan(3.325, 3.300, alpha=0.3, color='gray')
ax.axvspan(3.215, 3.190, alpha=0.3, color='gray')
ax.axvspan(3.085, 3.060, alpha=0.3, color='gray')
ax.axvspan(2.970, 2.945, alpha=0.3, color='gray')
#ax.set_title("Probstack")
ax.grid(True, linestyle='--', alpha=0.6)
ax.set_xlabel("Ma")
ax.set_xlim(3.4,2.9)
ax.set_ylabel("$\\delta^{18}O$ (" + u"\u2030" + ")")#  ") #($%_o)")
ax.set_ylim(5,1.5)
ax.legend()

savename = 'Probstack_test2'
plt.savefig(dir_plots + savename + '.png', transparent=False, dpi=1200)

#####################################################################################
### PLOT f22 CO2 ###
#####################################################################################
#https://www.annualreviews.org/content/journals/10.1146/annurev-earth-082420-063026#supplementary_data

filename_fort22 = 'fort.22_1-31c'
var = 'rco2'
file = os.path.join(dir_model_outputs,filename_fort22)
df = read_fort22(file)
df['Age_Ma'] = -1*df['time']/1000000

fig, ax = plt.subplots(figsize=(12,4))
ax.scatter(df['Age_Ma'], df[var], color='k', label='rCO2')
ax.axvspan(3.325, 3.300, alpha=0.3, color='gray')
ax.axvspan(3.215, 3.190, alpha=0.3, color='gray')
ax.axvspan(3.085, 3.060, alpha=0.3, color='gray')
ax.axvspan(2.970, 2.945, alpha=0.3, color='gray')
ax.grid(True, linestyle='--', alpha=0.6)
ax.set_xlabel("Ma")
ax.set_xlim(3.4,2.9)
ax.set_ylabel("rCO2")
ax.set_ylim(1,2)
ax.legend()

savename = 'rCO2_test2'
plt.savefig(dir_plots + savename + '.png', transparent=False, dpi=1200)

#####################################################################################
### PLOT Rae CO2 ###
#####################################################################################

filename_rae = 'raeco2.txt'
file_rae = os.path.join(dir_co2,filename_rae)

# Note: this skips the two more contemporary data points at the top (as they have a different delimiter so was easier)
df_rae = pd.read_csv(file_rae, sep='\t', skiprows=2, header=None)
df_rae.columns = ['time', 'rco2_rae']
df_rae['time'] = df_rae['time']/1000

fig, ax = plt.subplots(figsize=(12,4))

#ax.plot(df_rae['time'], df_rae['rco2_rae'], label='pCO2 Rae 2021')
ax.plot(df_rae['time'], df_rae['rco2_rae'], marker = 'o', mec = 'k', label='pCO2 Rae 2021')
#ax.scatter(df_rae['time'], df_rae['rco2_rae'], marker = 'o', edgecolors = 'k', label='pCO2 Rae 2021')
ax.axvspan(3.325, 3.300, alpha=0.3, color='gray')
ax.axvspan(3.215, 3.190, alpha=0.3, color='gray')
ax.axvspan(3.085, 3.060, alpha=0.3, color='gray')
ax.axvspan(2.970, 2.945, alpha=0.3, color='gray')
ax.grid(True, linestyle='--', alpha=0.6)
#ax.set_ylabel("$\it{p}$CO2 (ppm)")
ax.set_ylabel("Atmospheric $CO_2$ (ppm)")
ax.set_xlabel('Ma')
ax.set_xlim(3.4,2.9)
ax.set_ylim(250,600)
plt.legend(loc="best")
plt.savefig(dir_plots + "CO2_Rae_MPWP_scatterline" + '.png')



