import numpy as np
import os
import xarray as xr
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import geopandas as gpd

import cartopy.crs as ccrs  # Projections list
import cartopy.feature as cfe # for coastlines
import cmocean # for ocean colour maps
import matplotlib
import matplotlib as mpl

### Define directories and files
dir_path = 'XXX'
dir_model_outputs = os.path.join(dir_path,'Model_Output/')
dir_data_jha = os.path.join(dir_path,'Datasets/Jamieson_HighlandA')
dir_plot = os.path.join(dir_path,'Plots/Metrics/')
#dir_plot = os.path.join(dir_path,'Plots/Metrics_Testing/')
dir_metric_output = os.path.join(dir_path,'Analysis_Output/')
dir_data_bm = os.path.join(dir_path,'Datasets/BedMachine')
filename_bedmachine4 = 'BedMachineAntarctica-v4.0.nc'
filepath_bedmachine4 = os.path.join(dir_data_bm,filename_bedmachine4)
dir_data_measures = os.path.join(dir_path,'Datasets/Measures_Boundaries')

#####################################################################################
### PROCESSING FUNCTIONS ###
#####################################################################################

def import_file(experiment):
      filename_fort92 = 'fort.92_'+ experiment +'.nc'
      file = os.path.join(dir_model_outputs,filename_fort92)
      dsetin = xr.open_dataset(file)
      nt = np.arange(len(dsetin.time))
      dsetlat = dsetin.isel(time=0).alatd
      dsetlon = dsetin.isel(time=0).alond
      return dsetin, nt, dsetlat, dsetlon

def process_file_maskvars(dsetin):
      dsetin['fract_masked'] = xr.where(dsetin['maskwater'] == 0, dsetin['fract'], np.nan)
      dsetin['fract_masked'] = xr.where(dsetin['h'] != 0, dsetin['fract_masked'], np.nan)
      dsetin['fract_masked'] = dsetin['fract_masked'].assign_attrs(long_name = 'basal non-frozen fraction (water/land = n/a)',
                                                                   units = '0-1')
      dsetin['heatb_masked'] = xr.where(dsetin['maskwater'] == 0, dsetin['heatb'], np.nan)
      dsetin['heatb_masked'] = xr.where(dsetin['h'] != 0, dsetin['heatb_masked'], np.nan)
      dsetin['heatb_masked'] = dsetin['heatb_masked'].assign_attrs(long_name = 'basal non-frozen fraction (water/land = n/a)',
                                                                   units = '0-1')
      dsetin['h_masked'] = xr.where(dsetin['maskwater'] == 0, dsetin['h'], np.nan)
      dsetin['h_masked'] = xr.where(dsetin['h'] != 0, dsetin['h_masked'], np.nan)
      dsetin['h_masked'] = dsetin['h_masked'].assign_attrs(long_name = 'ice thickness (grounded)',
                                                                   units = 'm')
      return dsetin

def create_ASB_shapes(filepath_bedmachine4, domain, highlandA_blocks_3031, low_cutoff=0):

      xr_bedmachine4 = xr.open_dataset(filepath_bedmachine4)
      xr_bedmachine4_shp = xr_bedmachine4.sel(y = slice(domain['scene_y_max'], domain['scene_y_min']),
                                        x = slice(domain['scene_x_min'], domain['scene_x_max']))
      xr_bedmachine4_shp['bed'].values[(xr_bedmachine4_shp["mask"] != 2).values] = np.nan

      # Filter bedmachine to only records below cutoff
      xr_bed = xr_bedmachine4_shp['bed']
      #plot_single_map_topo("Test_0", "Calculation Domains", xr_bed)
      print(xr_bed.count())
      df_bed = xr_bed.to_dataframe().reset_index()

      # Convert to GeoDataFrame and drop points within Highland A
      gdf_bed = gpd.GeoDataFrame(df_bed['bed'],
                           geometry=gpd.points_from_xy(df_bed.x,df_bed.y),
                           crs="EPSG:3031")
      
      highlandA_blocks_South = highlandA_blocks_3031[highlandA_blocks_3031['Area'] == 7024].geometry.values[0]
      highlandA_blocks_Central = highlandA_blocks_3031[highlandA_blocks_3031['Area'] == 9712].geometry.values[0]
      highlandA_blocks_North = highlandA_blocks_3031[highlandA_blocks_3031['Area'] == 7093].geometry.values[0]
      
      inHA_mask = gdf_bed.within(highlandA_blocks_South | highlandA_blocks_Central | highlandA_blocks_North)
      outHA_mask = ~gdf_bed.within(highlandA_blocks_South | highlandA_blocks_Central | highlandA_blocks_North)
      gdf_bed['bed_inHA'] = gdf_bed['bed'].where(inHA_mask, np.nan)
      gdf_bed['bed_outHA'] = gdf_bed['bed'].where(outHA_mask, np.nan)
      gdf_bed['bed_lowoutHA'] = gdf_bed['bed_outHA'].where(gdf_bed.bed_outHA < low_cutoff, np.nan)
      gdf_bed['x'] = gdf_bed['geometry'].x
      gdf_bed['y'] = gdf_bed['geometry'].y
      gdf_bed = gdf_bed.set_index(['y', 'x'])
      gdf_bed = gdf_bed.drop(['geometry'], axis=1)

      xr_bed_HA = xr.Dataset.from_dataframe(gdf_bed)
      print(xr_bed_HA.bed_inHA.count())
      print(xr_bed_HA.bed_outHA.count())
      print(xr_bed_HA.bed_lowoutHA.count())

      return xr_bed_HA

def regrid_shape_masks(dsetin, xr_bed_HA):
      # Convert model output to m (from km) and regrid mask to model grid scale
      dsetin['x1'] = dsetin['x1']*1000
      dsetin['y1'] = dsetin['y1']*1000
      X_coord_new = dsetin.x1.values
      Y_coord_new = dsetin.y1.values
      xr_bed_regrid = xr_bed_HA.reindex(x=X_coord_new, method='nearest')
      xr_bed_regrid = xr_bed_regrid.reindex(y=Y_coord_new, method='nearest')
      #plot_single_map_topo("Test_3_Regrid", "Calculation Domains on Model Grid", xr_bed_regrid.bed_lowoutHA)
      #plot_single_map_topo("Test_4_Regrid", "Calculation Domains on Model Grid", xr_bed_regrid.bed_inHA)
      df_bed_regrid = xr_bed_regrid.to_dataframe().reset_index()
      return df_bed_regrid, xr_bed_regrid

def preprocess_model_output(dsetin, var):
      # Setup data structures for model output variables
      xr_var = dsetin[var]
      #dsetin_t0 = dsetin.isel(time=0)
      #plot_single_map_model(opt_dict,"Test_5_Maskwater", "Maskwater (0=ground/ice, 1=ocean)", dsetin_t0.maskwater, 'maskwater')

      df_var = xr_var.to_dataframe().reset_index()

      # Join dataframes so have one with fract along with variable to show if in area
      df_all = pd.merge(df_var,df_bed_regrid,left_on=['x1','y1'],right_on=['x','y'])
      #plot_dist(df_low['fract'],"Test_6_Fract_Dist","Fract whole domain")

      return df_all

def calculate_summary_per_timestep(df_var, var, isfract):

      timesteps = df_var['time'].unique()
      nt = len(timesteps)
      if isfract == True:
            boundary = 0.5
      else:
            boundary = 500000
      
      # Create summary dataframe to store results
      df_summary = pd.DataFrame(columns=['Year','ncells_HA', 'ncells_ASB',
                                         'ncells_warm_HA', 'ncells_cold_HA', 'ncells_retreat_HA',
                                         'Prop_warm_HA', 'Prop_cold_HA', 'Prop_retreat_HA',
                                         'ncells_retreat_ASBlow', 'Prop_retreat_ASBlow',
                                         'ncells_retreat_ASB', 'Prop_retreat_ASB',
                                         'Mean_HA'])
      df_summary['Year'] = timesteps
      df_summary['ncells_ASB'] = ncells_ASBall
      df_summary['ncells_ASBlow'] = ncells_ASB
      df_summary['ncells_HA'] = ncells_HA

      # Loop through each timestep in output
      for i in range(nt):
            df_t = df_var[df_var['time'] == timesteps[i]]
            ncells_warm_HA = len(df_t.loc[(~np.isnan(df_t['bed_inHA'])) &
                                          (df_t[var]>boundary)])
            ncells_cold_HA = len(df_t.loc[(~np.isnan(df_t['bed_inHA'])) &
                                          (df_t[var]<boundary)])
            ncells_retreat_HA = len(df_t.loc[(~np.isnan(df_t['bed_inHA'])) &
                                             (np.isnan(df_t[var]))])
            ncells_retreat_ASBlow = len(df_t.loc[(~np.isnan(df_t['bed_lowoutHA'])) &
                                              (np.isnan(df_t[var]))])
            ncells_retreat_ASB = len(df_t.loc[(np.isnan(df_t[var]))])
            cells_noretreat_HA = df_t.loc[(~np.isnan(df_t['bed_inHA']))]

            df_summary['ncells_warm_HA'][i] = ncells_warm_HA
            df_summary['ncells_cold_HA'][i] = ncells_cold_HA
            df_summary['ncells_retreat_HA'][i] = ncells_retreat_HA

            df_summary['Prop_warm_HA'][i] = ncells_warm_HA/ncells_HA
            df_summary['Prop_cold_HA'][i] = ncells_cold_HA/ncells_HA
            df_summary['Prop_retreat_HA'][i] = ncells_retreat_HA/ncells_HA

            df_summary['ncells_retreat_ASBlow'][i] = ncells_retreat_ASBlow
            df_summary['Prop_retreat_ASBlow'][i] = ncells_retreat_ASBlow/ncells_ASB

            df_summary['ncells_retreat_ASB'][i] = ncells_retreat_ASB
            df_summary['Prop_retreat_ASB'][i] = ncells_retreat_ASB/ncells_ASBall

            df_summary['Mean_HA'][i] = np.mean(cells_noretreat_HA[var])

      df_summary = df_summary.astype(float)
      return df_summary

def calcuate_mean_fract_over_time(dsetin,xr_bed_regrid,domain):
      xr_fract = dsetin['fract']
      xr_inHA = xr_bed_regrid['bed_inHA'].rename({'y':'y1','x':'x1'})
      xr_fractinHA = xr.combine_by_coords([xr_fract,xr_inHA])
      xr_fractinHA['fract_inHA'] = xr.where(~np.isnan(xr_fractinHA['bed_inHA']), xr_fractinHA['fract'], np.nan)
      xr_fractinHA['fract_inHA_mean'] = xr_fractinHA['fract_inHA'].mean('time', skipna=True)
      xr_fractinHA = xr_fractinHA.sel(y1 = slice(domain['scene_y_min'], domain['scene_y_max']),
                                      x1 = slice(domain['scene_x_min'], domain['scene_x_max']))
      return xr_fractinHA

def calcuate_mean_heatb_over_time(dsetin,xr_bed_regrid,domain):
      xr_heatb = dsetin['heatb']
      xr_inHA = xr_bed_regrid['bed_inHA'].rename({'y':'y1','x':'x1'})
      xr_heatbinHA = xr.combine_by_coords([xr_heatb,xr_inHA])
      xr_heatbinHA['heatb_inHA'] = xr.where(~np.isnan(xr_heatbinHA['bed_inHA']), xr_heatbinHA['heatb'], np.nan)
      xr_heatbinHA['heatb_inHA_mean'] = xr_heatbinHA['heatb_inHA'].mean('time', skipna=True)
      xr_heatbinHA = xr_heatbinHA.sel(y1 = slice(domain['scene_y_min'], domain['scene_y_max']),
                                      x1 = slice(domain['scene_x_min'], domain['scene_x_max']))
      return xr_heatbinHA

#####################################################################################
### PLOTTING FUNCTIONS ###
#####################################################################################

def plot_single_map_topo(savename, title, xr_bedmachine):
     
    XX, YY = np.meshgrid(xr_bedmachine.x, xr_bedmachine.y)
    im_ratio = xr_bedmachine.y.size / xr_bedmachine.x.size

    topo_cmap_abs = topo_colourmap_abs()

    fig, ax = plt.subplots()
    ax.pcolormesh(XX, YY, xr_bedmachine, cmap = topo_cmap_abs, vmin = -2000, vmax = 1000)
    fig.colorbar(mappable = ax.collections[0], fraction=0.046*im_ratio, pad=0.04)
    highlandA_boundaries_3031.plot(ax=ax, color = 'k')
    ax.set_aspect('equal')
    #ax.set_title(title)
    plt.axis('off')
    #ax.set_xlabel("m")

    plt.savefig(os.path.join(dir_plot,savename+".png"),transparent=True)

def plot_single_map_model(opt_dict, savename, title, xr_ISMout, var, domain):

      vmin = opt_dict[var]['vmin']
      vmax = opt_dict[var]['vmax']
      norm = opt_dict[var]['norm']
      cmap = opt_dict[var]['cmap']
      if vmin == 0:
            cbar_ext = 'max'
      else:
            cbar_ext = 'both'

      XX, YY = np.meshgrid(xr_ISMout.x1, xr_ISMout.y1)
      im_ratio = xr_ISMout.y1.size / xr_ISMout.x1.size

      fig = plt.figure()
      ax = plt.subplot(projection=ccrs.Stereographic(central_longitude=0., central_latitude=-90.))
      ax.pcolormesh(XX, YY, xr_ISMout, cmap = cmap, vmin = vmin, vmax = vmax)
      fig.colorbar(mappable = ax.collections[0], fraction=0.046*im_ratio, pad=0.04)
      highlandA_boundaries_3031.plot(ax=ax, color = 'k')
      ant_groundingline_b.plot(ax=ax, color='k')
      ax.set_extent([domain['scene_x_min'], domain['scene_x_max'],
                  domain['scene_y_min'], domain['scene_y_max']],
                  ccrs.Stereographic(central_longitude=0., central_latitude=-90.))    
      ax.set_aspect('equal')
      ax.set_title(title)
      plt.axis('off')
      #ax.set_xlabel("m")

      plt.savefig(os.path.join(dir_plot,savename+".png"),transparent=False, dpi=1200)

def plot_dist(df, savename, title):
      fig, ax = plt.subplots()
      ax = df.plot.hist()
      plt.savefig(os.path.join(dir_plot,savename+".png"))

def plot_timeseries_collection(calc_var,expt_collection,expt_collection_label, test_fract):
      
      n = len(expt_collection)
      colour_range = mpl.cm.rainbow(np.linspace(0,1,n))
      colour_range = colour_range[::-1]

      fig, ax = plt.subplots()
      i = 0
      for key in expt_collection:
            filename = 'Summary_fract_' + expt_collection[key]['savename'] + '.csv'
            load_path = dir_metric_output + filename
            temp_df = pd.read_csv(load_path)
            time = -1*temp_df['Year']/1000000
            ax.plot(time,temp_df[calc_var], color=colour_range[i], label=expt_collection[key]['label'])
            i += 1

      if test_fract==True:
            ax.set_title(expt_collection_label, size='large')
            ax.set_ylabel('Proportion of warm-based ice')
      else:
            ax.set_title(expt_collection_label, size='large')
            ax.set_ylabel("Proportion of domain retreated")
      ax.set_xlabel("Ma")
      ax.set_xlim(time.max(),time.min())
      ax.margins(x=0)
      ax.set_ylim(0,1)
      ax.grid(True, linestyle='--', alpha=0.6)
      ax.legend()
      savepath = dir_plot + '/TS_{}_{}.png'.format(calc_var, expt_collection_label) 
      plt.savefig(savepath, transparent=False, dpi=1200)

def plot_experiment_area(df_summary,experiment, experiment_label, isFract):
      fig, ax = plt.subplots()
      time = -1*df_summary['Year']/1000000
      if isFract == True:
            ax.stackplot(time,
                   df_summary['Prop_warm_HA'],
                   df_summary['Prop_cold_HA'],
                   df_summary['Prop_retreat_HA'],
                   labels=['Warm-based ice','Cold-based ice','No ice'],
                   colors=['crimson','deepskyblue','lightgrey'],
                   alpha = 0.9)
            #ax.set_title(experiment + ' Basal Thermal regime (fract) on Highland A', size='large')
            ax.set_title(experiment_label, size='large')
      else:
            ax.stackplot(time,
                   df_summary['Prop_warm_HA'],
                   df_summary['Prop_cold_HA'],
                   df_summary['Prop_retreat_HA'],
                   labels=['High Basal heat','Low Basal heat','No ice'],
                   colors=['maroon','royalblue','lightgrey'],
                   alpha = 0.9)
            #ax.set_title(experiment + ' Basal Thermal regime (heatb) on Highland A', size='large')
            ax.set_title(experiment_label, size='large')
      ax.set_xlim(time.max(),time.min())
      ax.set_ylim(0,1)
      ax.set_xlabel("Ma")
      ax.set_ylabel("Proportion")
      ax.grid(True, linestyle='--', alpha=1)
      ax.legend()
      if isFract == True:
            savepath = dir_plot + '/TSArea_fract_{}.png'.format(experiment_label)
      else:
            savepath = dir_plot + '/TSArea_heatb_{}.png'.format(experiment_label)
      plt.savefig(savepath, transparent=False, dpi=1200)

def topo_colourmap_abs():
    cvals  = [-2000, -1500, -1000, -500, -1,
              0, 
              1, 250, 500, 750, 1000]
    colors = ["#001079", "#0023c1", "#006cef", "#47bdfc", "#bee6fa", 
              "#7eb3a1",
              "#2D6E2A", "#DFFBBC", "#AFAD48", "#7D5F18", "#481B02"]

    # Create a colormap object
    norm = plt.Normalize(np.min(cvals), np.max(cvals))
    tuples = list(zip(map(norm, cvals), colors))
    topo_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", tuples)

    return topo_cmap

#####################################################################################
### SETUP PARAMETERS ###
#####################################################################################

topo_cmap_abs = topo_colourmap_abs()
opt_dict = {'heatb':{'cmap':'RdBu_r', 'conversion':0, 'AC':'r',
                     'vmin':0, 'vmax':1000000, 'norm':None,
                     'mask_opt':False},
            'heatb_masked':{'cmap':'RdBu_r', 'conversion':0, 'AC':'r',
                     'vmin':0, 'vmax':1000000, 'norm':None,
                     'mask_opt':False},
            'heatb_inHA_mean':{'cmap':'RdBu_r', 'conversion':0, 'AC':'r',
                     'vmin':0, 'vmax':1000000, 'norm':None,
                     'mask_opt':False},
            'hb':{'cmap':topo_cmap_abs, 'conversion':0, 'AC':'r',
                  'vmin':-2000, 'vmax':1000, 'norm':None,
                  'mask_opt':False},
            'h':{'cmap':cmocean.cm.ice, 'conversion':0, 'AC':'r',
                   'vmin':0, 'vmax':3500, 'norm':None,
                   'mask_opt':True},
            'h_masked':{'cmap':cmocean.cm.ice, 'conversion':0, 'AC':'r',
                   'vmin':0, 'vmax':3500, 'norm':None,
                   'mask_opt':True},
            'fract':{'cmap':'coolwarm', 'conversion':0, 'AC':'r',
                     'vmin':0, 'vmax':1, 'norm':None,
                     'mask_opt':False},
            'fract_masked':{'cmap':'coolwarm', 'conversion':0, 'AC':'r',
                     'vmin':0, 'vmax':1, 'norm':None,
                     'mask_opt':False},
            'fract_inHA_mean':{'cmap':'coolwarm', 'conversion':0, 'AC':'r',
                     'vmin':0, 'vmax':1, 'norm':None,
                     'mask_opt':False},
            'maskwater':{'cmap':'coolwarm', 'conversion':0, 'AC':'r',
                     'vmin':0, 'vmax':1, 'norm':None,
                     'mask_opt':False}}

domain_ASB = {                  # Focussed ASB domain 
    'label': 'ASB',
    'scene_y_min': -1100000,    # Lower edge y-coord (m)
    'scene_y_max': -300000,     # Upper edge y-coord (m)
    'scene_x_min': 1600000,     # Left edge x-coord (m)
    'scene_x_max': 2600000,     # Right edge x-coord (m)
    }
domain_HighlandA = {            # Highland A (within ASB)
    'label': 'Highland A',
    'scene_y_min': -700000,    # Lower edge y-coord (m)
    'scene_y_max': -450000,     # Upper edge y-coord (m)
    'scene_x_min': 1900000,     # Left edge x-coord (m)
    'scene_x_max': 2200000,     # Right edge x-coord (m)
    }

#####################################################################################
### LOAD FILES & CREATE ASB SHAPES ###
#####################################################################################

highlandA_ridges = gpd.read_file(os.path.join(dir_data_jha,'RADARSATRidges/'))
highlandA_valleys = gpd.read_file(os.path.join(dir_data_jha,'RADARSATRidges/'))
highlandA_blocks = gpd.read_file(os.path.join(dir_data_jha,'PlateauBoundaries/'))
highlandA_blocks_3031 = highlandA_blocks.set_crs("EPSG:3031")
highlandA_boundaries_3031 = highlandA_blocks_3031.boundary
#highlandA_blocks_4326 = highlandA_blocks_3031.to_crs("EPSG:4326")
#highlandA_boundaries_4326 = highlandA_blocks_4326.boundary

ant_boundaries = gpd.read_file(os.path.join(dir_data_measures,'IceBoundaries_Antarctica_v02/'))
ant_groundingline = gpd.read_file(os.path.join(dir_data_measures,'GroundingLine_Antarctica_v02/'))
ant_boundaries_b = ant_boundaries.boundary
ant_groundingline_b = ant_groundingline.boundary

xr_bed_HA = create_ASB_shapes(filepath_bedmachine4, domain_ASB, highlandA_blocks_3031, low_cutoff=0)
#plot_single_map_topo("Test_1", "Calculation Domains", xr_bed_HA.bed_lowoutHA)
#plot_single_map_topo("Test_2", "Calculation Domains", xr_bed_HA.bed_inHA)

#####################################################################################
### DEFINE COLLECTIONS ###
#####################################################################################

# single runs
expt_dict_single = {'OC2MedHi':{'savename':'1-18n_t1test4_3km','label':'OC2 MedHiOff'}}

# 5km vs 3km test
expt_dict_3_5km_t1 = {'3km':{'savename':'1-18n_t1test4_3km','label':'3km MG1 MedHi'},
                      '5km':{'savename':'1-18n_t1test2_5km','label':'5km MG1 MedHi'}}

# Rae t1 5km runs
expt_dict_Rae_5km_t1_all = {'Off':{'savename':'1-38n_Off_t1_5km','label':'MG1 Off'},
                            'Low':{'savename':'1-43n_Low_t1_5km','label':'MG1 Low'},
                            'MedLo':{'savename':'1-44n_MedLo_t1_5km','label':'MG1 Medlo'},
                            'MedHi':{'savename':'1-45n_MedHi_t1_5km','label':'MG1 MedHi'},
                            'Max':{'savename':'1-46n_Max_t1_5km','label':'MG1 Max'}}

# Rae t2 5km runs
expt_dict_Rae_5km_t2_all = {'Off':{'savename':'1-38n_Off_t2_5km','label':'K1 Off'},
                            'Low':{'savename':'1-43n_Low_t2_5km','label':'K1 Low'},
                            'MedLo':{'savename':'1-44n_MedLo_t2_5km','label':'K1 Medlo'},
                            'MedHi':{'savename':'1-45n_MedHi_t2_5km','label':'K1 MedHi'},
                            'Max':{'savename':'1-46n_Max_t2_5km','label':'K1 Max'}}

# Rae t3 5km runs
expt_dict_Rae_5km_t3_all = {'Off':{'savename':'1-38n_Off_t3_5km','label':'G17 Off'},
                            'Low':{'savename':'1-43n_Low_t3_5km','label':'G17 Low'},
                            'MedLo':{'savename':'1-44n_MedLo_t3_5km','label':'G17 Medlo'},
                            'MedHi':{'savename':'1-45n_MedHi_t3_5km','label':'G17 MedHi'},
                            'Max':{'savename':'1-46n_Max_t3_5km','label':'G17 Max'}}

# Rae t4 5km runs
expt_dict_Rae_5km_t4_all = {'Off':{'savename':'1-38n_Off_t4_5km','label':'KM5 Off'},
                            'Low':{'savename':'1-43n_Low_t4_5km','label':'KM5 Low'},
                            'MedLo':{'savename':'1-44n_MedLo_t4_5km','label':'KM5 Medlo'},
                            'MedHi':{'savename':'1-45n_MedHi_t4_5km','label':'KM5 MedHi'},
                            'Max':{'savename':'1-46n_Max_t4_5km','label':'KM5 Max'}}

# Rae Off 5km runs
expt_dict_Rae_5km_Off = {'t1':{'savename':'1-38n_Off_t1_5km','label':'MG1 Off'},
                         't2':{'savename':'1-38n_Off_t2_5km','label':'K1 Off'},
                         't3':{'savename':'1-38n_Off_t3_5km','label':'G17 Off'},
                         't4':{'savename':'1-38n_Off_t4_5km','label':'KM5 Off'}}

# Rae Low 5km runs
expt_dict_Rae_5km_Low = {'t1':{'savename':'1-43n_Low_t1_5km','label':'MG1 Low'},
                         't2':{'savename':'1-43n_Low_t2_5km','label':'K1 Low'},
                         't3':{'savename':'1-43n_Low_t3_5km','label':'G17 Low'},
                         't4':{'savename':'1-43n_Low_t4_5km','label':'KM5 Low'}}

# Rae MedLo 5km runs
expt_dict_Rae_5km_MedLo = {'t1':{'savename':'1-44n_MedLo_t1_5km','label':'MG1 MedLo'},
                           't2':{'savename':'1-44n_MedLo_t2_5km','label':'K1 MedLo'},
                           't3':{'savename':'1-44n_MedLo_t3_5km','label':'G17 MedLo'},
                           't4':{'savename':'1-44n_MedLo_t4_5km','label':'KM5 MedLo'}}

# Rae MedHi 5km runs
expt_dict_Rae_5km_MedHi = {'t1':{'savename':'1-45n_MedHi_t1_5km','label':'MG1 MedHi'},
                           't2':{'savename':'1-45n_MedHi_t2_5km','label':'K1 MedHi'},
                           't3':{'savename':'1-45n_MedHi_t3_5km','label':'G17 MedHi'},
                           't4':{'savename':'1-45n_MedHi_t4_5km','label':'KM5 MedHi'}}

# Rae Max 5km runs
expt_dict_Rae_5km_Max = {'t1':{'savename':'1-46n_Max_t1_5km','label':'MG1 Max'},
                         't2':{'savename':'1-46n_Max_t2_5km','label':'K1 Max'},
                         't3':{'savename':'1-46n_Max_t3_5km','label':'G17 Max'},
                         't4':{'savename':'1-46n_Max_t4_5km','label':'KM5 Max'}}

# Lisiecki t1 5km runs
expt_dict_Lis_5km_t1_all = {'Off':{'savename':'1-55n_Off_t1_5km','label':'Lisiecki MG1 Off'},
                            'Low':{'savename':'1-56n_Low_t1_5km','label':'Lisiecki MG1 Low'},
                            'MedLo':{'savename':'1-57n_MedLo_t1_5km','label':'Lisiecki MG1 Medlo'},
                            'MedHi':{'savename':'1-58n_MedHi_t1_5km','label':'Lisiecki MG1 MedHi'},
                            'Max':{'savename':'1-59n_Max_t1_5km','label':'Lisiecki MG1 Max'}}

# Lisiecki t2 5km runs
expt_dict_Lis_5km_t2_all = {'Off':{'savename':'1-55n_Off_t2_5km','label':'Lisiecki K1 Off'},
                            'Low':{'savename':'1-56n_Low_t2_5km','label':'Lisiecki K1 Low'},
                            'MedLo':{'savename':'1-57n_MedLo_t2_5km','label':'Lisiecki K1 Medlo'},
                            'MedHi':{'savename':'1-58n_MedHi_t2_5km','label':'Lisiecki K1 MedHi'},
                            'Max':{'savename':'1-59n_Max_t2_5km','label':'Lisiecki K1 Max'}}

# Lisiecki t4 5km runs
expt_dict_Lis_5km_t4_all = {'Off':{'savename':'1-55n_Off_t4_5km','label':'Lisiecki KM5 Off'},
                            'Low':{'savename':'1-56n_Low_t4_5km','label':'Lisiecki KM5 Low'},
                            'MedLo':{'savename':'1-57n_MedLo_t4_5km','label':'Lisiecki KM5 Medlo'},
                            'MedHi':{'savename':'1-58n_MedHi_t4_5km','label':'Lisiecki KM5 MedHi'},
                            'Max':{'savename':'1-59n_Max_t4_5km','label':'Lisiecki KM5 Max'}}


#####################################################################################
### PRODUCE TIMESERIES PLOTS ###
#####################################################################################

expt_collection_label = 'MG1'
expt_collection = expt_dict_3_5km_t1

for key in expt_collection:
      # Select the experiment and import file
      experiment = expt_collection[key]['savename']
      experiment_label = expt_collection[key]['label']
      print("Loading ",experiment)
      dsetin, nt, dsetlat, dsetlon = import_file(experiment)
      dsetin = process_file_maskvars(dsetin)
      df_bed_regrid, xr_bed_regrid = regrid_shape_masks(dsetin, xr_bed_HA)
      ncells_ASBall = df_bed_regrid.bed.count()
      ncells_ASB = df_bed_regrid.bed_lowoutHA.count()
      ncells_HA = df_bed_regrid.bed_inHA.count()

      # Use masks to process the model output
      df_fract = preprocess_model_output(dsetin, "fract_masked")
      df_heatb = preprocess_model_output(dsetin, "heatb_masked")

      # Create a timeseries summary
      df_summary_fract = calculate_summary_per_timestep(df_fract, "fract_masked", True)
      df_summary_heatb = calculate_summary_per_timestep(df_heatb, "heatb_masked", False)
      savepath_f = dir_metric_output + '/Summary_fract_{}.csv'.format(experiment)
      savepath_h = dir_metric_output + '/Summary_heatb_{}.csv'.format(experiment)
      df_summary_fract.to_csv(savepath_f)
      df_summary_heatb.to_csv(savepath_h)
      plot_experiment_area(df_summary_fract,experiment, experiment_label, True)
      plot_experiment_area(df_summary_heatb,experiment, experiment_label, False)

plot_timeseries_collection('Mean_HA',expt_collection,expt_collection_label,test_fract = True)
plot_timeseries_collection('Prop_retreat_ASB',expt_collection,expt_collection_label, test_fract = False)

#####################################################################################
### PRODUCE TIME AVERAGED HIGHLAND A MAPS ###
#####################################################################################

expt_collection_label = 'MG1'
expt_collection = expt_dict_3_5km_t1

for key in expt_collection:
      # Select the experiment and import file
      experiment = expt_collection[key]['savename']
      experiment_label = expt_collection[key]['label']
      print("Loading ",experiment)
      dsetin, nt, dsetlat, dsetlon = import_file(experiment)
      dsetin = process_file_maskvars(dsetin)
      df_bed_regrid, xr_bed_regrid = regrid_shape_masks(dsetin, xr_bed_HA)
      xr_fractinHA = calcuate_mean_fract_over_time(dsetin,xr_bed_regrid,domain_HighlandA)
      xr_heatbinHA = calcuate_mean_heatb_over_time(dsetin,xr_bed_regrid,domain_HighlandA)
      savename_f = "MapMeanFract_" + experiment
      savename_h = "MapMeanHeatB_" + experiment
      title_f = experiment_label #+ ": Mean Fract over Highland A"
      title_h = experiment_label #+ ": Mean HeatB over Highland A"
      plot_single_map_model(opt_dict,savename_f, title_f, xr_fractinHA['fract_inHA_mean'], 'fract_inHA_mean', domain_HighlandA)
      plot_single_map_model(opt_dict,savename_h, title_h, xr_heatbinHA['heatb_inHA_mean'], 'heatb_inHA_mean', domain_HighlandA)

#####################################################################################
### PRODUCE TIME AVERAGED HIGHLAND A MAPS ACROSS T1-T4  ###
#####################################################################################

expt_collection_savename = 'Off_Rae'
expt_collection_label = 'Off'
expt_collection = expt_dict_Rae_5km_t3_all

for key in expt_collection:
      # Select the experiment and import file
      experiment = expt_collection[key]['savename']
      print("Loading ",experiment)
      dsetin, nt, dsetlat, dsetlon = import_file(experiment)
      dsetin = process_file_maskvars(dsetin)
      dsetin_dropped = dsetin[['fract','heatb']]
      dsetin_dropped['fract'].max()
      if key == 't1':
            dsetin_new = dsetin_dropped
      else:
            dsetin_new = xr.concat([dsetin_new,dsetin_dropped], dim="time")

df_bed_regrid, xr_bed_regrid = regrid_shape_masks(dsetin_new, xr_bed_HA)
xr_fractinHA = calcuate_mean_fract_over_time(dsetin_new,xr_bed_regrid,domain_HighlandA)
xr_heatbinHA = calcuate_mean_heatb_over_time(dsetin_new,xr_bed_regrid,domain_HighlandA)

savename_f = "MapMeanFract_acrosst1-4_" + expt_collection_savename
savename_h = "MapMeanHeatB_acrosst1-4_" + expt_collection_savename
title_f = expt_collection_label #+ ": Mean Fract over Highland A"
title_h = expt_collection_label #+ ": Mean HeatB over Highland A"

plot_single_map_model(opt_dict,savename_f, title_f, xr_fractinHA['fract_inHA_mean'], 'fract_inHA_mean', domain_HighlandA)
plot_single_map_model(opt_dict,savename_h, title_h, xr_heatbinHA['heatb_inHA_mean'], 'heatb_inHA_mean', domain_HighlandA)

#####################################################################################
### MINIMUM EXTENT CALCULATIONS ###
#####################################################################################

expt_collection = expt_dict_Rae_5km_Off

for key in expt_collection:
      # Select the experiment and import file
      experiment = expt_collection[key]['savename']
      print("Loading ",experiment)
      loadpath_f = dir_metric_output + '/Summary_fract_{}.csv'.format(experiment)
      df_summary_fract = pd.read_csv(loadpath_f)
      max_val = df_summary_fract[['Prop_retreat_ASB']].max().values[0]
      max_id = df_summary_fract[['Prop_retreat_ASB']].idxmax()
      max_year = df_summary_fract['Year'].iloc[max_id].values[0]
      if key == 't1':
            max_val_tot = max_val
            max_id_tot = max_id
            max_year_tot = max_year
            max_t = key
      else:
            if max_val > max_val_tot:
                  max_val_tot = max_val
                  max_id_tot = max_id
                  max_year_tot = max_year
                  max_t = key
print(max_t, max_val_tot, max_id_tot, max_year_tot)

# For 'Max' maximum retreat
experiment = '1-46n_Max_t1_5km'
max_id_tot = 106
dsetin, nt, dsetlat, dsetlon = import_file(experiment)
dsetin = process_file_maskvars(dsetin)
dsetin['x1'] = dsetin['x1']*1000
dsetin['y1'] = dsetin['y1']*1000
dsetin_hmasked = dsetin['h_masked'].isel(time=max_id_tot)
dsetin_hmasked = dsetin_hmasked.drop_vars('time')
savename = "MapMaxRetreat_h_Max"
title = "Max" # (Rae CO2): Minimum Ice Extent for ASB domain"
plot_single_map_model(opt_dict,savename, title, dsetin_hmasked, 'h_masked', domain_ASB)

# For 'MedHi' maximum retreat
experiment = '1-45n_MedHi_t1_5km'
max_id_tot = 123
dsetin, nt, dsetlat, dsetlon = import_file(experiment)
dsetin = process_file_maskvars(dsetin)
dsetin['x1'] = dsetin['x1']*1000
dsetin['y1'] = dsetin['y1']*1000
dsetin_hmasked = dsetin['h_masked'].isel(time=max_id_tot)
dsetin_hmasked = dsetin_hmasked.drop_vars('time')
savename = "MapMaxRetreat_h_MedHi"
title = "MedHi" # (Rae CO2): Minimum Ice Extent for ASB domain"
plot_single_map_model(opt_dict,savename, title, dsetin_hmasked, 'h_masked', domain_ASB)

# For 'MedLo' maximum retreat
experiment = '1-44n_MedLo_t1_5km'
max_id_tot = 151
dsetin, nt, dsetlat, dsetlon = import_file(experiment)
dsetin = process_file_maskvars(dsetin)
dsetin['x1'] = dsetin['x1']*1000
dsetin['y1'] = dsetin['y1']*1000
dsetin_hmasked = dsetin['h_masked'].isel(time=max_id_tot)
dsetin_hmasked = dsetin_hmasked.drop_vars('time')
savename = "MapMaxRetreat_h_MedLo"
title = "MedLo" # (Rae CO2): Minimum Ice Extent for ASB domain"
plot_single_map_model(opt_dict,savename, title, dsetin_hmasked, 'h_masked', domain_ASB)

# For 'Low' maximum retreat
experiment = '1-43n_Low_t1_5km'
max_id_tot = 234
dsetin, nt, dsetlat, dsetlon = import_file(experiment)
dsetin = process_file_maskvars(dsetin)
dsetin['x1'] = dsetin['x1']*1000
dsetin['y1'] = dsetin['y1']*1000
dsetin_hmasked = dsetin['h_masked'].isel(time=max_id_tot)
dsetin_hmasked = dsetin_hmasked.drop_vars('time')
savename = "MapMaxRetreat_h_Low"
title = "Low" # (Rae CO2): Minimum Ice Extent for ASB domain"
plot_single_map_model(opt_dict,savename, title, dsetin_hmasked, 'h_masked', domain_ASB)

# For 'Off' maximum retreat
experiment = '1-38n_Off_t3_5km'
max_id_tot = 151
dsetin, nt, dsetlat, dsetlon = import_file(experiment)
dsetin = process_file_maskvars(dsetin)
dsetin['x1'] = dsetin['x1']*1000
dsetin['y1'] = dsetin['y1']*1000
dsetin_hmasked = dsetin['h_masked'].isel(time=max_id_tot)
dsetin_hmasked = dsetin_hmasked.drop_vars('time')
savename = "MapMaxRetreat_h_Off"
title = "Off" # (Rae CO2): Minimum Ice Extent for ASB domain"
plot_single_map_model(opt_dict,savename, title, dsetin_hmasked, 'h_masked', domain_ASB)

#####################################################################################
### SINGLE ANNOTATED AREA PLOT ###
#####################################################################################

expt_dict_single = {'MedHi':{'savename':'1-45n_MedHi_t1_5km','label':'MG1 MedHi'}}
experiment = expt_dict_single['MedHi']['savename']
experiment_label = expt_dict_single['MedHi']['label']

filename_f = 'Summary_fract_' + experiment + '.csv'
filename_h = 'Summary_heatb_' + experiment + '.csv'
load_path_f = dir_metric_output + filename_f
load_path_h = dir_metric_output + filename_h
df_summary_fract = pd.read_csv(load_path_f)
df_summary_heatb = pd.read_csv(load_path_h)

stage1_year = -3320000
stage2_year = -3314500
stage3_year = -3314100
stage4_year = -3313000
stage5_year = -3300000
stage1_id = df_summary_fract.index[df_summary_fract['Year']==stage1_year][0]
stage2_id = df_summary_fract.index[df_summary_fract['Year']==stage2_year][0]
stage3_id = df_summary_fract.index[df_summary_fract['Year']==stage3_year][0]
stage4_id = df_summary_fract.index[df_summary_fract['Year']==stage4_year][0]
stage5_id = df_summary_fract.index[df_summary_fract['Year']==stage5_year][0]

time_f = -1*df_summary_fract['Year']/1000000
time_h = -1*df_summary_heatb['Year']/1000000

stage1_year = -1*-3320000/1000000
stage2_year = -1*-3314500/1000000
stage3_year = -1*-3314100/1000000
stage4_year = -1*-3313000/1000000
stage5_year = -1*-3300000/1000000

fig, ax = plt.subplots()
ax.stackplot(time_f, df_summary_fract['Prop_warm_HA'],df_summary_fract['Prop_cold_HA'],df_summary_fract['Prop_retreat_HA'],
             labels=['Warm-based ice','Cold-based ice','No ice'],
             colors=['crimson','deepskyblue','lightgrey'], alpha = 0.9)
ax.plot(stage1_year, df_summary_fract['Prop_warm_HA'][stage1_id], marker = "x", color = 'black', markersize=5)
ax.plot(stage2_year, df_summary_fract['Prop_warm_HA'][stage2_id], marker = "x", color = 'black', markersize=5)
ax.plot(stage3_year, df_summary_fract['Prop_warm_HA'][stage3_id], marker = "x", color = 'black', markersize=5)
ax.plot(stage4_year, df_summary_fract['Prop_warm_HA'][stage4_id], marker = "x", color = 'black', markersize=5)
ax.plot(stage5_year, df_summary_fract['Prop_warm_HA'][stage5_id], marker = "x", color = 'black', markersize=5)
ax.set_title(experiment_label, size='large')
ax.set_xlim(time_f.max(),time_f.min())
ax.set_ylim(0,1)
ax.set_xlabel("Ma")
ax.set_ylabel("Proportion")
ax.grid(True, linestyle='--', alpha=1)
ax.legend()
savepath = dir_plot + '/TSAreaStages_fract_{}.png'.format(experiment_label)
plt.savefig(savepath, transparent=False, dpi=1200)

fig, ax = plt.subplots()
ax.stackplot(time_h, df_summary_heatb['Prop_warm_HA'], df_summary_heatb['Prop_cold_HA'],df_summary_heatb['Prop_retreat_HA'],
             labels=['High Basal heat','Low Basal heat','No ice'],
             colors=['maroon','royalblue','lightgrey'], alpha = 0.9)
ax.plot(stage1_year, df_summary_heatb['Prop_warm_HA'][stage1_id], marker = "x", color = 'black', markersize=5)
ax.plot(stage2_year, df_summary_heatb['Prop_warm_HA'][stage2_id], marker = "x", color = 'black', markersize=5)
ax.plot(stage3_year, df_summary_heatb['Prop_warm_HA'][stage3_id], marker = "x", color = 'black', markersize=5)
ax.plot(stage4_year, df_summary_heatb['Prop_warm_HA'][stage4_id], marker = "x", color = 'black', markersize=5)
ax.plot(stage5_year, df_summary_heatb['Prop_warm_HA'][stage5_id], marker = "x", color = 'black', markersize=5)
ax.set_title(experiment_label, size='large')
ax.set_xlim(time_h.max(),time_h.min())
ax.set_ylim(0,1)
ax.set_xlabel("Ma")
ax.set_ylabel("Proportion")
ax.grid(True, linestyle='--', alpha=1)
ax.legend()
savepath = dir_plot + '/TSAreaStages_heatb_{}.png'.format(experiment_label)
plt.savefig(savepath, transparent=False, dpi=1200)


#####################################################################################
### RETREAT STAGES (run with above section) ###
#####################################################################################

loadpath_f = dir_metric_output + '/Summary_fract_{}.csv'.format(experiment)
df_summary_fract = pd.read_csv(loadpath_f)

dsetin, nt, dsetlat, dsetlon = import_file(experiment)
dsetin = process_file_maskvars(dsetin)
dsetin['x1'] = dsetin['x1']*1000
dsetin['y1'] = dsetin['y1']*1000

dsetin_fract_masked = dsetin['fract_masked'].isel(time=stage1_id)
dsetin_fract_masked = dsetin_fract_masked.drop_vars('time')
dsetin_heatb_masked = dsetin['heatb_masked'].isel(time=stage1_id)
dsetin_heatb_masked = dsetin_heatb_masked.drop_vars('time')
savename_f = "MapStages_T1MedHi_S1_fract"
savename_h = "MapStages_T1MedHi_S1_heatb"
title = experiment_label + " Stage 1 (3.32 Ma)"
plot_single_map_model(opt_dict,savename_f, title, dsetin_fract_masked, 'fract_masked', domain_ASB)
plot_single_map_model(opt_dict,savename_h, title, dsetin_heatb_masked, 'heatb_masked', domain_ASB)

dsetin_fract_masked = dsetin['fract_masked'].isel(time=stage2_id)
dsetin_fract_masked = dsetin_fract_masked.drop_vars('time')
dsetin_heatb_masked = dsetin['heatb_masked'].isel(time=stage2_id)
dsetin_heatb_masked = dsetin_heatb_masked.drop_vars('time')
savename_f = "MapStages_T1MedHi_S2_fract"
savename_h = "MapStages_T1MedHi_S2_heatb"
title = experiment_label + " Stage 2 (3.3145 Ma)"
plot_single_map_model(opt_dict,savename_f, title, dsetin_fract_masked, 'fract_masked', domain_ASB)
plot_single_map_model(opt_dict,savename_h, title, dsetin_heatb_masked, 'heatb_masked', domain_ASB)

dsetin_fract_masked = dsetin['fract_masked'].isel(time=stage3_id)
dsetin_fract_masked = dsetin_fract_masked.drop_vars('time')
dsetin_heatb_masked = dsetin['heatb_masked'].isel(time=stage3_id)
dsetin_heatb_masked = dsetin_heatb_masked.drop_vars('time')
savename_f = "MapStages_T1MedHi_S3_fract"
savename_h = "MapStages_T1MedHi_S3_heatb"
title = experiment_label + " Stage 3 (3.3141 Ma)"
plot_single_map_model(opt_dict,savename_f, title, dsetin_fract_masked, 'fract_masked', domain_ASB)
plot_single_map_model(opt_dict,savename_h, title, dsetin_heatb_masked, 'heatb_masked', domain_ASB)

dsetin_fract_masked = dsetin['fract_masked'].isel(time=stage4_id)
dsetin_fract_masked = dsetin_fract_masked.drop_vars('time')
dsetin_heatb_masked = dsetin['heatb_masked'].isel(time=stage4_id)
dsetin_heatb_masked = dsetin_heatb_masked.drop_vars('time')
savename_f = "MapStages_T1MedHi_S4_fract"
savename_h = "MapStages_T1MedHi_S4_heatb"
title = experiment_label + " Stage 4 (3.313 Ma)"
plot_single_map_model(opt_dict,savename_f, title, dsetin_fract_masked, 'fract_masked', domain_ASB)
plot_single_map_model(opt_dict,savename_h, title, dsetin_heatb_masked, 'heatb_masked', domain_ASB)

dsetin_fract_masked = dsetin['fract_masked'].isel(time=stage5_id)
dsetin_fract_masked = dsetin_fract_masked.drop_vars('time')
dsetin_heatb_masked = dsetin['heatb_masked'].isel(time=stage5_id)
dsetin_heatb_masked = dsetin_heatb_masked.drop_vars('time')
savename_f = "MapStages_T1MedHi_S5_fract"
savename_h = "MapStages_T1MedHi_S5_heatb"
title = experiment_label + " Stage 5 (3.3 Ma)"
plot_single_map_model(opt_dict,savename_f, title, dsetin_fract_masked, 'fract_masked', domain_ASB)
plot_single_map_model(opt_dict,savename_h, title, dsetin_heatb_masked, 'heatb_masked', domain_ASB)

