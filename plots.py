import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy import stats

import RPV_model_benchmarking
path = RPV_model_benchmarking.__path__[0]

from RPV_model_benchmarking.models import EONY, E900, OWAY, JOWAY, GBR, GKRR, EnsembleNN_Jacobs23, EnsembleNN_Jacobs24, EnsembleNN_Jacobs25
models = {'EONY': EONY(), 'E900': E900(), 'OWAY': OWAY(), 'JOWAY': JOWAY(), 'GBR': GBR(), 'GKRR': GKRR(),
          'Jacobs23': EnsembleNN_Jacobs23(), 'Jacobs24': EnsembleNN_Jacobs24(), 'Jacobs25': EnsembleNN_Jacobs25()}
colors = {'Jacobs23': 'blue', 'Jacobs24': 'green', 'Jacobs25': 'red', 'OWAY': 'purple', 'GBR': 'red', 'EONY': 'orange',
          'E900': 'grey', 'GKRR': 'black', 'JOWAY': 'pink'}
colors_list = ['blue', 'green', 'red', 'purple', 'orange', 'grey', 'black']
linestyles = ['-', '--', '-.', ':']

def plot_embrittlement_curve(df, model_name_list, flux_list, style='log', num_points=200, ymax=350, include_ATR2=False):
    # style = log, linear, sqrt
    # model_name_list = list of model names, e.g., EONY, E900, OWAY, GBR, GKRR, Jacobs23, Jacobs24
    # flux_list = list of fluxes to evaluate each model at

    fluences = np.arange(16.5, 21, (21 - 16.5) / num_points)
    for i, data in df.iterrows():
        plt.clf()
        model_count = 0
        for model_name in model_name_list:
            model = models[model_name]
            features = model._features()
            flux_count = 0
            for flux in flux_list:
                d = dict()
                for f in features:
                    if f not in ['log(fluence_n_cm2)', 'fluence_n_cm2', 'flux_n_cm2_sec', 'log(flux_n_cm2_sec)',
                                       'Time', 'log(effective_fluence)']:
                        d[f] = [data[f] for i in range(num_points)]
                    else:
                        if f == 'log(fluence_n_cm2)':
                            d[f] = fluences
                        elif f == 'fluence_n_cm2':
                            d[f] = 10**fluences
                        elif f == 'log(flux_n_cm2_sec)':
                            d[f] = [np.log10(flux) for i in range(num_points)]
                        elif f == 'flux_n_cm2_sec':
                            d[f] = [flux for i in range(num_points)]
                        elif f == 'Time':
                            d[f] = 10**fluences/np.array([flux for i in range(num_points)])
                        elif f == 'log(effective_fluence)':
                            d[f] = np.log10(10**fluences*(3*10**10 / flux)**0.2)

                df_data = pd.DataFrame(d)

                # Make predictions
                preds, df_data = model.predict(df_data)

                if include_ATR2 == True:
                    temp_C = data['temperature_C']
                    cu = data['wt_percent_Cu']
                    ni = data['wt_percent_Ni']
                    mn = data['wt_percent_Mn']
                    si = data['wt_percent_Si']
                    p = data['wt_percent_P']
                    atr2_292 = OWAY().atr2cf292(cu, ni, mn, si, p)
                    atr2_T = OWAY().atr2cfti(temp_C, atr2_292)
                    atr2_tts = 0.00067*(atr2_T)**2 +0.49*atr2_T

                # Add alloy name to the plot
                alloy_comp = data['alloy'] + '_Cu_' + str(data['wt_percent_Cu']) + '_Ni_' + str(data['wt_percent_Ni']) + '_Mn_' + str(data['wt_percent_Mn'])
                if model_count == 0 and flux_count == 0:
                    plt.scatter([0], [0], s=0, c='white', label=alloy_comp)
                    if include_ATR2 == True:
                        if style == 'log':
                            plt.scatter([np.log10(1.38*10**20)], [atr2_tts], color='red', label='ATR2 CF')
                        if style == 'linear':
                            plt.scatter([1.38*10**20], [atr2_tts], color='red', label='ATR2 CF')
                        if style == 'sqrt':
                            plt.scatter([np.sqrt(1.38*10**20)], [atr2_tts], color='red', label='ATR2 CF')
                if style == 'log':
                    alloy_data = df[df['alloy'] == data['alloy']]
                    plt.scatter(alloy_data['log(fluence_n_cm2)'], alloy_data['Measured DT41J  [C]'], color='black')
                    plt.plot(fluences, preds, color=colors[model_name], linestyle=linestyles[flux_count], label=model_name + ' ' + 'Flux=' + str(round(flux, 3)))
                elif style == 'linear':
                    alloy_data = df[df['alloy'] == data['alloy']]
                    plt.scatter(10**alloy_data['log(fluence_n_cm2)'], alloy_data['Measured DT41J  [C]'], color='black')
                    plt.plot(10**fluences, preds, color=colors[model_name],  linestyle=linestyles[flux_count], label=model_name + ' ' + 'Flux=' + str(round(flux, 3)))
                elif style == 'sqrt':
                    alloy_data = df[df['alloy'] == data['alloy']]
                    plt.scatter(np.sqrt(10**alloy_data['log(fluence_n_cm2)']), alloy_data['Measured DT41J  [C]'], color='black')
                    plt.plot(np.sqrt(10**fluences), preds, color=colors[model_name],  linestyle=linestyles[flux_count], label=model_name + ' ' + 'Flux=' + str(round(flux, 3)))
                flux_count += 1
            model_count += 1

        # All models and fluxes plotted for this alloy
        if style == 'log':
            plt.ylim(-10, ymax)
            plt.ylabel('Predicted TTS (degC)')
            plt.legend(loc='best', fontsize=5)
            plt.xlim(16.5, 20.30103)  # 2*10**20
            plt.xlabel('log fluence (n/cm2)')
            plt.savefig('embrittlement_curve_' + alloy_comp + '_logfluence.png', bbox_inches='tight', dpi=300)
        elif style == 'linear':
            plt.ylim(-10, ymax)
            plt.ylabel('Predicted TTS (degC)')
            plt.legend(loc='best', fontsize=5)
            plt.xlim(0.0, 2.0 * 10 ** 20)  # 2*10**20
            plt.xlabel('fluence (n/cm2)')
            plt.savefig('embrittlement_curve_' + alloy_comp + '_linearfluence.png', bbox_inches='tight', dpi=300)
        elif style == 'sqrt':
            plt.ylim(-10, ymax)
            plt.ylabel('Predicted TTS (degC)')
            plt.legend(loc='best', fontsize=5)
            plt.xlim(0.0, 1.41 * 10 ** 10)
            plt.xlabel('sqrt fluence (n/cm2)')
            plt.savefig('embrittlement_curve_' + alloy_comp + '_sqrtfluence.png', bbox_inches='tight', dpi=300)
    return

def plot_crossplot(df, model_name_list, crossplot_variable_name, crossplot_variable_min, crossplot_variable_max,
                   grid_variables, grid_values, num_points=50, ymax=100):
    '''
    # Cu cross plot:
    crossplot_variable = 'wt_percent_Cu'
    crossplot_variable_min = 0
    crossplot_variable_max = 0.8
    grid_variables = ['wt_percent_Ni', ...]
    grid_values = [[0.4, 0.8, 1.2], [...]]
    '''

    if len(grid_variables)>1 or len(grid_values) > 1:
        raise ValueError('Error, cross plots only support a single variable at this time')

    for i, data in df.iterrows():
        plt.clf()

        for model_name in model_name_list:
            model = models[model_name]
            features = model._features()
            var_iter = 0
            model_iter = 0
            for j, grid_variable in enumerate(grid_variables):
                val_iter = 0
                for k, grid_value in enumerate(grid_values[j]):
                    d = dict()
                    for f in features:
                        if f == grid_variable:
                            d[f] = [grid_value for l in range(num_points)]
                        elif f == crossplot_variable_name:
                            x_vals = np.arange(crossplot_variable_min, crossplot_variable_max, crossplot_variable_max / num_points)
                            d[f] = x_vals
                        else:
                            d[f] = [data[f] for l in range(num_points)]

                    df_pred = pd.DataFrame(d)

                    preds, df_pred = model.predict(df_pred)

                    plt.plot(x_vals, preds, color=colors[model_name],  linestyle=linestyles[val_iter], label=model_name+'_'+grid_variable+'_'+str(grid_value))

                    val_iter += 1

                var_iter += 1
                model_iter += 1

        plt.legend(loc='best', fontsize=8)
        plt.xlabel(crossplot_variable_name, fontsize=14)
        plt.xticks(fontsize=12)
        plt.ylabel('Predicted TTS (degC)', fontsize=14)
        plt.yticks(fontsize=12)

        plt.ylim(0, ymax)

        grid_variable_str = ''
        for j, grid_variable in enumerate(grid_variables):
            grid_variable_str += grid_variable
            for grid_value in grid_values[j]:
                grid_variable_str += '_'+str(grid_value)

        alloy = data['alloy']
        plt.savefig('crossplot_'+alloy+'_'+crossplot_variable_name+'_'+grid_variable_str+'.png', dpi=300, bbox_inches='tight')
        #df.to_csv('crossplot_Cu_fluence_' + str(fluence) + '_Ni_' + str(Ni) + '_DATA.csv', index=False)

    return

def plot_flux_effect_crossover_histogram(df, model_name_list, fluence=2*10**19, flux_plotter=3*10**10, flux_atr2=3.68*10**12):
    # Make histograms of the predicted TTS difference at Plotter and ATR2 flux conditions for each model, for all alloys
    # Fluences of interest are 2*10**19 (60 year), 2.7*10**19 (80 year), 1*10**20 (approx highest US reactor)
    fluences = np.array([float(fluence) for i in range(df.shape[0])])
    fluxes_plotter = np.array([float(flux_plotter) for i in range(df.shape[0])])
    fluxes_atr2 = np.array([float(flux_atr2) for i in range(df.shape[0])])

    df_plotter = df.drop(['fluence_n_cm2', 'log(fluence_n_cm2)', 'flux_n_cm2_sec', 'log(flux_n_cm2_sec)', 'Time', 'log(effective_fluence)'], axis=1)
    df_plotter['fluence_n_cm2'] = fluences
    df_plotter['log(fluence_n_cm2)'] = np.log10(fluences)
    df_plotter['flux_n_cm2_sec'] = fluxes_plotter
    df_plotter['log(flux_n_cm2_sec)'] = np.log10(fluxes_plotter)
    df_plotter['Time'] = fluences/fluxes_plotter
    df_plotter['log(effective_fluence)'] = np.log10(fluences*(3*10**10/fluxes_plotter)**0.2)

    df_atr2 = df.drop(['fluence_n_cm2', 'log(fluence_n_cm2)', 'flux_n_cm2_sec', 'log(flux_n_cm2_sec)', 'Time', 'log(effective_fluence)'], axis=1)
    df_atr2['fluence_n_cm2'] = fluences
    df_atr2['log(fluence_n_cm2)'] = np.log10(fluences)
    df_atr2['flux_n_cm2_sec'] = fluxes_atr2
    df_atr2['log(flux_n_cm2_sec)'] = np.log10(fluxes_atr2)
    df_atr2['Time'] = fluences / fluxes_atr2
    df_atr2['log(effective_fluence)'] = np.log10(fluences * (3 * 10 ** 10 / fluxes_atr2) ** 0.2)

    plt.clf()
    model_count = 0
    for model_name in model_name_list:
        model = models[model_name]
        preds_plotter, _ = model.predict(df_plotter)
        preds_atr2, _ = model.predict(df_atr2)

        tts_diff = preds_plotter - preds_atr2

        plt.hist(bins=np.arange(-80, 30, 1), x=tts_diff, edgecolor='black', color=colors[model_name], alpha=0.5,
                 label=model_name)
        plt.text(-80, 300-50*model_count, model_name+' mean = ' + str(round(np.mean(tts_diff), 3)))
        model_count += 1

    plt.xlabel('Delta Predicted TTS @ ATR2 fluence (Plotter flux - ATR2 flux curves) (degC)', fontsize=10)
    plt.xticks(fontsize=12)
    plt.ylabel('Number of occurrences', fontsize=14)
    plt.yticks(fontsize=12)
    plt.legend(loc='best')
    plt.savefig('TTS_difference_histogram_fluence_'+str(fluence)+'.png', dpi=300, bbox_inches='tight')
    return

def plot_residuals(trues, preds, name):
    res = preds-trues
    lin = LinearRegression()
    lin.fit(np.array(trues).reshape(-1, 1), np.array(res).reshape(-1, 1))
    slope = lin.coef_[0][0]
    intercept = lin.intercept_[0]
    plt.clf()
    plt.scatter(trues, res, s=25, alpha=0.5, color='blue')
    plt.plot(trues, slope * trues + intercept, 'k')
    plt.xlabel('True TTS (degC)', fontsize=14)
    plt.ylabel('Residuals (degC)', fontsize=14)
    plt.text(0, -40, 'Slope=' + str(round(slope, 3)))
    plt.text(0, -50, 'Intercept=' + str(round(intercept, 3)))
    plt.ylim(-60, 60)
    plt.xlim(-100, 375)
    plt.savefig('Res_vs_TTS_'+name+'.png', dpi=300, bbox_inches='tight')
    return

def plot_model_noise(trues, noise, num_points):
    # noise is stdev to add to true values, and 9 degC gets closest to observed spread from ML fits

    #size = 1853
    # noises = np.arange(0, 15, 1)
    def model(x):
        return x
        # return np.array([0 for i in x])
        # return 20*np.sqrt(x)

    eps = np.random.normal(loc=0.0, scale=noise, size=num_points)

    # Create a KDE object
    kde = stats.gaussian_kde(trues)

    # Evaluate the KDE on a grid of points
    x = np.linspace(50, 350, 100)
    y = kde(x)

    # Sample 50 points from the KDE
    samples = kde.resample(num_points)

    x = samples

    TTSmeas = model(x) + eps

    TTScalc = model(x)

    res = TTScalc - TTSmeas

    lin = LinearRegression()
    lin.fit(TTSmeas.reshape(-1, 1), res.reshape(-1, 1))
    slope = lin.coef_[0][0]
    intercept = lin.intercept_[0]

    plt.clf()
    plt.scatter(TTSmeas, res, s=25, alpha=0.5, color='blue')
    print(TTSmeas.shape)
    #print(slope * TTSmeas + intercept)
    plt.plot(TTSmeas[0], slope * TTSmeas[0] + intercept, 'k')
    plt.text(0, -40, 'Slope=' + str(round(slope, 3)))
    plt.text(0, -50, 'Intercept=' + str(round(intercept, 3)))
    plt.ylim(-60, 60)
    plt.xlim(-100, 375)
    plt.xlabel('True TTS (degC)', fontsize=14)
    plt.ylabel('Residuals (degC)', fontsize=14)
    plt.savefig('Res_vs_TTSmeas_synth_noise_' + str(noise) + '.png', dpi=300, bbox_inches='tight')
    return

def plot_best_worst_alloys(model_name, num_alloys, metric='RMSE', plot_type='best'):
    # Example
    # num_alloys = 25
    # metric = RMSE, MAE, ME
    # model_name = EONY, E900, GKRR, GBR, Jacobs23, Jacobs24
    # plot_type = best, worst

    # Get the 5fold df based on model name
    model_path = os.path.join(os.path.join(path, 'model_files'), model_name + '/5fold')

    if model_name == 'E900':
        df0 = pd.read_csv(os.path.join(model_path, 'e900_5fold_split_0.csv'))
        df1 = pd.read_csv(os.path.join(model_path, 'e900_5fold_split_1.csv'))
        df2 = pd.read_csv(os.path.join(model_path, 'e900_5fold_split_2.csv'))
        df3 = pd.read_csv(os.path.join(model_path, 'e900_5fold_split_3.csv'))
        df4 = pd.read_csv(os.path.join(model_path, 'e900_5fold_split_4.csv'))
        df = pd.concat([df0, df1, df2, df3, df4])
    elif model_name == 'EONY':
        df0 = pd.read_csv(os.path.join(model_path, 'eony_5fold_split_0.csv'))
        df1 = pd.read_csv(os.path.join(model_path, 'eony_5fold_split_1.csv'))
        df2 = pd.read_csv(os.path.join(model_path, 'eony_5fold_split_2.csv'))
        df3 = pd.read_csv(os.path.join(model_path, 'eony_5fold_split_3.csv'))
        df4 = pd.read_csv(os.path.join(model_path, 'eony_5fold_split_4.csv'))
        df = pd.concat([df0, df1, df2, df3, df4])
    else:
        X = pd.read_csv(os.path.join(model_path, 'X_test.csv'))
        Xextra = pd.read_csv(os.path.join(model_path, 'X_extra_test.csv'))
        ytrue = pd.read_csv(os.path.join(model_path, 'y_test.csv'))
        ypred = pd.read_csv(os.path.join(model_path, 'y_pred.csv'))
        df = pd.concat([ytrue, ypred, X, Xextra], axis=1)
        if model_name in ['Jacobs23', 'Jacobs24']:
            pred_name = 'NN ensemble predicted TTS (degC) ' + model_name
        else:
            pred_name = model_name + ' predicted TTS (degC)'
        df = df.rename(columns={'y_test': 'Measured DT41J  [C]', 'y_pred': pred_name})

    X_extra_plotter = df[df['datatype'] == 'Plotter']
    #trues = df['Measured DT41J  [C]']
    #preds = df[pred_name]
    trues_plotter = X_extra_plotter['Measured DT41J  [C]']
    preds_plotter = X_extra_plotter[pred_name]

    res_plotter = np.array(trues_plotter) - np.array(preds_plotter)
    X_extra_plotter['residuals'] = res_plotter
    X_extra_plotter['squared residuals'] = res_plotter ** 2
    df_data = X_extra_plotter[['alloy', 'squared residuals', 'residuals']]
    df_data['trues'] = trues_plotter
    df_data['preds'] = preds_plotter

    alloys = np.unique(df_data['alloy'])
    rmses = list()
    maes = list()
    mes = list()
    num_points = list()
    true_avg = list()
    pred_avg = list()
    for alloy in alloys:
        data = df_data[df_data['alloy'] == alloy]
        num_points.append(data.shape[0])
        rmses.append(np.sqrt(np.mean(data['squared residuals'])))
        maes.append(np.mean(abs(data['residuals'])))
        mes.append(np.mean(data['residuals']))
        true_avg.append(np.mean(data['trues']))
        pred_avg.append(np.mean(data['preds']))

    df_data_plot = pd.DataFrame(
        {'alloy': alloys, 'RMSE': rmses, 'MAE': maes, 'ME': mes, 'True avg': true_avg, 'Pred avg': pred_avg,
         'num_points': num_points})
    df_data_plot_sorted = df_data_plot.sort_values(by=metric)

    # Plot good alloys
    plt.clf()
    # x = alloys[0:25]
    if plot_type == 'best':
        x = np.array(df_data_plot_sorted['alloy'][0:num_alloys])
        y = np.array(df_data_plot_sorted[metric][0:num_alloys])
        counts = np.array(df_data_plot_sorted['num_points'][0:num_alloys])
    elif plot_type == 'worst':
        x = np.array(df_data_plot_sorted['alloy'][alloys.shape[0] - num_alloys:])
        y = np.array(df_data_plot_sorted[metric][alloys.shape[0] - num_alloys:])
        counts = np.array(df_data_plot_sorted['num_points'][alloys.shape[0] - num_alloys:])
    plt.scatter(x, y, c=counts, cmap='viridis')
    plt.ylabel('5-fold ' + metric + ' (degC)', fontsize=14)
    plt.yticks(fontsize=12)
    plt.xlabel('Plotter alloy', fontsize=14)
    plt.xticks(fontsize=12, rotation=90)

    shift = 0.025 * (max(y) - min(y))
    for i, j, k in zip(x, y, counts):
        plt.text(i, j + shift, str(k), fontsize=8)

    # Add a color bar
    cbar = plt.colorbar()

    # Set the colorbar label
    cbar.ax.set_ylabel('Number of data points', rotation=270, labelpad=15, fontsize=12)

    if plot_type == 'best':
        plt.savefig('alloy_errors_best_' + model_name + '_' + metric + '.png', dpi=300, bbox_inches='tight')
        return df_data_plot, df_data_plot_sorted.iloc[0:num_alloys]
    elif plot_type == 'worst':
        plt.savefig('alloy_errors_worst' + model_name + '_' + metric + '.png', dpi=300, bbox_inches='tight')
        return df_data_plot, df_data_plot_sorted.iloc[alloys.shape[0] - num_alloys:].sort_values(by=metric,
                                                                                                 ascending=False)