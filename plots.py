import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy import stats

import RPV_model_benchmarking
path = RPV_model_benchmarking.__path__[0]

from RPV_model_benchmarking.models import XGBoost, EONY, E900, OWAY, JOWAY, GBR, GKRR, EnsembleNN_Jacobs23, EnsembleNN_Jacobs24, EnsembleNN_Jacobs25, EnsembleNN_Jacobs25_BWR, EnsembleNN_Jacobs25_PWR, EnsembleNN_Jacobs25_Bob, EnsembleNN_Jacobs26
models = {'EONY': EONY(), 'E900': E900(), 'OWAY': OWAY(), 'ML-OWAY': JOWAY(), 'GBR': GBR(), 'GKRR': GKRR(),
          'Jacobs23': EnsembleNN_Jacobs23(), 'Jacobs24': EnsembleNN_Jacobs24(), 'Jacobs25': EnsembleNN_Jacobs25(),
          'Jacobs25_BWR': EnsembleNN_Jacobs25_BWR(), 'Jacobs25_PWR': EnsembleNN_Jacobs25_PWR(),
          'Jacobs25_Bob': EnsembleNN_Jacobs25_Bob(),
          'XGBoost': XGBoost(), 'Jacobs26': EnsembleNN_Jacobs26()}
colors = {'Jacobs23': 'blue', 'Jacobs24': 'green', 'Jacobs25': 'red', 'Jacobs25_BWR': 'red', 'Jacobs25_PWR': 'red', 'Jacobs25_Bob': 'red', 'OWAY': 'purple', 'GBR': 'red', 'EONY': 'orange',
          'E900': 'grey', 'GKRR': 'black', 'ML-OWAY': 'blue', 'XGBoost': 'blue', 'Jacobs26': 'red'}
colors_list = ['blue', 'green', 'red', 'purple', 'orange', 'grey', 'black']
linestyles = ['-', '--', '-.', ':']

def get_effective_fluence(df, p=0.2):
    eff_fluence = df['fluence_n_cm2']*(3*10**10 / df['flux_n_cm2_sec'])**p
    df['effective_fluence'] = eff_fluence
    return df
def get_atom_percent(df):
    Fe = 55.845
    Cu = 63.546
    Ni = 58.693
    Mn = 54.938
    P = 30.974
    C = 12.011
    Si = 28.085
    w_Fe = 100 - df['wt_percent_Cu'] - df['wt_percent_Ni'] - df['wt_percent_Mn'] - df['wt_percent_Si'] - df['wt_percent_C'] - df['wt_percent_P']
    denom = w_Fe/Fe + df['wt_percent_Cu']/Cu + df['wt_percent_Ni']/Ni + df['wt_percent_Mn']/Mn + df['wt_percent_Si']/Si + df['wt_percent_C']/C + df['wt_percent_P']/P
    a_Cu = 100*((df['wt_percent_Cu']/Cu) / denom)
    a_Ni = 100 * ((df['wt_percent_Ni'] / Ni) / denom)
    a_Mn = 100 * ((df['wt_percent_Mn'] / Mn) / denom)
    a_P = 100 * ((df['wt_percent_P'] / P) / denom)
    a_C = 100 * ((df['wt_percent_C'] / C) / denom)
    a_Si = 100 * ((df['wt_percent_Si'] / Si) / denom)
    df['at_percent_Cu'] = a_Cu
    df['at_percent_Ni'] = a_Ni
    df['at_percent_Mn'] = a_Mn
    df['at_percent_P'] = a_P
    df['at_percent_Si'] = a_Si
    df['at_percent_C'] = a_C
    return df

def get_thermofeatures(df):
    D_Mn = (1.49*10**-4)*np.exp(-(234/(8.314*(273+df['temperature_C']))))
    D_Ni = (1.4 * 10 ** -4) * np.exp(-(245.7 / (8.314 * (273 + df['temperature_C']))))
    D_Si = (0.78 * 10 ** -4) * np.exp(-(231.5 / (8.314 * (273 + df['temperature_C']))))
    D_Fe = (27.5 * 10 ** -4) * np.exp(-(254 / (8.314 * (273 + df['temperature_C']))))
    D_Cu = (1.43 * 10 ** -7) * np.exp(-(184.28 / (8.314 * (273 + df['temperature_C']))))
    Cu_mins = list()
    ln_ksps = list()
    for i in df.iterrows():
        Cu_min = min(i[1]['at_percent_Cu'], 0.25)
        Cu_mins.append(Cu_min)
        ln_ksp = np.log((i[1]['at_percent_Ni']*i[1]['at_percent_Mn']*i[1]['at_percent_Si'])/ (2.4*10**-3))
        ln_ksps.append(ln_ksp)
    #print(df)
    #ln_ksp = np.log((df['at_percent_Ni']*df['at_percent_Mn']*df['at_percent_Si']) / 2.4*10**-3)
    Cu_sol = np.exp(1)*np.exp(-0.545/(8.6173*10**-5*(273+df['temperature_C'])))
    df['D_Mn (m2/s)'] = D_Mn
    df['D_Ni (m2/s)'] = D_Ni
    df['D_Si (m2/s)'] = D_Si
    df['D_Fe (m2/s)'] = D_Fe
    df['D_Cu (m2/s)'] = D_Cu
    df['Cu_min (at%)'] = Cu_mins
    df['Ln(Ksp/Kspbar)'] = ln_ksps
    df['Cu solubility'] = Cu_sol
    df['Cu/Cu solubility'] = df['at_percent_Cu'] / df['Cu solubility']
    return df

def get_avrami_features(df):
    opt_params = [908.7646764525934, 78.70579362285888, 11.793443181426348,
       -8.972747439739809, 1e+20, 1e+19, 3.162277660168379e+19,
       0.41511109433075516, 0.4215949101259248, 4.111020232421935]

    def model_contributions(params, df):
        A_1 = list()
        A_2 = list()
        A_3 = list()
        A_4 = list()
        TTS_pred = list()
        for i in df.iterrows():
            a1 = params[0] * np.array(i[1]['at_percent_Cu']) * (
                        1 - np.exp(-(np.array(i[1]['effective_fluence']) / params[4]) ** params[7]))
            a2 = params[1] * min(np.array(i[1]['at_percent_Ni']),
                                 np.array(i[1]['at_percent_Mn']) + np.array(i[1]['at_percent_Si'])) * (
                             1 - np.exp(-(np.array(i[1]['effective_fluence']) / params[5]) ** params[8]))
            a3 = params[2] * (1 - np.exp(-(np.array(i[1]['effective_fluence']) / params[6]) ** params[9]))
            a4 = params[3] * (np.exp(-(np.array(i[1]['effective_fluence']) / params[6]) ** params[9]))

            A_1.append(a1)
            A_2.append(a2)
            A_3.append(a3)
            A_4.append(a4)
            tts = a1 + a2 + a3 + a4
            TTS_pred.append(tts)
        return np.array(TTS_pred), np.array(A_1), np.array(A_2), np.array(A_3), np.array(A_4)

    pred_TTS, A1, A2, A3, A4 = model_contributions(opt_params, df)
    df['Avrami_TTS'] = pred_TTS
    df['Avrami_A1'] = A1
    df['Avrami_A2'] = A2
    df['Avrami_A3'] = A3
    df['Avrami_A4'] = A4
    return df

def plot_embrittlement_curve(df,
                             model_name_list,
                             flux_list,
                             style='log',
                             num_points=200,
                             ymax=350,
                             xmin=16.5,
                             xmax=21.5,
                             include_ATR2=False,
                             include_database_preds=True,
                             legendfontsize=5,
                             anchors=['2025'],
                             apply_smoothing=False,
                             smoothing_window=11,
                             smoothing_order=3,
                             use_alloy_flux=False,
                             joway_model='Jacobs26',
                             nn_ebars=False,
                             nn_domain=False,
                             domain_d=0.99):
    # style = log, linear, sqrt
    # model_name_list = list of model names, e.g., EONY, E900, OWAY, GBR, GKRR, Jacobs23, Jacobs24
    # flux_list = list of fluxes to evaluate each model at
    # This used to just iterate over all rows in df, but that repeats many alloys. Now, just iterate over unique alloys

    alloys = df['alloy'].unique()
    fluences = np.arange(16.5, xmax, (xmax - 16.5) / num_points)
    #for i, data in df.iterrows():
    for alloy in alloys:
        print('On alloy', alloy)
        plt.clf()
        model_count = 0
        data = df[df['alloy'] == alloy]
        if use_alloy_flux == True:
            flux_list = list()
            alloy_flux = np.mean(data['flux_n_cm2_sec'])
            flux_list.append(alloy_flux)
        dfs = list()
        for i, model_name in enumerate(model_name_list):
            model = models[model_name]
            if 'Jacobs25' in model_name:
                features = model._features(anchors=anchors[i])
            elif 'ML-OWAY' in model_name:
                features = model._features()
                model_nn = models[joway_model]
                features_nn = model_nn._features()
                for f in features_nn:
                    if f not in features:
                        features.append(f)
            else:
                features = model._features()

            flux_count = 0
            for flux in flux_list:
                d = dict()
                for f in features:
                    if f not in ['log(fluence_n_cm2)', 'fluence_n_cm2', 'flux_n_cm2_sec', 'log(flux_n_cm2_sec)',
                                'Time', 'log(effective_fluence)', 'D_Mn (m2/s)','D_Ni (m2/s)', 'D_Si (m2/s)', 'D_Fe (m2/s)',
                                'D_Cu (m2/s)', 'Cu_min (at%)', 'Ln(Ksp/Kspbar)', 'Cu solubility', 'Cu/Cu solubility',
                                'Avrami_TTS', 'Avrami_A1', 'Avrami_A2', 'Avrami_A3', 'Avrami_A4',
                                 'effective_fluence_p02', 'log(effective_fluence_p02)',
                                 'effective_fluence_p0', 'log(effective_fluence_p0)', 'effective_fluence_p005',
                                 'log(effective_fluence_p005)', 'effective_fluence_p01', 'log(effective_fluence_p01)',
                                 'EONY predicted TTS (degC)', 'E900 predicted TTS (degC)']:
                        d[f] = [data[f].iloc[0] for i in range(num_points)]
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
                        # Here other effective fluence features
                        elif f == 'effective_fluence_p02':
                            d[f] = 10**fluences*(3*10**10 / flux)**0.2
                        elif f == 'effective_fluence_p01':
                            d[f] = 10**fluences*(3*10**10 / flux)**0.1
                        elif f == 'effective_fluence_p005':
                            d[f] = 10**fluences*(3*10**10 / flux)**0.05
                        elif f == 'effective_fluence_p0':
                            d[f] = 10**fluences*(3*10**10 / flux)**0.0
                        elif f == 'log(effective_fluence_p02)':
                            d[f] = np.log10(10**fluences*(3*10**10 / flux)**0.2)
                        elif f == 'log(effective_fluence_p01)':
                            d[f] = np.log10(10**fluences*(3*10**10 / flux)**0.1)
                        elif f == 'log(effective_fluence_p005)':
                            d[f] = np.log10(10**fluences*(3*10**10 / flux)**0.05)
                        elif f == 'log(effective_fluence_p0)':
                            d[f] = np.log10(10**fluences*(3*10**10 / flux)**0.0)

                df_data = pd.DataFrame(d)

                if 'e900eony' in anchors:
                    df_data['fluence_n_cm2'] = 10**fluences
                    df_data['flux_n_cm2_sec'] = [flux for i in range(num_points)]
                    df_data['Product Form'] = [data['Product Form'].iloc[0] for i in range(num_points)]

                    _, df_data = EONY().predict(df_data)
                    _, df_data = E900().predict(df_data)
                    df_data = df_data.drop(['fluence_n_cm2', 'flux_n_cm2_sec', 'Product Form'], axis=1)

                if 'thermofeatures' in anchors:
                    df_data = get_atom_percent(df_data)
                    df_data = get_thermofeatures(df_data)
                    # Drop the atom percent columns as they aren't used as features
                    df_data = df_data.drop(['at_percent_Cu', 'at_percent_Ni', 'at_percent_Mn', 'at_percent_Si', 'at_percent_C', 'at_percent_P'], axis=1)

                if 'Avrami' in anchors:
                    df_data = get_atom_percent(df_data)
                    df_data = get_thermofeatures(df_data)
                    df_data = get_effective_fluence(df_data, p=0.2)
                    df_data = get_avrami_features(df_data)
                    # Drop the atom percent columns as they aren't used as features
                    df_data = df_data.drop(['effective_fluence', 'at_percent_Cu', 'at_percent_Ni', 'at_percent_Mn', 'at_percent_Si', 'at_percent_C', 'at_percent_P'], axis=1)

                # Make predictions
                if model_name == 'Jacobs25' or model_name == 'Jacobs26':
                    preds, df_data = model.predict(df_data, anchors=anchors[i], return_ebars=nn_ebars, return_domains=nn_domain)
                elif model_name == 'XGBoost':
                    preds, df_data = model.predict(df_data, anchors=anchors[i])
                elif model_name == 'ML-OWAY':
                    preds, df_data = model.predict(df_data, nn_model=joway_model)
                else:
                    preds, df_data = model.predict(df_data)
                dfs.append(df_data)

                # Also make ML preds of the data from the database and add to plot
                try:
                    if model_name == 'Jacobs25' or model_name == 'Jacobs26':
                        preds_data, _ = model.predict(data, anchors=anchors[i], return_ebars=nn_ebars, return_domains=nn_domain)
                    elif model_name == 'XGBoost':
                        preds_data, _ = model.predict(data, anchors=anchors[i])
                    elif model_name == 'ML-OWAY':
                        preds_data, _ = model.predict(data, nn_model=joway_model)
                    else:
                        preds_data, _ = model.predict(data)
                except:
                    preds_data = None

                # Here, apply smoothing to preds_data
                if apply_smoothing == True:
                    from scipy.signal import savgol_filter

                    # Choose window_length (odd) and polynomial order
                    window_length = smoothing_window  # must be odd, controls smoothness (larger = smoother)
                    polyorder = smoothing_order # usually 2 or 3 is fine

                    idx = np.argsort(fluences)
                    #fluence_sorted = fluences[idx]
                    y_pred_sorted = np.array(preds)[idx]

                    preds = savgol_filter(y_pred_sorted, window_length=window_length, polyorder=polyorder)

                if include_ATR2 == True:
                    temp_C = data['temperature_C'].iloc[0]
                    cu = data['wt_percent_Cu'].iloc[0]
                    ni = data['wt_percent_Ni'].iloc[0]
                    mn = data['wt_percent_Mn'].iloc[0]
                    si = data['wt_percent_Si'].iloc[0]
                    p = data['wt_percent_P'].iloc[0]
                    atr2_292 = OWAY().atr2cf292(cu, ni, mn, si, p)
                    atr2_T = OWAY().atr2cfti(temp_C, atr2_292)
                    atr2_tts = 0.00067*(atr2_T)**2 +0.49*atr2_T

                # Add alloy name to the plot
                alloy_comp = data['alloy'].iloc[0] + '_Cu_' + str(data['wt_percent_Cu'].iloc[0]) + '_Ni_' + str(data['wt_percent_Ni'].iloc[0]) + '_Mn_' + str(data['wt_percent_Mn'].iloc[0])
                if model_count == 0 and flux_count == 0:
                    plt.scatter([0], [0], s=0, c='white', label=alloy_comp)
                    if style == 'log':
                        try:
                            plt.scatter(data['log(fluence_n_cm2)'], data['Measured DT41J  [C]'], color='black', label='Database values', s=50)
                        except:
                            pass
                    if style == 'linear':
                        try:
                            plt.scatter(10 ** data['log(fluence_n_cm2)'], data['Measured DT41J  [C]'], color='black', label='Database values', s=50)
                        except:
                            pass
                    if style == 'sqrt':
                        try:
                            plt.scatter(np.sqrt(10 ** data['log(fluence_n_cm2)']), data['Measured DT41J  [C]'], color='black', label='Database values', s=50)
                        except:
                            pass
                    if include_ATR2 == True:
                        if style == 'log':
                            plt.scatter([np.log10(1.38*10**20)], [atr2_tts], color='red', label='ATR2 CF', s=50 )
                        if style == 'linear':
                            plt.scatter([1.38*10**20], [atr2_tts], color='red', label='ATR2 CF', s=50)
                        if style == 'sqrt':
                            plt.scatter([np.sqrt(1.38*10**20)], [atr2_tts], color='red', label='ATR2 CF', s=50)
                flux_label = label = f"{flux:.2e}"
                if style == 'log':
                    if include_database_preds == True and preds_data is not None and flux_count == 0:
                        plt.scatter(data['log(fluence_n_cm2)'], preds_data, color=colors[model_name], marker='^', s=50, label='Database predictions', alpha=0.2)
                    if model_name in ['Jacobs23', 'Jacobs26']:
                        #if i == 1:
                        color = colors[model_name]
                        #else:
                        #    color = 'blue'
                        plt.plot(fluences, preds, color=color, linestyle=linestyles[flux_count],
                                 label=model_name + ' ' + anchors[i] + ' Flux=' + str(flux_label))
                        if nn_ebars == True:
                            ebars = df_data[model_name+' NN ensemble error bars (degC)']
                            plt.fill_between(fluences, preds+ebars, preds-ebars, color=color, alpha=0.5)
                        if nn_domain == True:
                            domains = df_data['Jacobs26 NN ensemble domain d']
                            for i in range(fluences.shape[0]):
                                # Plot each point with its specific marker
                                if domains[i] > domain_d: #0.902
                                    marker = 'x'
                                else:
                                    marker = 'o'
                                plt.scatter(fluences[i], preds[i], marker=marker, color=colors[model_name], s=50, alpha=0.2)  # s is marker size

                    else:
                        plt.plot(fluences, preds, color=colors[model_name], linestyle=linestyles[flux_count], label=model_name + ' ' + 'Flux=' + str(flux_label))
                elif style == 'linear':
                    if include_database_preds == True and preds_data is not None and flux_count == 0:
                        plt.scatter(10**data['log(fluence_n_cm2)'], preds_data, color=colors[model_name], marker='^', s=50, label='Database predictions', alpha=0.2)
                    if model_name in ['Jacobs23', 'Jacobs26']:
                        #if i == 1:
                        color = colors[model_name]
                        #else:
                        #    color = 'blue'
                        plt.plot(10 ** fluences, preds, color=color, linestyle=linestyles[flux_count],
                                 label=model_name + ' ' + anchors[i] + ' Flux=' + str(flux_label))
                        if nn_ebars == True:
                            ebars = df_data[model_name+' NN ensemble error bars (degC)']
                            plt.fill_between(10**fluences, preds+ebars, preds-ebars, color=color, alpha=0.5)
                        if nn_domain == True:
                            domains = df_data['Jacobs26 NN ensemble domain d']
                            for i in range(fluences.shape[0]):
                                # Plot each point with its specific marker
                                if domains[i] > domain_d:
                                    marker = 'x'
                                else:
                                    marker = 'o'
                                plt.scatter(10**fluences[i], preds[i], marker=marker, color=colors[model_name], s=50, alpha=0.2)  # s is marker size

                    else:
                        plt.plot(10**fluences, preds, color=colors[model_name],  linestyle=linestyles[flux_count], label=model_name + ' ' + 'Flux=' + str(flux_label))
                elif style == 'sqrt':
                    if include_database_preds == True and preds_data is not None and flux_count == 0:
                        plt.scatter(np.sqrt(10**data['log(fluence_n_cm2)']), preds_data, color=colors[model_name], s=50, marker='^', label='Database predictions', alpha=0.2)
                    if model_name in ['Jacobs23', 'Jacobs26']:
                        #if i == 1:
                        color = colors[model_name]
                        #else:
                        #    color = 'blue'
                        plt.plot(np.sqrt(10 ** fluences), preds, color=color,
                                 linestyle=linestyles[flux_count], label=model_name + ' ' + anchors[i] + ' Flux=' + str(flux_label))
                        if nn_ebars == True:
                            ebars = df_data[model_name+' NN ensemble error bars (degC)']
                            plt.fill_between(np.sqrt(10**fluences), preds+ebars, preds-ebars, color=color, alpha=0.5)
                        if nn_domain == True:
                            domains = df_data['Jacobs26 NN ensemble domain d']
                            for i in range(fluences.shape[0]):
                                # Plot each point with its specific marker
                                if domains[i] > 0.902:
                                    marker = 'x'
                                else:
                                    marker = 'o'
                                plt.scatter(np.sqrt(10**fluences[i]), preds[i], marker=marker, color=colors[model_name], s=50, alpha=0.2)  # s is marker size

                    else:
                        plt.plot(np.sqrt(10**fluences), preds, color=colors[model_name],  linestyle=linestyles[flux_count], label=model_name + ' ' + 'Flux=' + str(flux_label))
                flux_count += 1
            model_count += 1

        # All models and fluxes plotted for this alloy
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        if style == 'log':
            plt.ylim(-10, ymax)
            plt.xlim(xmin, xmax)
            plt.ylabel('Predicted TTS ($\degree$C)', fontsize=10)
            plt.legend(loc='best', fontsize=legendfontsize)
            #plt.xlim(16.5, 20.30103)  # 2*10**20
            plt.xlabel('log fluence (n/cm$^2$)', fontsize=10)
            plt.savefig('embrittlement_curve_' + alloy_comp + '_logfluence.png', bbox_inches='tight', dpi=300)
        elif style == 'linear':
            plt.ylim(-10, ymax)
            plt.xlim(10**xmin, 10**xmax)
            plt.ylabel('Predicted TTS ($\degree$C)', fontsize=10)
            plt.legend(loc='best', fontsize=legendfontsize)
            #plt.xlim(0.0, 2.0 * 10 ** 20)  # 2*10**20
            plt.xlabel('fluence (n/cm$^2$)', fontsize=10)
            #plt.savefig('embrittlement_curve_' + alloy_comp + '_linearfluence.png', bbox_inches='tight', dpi=300)
            plt.savefig(alloy_comp + '_linearfluence.png', bbox_inches='tight', dpi=300)
        elif style == 'sqrt':
            plt.ylim(-10, ymax)
            plt.xlim(np.sqrt(10**xmin), np.sqrt(10**xmax))
            plt.ylabel('Predicted TTS ($\degree$C)', fontsize=10)
            plt.legend(loc='best', fontsize=legendfontsize)
            #plt.xlim(0.0, 1.41 * 10 ** 10)
            plt.xlabel('sqrt fluence (n/cm$^2$)', fontsize=10)
            plt.savefig('embrittlement_curve_' + alloy_comp + '_sqrtfluence.png', bbox_inches='tight', dpi=300)

        df_all = pd.concat(dfs)
        df_all.to_csv('embrittlement_curve_' + alloy_comp + '.csv', index=False)
    return df_all

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
                        d[f] = [data[f] for l in range(num_points)]
                    for f in features:
                        if f == grid_variable:
                            d[f] = [grid_value for l in range(num_points)]
                            if grid_variable == 'log(fluence_n_cm2)':
                                if 'fluence_n_cm2' in features:
                                    d['fluence_n_cm2'] = [10**grid_value for l in range(num_points)]
                            if grid_variable == 'log(flux_n_cm2_sec)':
                                if 'flux_n_cm2_sec' in features:
                                    d['flux_n_cm2_sec'] = [10**grid_value for l in range(num_points)]
                        elif f == crossplot_variable_name:
                            #print('here', crossplot_variable_name)
                            x_vals = np.arange(crossplot_variable_min, crossplot_variable_max, (crossplot_variable_max-crossplot_variable_min) / num_points)
                            #print(x_vals)
                            d[f] = x_vals

                    # Update these other features
                    if crossplot_variable_name == 'log(fluence_n_cm2)':
                        if 'fluence_n_cm2' in features:
                            d['fluence_n_cm2'] = 10**(x_vals)
                    if crossplot_variable_name == 'log(flux_n_cm2_sec)':
                        if 'flux_n_cm2_sec' in features:
                            d['flux_n_cm2_sec'] = 10**(x_vals)
                    # Re-do the Time feature, if applicable to the model
                    if 'Time' in features:
                        d['Time'] = np.array(d['fluence_n_cm2']) / np.array(d['flux_n_cm2_sec'])

                    df_pred = pd.DataFrame(d)

                    preds, df_pred = model.predict(df_pred)

                    plt.plot(x_vals, preds, color=colors[model_name],  linestyle=linestyles[val_iter], label=model_name+'_'+grid_variable+'_'+str(grid_value))

                    grid_variable_str = grid_variable+'_'+str(grid_value)
                    try:
                        alloy = data['alloy']
                    except:
                        alloy = 'test'
                    df_pred.to_csv('crossplot_' + alloy + '_' + crossplot_variable_name + '_' + grid_variable_str + '.csv', index=False)

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
        try:
            alloy = data['alloy']
        except:
            alloy = 'test'
        plt.savefig('crossplot_'+alloy+'_'+crossplot_variable_name+'_'+grid_variable_str+'.png', dpi=300, bbox_inches='tight')

    return

def plot_flux_effect_crossover_histogram(df, model_name_list, fluence=2*10**19, flux_plotter=3*10**10, flux_atr2=3.68*10**12, ymax=50, anchors='2025'):
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
    tts_diff_lst = list()
    for model_name in model_name_list:
        model = models[model_name]
        if model_name == 'Jacobs25' or model_name == 'Jacobs26':
            preds_plotter, _ = model.predict(df_plotter, anchors=anchors)
            preds_atr2, _ = model.predict(df_atr2, anchors=anchors)
        else:
            preds_plotter, _ = model.predict(df_plotter)
            preds_atr2, _ = model.predict(df_atr2)

        tts_diff = preds_plotter - preds_atr2
        tts_diff_lst.append(tts_diff)

        plt.hist(bins=np.arange(-80, 30, 1), x=tts_diff, edgecolor='black', color=colors[model_name], alpha=0.5,
                 label=model_name)
        plt.text(-80, ymax-0.1*ymax-5*model_count, model_name+' mean = ' + str(round(np.mean(tts_diff), 2)))
        model_count += 1

    plt.ylim(0, ymax)
    plt.xlabel('$\Delta$TTS ($\degree$C)', fontsize=14)
    plt.xticks(fontsize=12)
    plt.ylabel('Number of occurrences', fontsize=14)
    plt.yticks(fontsize=12)
    plt.legend(loc='best')
    plt.savefig('TTS_difference_histogram_fluence_'+str(fluence)+'.png', dpi=300, bbox_inches='tight')
    return tts_diff_lst

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