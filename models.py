import pandas as pd
import numpy as np
import math
import os
import joblib
import tensorflow as tf
from mastml.models import EnsembleModel
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from scikeras.wrappers import KerasRegressor

import RPV_model_benchmarking
path = RPV_model_benchmarking.__path__[0]

class E900():
    # E900 MODEL

    # T = temp in deg C
    # PHI = fluence in n/m2 (NOT n/cm2)
    # TTS = transition temp shift in deg C
    # P, Ni, Mn, Cu are alloy fractions in weight percent

    def __init__(self):
        return

    def _get_e900_tts_partone(self, product_form, temp, fluence, P, Ni, Mn, Cu):
        if product_form == 'P':
            A = 1.080
        elif product_form == 'SRM':
            A = 1.080
        elif product_form == 'F':
            A = 1.011
        elif product_form == 'W':
            A = 0.919
        else:
            A = 1.080
        tts = A * (5 / 9) * (1.8943 * 10 ** -12) * ((fluence) ** 0.5695) * (((1.8 * temp + 32) / 550) ** -5.47) * (
                    (0.09 + (P / 0.012)) ** 0.216) * ((1.66 + ((Ni ** 8.54) / 0.63)) ** 0.39) * (Mn / 1.36) ** 0.3
        return tts


    def _get_e900_tts_parttwo(self, product_form, temp, fluence, P, Ni, Mn, Cu):
        if product_form == 'P':
            B = 0.819
        elif product_form == 'SRM':
            B = 0.819
        elif product_form == 'F':
            B = 0.738
        elif product_form == 'W':
            B = 0.968
        else:
            B = 0.819
        M = B * max([min([113.87 * (np.log(fluence) - np.log(4.5 * 10 ** 20)), 612.6]), 0]) * (
                    ((1.8 * temp + 32) / 550) ** -5.45) * ((0.1 + (P / 0.012)) ** -0.098) * (
                        0.168 + ((Ni ** 0.58) / 0.63)) ** 0.73
        tts = (5 / 9) * max([min([Cu, 0.28]) - 0.053, 0]) * M
        return tts


    def _get_e900_tts(self, product_form, temp, fluence, P, Ni, Mn, Cu):
        tts1 = self._get_e900_tts_partone(product_form, temp, fluence, P, Ni, Mn, Cu)
        tts2 = self._get_e900_tts_parttwo(product_form, temp, fluence, P, Ni, Mn, Cu)
        tts = tts1 + tts2
        return tts

    def _features(self):
        features = ['Product Form', 'temperature_C', 'wt_percent_Cu', 'wt_percent_Ni', 'wt_percent_Mn', 'wt_percent_P',
                    'flux_n_cm2_sec', 'fluence_n_cm2']
        return features

    def predict(self, df):
        features = self._features()
        df_features = df[features]
        e900_tts = list()
        for i, d in df_features.iterrows():
            tts = self._get_e900_tts(product_form=d['Product Form'],
                               temp=d['temperature_C'],
                               fluence=100 * 100 * d['fluence_n_cm2'],
                               P=d['wt_percent_P'],
                               Ni=d['wt_percent_Ni'],
                               Mn=d['wt_percent_Mn'],
                               Cu=d['wt_percent_Cu'])
            e900_tts.append(tts)
        df['E900 predicted TTS (degC)'] = e900_tts

        return np.array(e900_tts), df

class EONY():
    # EONY MODEL

    # T = temp in deg F (NOT deg C)
    # PHI = flux in n/cm2-sec
    # t = time in seconds
    # PHI*te = flux-adjusted effective fluence
    # TTS = transition temp shift in deg F (I guess???)
    # P, Ni, Mn, Cu are alloy fractions in weight percent

    def __init__(self):
        return

    def _get_eony_tts_partone(self, product_form, temp, flux, fluence, P, Ni, Mn, Cu):
        if product_form == 'P':
            A = 1.561 * 10 ** -7
        elif product_form == 'F':
            A = 1.140 * 10 ** -7
        elif product_form == 'W':
            A = 1.417 * 10 ** -7
        else:
            A = 1.561 * 10 ** -7

        if flux >= 4.39 * 10 ** 10:
            eff_flu = fluence
        else:
            eff_flu = fluence * ((4.39 * 10 ** 10) / flux) ** 0.259

        tts = A * (1 - 0.001718 * temp) * (1 + 6.13 * P * Mn ** 2.47) * np.sqrt(eff_flu)

        return tts

    def _get_eony_tts_parttwo(self, product_form, temp, flux, fluence, P, Ni, Mn, Cu):
        if product_form == 'PCE':
            B = 135.2
        elif product_form == 'P':
            B = 102.5
        elif product_form == 'SRM':
            B = 128.2
        elif product_form == 'F':
            B = 102.3
        elif product_form == 'W':
            B = 155.0
        elif product_form == 'W80':
            B = 155.0
        else:
            B = 128.2

        if product_form == 'W80':
            max_Cu_e = 0.243
        else:
            max_Cu_e = 0.301

        if Cu <= 0.072:
            Cu_e = 0
        else:
            Cu_e = min([Cu, max_Cu_e])

        if flux >= 4.39 * 10 ** 10:
            eff_flu = fluence
        else:
            eff_flu = fluence * ((4.39 * 10 ** 10) / flux) ** 0.259

        if Cu <= 0.072:
            func_Cu_e = 0
        elif Cu > 0.072:
            if P <= 0.008:
                func_Cu_e = (Cu_e - 0.072) ** 0.668
            else:
                func_Cu_e = (Cu_e - 0.072 + 1.359 * (P - 0.008)) ** 0.668

        gfunc_Cu_e = 0.5 + 0.5 * math.tanh((np.log10(eff_flu) + 1.139 * Cu_e - 0.448 * Ni - 18.120) / 0.629)

        tts = B * (1 + 3.77 * Ni ** 1.191) * func_Cu_e * gfunc_Cu_e

        return tts

    def _get_eony_tts(self, product_form, temp, flux, fluence, P, Ni, Mn, Cu):
        tts1 = self._get_eony_tts_partone(product_form, temp, flux, fluence, P, Ni, Mn, Cu)
        tts2 = self._get_eony_tts_parttwo(product_form, temp, flux, fluence, P, Ni, Mn, Cu)
        tts = (tts1 + tts2) * (5 / 9)
        return tts

    def _features(self):
        features = ['Product Form', 'temperature_C', 'wt_percent_Cu', 'wt_percent_Ni', 'wt_percent_Mn', 'wt_percent_P',
                    'flux_n_cm2_sec', 'fluence_n_cm2']
        return features

    def predict(self, df):
        features = self._features()
        df_features = df[features]
        eony_tts = list()
        for i, d in df_features.iterrows():
            tts = self._get_eony_tts(product_form=d['Product Form'],
                               temp=32 + (9 / 5) * d['temperature_C'],
                               flux=d['flux_n_cm2_sec'],
                               fluence=d['fluence_n_cm2'],
                               P=d['wt_percent_P'],
                               Ni=d['wt_percent_Ni'],
                               Mn=d['wt_percent_Mn'],
                               Cu=d['wt_percent_Cu'])
            eony_tts.append(tts)

        df['EONY predicted TTS (degC)'] = eony_tts
        return np.array(eony_tts), df

class OWAY():

    def __init__(self):
        return

    def atr2cf292(self, cu, ni, mn, si, p):
        #
        # ATR2 Chemistry factor at 292 and fte 1 to 1.4 x 10^20
        #
        # DYS=A + max(0,Cueff-Cumin)B+[max(0,Cueff-Cumin)C+D]*(Ni-0.75) +E*Mn+F*Si+G(1-H*max(0,Cueff-Cumin))*max(0, P-Pmin)
        # A = 127.538, B = 570.314, C = 504.807, D = 82.764, Cumin = 0.04, Cumax = 0.239, E = 20.69870768,
        # F = 24.83844922, G = 1481.190317, H = 3.730888874, Pmin = 0.004
        A2CF_A = 127.538
        A2CF_B = 570.314
        A2CF_C = 504.807
        A2CF_D = 82.764
        Cumin = 0.04
        Cumax = 0.239
        A2CF_E = 20.69870768
        A2CF_F = 24.83844922
        A2CF_G = 1481.190317
        A2CF_H = 3.730888874
        Pmin = 0.004
        cueff = min(cu, Cumax)
        a2cf = A2CF_A + A2CF_B * max(0, cueff - Cumin) + (A2CF_C * max(0, cueff - Cumin) + A2CF_D) * (
                    ni - 0.75) + A2CF_E * mn
        a2cf = a2cf + A2CF_F * si + A2CF_G * (1 - A2CF_H * max(0, cueff - Cumin)) * max(0, p - Pmin)
        return a2cf

    def atr2cfti(self, temp_C, dsy292):
        #
        # ATR CF adjusted for another temperature, temp_C
        # 1) First, get a DSY estimate for 255C using a polynomial fitting of DSY(255) vs DSY(292) data set
        #      ATR2 DSY(255) = CFT0 + CFT1 x DSY(292) + CFT2 x DSY(292)
        #       =A2FT20+A2FT21*BL55+A2FT22*BL55^2
        #      A2FT20	A2FT21	A2FT22
        #       0	1.407	-0.0005029
        # 2) Then, DSY(temp_C) is from linear interplation between DSY(255) and DSY(292)
        #
        a2ft20 = 0
        a2ft21 = 1.407
        a2ft22 = -0.0005029
        dsy255 = a2ft20 + a2ft21 * dsy292 + a2ft22 * dsy292 ** 2
        #    print(dsy255)
        a2cfti = dsy255 + (dsy255 - dsy292) * (temp_C - 255) / (255 - 292)
        return a2cfti

    def tts2dsy(self, pf, tts):
        # Converting EONY TTS to DSY using dsy = tts/cc
        # cc = IF(($K55="W")+($K55="W80"),MIN(_WCc3*BD55^3+_WCc2*BD55^2+_WCc1*BD55+_WCc0,WCcmax),MIN(_Cc3*BD55^3+_Cc2*BD55^2+_Cc1*BD55+Cc0,Ccmax)))
        # Plates	Cc3	Cc2	Cc1	Cc0				limit
        #	8.473E-09	-5.496E-06	1.945E-03	4.496E-01				0.7
        # WELDS	WCc3	WCc2	WCc1	WCc0				WCcmax
        #	0	-0.00000133	0.001197	0.55				0.8
        # predtts = (atr2tts-eonytts)/(atr2fte-4e19)*(fluence-4e19)+eonytts
        Cc3 = 8.473E-09
        Cc2	= -5.496E-06
        Cc1	= 1.945E-03
        Cc0	= 4.496E-01
        Ccmax = 0.7
        WCc3 = 0
        WCc2 = -0.00000133
        WCc1 = 0.001197
        WCc0 = 0.55
        WCcmax = 0.8
        if pf == 'W' or pf == 'W80':
            cc = min(WCc3*tts**3 + WCc2*tts**2 + WCc1*tts + WCc0, WCcmax)
        else:
            cc = min(Cc3*tts**3 + Cc2*tts**2 + Cc1*tts + Cc0, Ccmax)
        #print(cc)
        dsy = tts/cc
        return dsy, cc

    def _features(self):
        features = ['Product Form', 'temperature_C', 'wt_percent_Cu', 'wt_percent_Ni', 'wt_percent_Mn', 'wt_percent_Si',
                    'wt_percent_P', 'flux_n_cm2_sec', 'fluence_n_cm2']
        return features

    def predict(self, df, atr2fte=1.38e20):
        #
        #   OWAY model using linear interpolation between EONY DSY@4E19 and ATR2CF at ATR2fte
        #   for a desired fluence for prediction
        #   (ATRCF - eonydsy)/(atr2fte - eonyft)
        #   - EONY part can be replace with ML
        #   - Can use more recent cc = TTS/DSY
        #   - All can be done on TTS base instead of DSY done here
        #   - Temperature dependence can be replaced with ML
        #
        #tts_4e19 = eony_tts(pf, temp_c, wt_cu, wt_ni, wt_mn, wt_p, 4e19, flux)
        oway_tts = list()
        for i, d in df.iterrows():
            pf = d['Product Form']
            temp_c = d['temperature_C']
            wt_cu = d['wt_percent_Cu']
            wt_ni = d['wt_percent_Ni']
            wt_mn = d['wt_percent_Mn']
            wt_si = d['wt_percent_Si']
            wt_p = d['wt_percent_P']
            flux = d['flux_n_cm2_sec']
            fluence = d['fluence_n_cm2']
            df_eony = pd.DataFrame({'Product Form': [pf],
                               'temperature_C': [temp_c],
                               'wt_percent_Cu': [wt_cu],
                               'wt_percent_Ni': [wt_ni],
                               'wt_percent_Mn': [wt_mn],
                               'wt_percent_P': [wt_p],
                               'fluence_n_cm2': [4e19],
                               'flux_n_cm2_sec': [flux]})
            preds, _ = EONY().predict(df_eony)
            dsy_4e19, cc = self.tts2dsy(pf, preds)
            atr2dsy292 = self.atr2cf292(wt_cu, wt_ni, wt_mn, wt_si, wt_p)
            atr2dsyti = self.atr2cfti(temp_c, atr2dsy292)
            owaydsy = (atr2dsyti - dsy_4e19) / (atr2fte - 4e19) * (fluence - 4e19) + dsy_4e19
            owaytts = owaydsy*cc
            oway_tts.append(owaytts[0])
        df['OWAY predicted TTS (degC)'] = oway_tts
        return np.array(oway_tts), df

class JOWAY(OWAY):

    def __init__(self):
        super().__init__()

    def _features(self):
        features = ['Product Form', 'temperature_C', 'wt_percent_Cu', 'wt_percent_Ni', 'wt_percent_Mn', 'wt_percent_P',
                        'wt_percent_Si', 'wt_percent_C', 'log(fluence_n_cm2)', 'log(flux_n_cm2_sec)']
        return features

    def tts2dsy(self, pf, tts):
        # TTS = 0.00067*dSy**2 + 0.49*dSy
        a = 0.00067
        b = 0.49
        c = -1*tts
        dsy = (-b + np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
        #print(dsy)
        cc = tts/dsy
        return dsy, cc

    def predict(self, df, atr2fte=1.38e20, nn_model='Jacobs23'):
        #
        #   OWAY model using linear interpolation between EONY DSY@4E19 and ATR2CF at ATR2fte
        #   for a desired fluence for prediction
        #   (ATRCF - eonydsy)/(atr2fte - eonyft)
        #   - EONY part can be replace with ML
        #   - Can use more recent cc = TTS/DSY
        #   - All can be done on TTS base instead of DSY done here
        #   - Temperature dependence can be replaced with ML
        #
        #tts_4e19 = eony_tts(pf, temp_c, wt_cu, wt_ni, wt_mn, wt_p, 4e19, flux)
        models = {'Jacobs23': EnsembleNN_Jacobs23(), 'Jacobs24': EnsembleNN_Jacobs24(), 'Jacobs25': EnsembleNN_Jacobs25()}
        model = models[nn_model]
        oway_tts = list()
        atr2cf_preds = list()
        features = model._features()
        df_nn = df.drop(['log(fluence_n_cm2)'], axis=1)
        log_fluence = [np.log10(4e19) for i in range(df_nn.shape[0])]
        df_nn['log(fluence_n_cm2)'] = log_fluence
        if 'fluence_n_cm2' in features:
            df_nn = df.drop(['fluence_n_cm2'], axis=1)
            fluence = [4e19 for i in range(df_nn.shape[0])]
            df_nn['fluence_n_cm2'] = fluence

        df_nn = df_nn[features]
        preds, _ = model.predict(df_nn)
        for i, d in df.iterrows():
            pf = d['Product Form']
            temp_c = d['temperature_C']
            wt_cu = d['wt_percent_Cu']
            wt_ni = d['wt_percent_Ni']
            wt_mn = d['wt_percent_Mn']
            wt_p = d['wt_percent_P']
            wt_si = d['wt_percent_Si']
            fluence = 10**d['log(fluence_n_cm2)']
            log_flux = d['log(flux_n_cm2_sec)']

            dsy_4e19, cc = self.tts2dsy(pf, preds[i])
            atr2dsy292 = self.atr2cf292(wt_cu, wt_ni, wt_mn, wt_si, wt_p)
            atr2dsyti = self.atr2cfti(temp_c, atr2dsy292)
            owaydsy = (atr2dsyti - dsy_4e19) / (atr2fte - 4e19) * (fluence - 4e19) + dsy_4e19
            owaytts = owaydsy*cc
            oway_tts.append(owaytts)
            atr2cf_preds.append(atr2dsyti*cc)
        df['NN predicted TTS (degC) at 4e19 fluence'] = preds
        df['ATR2 CF TTS (degC) at 1.38e20 fluence'] = atr2cf_preds
        df['JOWAY predicted TTS (degC)'] = oway_tts
        return np.array(oway_tts), df

class GBR():
    # GBR MODEL (Diego params: https://www.mdpi.com/2075-4701/12/2/186)
    def __init__(self):
        return

    def _features(self):
        features = ['temperature_C', 'wt_percent_Cu', 'wt_percent_Ni', 'wt_percent_Mn', 'wt_percent_P', 'fluence_n_cm2',
                    'flux_n_cm2_sec', 'Product Form', 'Reactor Type']
        return features

    def predict(self, df):
        model_folder = os.path.join(path, 'model_files/GBR/fullfit/')
        features = self._features()
        df_features = df[features]

        pfs = np.array(df['Product Form'])
        rts = np.array(df['Reactor Type'])

        pf_0 = list()
        pf_1 = list()
        pf_2 = list()
        pf_3 = list()
        pf_4 = list()
        pf_5 = list()
        rt_0 = list()
        rt_1 = list()

        for i in pfs:
            if i == 'F':
                pf_0.append(1)
                pf_1.append(0)
                pf_2.append(0)
                pf_3.append(0)
                pf_4.append(0)
                pf_5.append(0)
            elif i == 'HAZ':
                pf_0.append(0)
                pf_1.append(1)
                pf_2.append(0)
                pf_3.append(0)
                pf_4.append(0)
                pf_5.append(0)
            elif i == 'P':
                pf_0.append(0)
                pf_1.append(0)
                pf_2.append(1)
                pf_3.append(0)
                pf_4.append(0)
                pf_5.append(0)
            elif i == 'SRM':
                pf_0.append(0)
                pf_1.append(0)
                pf_2.append(0)
                pf_3.append(1)
                pf_4.append(0)
                pf_5.append(0)
            elif i == 'W':
                pf_0.append(0)
                pf_1.append(0)
                pf_2.append(0)
                pf_3.append(0)
                pf_4.append(4)
                pf_5.append(0)
            elif i == 'PCE':
                pf_0.append(0)
                pf_1.append(0)
                pf_2.append(0)
                pf_3.append(0)
                pf_4.append(0)
                pf_5.append(1)

        for i in rts:
            if i == 'PWR':
                rt_0.append(0)
                rt_1.append(1)
            elif i == 'BWR':
                rt_0.append(1)
                rt_1.append(0)
            else:
                print('No home for', i)

        df_features['Product Form_0'] = pf_0
        df_features['Product Form_1'] = pf_1
        df_features['Product Form_2'] = pf_2
        df_features['Product Form_3'] = pf_3
        df_features['Product Form_4'] = pf_4
        df_features['Product Form_5'] = pf_5

        df_features['Reactor Type_0'] = rt_0
        df_features['Reactor Type_1'] = rt_1

        df_features = df_features.drop(['Product Form', 'Reactor Type'], axis=1)

        # Normalize the input features
        preprocessor = joblib.load(os.path.join(model_folder, 'StandardScaler.pkl'))

        # Get predictions and error bars from model
        model = joblib.load(os.path.join(model_folder, 'GradientBoostingRegressor.pkl'))
        preds = model.predict(preprocessor.transform(df_features))

        df['GBR predicted TTS (degC)'] = preds

        return preds, df

class GKRR():
    # GKRR MODEL (Yu-chen params: https://www.nature.com/articles/s41524-022-00760-4)
    def __init__(self):
        return

    def _features(self):
        features = ['temperature_C', 'log(fluence_n_cm2)', 'log(effective_fluence)', 'at_percent_Cu', 'at_percent_Ni',
                    'at_percent_Mn', 'at_percent_P', 'at_percent_Si', 'at_percent_C']
        return features

    def predict(self, df):
        model_folder = os.path.join(path, 'model_files/GKRR/fullfit/')
        features = self._features()
        df_features = df[features]

        # Normalize the input features
        preprocessor = joblib.load(os.path.join(model_folder, 'StandardScaler.pkl'))

        # Get predictions and error bars from model
        model = joblib.load(os.path.join(model_folder, 'KernelRidge.pkl'))
        preds = model.predict(preprocessor.transform(df_features))

        df['GKRR predicted TTS (degC)'] = preds

        return preds, df

class EnsembleNN_Jacobs23():
    # NN ENSEMBLE MODEL

    def __init__(self):
        return

    def _rebuild_model(self, n_features, model_folder):

        # We need to define the function that builds the network architecture
        def keras_model(n_features):
            model = Sequential()
            model.add(Dense(1024, input_dim=n_features, kernel_initializer='normal', activation='relu'))
            model.add(Dropout(0.3))
            model.add(Dense(1024, kernel_initializer='normal', activation='relu'))
            model.add(Dropout(0.3))
            model.add(Dense(1, kernel_initializer='normal'))
            model.compile(loss='mean_squared_error', optimizer='adam')

            return model

        model_keras = KerasRegressor(build_fn=keras_model, epochs=250, batch_size=100, verbose=0)
        model_bagged_keras_rebuild = EnsembleModel(model=model_keras, n_estimators=10)

        num_models = 10
        models = list()
        for i in range(num_models):
            models.append(tf.keras.models.load_model(os.path.join(model_folder, 'keras_model_' + str(i) + '.keras')))

        model_bagged_keras_rebuild.model.estimators_ = models
        model_bagged_keras_rebuild.model.estimators_features_ = [np.arange(0, n_features) for i in models]

        return model_bagged_keras_rebuild

    def _get_preds_ebars(self, model, df_featurized, preprocessor, return_ebars=True):
        preds_each = list()
        ebars_each = list()

        df_featurized_scaled = preprocessor.transform(pd.DataFrame(df_featurized))

        if return_ebars == True:
            for i, x in df_featurized_scaled.iterrows():
                preds_per_data = list()
                for m in model.model.estimators_:
                    preds_per_data.append(m.predict(pd.DataFrame(x).T, verbose=0))  # pd.DataFrame(x).T
                preds_each.append(np.mean(preds_per_data))
                ebars_each.append(np.std(preds_per_data))

        else:
            preds_each = model.predict(df_featurized_scaled) # Can't seem to pass verbose=0 to EnsembleModel
            try:
                ebars_each = [np.nan for i in range(preds_each.shape[0])]
            except:
                ebars_each = [np.nan]

        if return_ebars == True:
            # Jacobs 23 model recalibration
            a = -0.041
            b = 2.041
            c = 3.124
            ebars_each_recal = a * np.array(ebars_each) ** 2 + b * np.array(ebars_each) + c
        else:
            ebars_each_recal = ebars_each

        return np.array(preds_each).ravel(), np.array(ebars_each_recal).ravel()


    def _features(self):
        features = ['temperature_C', 'wt_percent_Cu', 'wt_percent_Ni', 'wt_percent_Mn', 'wt_percent_P',
                        'wt_percent_Si', 'wt_percent_C', 'log(fluence_n_cm2)', 'log(flux_n_cm2_sec)']
        return features

    def predict(self, df, return_ebars=False):
        # model_name = Jacobs23, Jacobs24

        features = self._features()
        model_folder = os.path.join(path, 'model_files/Jacobs23/fullfit')

        df_features = df[features]

        # Rebuild the saved model
        n_features = df_features.shape[1]
        model = self._rebuild_model(n_features, model_folder)

        # Normalize the input features
        preprocessor = joblib.load(os.path.join(model_folder, 'StandardScaler.pkl'))

        # Get predictions and error bars from model
        preds, ebars = self._get_preds_ebars(model, df_features, preprocessor, return_ebars=return_ebars)

        pred_dict = {'Jacobs23 NN ensemble predicted TTS (degC)': preds,
                     'Jacobs23 NN ensemble error bars (degC)': ebars}

        for k, v in pred_dict.items():
            df[k] = v

        return preds, df

class EnsembleNN_Jacobs24():
    # NN ENSEMBLE MODEL

    def __init__(self):
        return

    def _rebuild_model(self, n_features, model_folder):

        # We need to define the function that builds the network architecture
        def keras_model(n_features):
            model = Sequential()
            model.add(Dense(1024, input_dim=n_features, kernel_initializer='normal', activation='relu'))
            model.add(Dropout(0.3))
            model.add(Dense(1024, kernel_initializer='normal', activation='relu'))
            model.add(Dropout(0.3))
            model.add(Dense(1, kernel_initializer='normal'))
            model.compile(loss='mean_squared_error', optimizer='adam')

            return model

        model_keras = KerasRegressor(build_fn=keras_model, epochs=250, batch_size=100, verbose=0)
        model_bagged_keras_rebuild = EnsembleModel(model=model_keras, n_estimators=10)

        num_models = 10
        models = list()
        for i in range(num_models):
            models.append(tf.keras.models.load_model(os.path.join(model_folder, 'keras_model_' + str(i) + '.keras')))

        model_bagged_keras_rebuild.model.estimators_ = models
        model_bagged_keras_rebuild.model.estimators_features_ = [np.arange(0, n_features) for i in models]

        return model_bagged_keras_rebuild

    def _get_preds_ebars(self, model, df_featurized, preprocessor, return_ebars=True):
        preds_each = list()
        ebars_each = list()

        df_featurized_scaled = preprocessor.transform(pd.DataFrame(df_featurized))

        if return_ebars == True:
            for i, x in df_featurized_scaled.iterrows():
                preds_per_data = list()
                for m in model.model.estimators_:
                    preds_per_data.append(m.predict(pd.DataFrame(x).T, verbose=0))  # pd.DataFrame(x).T
                preds_each.append(np.mean(preds_per_data))
                ebars_each.append(np.std(preds_per_data))

        else:
            preds_each = model.predict(df_featurized_scaled)  # Can't seem to pass verbose=0 to EnsembleModel
            try:
                ebars_each = [np.nan for i in range(preds_each.shape[0])]
            except:
                ebars_each = [np.nan]

        if return_ebars == True:
            #TODO: need to update these for final model!
            # Jacobs 23 model recalibration
            a = -0.041
            b = 2.041
            c = 3.124
            ebars_each_recal = a * np.array(ebars_each) ** 2 + b * np.array(ebars_each) + c
        else:
            ebars_each_recal = ebars_each

        return np.array(preds_each).ravel(), np.array(ebars_each_recal).ravel()

    def _features(self):
        features = ['temperature_C', 'wt_percent_Cu', 'wt_percent_Ni', 'wt_percent_Mn', 'wt_percent_P',
                    'wt_percent_Si', 'wt_percent_C', 'log(fluence_n_cm2)', 'log(flux_n_cm2_sec)', 'fluence_n_cm2',
                    'flux_n_cm2_sec', 'Time']
        return features

    def predict(self, df, return_ebars=False):
        features = self._features()
        model_folder = os.path.join(path, 'model_files/Jacobs24/fullfit')

        df_features = df[features]

        # Rebuild the saved model
        n_features = df_features.shape[1]
        model = self._rebuild_model(n_features, model_folder)

        # Normalize the input features
        preprocessor = joblib.load(os.path.join(model_folder, 'StandardScaler.pkl'))

        # Get predictions and error bars from model
        preds, ebars = self._get_preds_ebars(model, df_features, preprocessor, return_ebars=return_ebars)

        pred_dict = {'Jacobs24 NN ensemble predicted TTS (degC)': preds,
                     'Jacobs24 NN ensemble error bars (degC)': ebars}

        for k, v in pred_dict.items():
            df[k] = v

        return preds, df

class EnsembleNN_Jacobs25():
    # NN ENSEMBLE MODEL

    def __init__(self):
        return

    def _rebuild_model(self, n_features, model_folder):

        # We need to define the function that builds the network architecture
        def keras_model(n_features):
            model = Sequential()
            model.add(Dense(1024, input_dim=n_features, kernel_initializer='normal', activation='relu'))
            model.add(Dropout(0.3))
            model.add(Dense(1024, kernel_initializer='normal', activation='relu'))
            model.add(Dropout(0.3))
            model.add(Dense(1, kernel_initializer='normal'))
            model.compile(loss='mean_squared_error', optimizer='adam')

            return model

        model_keras = KerasRegressor(build_fn=keras_model, epochs=250, batch_size=100, verbose=0)
        model_bagged_keras_rebuild = EnsembleModel(model=model_keras, n_estimators=10)

        num_models = 10
        models = list()
        for i in range(num_models):
            models.append(
                tf.keras.models.load_model(os.path.join(model_folder, 'keras_model_' + str(i) + '.keras')))

        model_bagged_keras_rebuild.model.estimators_ = models
        model_bagged_keras_rebuild.model.estimators_features_ = [np.arange(0, n_features) for i in models]

        return model_bagged_keras_rebuild

    def _get_preds_ebars(self, model, df_featurized, preprocessor, return_ebars=True):
        preds_each = list()
        ebars_each = list()

        df_featurized_scaled = preprocessor.transform(pd.DataFrame(df_featurized))

        if return_ebars == True:
            for i, x in df_featurized_scaled.iterrows():
                preds_per_data = list()
                for m in model.model.estimators_:
                    preds_per_data.append(m.predict(pd.DataFrame(x).T, verbose=0))  # pd.DataFrame(x).T
                preds_each.append(np.mean(preds_per_data))
                ebars_each.append(np.std(preds_per_data))

        else:
            preds_each = model.predict(df_featurized_scaled)  # Can't seem to pass verbose=0 to EnsembleModel
            try:
                ebars_each = [np.nan for i in range(preds_each.shape[0])]
            except:
                ebars_each = [np.nan]

        if return_ebars == True:
            # TODO: need to update these for final model!
            # Jacobs 23 model recalibration
            a = -0.041
            b = 2.041
            c = 3.124
            ebars_each_recal = a * np.array(ebars_each) ** 2 + b * np.array(ebars_each) + c
        else:
            ebars_each_recal = ebars_each

        return np.array(preds_each).ravel(), np.array(ebars_each_recal).ravel()

    def _features(self):
        features = ['temperature_C', 'wt_percent_Cu', 'wt_percent_Ni', 'wt_percent_Mn', 'wt_percent_P',
                    'wt_percent_Si', 'wt_percent_C', 'log(fluence_n_cm2)', 'log(flux_n_cm2_sec)', 'fluence_n_cm2',
                    'flux_n_cm2_sec', 'Time']
        return features

    def predict(self, df, return_ebars=False):
        features = self._features()
        model_folder = os.path.join(path, 'model_files/Jacobs25/fullfit')

        df_features = df[features]

        # Rebuild the saved model
        n_features = df_features.shape[1]
        model = self._rebuild_model(n_features, model_folder)

        # Normalize the input features
        preprocessor = joblib.load(os.path.join(model_folder, 'StandardScaler.pkl'))

        # Get predictions and error bars from model
        preds, ebars = self._get_preds_ebars(model, df_features, preprocessor, return_ebars=return_ebars)

        pred_dict = {'Jacobs25 NN ensemble predicted TTS (degC)': preds,
                     'Jacobs25 NN ensemble error bars (degC)': ebars}

        for k, v in pred_dict.items():
            df[k] = v

        return preds, df