import pandas as pd
import numpy as np
import math
import os
import joblib
import tensorflow as tf
from tensorflow.keras import layers
from copy import copy
import dill

import io
import zipfile
import tempfile
import scikeras._saving_utils as scu

from mastml.models import EnsembleModel
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from scikeras.wrappers import KerasRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

import RPV_model_benchmarking
from RPV_model_benchmarking.data import *

path = RPV_model_benchmarking.__path__[0]
@keras.saving.register_keras_serializable(package="Custom")
class DerivativePenaltyModel(tf.keras.Model):
    """
    Graph-network model (has .inputs/.outputs) + custom train_step enforcing:
      - dy/d(fluence) >= 0  (penalize negative gradients)
      - dy/d(flux)    <= 0  (penalize positive gradients)
    """

    def __init__(
            self,
            input_dim: int,
            dropout_rate: float = 0.3,
            fluence_idx=(),
            flux_idx=(),
            alpha_fluence: float = 0.0,
            alpha_flux: float = 0.0,
            **kwargs,
    ):
        self.fluence_idx = tuple(int(i) for i in fluence_idx)
        self.flux_idx = tuple(int(i) for i in flux_idx)
        self.alpha_fluence = float(alpha_fluence)
        self.alpha_flux = float(alpha_flux)

        # Build graph network so SciKeras can inspect .inputs/.outputs
        inp = layers.Input(shape=(input_dim,), name="X")
        x = layers.Dense(1024, activation="relu")(inp)
        x = layers.Dropout(dropout_rate)(x)
        x = layers.Dense(1024, activation="relu")(x)
        x = layers.Dropout(dropout_rate)(x)
        out = layers.Dense(1, name="y")(x)

        super().__init__(inputs=inp, outputs=out, **kwargs)

    def train_step(self, data):
        # Support (x, y) or (x, y, sample_weight)
        if len(data) == 3:
            x, y_true, sample_weight = data
        else:
            x, y_true = data
            sample_weight = None

        x = tf.cast(x, tf.float32)
        y_true = tf.cast(y_true, tf.float32)

        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)

            y_pred = self(x, training=True)

            base_loss = self.compiled_loss(
                y_true, y_pred,
                sample_weight=sample_weight,
                regularization_losses=self.losses
            )

            # Input gradients for derivative penalties
            dy_dx = tape.gradient(y_pred, x)  # (batch, n_features)

            fluence_pen = tf.constant(0.0, dtype=tf.float32)
            flux_pen = tf.constant(0.0, dtype=tf.float32)

            # dy/d(fluence) >= 0 -> penalize negative slopes
            if self.fluence_idx and self.alpha_fluence > 0:
                g = tf.gather(dy_dx, indices=list(self.fluence_idx), axis=1)
                fluence_pen = tf.reduce_mean(tf.nn.relu(-g))

            # dy/d(flux) <= 0 -> penalize positive slopes
            if self.flux_idx and self.alpha_flux > 0:
                g = tf.gather(dy_dx, indices=list(self.flux_idx), axis=1)
                flux_pen = tf.reduce_mean(tf.nn.relu(g))

            total_loss = base_loss + self.alpha_fluence * fluence_pen + self.alpha_flux * flux_pen

        # Weight gradients
        grads = tape.gradient(total_loss, self.trainable_variables)
        del tape
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        # Keep Keras metrics updated (safe across keras versions)
        try:
            self.compiled_metrics.update_state(y_true, y_pred, sample_weight=sample_weight)
        except Exception:
            pass

        # Only return scalars (progbar-safe)
        mae = tf.reduce_mean(tf.abs(y_true - y_pred))
        return {
            "loss": total_loss,
            "base_loss": base_loss,
            "fluence_pen": fluence_pen,
            "flux_pen": flux_pen,
            "mae": mae,
        }

# Fixing issue with loading model.dill domain file for Jacobs26
# ------------------------------------------------------------
# 1) Rebuild the SAME architecture used inside model.dill
#    Adjust this if the domain model architecture differs.
# ------------------------------------------------------------
def build_domain_infer_model(input_dim=11):
    inp = tf.keras.layers.Input(shape=(input_dim,))
    x = tf.keras.layers.Dense(1024, activation="relu")(inp)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(1024, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    # Adjust final layer if needed:
    # regression / single score:
    out = tf.keras.layers.Dense(1)(x)

    # binary classifier alternative:
    # out = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    # multiclass alternative:
    # out = tf.keras.layers.Dense(n_classes, activation="softmax")(x)

    model = tf.keras.Model(inp, out, name="domain_infer_model")
    return model


# ------------------------------------------------------------
# 2) Monkeypatch SciKeras unpacker
# ------------------------------------------------------------

def patched_unpack_keras_model(packed_keras_model):
    """
    SciKeras stores a packed .keras archive in bytes.
    This function reconstructs a plain Functional model
    directly from model.weights.h5 inside that archive.
    """
    b = io.BytesIO(packed_keras_model)

    with zipfile.ZipFile(b, "r") as zf:
        names = zf.namelist()
        print("Packed archive contents:", names)

        with tempfile.TemporaryDirectory() as td:
            zf.extract("model.weights.h5", path=td)
            wpath = os.path.join(td, "model.weights.h5")

            # IMPORTANT: set the correct input dimension
            input_dim = 11

            model = build_domain_infer_model(input_dim=input_dim)

            # build variables before loading weights
            _ = model(tf.zeros((1, input_dim), dtype=tf.float32))

            model.load_weights(wpath)
            return model

class NaiveLinear():
    '''
    Naive estimator of per-plan TTS given 2 or more fluences, assuming linear scaling
    '''
    def __init__(self):
        alloys_multifluence = self._get_multifluence_alloys()
        self.alloys_multifluence = alloys_multifluence
        return

    def _get_multifluence_alloys(self):
        # Iterate over alloys and try to find alloys with 3 different fluences, used to test naive linear model and OWAY model
        df = DataLoader().load_rpv_data()
        alloys = df['alloy'].unique()

        alloys_multifluence = list()
        for alloy in alloys:
            df_sub = df[df['alloy'] == alloy]
            fluences = df_sub['log(fluence_n_cm2)']
            if fluences.unique().shape[0] >= 3:
                alloys_multifluence.append(alloy)

        return alloys_multifluence
    def predict(self, df, alloy, fluence, fit_log_scale=False, fluence_threshold=3e19):
        # Check the specified alloy is a multifluence alloy
        # Update: don't need this, just guess the average of available data if don't have 3 unique fluences
        if alloy not in self.alloys_multifluence:
            use_average_tts = True
        else:
            use_average_tts = False
            #raise ValueError('The specified alloy does not have 3 or more unique fluences, change to a suitable alloy')

        #print(alloy, use_average_tts)
        df_sub = df[df['alloy'] == alloy]
        # Invoke the fluence threshold for what points to include in linear fit
        df_sub = df_sub[df_sub['fluence_n_cm2'] >= fluence_threshold]
        fluences = df_sub['fluence_n_cm2']
        trues = df_sub['Measured DT41J  [C]']

        # Check the length again
        if df_sub.shape[0] < 2:
            use_average_tts = True

        # Sometimes there are multiple points at one fluence. For now, just average the true TTS
        x = list()
        y = list()
        for f in fluences.unique():
            if fit_log_scale == True:
                x.append(np.log10(f))
            else:
                x.append(f)
            y.append(np.mean(df_sub[df_sub['fluence_n_cm2'] == f]['Measured DT41J  [C]']))

        print(x, y)

        if use_average_tts == False:
            print('Doing linear')
            # Do linear full fit
            data = pd.DataFrame({'x': x, 'y': y})
            linear = LinearRegression().fit(np.array(data['x']).reshape(-1, 1), np.array(data['y']).reshape(-1, 1))
            if fit_log_scale == True:
                preds_data = linear.predict(np.log10(np.array(df_sub['fluence_n_cm2'])).reshape(-1, 1))
            else:
                preds_data = linear.predict(np.array(df_sub['fluence_n_cm2']).reshape(-1, 1))
            slope = linear.coef_[0]

            # If slope is positive, use linear model to predict the desired fluence. Otherwise, just use max value
            if slope > 0:
                print('Slope is positive')
                if fit_log_scale == True:
                    preds = linear.predict(np.array([np.log10(fluence)]).reshape(-1, 1))[0]
                else:
                    preds = linear.predict(np.array([fluence]).reshape(-1, 1))[0]
            else:
                print('Slope not positive, using max')
                preds = np.max(y)
        else:
            print('Use max TTS')
            preds_data = df_sub['Measured DT41J  [C]']
            try:
                preds = np.max(y)
            except:
                # Probably only one fluence passing the criteria, so just return max of rest of the data
                try:
                    preds = np.max(preds_data)
                except:
                    try:
                        preds = np.max(trues)
                    except:
                        #If no data, then nan
                        preds = np.nan

        df_sub['Naive linear predicted TTS (degC)'] = preds_data

        # Add a new line to the alloy df that contains the new fluence and its prediction
        df_pred = pd.DataFrame({'fluence_n_cm2': [fluence], 'Naive linear predicted TTS (degC)': preds})

        df_together = pd.concat([df_sub, df_pred])

        return preds, df_together

class NaiveLinear_OLD():
    '''
    Naive estimator of per-plan TTS given 2 or more fluences, assuming linear scaling
    '''
    def __init__(self):
        alloys_multifluence = self._get_multifluence_alloys()
        self.alloys_multifluence = alloys_multifluence
        return

    def _get_multifluence_alloys(self):
        # Iterate over alloys and try to find alloys with 3 different fluences, used to test naive linear model and OWAY model
        df = DataLoader().load_rpv_data()
        alloys = df['alloy'].unique()

        alloys_multifluence = list()
        for alloy in alloys:
            df_sub = df[df['alloy'] == alloy]
            fluences = df_sub['log(fluence_n_cm2)']
            if fluences.unique().shape[0] >= 3:
                alloys_multifluence.append(alloy)

        return alloys_multifluence
    def predict(self, df, alloy, fluence):
        # Check the specified alloy is a multifluence alloy
        # Update: don't need this, just guess the average of available data if don't have 3 unique fluences
        if alloy not in self.alloys_multifluence:
            use_average_tts = True
        else:
            use_average_tts = False
            #raise ValueError('The specified alloy does not have 3 or more unique fluences, change to a suitable alloy')

        print(alloy, use_average_tts)
        df_sub = df[df['alloy'] == alloy]
        fluences = df_sub['fluence_n_cm2']
        trues = df_sub['Measured DT41J  [C]']
        # Sometimes there are multiple points at one fluence. For now, just average the true TTS
        x = list()
        y = list()
        for f in fluences.unique():
            x.append(f)
            y.append(np.mean(df_sub[df_sub['fluence_n_cm2'] == f]['Measured DT41J  [C]']))

        print(x, y)

        if use_average_tts == False:
            print('Doing linear')
            # Do linear full fit
            data = pd.DataFrame({'x': x, 'y': y})
            linear = LinearRegression().fit(np.array(data['x']).reshape(-1, 1), np.array(data['y']).reshape(-1, 1))
            preds_data = linear.predict(np.array(df_sub['fluence_n_cm2']).reshape(-1, 1))
            slope = linear.coef_[0]

            # If slope is positive, use linear model to predict the desired fluence. Otherwise, just use average value
            if slope > 0:
                print('Slope is positive')
                preds = linear.predict(np.array([fluence]).reshape(-1, 1))[0]
            else:
                print('Slope not positive, using mean')
                preds = np.mean(y)
        else:
            print('Use average TTS')
            preds_data = df_sub['Measured DT41J  [C]']
            preds = np.mean(y)

        df_sub['Naive linear predicted TTS (degC)'] = preds_data

        # Add a new line to the alloy df that contains the new fluence and its prediction
        df_pred = pd.DataFrame({'fluence_n_cm2': [fluence], 'Naive linear predicted TTS (degC)': preds})

        df_together = pd.concat([df_sub, df_pred])

        return preds, df_together

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

    def tts2dsy_OLD(self, pf, tts):
        '''
        Below is old way from Takuya
        '''
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

    def tts2dsy(self, pf, tts):
        # Converting TTS to DSY using quadratic equation from Jacobs Mat & Des 2023
        # TTS = 0.00067*DSY**2 +0.49*DSY

        import cmath

        def positive_real_roots(a, b, c):
            """Calculate and return only the positive, real roots of ax^2 + bx + c = 0."""
            # Calculate the discriminant
            discriminant = b ** 2 - 4 * a * c

            # Compute the roots (could be complex)
            root1 = (-b + cmath.sqrt(discriminant)) / (2 * a)
            root2 = (-b - cmath.sqrt(discriminant)) / (2 * a)

            # Filter for roots that are real (imag part zero) and positive
            positive_real = []
            for root in (root1, root2):
                if abs(root.imag) < 1e-9 and root.real >= 0:
                    positive_real.append(root.real)

            return positive_real[0]

        if tts < 0:
            tts = 0

        dsy = positive_real_roots(a=0.00067, b=0.49, c=-1*tts)

        if tts == 0:
            cc = 1
        else:
            cc = tts/dsy

        return dsy, cc

    def _features(self):
        features = ['Product Form', 'temperature_C', 'wt_percent_Cu', 'wt_percent_Ni', 'wt_percent_Mn', 'wt_percent_Si',
                    'wt_percent_P', 'flux_n_cm2_sec', 'fluence_n_cm2']
        return features

    def predict(self, df, atr2fte=1.38e20, query_database=True, fluence_threshold=3e19, remove_highest_fluence=False):
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
        oway_r2 = list()
        oway_mae = list()
        oway_usedeony = list()
        if query_database == True:
            df_data = DataLoader().load_rpv_data()

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

            # Try to query database to get TTS of actual material for OWAY prediction, otherwise use EONY model
            if query_database == True:
                # Try to find the exact composition in the database
                filtered_df = df_data[(df_data['fluence_n_cm2'] > fluence_threshold) & (df_data['wt_percent_Cu'] == wt_cu) & (df_data['wt_percent_Ni'] == wt_ni) & (df_data['wt_percent_Mn'] == wt_mn) & (df_data['wt_percent_P'] == wt_p) & (df_data['wt_percent_Si'] == wt_si)]

                if filtered_df.shape[0] >= 1:
                    if remove_highest_fluence == True:
                        max_fluence = max(filtered_df['log(fluence_n_cm2)'])
                        filtered_df = filtered_df[filtered_df['log(fluence_n_cm2)'] != max_fluence]

                if filtered_df.shape[0] < 1:
                    oway_usedeony.append(True)
                    preds, _ = EONY().predict(df_eony)
                    dsy_4e19, cc = self.tts2dsy(pf, preds)
                    atr2dsy292 = self.atr2cf292(wt_cu, wt_ni, wt_mn, wt_si, wt_p)
                    atr2dsyti = self.atr2cfti(temp_c, atr2dsy292)
                    owaydsy = (atr2dsyti - dsy_4e19) / (atr2fte - 4e19) * (fluence - 4e19) + dsy_4e19
                    #owaytts = owaydsy * cc
                    owaytts = 0.00067*owaydsy**2 +0.49*owaydsy
                    try:
                        oway_tts.append(owaytts[0])
                    except:
                        oway_tts.append(owaytts)
                    oway_mae.append(np.nan)
                    oway_r2.append(np.nan)
                else:
                    # Get fit line of TTS vs. fluence and ATR2 point, then predict the prediction of interest back
                    # Just need this to get cc value
                    oway_usedeony.append(False)
                    preds_ = np.array([np.mean(filtered_df['Measured DT41J  [C]'])])
                    dsy_4e19, cc = self.tts2dsy(pf, preds_)
                    atr2dsy292 = self.atr2cf292(wt_cu, wt_ni, wt_mn, wt_si, wt_p)
                    atr2dsyti = self.atr2cfti(temp_c, atr2dsy292)
                    #atr2_tts = cc*atr2dsyti
                    atr2_tts = 0.00067*atr2dsyti**2 + 0.49*atr2dsyti
                    x = list(filtered_df['fluence_n_cm2'])
                    x.append(1.38e20)
                    y = list(filtered_df['Measured DT41J  [C]'])
                    y.append(float(atr2_tts))
                    linear = LinearRegression()
                    linear.fit(np.array(x).reshape(-1,1), np.array(y).reshape(-1, 1))
                    owaytts = linear.predict(np.array([fluence]).reshape(-1,1))
                    oway_preds = linear.predict(np.array(x).reshape(-1,1))
                    oway_tts.append(owaytts[0][0])
                    trues = np.array(y).reshape(-1, 1)
                    r2 = r2_score(trues, oway_preds)
                    mae = mean_absolute_error(trues, oway_preds)
                    oway_mae.append(mae)
                    oway_r2.append(r2)

            else:
                oway_usedeony.append(True)
                preds, _ = EONY().predict(df_eony)
                dsy_4e19, cc = self.tts2dsy(pf, preds)
                atr2dsy292 = self.atr2cf292(wt_cu, wt_ni, wt_mn, wt_si, wt_p)
                atr2dsyti = self.atr2cfti(temp_c, atr2dsy292)
                owaydsy = (atr2dsyti - dsy_4e19) / (atr2fte - 4e19) * (fluence - 4e19) + dsy_4e19
                #owaytts = owaydsy*cc
                owaytts = 0.00067*owaydsy**2 +0.49*owaydsy
                try:
                    oway_tts.append(owaytts[0])
                except:
                    oway_tts.append(owaytts)
                oway_mae.append(np.nan)
                oway_r2.append(np.nan)
        #print(oway_tts)
        df['OWAY predicted TTS (degC)'] = oway_tts
        df['OWAY EONY model used'] = oway_usedeony
        df['OWAY linear fit R2'] = oway_r2
        df['OWAY linear fit MAE'] = oway_mae

        return np.array(oway_tts), df

class JOWAY(OWAY):

    def __init__(self):
        super().__init__()

    def _features(self):
        features = ['Product Form', 'temperature_C', 'wt_percent_Cu', 'wt_percent_Ni', 'wt_percent_Mn', 'wt_percent_P',
                        'wt_percent_Si', 'wt_percent_C', 'log(fluence_n_cm2)', 'log(flux_n_cm2_sec)']
        return features

    '''
    def tts2dsy(self, pf, tts):
        # TTS = 0.00067*dSy**2 + 0.49*dSy
        a = 0.00067
        b = 0.49
        c = -1*tts
        dsy = (-b + np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
        #print(dsy)
        cc = tts/dsy
        return dsy, cc
    '''

    def predict(self, df, atr2fte=1.38e20, nn_model='Jacobs26'):
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
        models = {'Jacobs23': EnsembleNN_Jacobs23(), 'Jacobs24': EnsembleNN_Jacobs24(),
                  'Jacobs25': EnsembleNN_Jacobs25(), 'Jacobs26': EnsembleNN_Jacobs26()}
        model = models[nn_model]
        oway_tts = list()
        atr2cf_preds = list()
        features = model._features()

        df_nn = copy(df)
        print('joway df_nn', df_nn.shape)
        df_nn = df_nn.drop(['log(fluence_n_cm2)'], axis=1)
        log_fluence = [np.log10(4e19) for i in range(df_nn.shape[0])]
        df_nn['log(fluence_n_cm2)'] = log_fluence

        if 'fluence_n_cm2' in features:
            df_nn = df_nn.drop(['fluence_n_cm2'], axis=1)
            fluence = [4e19 for i in range(df_nn.shape[0])]
            df_nn['fluence_n_cm2'] = fluence

        df_nn = df_nn[features]

        preds, _ = model.predict(df_nn)

        if 'Product Form' not in features:
            # Assume plate
            df_nn['Product Form'] = ['P' for i in range(df_nn.shape[0])]

        for i, d in df.iterrows():
        #for i, d in df_nn.iterrows():
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
            #owaytts = owaydsy*cc
            owaytts = 0.00067*owaydsy**2 + 0.49*owaydsy
            oway_tts.append(owaytts)
            #atr2cf_preds.append(atr2dsyti*cc)
            atr2ttsti =  0.00067*atr2dsyti**2 + 0.49*atr2dsyti
            atr2cf_preds.append(atr2ttsti)
        df['NN predicted TTS (degC) at 4e19 fluence'] = preds
        df['ATR2 CF TTS (degC) at 1.38e20 fluence'] = atr2cf_preds
        df['JOWAY predicted TTS (degC)'] = oway_tts
        return np.array(oway_tts), df

class XGBoost():
    # XGBoost model with fitted derivatives to have dTTS/dfluence > 0 and dTTS/dflux < 0
    def __init__(self):
        return

    def _features(self):
        features = ['temperature_C', 'wt_percent_Cu', 'wt_percent_Ni', 'wt_percent_Mn', 'wt_percent_P', 'wt_percent_Si',
                    'wt_percent_C', 'log(fluence_n_cm2)', 'log(flux_n_cm2_sec)', 'fluence_n_cm2', 'flux_n_cm2_sec']
        return features

    def predict(self, df, anchors=None):
        if anchors == '2025_v2':
            model_folder = os.path.join(path, 'model_files/XGBoost/fullfit_2025anchors_v2/')
        else:
            model_folder = os.path.join(path, 'model_files/XGBoost/fullfit/')
        print('Using model folder', model_folder)
        features = self._features()
        df_features = df[features]

        # Normalize the input features
        preprocessor = joblib.load(os.path.join(model_folder, 'StandardScaler.pkl'))

        # Get predictions and error bars from model
        model = joblib.load(os.path.join(model_folder, 'XGBRegressor.pkl'))
        preds = model.predict(preprocessor.transform(df_features))

        df['XGBoost predicted TTS (degC)'] = preds

        return preds, df

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
            # Jacobs 25 model recalibration
            a = 1.6449593406320036
            b = 4.996466269117626
            #a = -0.041
            #b = 2.041
            #c = 3.124
            #ebars_each_recal = a * np.array(ebars_each) ** 2 + b * np.array(ebars_each) + c
            ebars_each_recal = a*np.array(ebars_each) + b
        else:
            ebars_each_recal = ebars_each

        return np.array(preds_each).ravel(), np.array(ebars_each_recal).ravel()

    def _features(self, anchors):
        if anchors in ['2023', '2025', '2025_v2', 'paperdraft']:
            features = ['temperature_C', 'wt_percent_Cu', 'wt_percent_Ni', 'wt_percent_Mn', 'wt_percent_P',
                    'wt_percent_Si', 'wt_percent_C', 'log(fluence_n_cm2)', 'log(flux_n_cm2_sec)', 'fluence_n_cm2',
                    'flux_n_cm2_sec', 'Time']
        elif anchors in ['2023_thermofeatures', '2025_v2_thermofeatures']:
            features = ['temperature_C', 'wt_percent_Cu', 'wt_percent_Ni', 'wt_percent_Mn', 'wt_percent_P',
                    'wt_percent_Si', 'wt_percent_C', 'log(fluence_n_cm2)', 'log(flux_n_cm2_sec)', 'fluence_n_cm2',
                    'flux_n_cm2_sec', 'Time', 'D_Mn (m2/s)','D_Ni (m2/s)', 'D_Si (m2/s)', 'D_Fe (m2/s)',
                        'D_Cu (m2/s)', 'Cu_min (at%)', 'Ln(Ksp/Kspbar)', 'Cu solubility', 'Cu/Cu solubility']
        elif anchors in ['2023_thermofeatures_Avrami', '2025_v2_thermofeatures_Avrami']:
            features = ['temperature_C', 'wt_percent_Cu', 'wt_percent_Ni', 'wt_percent_Mn', 'wt_percent_P',
                    'wt_percent_Si', 'wt_percent_C', 'log(fluence_n_cm2)', 'log(flux_n_cm2_sec)', 'fluence_n_cm2',
                    'flux_n_cm2_sec', 'Time', 'D_Mn (m2/s)','D_Ni (m2/s)', 'D_Si (m2/s)', 'D_Fe (m2/s)',
                        'D_Cu (m2/s)', 'Cu_min (at%)', 'Ln(Ksp/Kspbar)', 'Cu solubility', 'Cu/Cu solubility',
                        'Avrami_TTS', 'Avrami_A1', 'Avrami_A2', 'Avrami_A3', 'Avrami_A4']
        elif anchors in ['2025_v2_efffluence']:
            features = ['temperature_C', 'wt_percent_Cu', 'wt_percent_Ni', 'wt_percent_Mn', 'wt_percent_P',
                        'wt_percent_Si', 'wt_percent_C', 'effective_fluence_p02', 'log(effective_fluence_p02)',
                        'effective_fluence_p0', 'log(effective_fluence_p0)', 'effective_fluence_p005',
                        'log(effective_fluence_p005)', 'effective_fluence_p01', 'log(effective_fluence_p01)']
        elif anchors in ['2025_v2_efffluence_e900eony']:
            features = ['temperature_C', 'wt_percent_Cu', 'wt_percent_Ni', 'wt_percent_Mn', 'wt_percent_P',
                        'wt_percent_Si', 'wt_percent_C', 'effective_fluence_p02', 'log(effective_fluence_p02)',
                        'effective_fluence_p0', 'log(effective_fluence_p0)', 'effective_fluence_p005',
                        'log(effective_fluence_p005)', 'effective_fluence_p01', 'log(effective_fluence_p01)',
                        'EONY predicted TTS (degC)', 'E900 predicted TTS (degC)']
        return features

    def predict(self, df, anchors='2025', return_ebars=False):
        # anchors = 2023 or 2025
        features = self._features(anchors)
        if anchors == '2023':
            model_folder = os.path.join(path, 'model_files/Jacobs25/fullfit_2023anchors')
        elif anchors == '2025':
            model_folder = os.path.join(path, 'model_files/Jacobs25/fullfit_2025anchors')
        elif anchors == '2025_v2':
            model_folder = os.path.join(path, 'model_files/Jacobs25/fullfit_2025anchors_v2')
        elif anchors == '2025_v2_effectivefluence':
            model_folder = os.path.join(path, 'model_files/Jacobs25/fullfit_2025anchors_v2_effectivefluence')
        elif anchors == '2023_thermofeatures':
            model_folder = os.path.join(path, 'model_files/Jacobs25/fullfit_2023anchors_thermofeatures')
        elif anchors == '2025_v2_thermofeatures':
            model_folder = os.path.join(path, 'model_files/Jacobs25/fullfit_2025anchors_v2_thermofeatures')
        elif anchors == '2023_thermofeatures_Avrami':
            model_folder = os.path.join(path, 'model_files/Jacobs25/fullfit_2023anchors_thermofeatures_Avrami')
        elif anchors == '2025_v2_thermofeatures_Avrami':
            model_folder = os.path.join(path, 'model_files/Jacobs25/fullfit_2025anchors_v2_thermofeatures_Avrami')
        elif anchors == 'paperdraft':
            model_folder = os.path.join(path, 'model_files/Jacobs25/fullfit') # Version used in the paper draft
        elif anchors == '2025_v2_efffluence':
            model_folder = os.path.join(path, 'model_files/Jacobs25/fullfit_2025anchors_v2_efffluence')
        elif anchors == '2025_v2_efffluence_e900eony':
            model_folder = os.path.join(path, 'model_files/Jacobs25/fullfit_2025anchors_v2_efffluence_e900eony')

        print('model folder', model_folder)
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

class EnsembleNN_Jacobs26():
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
                tf.keras.models.load_model(os.path.join(model_folder, 'keras_model_' + str(i) + '_functional.keras')))

        model_bagged_keras_rebuild.model.estimators_ = models
        model_bagged_keras_rebuild.model.estimators_features_ = [np.arange(0, n_features) for i in models]

        return model_bagged_keras_rebuild

    def _get_preds_ebars(self, model, df_featurized, preprocessor, return_ebars=True, return_domains=True):
        preds_each = list()
        ebars_each = list()
        domains_each = list()

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

        if return_domains == True:
            # Get domains
            _original_unpack = scu.unpack_keras_model
            scu.unpack_keras_model = patched_unpack_keras_model
            with open(os.path.join('RPV_model_benchmarking/model_files/Jacobs26/domain', 'model.dill'), 'rb') as f:
                model_domain = dill.load(f)
            domains_each = model_domain.predict(df_featurized_scaled)
            domains_each = domains_each['d_pred']
        else:
            try:
                domains_each = [np.nan for i in range(np.array(preds_each).shape[0])]
            except:
                domains_each = [np.nan]

        if return_ebars == True:
            # Jacobs 26 model recalibration
            a = 1.300052530566834
            b = 5.537998189600857
            #a = -0.041
            #b = 2.041
            #c = 3.124
            #ebars_each_recal = a * np.array(ebars_each) ** 2 + b * np.array(ebars_each) + c
            ebars_each_recal = a*np.array(ebars_each) + b
        else:
            ebars_each_recal = ebars_each

        return np.array(preds_each).ravel(), np.array(ebars_each_recal).ravel(), np.array(domains_each).ravel()

    def _features(self):
        features = ['temperature_C', 'wt_percent_Cu', 'wt_percent_Ni', 'wt_percent_Mn', 'wt_percent_P',
                    'wt_percent_Si', 'wt_percent_C', 'log(fluence_n_cm2)', 'log(flux_n_cm2_sec)', 'fluence_n_cm2',
                    'flux_n_cm2_sec']
        return features

    def predict(self, df, anchors='2026', return_ebars=False, return_domains=False):
        # anchors = 2023 or 2025
        features = self._features()
        if anchors == '2026':
            model_folder = os.path.join(path, 'model_files/Jacobs26/fullfit')
        elif anchors == 'None':
            model_folder = os.path.join(path, 'model_files/Jacobs26/fullfit_noanchors')

        df_features = df[features]

        # Rebuild the saved model
        n_features = df_features.shape[1]
        model = self._rebuild_model(n_features, model_folder)

        # Normalize the input features
        preprocessor = joblib.load(os.path.join(model_folder, 'StandardScaler.pkl'))

        # Get predictions and error bars from model
        preds, ebars, domains = self._get_preds_ebars(model, df_features, preprocessor, return_ebars=return_ebars, return_domains=return_domains)

        pred_dict = {'Jacobs26 NN ensemble predicted TTS (degC)': preds,
                     'Jacobs26 NN ensemble error bars (degC)': ebars,
                     'Jacobs26 NN ensemble domain d': domains}

        for k, v in pred_dict.items():
            df[k] = v

        return preds, df

class EnsembleNN_Jacobs25_BWR(EnsembleNN_Jacobs25):
    # NN ENSEMBLE MODEL
    def __init__(self):
        super().__init__()

    def predict(self, df, return_ebars=False):
        features = self._features()
        model_folder = os.path.join(path, 'model_files/Jacobs25_BWR/fullfit')

        df_features = df[features]

        # Rebuild the saved model
        n_features = df_features.shape[1]
        model = self._rebuild_model(n_features, model_folder)

        # Normalize the input features
        preprocessor = joblib.load(os.path.join(model_folder, 'StandardScaler.pkl'))

        # Get predictions and error bars from model
        preds, ebars = self._get_preds_ebars(model, df_features, preprocessor, return_ebars=return_ebars)

        pred_dict = {'Jacobs25 BWR NN ensemble predicted TTS (degC)': preds,
                     'Jacobs25 BWR NN ensemble error bars (degC)': ebars}

        for k, v in pred_dict.items():
            df[k] = v

        return preds, df

class EnsembleNN_Jacobs25_PWR(EnsembleNN_Jacobs25):
    # NN ENSEMBLE MODEL
    def __init__(self):
        super().__init__()

    def predict(self, df, return_ebars=False):
        features = self._features()
        model_folder = os.path.join(path, 'model_files/Jacobs25_PWR/fullfit')

        df_features = df[features]

        # Rebuild the saved model
        n_features = df_features.shape[1]
        model = self._rebuild_model(n_features, model_folder)

        # Normalize the input features
        preprocessor = joblib.load(os.path.join(model_folder, 'StandardScaler.pkl'))

        # Get predictions and error bars from model
        preds, ebars = self._get_preds_ebars(model, df_features, preprocessor, return_ebars=return_ebars)

        pred_dict = {'Jacobs25 PWR NN ensemble predicted TTS (degC)': preds,
                     'Jacobs25 PWR NN ensemble error bars (degC)': ebars}

        for k, v in pred_dict.items():
            df[k] = v

        return preds, df

class EnsembleNN_Jacobs25_Bob(EnsembleNN_Jacobs25):
# NN ENSEMBLE MODEL
    def __init__(self):
        super().__init__()

    def predict(self, df, return_ebars=False):
        features = self._features()
        model_folder = os.path.join(path, 'model_files/Jacobs25_Bob/fullfit')

        df_features = df[features]

        # Rebuild the saved model
        n_features = df_features.shape[1]
        model = self._rebuild_model(n_features, model_folder)

        # Normalize the input features
        preprocessor = joblib.load(os.path.join(model_folder, 'StandardScaler.pkl'))

        # Get predictions and error bars from model
        preds, ebars = self._get_preds_ebars(model, df_features, preprocessor, return_ebars=return_ebars)

        pred_dict = {'Jacobs25 BWR NN ensemble predicted TTS (degC)': preds,
                     'Jacobs25 BWR NN ensemble error bars (degC)': ebars}

        for k, v in pred_dict.items():
            df[k] = v

        return preds, df
    def _features(self):
        # Use this feature order for model trained on Bob's data only
        features = ['temperature_C', 'wt_percent_Cu', 'wt_percent_Ni', 'wt_percent_Mn', 'wt_percent_P',
                    'wt_percent_Si', 'wt_percent_C', 'fluence_n_cm2',
                    'flux_n_cm2_sec', 'log(fluence_n_cm2)', 'log(flux_n_cm2_sec)',  'Time']
        return features
