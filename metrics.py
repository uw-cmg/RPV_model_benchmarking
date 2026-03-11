import numpy as np
import pandas as pd
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error

import RPV_model_benchmarking
path = RPV_model_benchmarking.__path__[0]

class Metrics():

    def __init__(self):
        return

    def get_me(self, df, model_name):
        # model_name = EONY, E900, OWAY, GKRR, GBR, Jacobs23, Jacobs24
        trues = df['Measured DT41J  [C]']
        if model_name in ['Jacobs23', 'Jacobs24', 'Jacobs25', 'Jacobs26']:
            try:
                preds = df[model_name+' NN ensemble predicted TTS (degC)']
            except:
                preds = df['NN ensemble predicted TTS (degC) '+model_name]
        else:
            preds = df[model_name+' predicted TTS (degC)']
        me = np.mean(preds-trues) # or maybe preds-trues?
        return me

    def get_mae(self, df, model_name):
        # model_name = EONY, E900, OWAY, GKRR, GBR, Jacobs23, Jacobs24
        trues = df['Measured DT41J  [C]']
        if model_name in ['Jacobs23', 'Jacobs24', 'Jacobs25', 'Jacobs26']:
            try:
                preds = df[model_name+' NN ensemble predicted TTS (degC)']
            except:
                preds = df['NN ensemble predicted TTS (degC) '+model_name]
        else:
            preds = df[model_name+' predicted TTS (degC)']
        mae = mean_absolute_error(trues, preds)
        return mae

    def get_rmse(self, df, model_name):
        # model_name = EONY, E900, OWAY, GKRR, GBR, Jacobs23, Jacobs24
        trues = df['Measured DT41J  [C]']
        if model_name in ['Jacobs23', 'Jacobs24', 'Jacobs25', 'Jacobs26']:
            try:
                preds = df[model_name+' NN ensemble predicted TTS (degC)']
            except:
                preds = df['NN ensemble predicted TTS (degC) '+model_name]
        else:
            preds = df[model_name+' predicted TTS (degC)']

        rmse = np.sqrt(mean_squared_error(trues, preds))
        return rmse

    def filter_df(self, df, column_name, filter_val, filter_operation):
        # filter_operation = 'equal', 'greater', 'less'
        if filter_operation == 'equal':
            df = df[df[column_name] == filter_val]
        elif filter_operation == 'greater':
            df = df[df[column_name] > filter_val]
        elif filter_operation == 'less':
            df = df[df[column_name] < filter_val]
        else:
            raise ValueError('filter_operation should be one of: equal, greater, less')
        return df

    def get_metric_per_alloy(self, df, metric_name, model_name):
        # Per alloy metric
        # metric_name one of ME, MAE, RMSE
        alloys = df['alloy'].unique()
        errors = list()
        for alloy in alloys:
            df_sub = df[df['alloy'] == alloy]
            if metric_name == 'ME':
                error = self.get_me(df_sub, model_name=model_name)
            elif metric_name == 'MAE':
                error = self.get_mae(df_sub, model_name=model_name)
            elif metric_name == 'RMSE':
                error = self.get_rmse(df_sub, model_name=model_name)
            errors.append(error)
        return errors, alloys

class Benchmarking(Metrics):

    def __init__(self):
        super(Benchmarking, self).__init__()
        return

    def get_5fold_benchmarks(self, model_name, anchors='2025'):
        # model_name = EONY, E900, GKRR, GBR, Jacobs23, Jacobs24, Jacobs25
        # If using Jacobs25, need to specify 2023 or 2025 for anchor types

        # Get the 5fold df based on model name
        if model_name != 'Jacobs25':
            if model_name == 'Jacobs26':
                if anchors == '2026':
                    model_path = os.path.join(os.path.join(path, 'model_files'), model_name+'/5fold')
                elif anchors == 'None':
                    model_path = os.path.join(os.path.join(path, 'model_files'), model_name+'/5fold_noanchors')
            else:
                model_path = os.path.join(os.path.join(path, 'model_files'), model_name+'/5fold')
        else:
            if anchors == '2023':
                model_path = os.path.join(os.path.join(path, 'model_files'), model_name + '/5fold_2023anchors')
            elif anchors == '2025':
                model_path = os.path.join(os.path.join(path, 'model_files'), model_name + '/5fold_2025anchors')
            elif anchors == '2025_v2':
                model_path = os.path.join(os.path.join(path, 'model_files'), model_name + '/5fold_2025anchors_v2')
            elif anchors == '2023_thermofeatures':
                model_path = os.path.join(os.path.join(path, 'model_files'), model_name + '/5fold_2023anchors_thermofeatures')
            elif anchors == '2025_v2_thermofeatures':
                model_path = os.path.join(os.path.join(path, 'model_files'), model_name + '/5fold_2025anchors_v2_thermofeatures')
            elif anchors == '2023_thermofeatures_Avrami':
                model_path = os.path.join(os.path.join(path, 'model_files'), model_name + '/5fold_2023anchors_thermofeatures_Avrami')
            elif anchors == '2025_v2_thermofeatures_Avrami':
                model_path = os.path.join(os.path.join(path, 'model_files'), model_name + '/5fold_2025anchors_v2_thermofeatures_Avrami')
            elif anchors == 'paperdraft':
                model_path = os.path.join(os.path.join(path, 'model_files'), model_name + '/5fold')
            elif anchors == '2025_v2_efffluence':
                model_path = os.path.join(os.path.join(path, 'model_files'), model_name + '/5fold_2025anchors_v2_efffluence')
            elif anchors == '2025_v2_efffluence_e900eony':
                model_path = os.path.join(os.path.join(path, 'model_files'), model_name + '/5fold_2025anchors_v2_efffluence_e900eony')

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
            if model_name in ['Jacobs23', 'Jacobs24', 'Jacobs25', 'Jacobs26']:
                #pred_name = 'NN ensemble predicted TTS (degC) ' + model_name
                pred_name = model_name + ' NN ensemble predicted TTS (degC)'
            else:
                pred_name = model_name + ' predicted TTS (degC)'
            df = df.rename(columns={'y_test': 'Measured DT41J  [C]', 'y_pred': pred_name})

        df_plotter = Metrics().filter_df(df, column_name='datatype', filter_val='Plotter', filter_operation='equal')
        df_highfluence = Metrics().filter_df(df, column_name='fluence_n_cm2', filter_val=6e19,
                                             filter_operation='greater')
        df_highfluence_plotter = Metrics().filter_df(df_highfluence, column_name='datatype', filter_val='Plotter',
                                                     filter_operation='equal')
        df_hightts = Metrics().filter_df(df, column_name='Measured DT41J  [C]', filter_val=150,
                                         filter_operation='greater')
        df_hightts_plotter = Metrics().filter_df(df_hightts, column_name='datatype', filter_val='Plotter',
                                                 filter_operation='equal')
        df_lowCu_plotter = Metrics().filter_df(df_plotter, column_name='wt_percent_Cu', filter_val=0.08,
                                               filter_operation='less')

        rmse_all = Metrics().get_rmse(df, model_name=model_name)
        rmse_plotter = Metrics().get_rmse(df_plotter, model_name=model_name)
        rmse_highfluence = Metrics().get_rmse(df_highfluence, model_name=model_name)
        rmse_highfluence_plotter = Metrics().get_rmse(df_highfluence_plotter, model_name=model_name)
        rmse_hightts = Metrics().get_rmse(df_hightts, model_name=model_name)
        rmse_hightts_plotter = Metrics().get_rmse(df_hightts_plotter, model_name=model_name)
        rmse_lowCu_plotter = Metrics().get_rmse(df_lowCu_plotter, model_name=model_name)
        rmse_peralloy, alloys = Metrics().get_metric_per_alloy(df_plotter, metric_name='RMSE', model_name=model_name)

        data = {'RMSE, all': rmse_all,
                'RMSE, Plotter': rmse_plotter,
                'RMSE, high fluence': rmse_highfluence,
                'RMSE, high fluence Plotter': rmse_highfluence_plotter,
                'RMSE, high TTS': rmse_hightts,
                'RMSE, high TTS Plotter': rmse_hightts_plotter,
                'RMSE, low Cu Plotter': rmse_lowCu_plotter,
                'RMSE, per alloy (average +/- stdev)': [np.mean(rmse_peralloy), np.std(rmse_peralloy)]}

        return data
