import pandas as pd
import os

import RPV_model_benchmarking
path = RPV_model_benchmarking.__path__[0]

class DataLoader():

    def __init__(self):
        return

    def load_rpv_data(self):
        df = pd.read_excel(os.path.join(path, 'data_files/RPV_UCSB_Plotter_combined_2025-03-04_quad2term_noATR1_dropdups.xlsx'))
        return df

    def load_Jacobs23_anchor_data(self):
        df = pd.read_excel(os.path.join(path, 'data_files/RPV_UCSB_Plotter_combined_onlyanchors_noATR2.xlsx'))
        return df

    def load_Jacobs24_anchor_data(self):
        df = pd.read_excel(os.path.join(path, 'data_files/RPV_UCSB_Plotter_combined_onlyanchors_noATR2_newfeatures_reorderPF_withATR2anchors_ATR2points.xlsx'))
        return df

    def load_GKRR_anchor_data(self):
        df = pd.read_excel(os.path.join(path, 'data_files/RPV_UCSB_Plotter_combined_onlyanchors_Yuchen.xlsx'))
        return df