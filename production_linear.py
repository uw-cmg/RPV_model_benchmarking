"""Flux-independent OWAY fits; production includes all high-fluence data."""
from dataclasses import dataclass

import numpy as np
import pandas as pd



def linear_fit_prediction(fluences, tts, target_fluence):
    """Unweighted OLS in TTS versus linear fluence, scaled only for conditioning."""
    x = np.asarray(fluences, dtype=float) / 1e20
    y = np.asarray(tts, dtype=float)
    if len(x) < 2 or len(np.unique(x)) < 2 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('A line requires finite observations at two distinct fluences.')
    slope, intercept = np.linalg.lstsq(np.column_stack([x, np.ones(len(x))]), y, rcond=None)[0]
    return float(slope * target_fluence / 1e20 + intercept), float(slope), float(intercept)


def select_high_fluence_rows(database_rows, *, fluence_threshold_n_cm2=3e19,
                             include_threshold=False, hold_out_highest=False):
    """Production: fluence > threshold, including the highest observation.

    Flux is deliberately not an eligibility condition. Evaluation can explicitly
    request highest-point holdout and a different threshold/inclusive boundary.
    """
    if not np.isfinite(fluence_threshold_n_cm2) or fluence_threshold_n_cm2 < 0:
        raise ValueError('Measured fluence threshold must be finite and nonnegative.')
    rows = database_rows.copy()
    if 'alloy' in rows and len(rows) and rows.alloy.nunique() != 1:
        raise ValueError('Supply measurements for exactly one alloy.')
    fluence = pd.to_numeric(rows.fluence_n_cm2,errors='coerce')
    tts = pd.to_numeric(rows['Measured DT41J  [C]'],errors='coerce')
    above = fluence.ge(fluence_threshold_n_cm2) if include_threshold else fluence.gt(fluence_threshold_n_cm2)
    highest = fluence.eq(fluence.max())
    held_out = highest if hold_out_highest else pd.Series(False,index=rows.index)
    used = np.isfinite(fluence) & np.isfinite(tts) & above & ~held_out
    audit = rows[['fluence_n_cm2','Measured DT41J  [C]']].copy()
    if 'flux_n_cm2_sec' in rows:
        audit['flux_n_cm2_sec'] = rows.flux_n_cm2_sec
    audit['workbook_row_index'] = rows.index
    audit['passes_fluence_threshold'] = above
    audit['is_highest_fluence'] = highest
    audit['held_out_highest_fluence'] = held_out
    audit['used_in_OWAY_fit'] = used
    return rows.loc[used].copy(),audit


@dataclass
class OWAYFit:
    slope_degC_per_1e20: float
    intercept_degC: float
    atr2_fluence_n_cm2: float
    atr2_tts_degC: float
    measured_fit_rows: pd.DataFrame
    measured_row_audit: pd.DataFrame

    def predict(self,fluence_n_cm2):
        fluence = np.asarray(fluence_n_cm2,dtype=float)
        if not np.isfinite(fluence).all() or (fluence < 0).any():
            raise ValueError('Prediction fluence must be finite and nonnegative.')
        return self.slope_degC_per_1e20*fluence/1e20+self.intercept_degC


def fit_oway(database_rows, *, atr2_endpoint, query_profiles=None, fluence_threshold_n_cm2=3e19,
             include_threshold=False, hold_out_highest=False, atr2_fluence_n_cm2=1.38e20):
    """OLS in TTS versus linear fluence, using measured data plus ATR2.

    No flux input or filtering is used. No EONY fallback is supplied when
    measured high-fluence data are unavailable. Production is the default;
    highest-fluence holdout is an explicit evaluation option only.
    """
    measured,audit = select_high_fluence_rows(database_rows,
        fluence_threshold_n_cm2=fluence_threshold_n_cm2,
        include_threshold=include_threshold,hold_out_highest=hold_out_highest)
    if measured.empty:
        raise ValueError('OWAY requires a measured point passing the high-fluence threshold.')
    if query_profiles is None:
        query_profiles = database_rows[database_rows.fluence_n_cm2.eq(database_rows.fluence_n_cm2.max())]
    atr2 = float(np.mean([atr2_endpoint(row) for _,row in query_profiles.iterrows()]))
    x = np.r_[measured.fluence_n_cm2.to_numpy(dtype=float),atr2_fluence_n_cm2]
    y = np.r_[measured['Measured DT41J  [C]'].to_numpy(dtype=float),atr2]
    if len(np.unique(x)) < 2 or not np.isfinite(atr2):
        raise ValueError('OWAY needs two distinct finite fit fluences.')
    _,slope,intercept = linear_fit_prediction(x,y,0)
    return OWAYFit(slope,intercept,atr2_fluence_n_cm2,atr2,measured,audit)


@dataclass
class MLOWAYFit:
    flux_n_cm2_sec: float
    flux_source: str
    flux_tolerance_factor: float
    jacobs_anchor_fluence_n_cm2: float
    jacobs_anchor_tts_degC: float
    atr2_fluence_n_cm2: float
    atr2_tts_degC: float
    slope_degC_per_1e20: float
    intercept_degC: float
    measured_fit_rows: pd.DataFrame
    measured_row_audit: pd.DataFrame

    def predict(self, fluence_n_cm2):
        fluence = np.asarray(fluence_n_cm2, dtype=float)
        if not np.isfinite(fluence).all() or (fluence < 0).any():
            raise ValueError('Prediction fluence must be finite and nonnegative.')
        return self.slope_degC_per_1e20 * fluence / 1e20 + self.intercept_degC


def select_fit_rows(database_rows, flux_n_cm2_sec, *, fluence_threshold_n_cm2=3e19,
                    hold_out_highest=False, flux_tolerance_factor=1.25, include_threshold=False):
    """Accept ratios in [1/factor, factor], inclusive; never relabel row fluxes."""
    if not np.isfinite(flux_n_cm2_sec) or flux_n_cm2_sec <= 0:
        raise ValueError('Query flux must be finite and positive.')
    if not np.isfinite(flux_tolerance_factor) or flux_tolerance_factor < 1:
        raise ValueError('Flux tolerance factor must be at least 1.')
    if not np.isfinite(fluence_threshold_n_cm2) or fluence_threshold_n_cm2 < 0:
        raise ValueError('Measured fluence threshold must be finite and nonnegative.')
    rows = database_rows.copy()
    _,high_fluence_audit = select_high_fluence_rows(rows,
        fluence_threshold_n_cm2=fluence_threshold_n_cm2,
        include_threshold=include_threshold,hold_out_highest=hold_out_highest)
    flux = pd.to_numeric(rows.flux_n_cm2_sec, errors='coerce')
    valid_flux = np.isfinite(flux) & (flux > 0)
    ratio = pd.Series(np.inf, index=rows.index, dtype=float)
    ratio.loc[valid_flux] = np.maximum(flux.loc[valid_flux]/flux_n_cm2_sec,
                                     flux_n_cm2_sec/flux.loc[valid_flux])
    close = valid_flux & (ratio <= flux_tolerance_factor*(1+1e-12))
    above = high_fluence_audit.passes_fluence_threshold
    held_out = high_fluence_audit.held_out_highest_fluence
    used = close & high_fluence_audit.used_in_OWAY_fit
    audit = rows[['fluence_n_cm2','flux_n_cm2_sec','Measured DT41J  [C]']].copy()
    audit['workbook_row_index'] = rows.index
    audit['query_flux_n_cm2_sec'] = flux_n_cm2_sec
    audit['flux_ratio_distance'] = ratio
    audit['flux_matches'] = close
    audit['passes_fluence_threshold'] = above
    audit['is_highest_fluence'] = high_fluence_audit.is_highest_fluence
    audit['held_out_highest_fluence'] = held_out
    audit['used_in_ML_OWAY_fit'] = used
    selected = rows.loc[used].copy()
    assert not np.intersect1d(selected.index, rows.index[held_out]).size
    return selected, audit


def fit_ml_oway(*, jacobs_endpoint, atr2_endpoint, database_rows=None, query_profiles=None,
                flux_n_cm2_sec=None, flux_tolerance_factor=1.25,
                fluence_threshold_n_cm2=3e19, hold_out_highest=False, include_threshold=False,
                jacobs_anchor_fluence_n_cm2=4e19, atr2_fluence_n_cm2=1.38e20):
    """Fit production ML-OWAY at one query flux.

    Production includes all flux-matched measurements strictly above 3e19,
    including the highest point. Holdout is an explicit evaluation option.

    For a database alloy, omitted flux defaults to its highest-fluence row's
    flux. If those rows have different fluxes, require an explicit query flux.
    Without database rows, both query_profiles and user-specified flux are
    required. Chemistry and temperature come from query_profiles (defaulting
    to the highest-fluence database rows); their original fluxes are not used
    for the Jacobs26 anchor when a query flux is supplied.
    """
    has_database = database_rows is not None and len(database_rows) > 0
    if has_database:
        if 'alloy' in database_rows and database_rows.alloy.nunique() != 1:
            raise ValueError('Supply measurements for exactly one alloy per fit.')
        highest = database_rows[database_rows.fluence_n_cm2.eq(database_rows.fluence_n_cm2.max())]
        if query_profiles is None:
            query_profiles = highest
        if flux_n_cm2_sec is None:
            fluxes = highest.flux_n_cm2_sec.unique()
            if len(fluxes) != 1:
                raise ValueError('Highest-fluence rows have different fluxes; specify query flux.')
            flux_n_cm2_sec = float(fluxes[0])
            flux_source = 'highest_fluence_database_point'
        else:
            flux_source = 'user_specified'
    else:
        if flux_n_cm2_sec is None:
            raise ValueError('Specify query flux when there are no database measurements.')
        flux_source = 'user_specified'
    if query_profiles is None or len(query_profiles) == 0:
        raise ValueError('Provide alloy chemistry and temperature for the anchor.')
    if not np.isfinite(flux_n_cm2_sec) or flux_n_cm2_sec <= 0:
        raise ValueError('Query flux must be finite and positive.')
    if not 0 < jacobs_anchor_fluence_n_cm2 < atr2_fluence_n_cm2:
        raise ValueError('Anchor fluences must be positive, distinct, and ordered.')
    if has_database:
        measured, audit = select_fit_rows(database_rows,flux_n_cm2_sec,
            fluence_threshold_n_cm2=fluence_threshold_n_cm2,
            hold_out_highest=hold_out_highest,flux_tolerance_factor=flux_tolerance_factor,
            include_threshold=include_threshold)
    else:
        empty = pd.DataFrame(columns=['fluence_n_cm2','flux_n_cm2_sec','Measured DT41J  [C]'])
        measured, audit = select_fit_rows(empty,flux_n_cm2_sec,
            fluence_threshold_n_cm2=fluence_threshold_n_cm2,
            hold_out_highest=False,flux_tolerance_factor=flux_tolerance_factor,
            include_threshold=include_threshold)
    jacobs_tts = float(np.mean([jacobs_endpoint(row, jacobs_anchor_fluence_n_cm2, flux_n_cm2_sec)
                               for _,row in query_profiles.iterrows()]))
    atr2_tts = float(np.mean([atr2_endpoint(row) for _,row in query_profiles.iterrows()]))
    if not np.isfinite([jacobs_tts,atr2_tts]).all():
        raise ValueError('Nonfinite model anchor.')
    x = np.r_[jacobs_anchor_fluence_n_cm2,measured.fluence_n_cm2.to_numpy(dtype=float),atr2_fluence_n_cm2]
    y = np.r_[jacobs_tts,measured['Measured DT41J  [C]'].to_numpy(dtype=float),atr2_tts]
    _,slope,intercept = linear_fit_prediction(x,y,0)
    return MLOWAYFit(float(flux_n_cm2_sec),flux_source,float(flux_tolerance_factor),
        float(jacobs_anchor_fluence_n_cm2),jacobs_tts,float(atr2_fluence_n_cm2),atr2_tts,
        slope,intercept,measured,audit)
