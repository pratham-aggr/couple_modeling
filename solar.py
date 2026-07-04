"""
solar.py
========
Analytic daily-mean top-of-atmosphere insolation (W/m2) as a function of
day-of-year and latitude.  Shared by preprocess_cice.py (cache build) and
cice_coupler.py (online) so the offline and online SOLIN input channel are
computed by IDENTICAL math -> no train/serve skew.

Why the CICE emulator needs this: under the ice pack SST is pinned at the
freezing point (~-1.8 C) everywhere, so it carries almost no information about
the seasonal freeze/melt cycle.  TOA insolation encodes season x latitude
directly and is trivially reproducible online (the server knows the model date
and the grid latitudes), making it the ideal "real input" to break the SST
saturation that left the SST-only emulator seasonally unresponsive.
"""
import numpy as np

S0 = 1361.0  # solar constant, W/m2 (scale is irrelevant post-normalization)


def solar_declination(doy):
    """Solar declination (radians) for day-of-year doy (1..365/366).
    Spencer (1971) Fourier series; accurate to ~0.01 deg."""
    g = 2.0 * np.pi * (np.asarray(doy, dtype=np.float64) - 1.0) / 365.0
    return (0.006918
            - 0.399912 * np.cos(g)   + 0.070257 * np.sin(g)
            - 0.006758 * np.cos(2*g) + 0.000907 * np.sin(2*g)
            - 0.002697 * np.cos(3*g) + 0.001480 * np.sin(3*g))


def daily_insolation(doy, lat_deg):
    """Daily-mean TOA insolation (W/m2).

    doy      : scalar day-of-year.
    lat_deg  : array of latitudes (deg); any shape.
    Returns an array shaped like lat_deg.  Polar night -> 0; polar day handled
    via the clipped sunset hour angle.
    """
    phi = np.deg2rad(np.asarray(lat_deg, dtype=np.float64))
    decl = float(solar_declination(doy))
    # sunset hour angle h0 = arccos(-tan phi tan decl), clipped for polar day/night
    x = np.clip(-np.tan(phi) * np.tan(decl), -1.0, 1.0)
    h0 = np.arccos(x)
    q = (S0 / np.pi) * (h0 * np.sin(phi) * np.sin(decl)
                        + np.cos(phi) * np.cos(decl) * np.sin(h0))
    return np.maximum(q, 0.0)


def representative_doy_from_stamp(stamp):
    """CICE monthly means are stamped at the period END (the first day of the
    FOLLOWING month), so a stamp's data month is (stamp - 1 day)'s month.  Return
    the day-of-year of the 15th of that data month (mid-month, season-representative).

    stamp : a cftime/datetime object with .year/.month and (cftime) .dayofyr.
    """
    import datetime
    prev = stamp - datetime.timedelta(days=1)      # lands in the data month
    mid = prev.replace(day=15)
    # cftime objects expose .dayofyr; std datetime exposes .timetuple().tm_yday
    return getattr(mid, "dayofyr", None) or mid.timetuple().tm_yday


def representative_doy_from_month(year, month):
    """Online analogue: the mid-month (15th) day-of-year for the current model
    month being predicted.  Uses a proleptic-Gregorian day count (calendar-
    agnostic to within ~1 day, negligible for insolation)."""
    import datetime
    return datetime.date(int(year), int(month), 15).timetuple().tm_yday
