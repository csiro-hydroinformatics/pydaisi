from pathlib import Path
import numpy as np
import pandas as pd
import zipfile

from hydrodiy.io import csv

source_file = Path(__file__).resolve()
# Path to data within pydaisi repository
FDATA = source_file.parent.parent.parent / "data"

class Period():
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def __str__(self):
        return f"{self.start.date()}-{self.end.date()}"

    def select_index(self, time):
        return (time>=self.start) & (time<=self.end)


class Periods():
    def __init__(self, data_start=1975, data_end=2019, \
                            warmup_years=5, wateryear_start=7):
        self.warmup_years = warmup_years
        self.wateryear_start = wateryear_start
        self.data_start = data_start
        self.data_end = data_end
        self.periods = {}

        self.set_periods()


    @property
    def names(self):
        return list(self.periods.keys())


    def add_period(self, pername, start_active, end_active):
        start_active = pd.to_datetime(start_active)
        end_active = pd.to_datetime(end_active)
        start_total = start_active-pd.DateOffset(months=self.warmup_years*12)
        end_total = end_active
        self.periods[pername] = {\
                    "active": [start_active, end_active],
                    "total": [start_total, end_total]
                }

    def set_periods(self):
        years = np.arange(self.data_start, self.data_end)
        nwarm = self.warmup_years
        nper = (len(years)-nwarm)//2
        ys = self.wateryear_start

        s1 = pd.to_datetime(f"{years[0]}-{ys:02d}-01")\
                    +pd.DateOffset(years=nwarm)-pd.DateOffset(months=1)
        e1 = s1+pd.DateOffset(years=nper)
        self.add_period("per1", s1, e1)

        s2 = s1+pd.DateOffset(years=nper, months=1)
        e2 = s2+pd.DateOffset(years=nper)
        self.add_period("per2", s2, e2)

        s3 = s1
        e3 = e2
        self.add_period("per3", s3, e3)


    def get_periodset(self, pername):
        sa, se = self.periods[pername]["active"]
        active = Period(sa, se)

        st, et = self.periods[pername]["total"]
        total = Period(st, et)

        periodset = type("PeriodSet", (), {"active": active, "total": total})
        return periodset


    def get_validation(self, pername):
        for pn in ["per1", "per2", "per3"]:
            errmsg = f"Period {pn} is not in the list of periods."
            assert pn in self.periods, errmsg

        if pername == "per1":
            return self.get_periodset("per2")
        elif pername == "per2":
            return self.get_periodset("per1")
        else:
            return self.get_periodset("per3")



def get_sites():
    fs = FDATA / "werp_non_stationarity_sites.csv"
    sites, _ = csv.read_csv(fs, index_col="siteid", encoding="cp1252")

    # Add regions
    coords = sites.loc[:, ["lon_centroid_dem", \
                                    "lat_centroid_dem"]]

    sites.loc[:, "isALL"] = True
    sites.loc[:, "isWVIC"] = (coords.lon_centroid_dem<146.) \
                                & (coords.lat_centroid_dem<-36.)

    sites.loc[:, "isNNSW"] = (coords.lon_centroid_dem>149.) \
                                & (coords.lat_centroid_dem>-33.)

    return sites


def get_data(siteid, start=1970):
    version = 2
    fz = FDATA / f"hydro_monthly_v{version}.zip"
    fm = f"monthly_{siteid}.csv"
    with zipfile.ZipFile(fz, "r") as archive:
        mthly, _ = csv.read_csv(fm, parse_dates=True, index_col=0, archive=archive)
        mthly = mthly.rename(columns={"rain[mm/m]": "Rain", \
                                "evap[mm/m]": "Evap", \
                                "runoff[mm/m]": "Qobs"})
    return mthly.loc[f"{start}-07-01":, :]


def get_inputs_and_obs(mthly, pername):
    periodset = Periods().get_periodset(pername)
    itotal = periodset.total.select_index(mthly.index)
    iactive = periodset.active.select_index(mthly.index)

    inputs = np.ascontiguousarray(\
                    mthly.loc[itotal, ["Rain", "Evap"]].values)
    obs = mthly.loc[itotal, "Qobs"].values

    ieval = pd.notnull(obs) & iactive[itotal]

    return inputs, obs, itotal, iactive, ieval

