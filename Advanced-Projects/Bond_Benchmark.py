import os
import json
from collections import defaultdict
from datetime import datetime
import datetime

import numpy as np
import pandas as pd
from typing import Dict, Any
import warnings

import pytz
import seaborn as sns
from matplotlib import pyplot as plt

warnings.filterwarnings("ignore")


class FileManager:
    def __init__(self, base_path: str = r'mini_poc'):
        self.base_path = base_path
        self.files = {
            'listbenchmark': os.path.join(os.path.dirname(os.path.dirname(__file__)), r'/listbenchmark.xlsx'),
            'mts_20jan':os.path.join(os.path.dirname(os.path.dirname(__file__)), r'/MTS20jan17mars.xlsx'),
            'mts_18mar': os.path.join(os.path.dirname(os.path.dirname(__file__)), r'/MTS18mars19juin.xlsx'),
            'pricebenchmark20jan': os.path.join(os.path.dirname(os.path.dirname(__file__)), r'/pricebenchmarkCash.xlsx'),
            'pricebenchmark2may': os.path.join(os.path.dirname(os.path.dirname(__file__)), r'/pricebenchmarkCash02mai.xlsx')
        }

    def read_transac_files(self) -> Dict[str, pd.DataFrame]:
        # Read mts files
        mts_20jan_data = pd.read_excel(self.files['mts_20jan'], sheet_name='Data', usecols='A:O', skiprows=3)
        mts_18mar_data = pd.read_excel(self.files['mts_18mar'], sheet_name='DATA', usecols='A:O', skiprows=3)
        mts_20jan_bond_spec = pd.read_excel(self.files['mts_20jan'], sheet_name='Securities')
        mts_18mar_bond_spec = pd.read_excel(self.files['mts_18mar'], sheet_name='Securities')
        mts_data = pd.concat([mts_20jan_data, mts_18mar_data])
        mts_data.to_pickle('mts_data.pkl')
        mts_bond_spec = pd.concat([mts_20jan_bond_spec, mts_18mar_bond_spec]).drop_duplicates('InstrumentCode',
                                                                                    keep='last')
        mts_bond_spec.to_pickle('mts_bond_spec.pkl')
        return mts_data, mts_bond_spec


    def read_bench_files(self) -> Dict[str, pd.DataFrame]:
            # Read bench isin files
            listbenchmark_per_date = pd.read_excel(self.files['listbenchmark'], sheet_name='perdate', skiprows=1)
            listbenchmark_per_date = listbenchmark_per_date.set_index(['Unnamed: 0'], drop=True)
            listbenchmark_per_date.to_pickle('listbenchmark_per_date.pkl')

            # Read bench yield files
            dictbenchyield = {}
            pricebenchmark20jan_data = pd.ExcelFile(self.files['pricebenchmark20jan'])
            pricebenchmark2may_data = pd.ExcelFile(self.files['pricebenchmark2may'])
            for sheet_name in pricebenchmark20jan_data.sheet_names:
                df = pd.read_excel(pricebenchmark20jan_data, sheet_name=sheet_name, skiprows=6).set_index(
                    'Unnamed: 0').iloc[:, 603:]
                df.columns = pd.to_datetime(df.columns, dayfirst=True).floor('T')
                dictbenchyield[sheet_name] = df
            for sheet_name in pricebenchmark2may_data.sheet_names:
                df = pd.read_excel(pricebenchmark2may_data, sheet_name=sheet_name, skiprows=6).set_index(
                    'Unnamed: 0').iloc[:, 603:]
                df.columns = pd.to_datetime(df.columns, dayfirst=True).floor('T')
                dictbenchyield[sheet_name] = df
            json_data = {key: value.to_json() for key, value in dictbenchyield.items()}
            with open('dictbenchyield.json', 'w') as f:
                json.dump(json_data, f, indent=4)
            return listbenchmark_per_date, dictbenchyield



class DataProcessor:
    def __init__(self, file_manager: FileManager):
        self.file_manager = file_manager

    @staticmethod
    def identify_flagged_rows(df):
        # Convert str dates to datetime
        df['Trading date and time'] = pd.to_datetime(df['Trading date and time'])
        # delete dates where benchmark not available
        df = df[df['Trading date and time'] <= pd.Timestamp('2023-06-14', tz='UTC')]
        # Calculate time difference
        df['time_diff'] = df['Trading date and time'].diff().dt.total_seconds()
        # Create condition for group
        df['new_group'] = (df['time_diff'] >= 0.01) | pd.isna(df['time_diff'])
        df['group_id'] = df['new_group'].cumsum()
        # Only take groups with 2 rows or more
        grouped = df.groupby('group_id')
        grouped_dfs = [group_df for _, group_df in grouped if len(group_df) >= 2]
        # Add order type
        list_modified_groups = []
        for group in grouped_dfs:
            price_changes = group['Price'].diff()
            price_changes.iloc[0] = 0
            group['Order'] = None
            if all(price_changes >= 0):
                group['Order'] = 'buy'
            elif all(price_changes <= 0):
                group['Order'] = 'sell'
            else:
                group['Order'] = 'N/A'
            list_modified_groups.append(group)
        return pd.concat(list_modified_groups, ignore_index=True).drop(['time_diff', 'new_group'], axis=1)

    def sort_by_duration(self, mts_bond_spec: pd.DataFrame, mts_data: pd.DataFrame) -> pd.DataFrame:
        # Flag rows in transaction data based on specific criteria using identify_flagged_rows method
        mts_data_flagged = self.identify_flagged_rows(mts_data)
        data_flagged_matu = (mts_data_flagged.merge(mts_bond_spec, left_on='Instrument id code',
                                                    right_on='InstrumentCode', how='left')).drop(['InstrumentCode'], axis=1)
        data_flagged_matu['duration'] = data_flagged_matu[' Notional amount '] * data_flagged_matu['matu'] * 0.001
        grouped_sum_sorted = data_flagged_matu.groupby('group_id', as_index=False)['duration'].sum().sort_values(
            by='duration')
        # Sort by duration
        data_flagged_matu_sorted = (
            pd.merge(data_flagged_matu, grouped_sum_sorted, on='group_id', suffixes=('', '_sum'))). \
            sort_values(by='duration_sum', ascending=False)
        # Add constraints of mini-poc
        sorted_constrained_data = data_flagged_matu_sorted[(data_flagged_matu_sorted['Price currency'] == 'EUR') &
                                                    (data_flagged_matu_sorted['matu'] >= 2) & (
                                                                data_flagged_matu_sorted['CPN_TYP'] == 'FIXED')
                                                    & (data_flagged_matu_sorted['Country'].isin(
                                                        ['DE', 'IT', 'ES', 'PT', 'BE', 'NL', 'FR']))
                                                    & (data_flagged_matu_sorted['INFLATION_LINKED_INDICATOR'] == 'N')]
        return sorted_constrained_data

    def build_entry_data(self, df_flagged):
        dict_transactions = defaultdict(dict)
        for group in list(df_flagged['group_id'].unique()):
            # Discard group if more than 1 country and Create Entry Data
            if len((df_flagged.loc[df_flagged['group_id'] == group, 'Country']).unique()) > 1:
                pass
            else:
                dict_transactions[group]['Datetime'] = \
                list(df_flagged.loc[df_flagged['group_id'] == group, 'Trading date and time'])[0]
                dict_transactions[group]['Group_ID'] = group
                dict_transactions[group]['Issuing_Country'] = \
                list(df_flagged.loc[df_flagged['group_id'] == group, 'Country'])[0]
                dict_transactions[group]['Order'] = list(df_flagged.loc[df_flagged['group_id'] == group, 'Order'])[0]
                dict_transactions[group]['Lower_Bond_Maturity'] = (
                df_flagged.loc[df_flagged['group_id'] == group, 'matu']).min()
                dict_transactions[group]['Upper_Bond_Maturity'] = (
                df_flagged.loc[df_flagged['group_id'] == group, 'matu']).max()
                dict_transactions[group]['Total_Notional_Size'] = (df_flagged.loc[df_flagged['group_id'] == group,
                                                                                  ' Notional amount ']).sum()
                dict_transactions[group]['Total_Duration_Size'] = (df_flagged.loc[df_flagged['group_id'] == group,
                                                                                  'duration']).sum()
                dict_transactions[group]['Weighted_Maturity'] = (
                            ((df_flagged.loc[df_flagged['group_id'] == group, 'matu'])
                             * (df_flagged.loc[df_flagged['group_id'] == group, 'Quantity'])) / \
                            ((df_flagged.loc[df_flagged['group_id'] == group, 'Quantity']).sum())).sum()
        entry_data = pd.DataFrame.from_dict(dict_transactions, orient='index').reset_index(drop=True)
        return entry_data

    def process_data(self) -> Dict[str, Any]:
        # mts_data, mts_bond_spec = self.file_manager.read_transac_files()
        mts_bond_spec = pd.read_pickle("mts_bond_spec.pkl")
        mts_data = pd.read_pickle("mts_data.pkl")

        if mts_bond_spec is None or mts_data is None:
            raise ValueError("Missing required data for processing.")

        sorted_constrained_data = self.sort_by_duration(mts_bond_spec, mts_data)
        entry_data = self.build_entry_data(sorted_constrained_data)
        return entry_data

class BenchmarkMatcher:
    def __init__(self, data_processor: DataProcessor):
        self.data_processor = data_processor
        self.file_manager = data_processor.file_manager
        self.dict_country_code={"DE": "German", "IT":"Italy", "ES":"Spanish", "PT":"Portuguese", "BE":"Belgian", "NL":"Netherlands",
                           "FR":"France"}
        self.maturity_benchmarks = {2: "2Y", (2, 5): ("2Y", "5Y"),5: "5Y", (5, 10): ("5Y", "10Y"),
                                    10: "10Y", (10, 15): ("10Y", "15Y"), 15: "15Y", (15, 30): ("15Y", "30Y"), 30: "30Y"}
        self.timings={"10min_impact": 10, '2h_impact': 120, 'EOD_impact': -1, 'next_EOD_impact': -1}
        self.country_spreads={'IT': ['DE'], 'ES': ['DE', 'IT'], 'PT':['DE', 'IT'], 'BE':['DE', 'FR'], 'NL': ['DE', 'FR'],
                              'FR': ['DE']}

    def get_benchmark_curves(self, country, weighted_maturity):
        for key, benchmarks in self.maturity_benchmarks.items():
            if isinstance(key, tuple) and key[0] < weighted_maturity < key[1]:
                return [f"{country} Sovereign Curve  -  {yr}" for yr in benchmarks]
            elif weighted_maturity == key:
                return [f"{country} Sovereign Curve  -  {yr}" for yr in benchmarks]
            elif weighted_maturity > 30:
                return [f"{country} Sovereign Curve  -  30Y"]
        return ('N/A',)

    def match_bond_bench(self, entry_data):
        data_bond_bench = entry_data.copy()
        data_bond_bench['Benchmark_Curves'] = None
        for row in range(len(data_bond_bench)):
            country_code = data_bond_bench.loc[row, 'Issuing_Country']
            country = self.dict_country_code[country_code]
            weighted_maturity = float(data_bond_bench.loc[row, 'Weighted_Maturity'])
            data_bond_bench.at[row, 'Benchmark_Curves'] = self.get_benchmark_curves(country, weighted_maturity)
        return data_bond_bench

    def match_bench_isin(self, listbench, data_bond_bench):
        data_bench_isin = data_bond_bench.copy()
        data_bench_isin['Bench_Isin'] = None
        for row in range(0, len(data_bench_isin)):
            list_bench_isin = []
            date = data_bench_isin.loc[row, 'Datetime'].normalize().to_pydatetime().replace(tzinfo=None)
            for bench in data_bench_isin.loc[row, 'Benchmark_Curves']:
                list_bench_isin.append((listbench.loc[listbench.index == bench, date])[0])
            data_bench_isin.at[row, 'Bench_Isin'] = list_bench_isin
        return data_bench_isin

    @staticmethod
    def identify_matu_bounds(matu):
        if 2 < matu < 5:
            bounds = [2, 5]
        elif 5 < matu < 10:
            bounds = [5, 10]
        elif 10 < matu < 15:
            bounds = [10, 15]
        else:
            bounds = [15, 30]
        return bounds

    @staticmethod
    def equivalent_benchmark_yield(maturity, bounds):
        weights = []
        for threshold in bounds:
            if threshold > maturity:
                allocation = 1 - ((bounds[1] - maturity) / bounds[0])
            else:
                allocation = 1 - ((maturity - bounds[0]) / bounds[0])
            weights.append(allocation)
        return weights

    def get_curve_benchmark_dates(self, maturity, country_code, listbench, dictbenchyield, date_normalized,
                                  date_sheet_select_before_exec,date_before_exec, date_sheet_select_after_exec, date_after_exec):
        country = self.dict_country_code[country_code]
        # Classify according to poc table benchmark dates for each country
        if maturity in [2, 5] or maturity in [5,10]:
            benchmark_10y = listbench.loc[listbench.index == f"{country} Sovereign Curve  -  10Y", date_normalized][0]
            benchmark_5y = listbench.loc[listbench.index == f"{country} Sovereign Curve  -  5Y", date_normalized][0]
            yield_after_exec_10y = float(dictbenchyield[date_sheet_select_after_exec].loc[benchmark_10y, date_after_exec])
            yield_before_exec_10y = float(dictbenchyield[date_sheet_select_before_exec].loc[benchmark_10y, date_before_exec])
            yield_after_exec_5y = float(dictbenchyield[date_sheet_select_after_exec].loc[benchmark_5y, date_after_exec])
            yield_before_exec_5y = float(dictbenchyield[date_sheet_select_before_exec].loc[benchmark_5y, date_before_exec])
            yield_10y=yield_after_exec_10y-yield_before_exec_10y
            yield_5y=yield_after_exec_5y-yield_before_exec_5y
            curve_benchmark_dates = yield_10y - yield_5y
        elif maturity in [10,15]:
            benchmark_15y = listbench.loc[listbench.index == f"{country} Sovereign Curve  -  15Y", date_normalized][0]
            benchmark_10y = listbench.loc[listbench.index == f"{country} Sovereign Curve  -  10Y", date_normalized][0]
            yield_after_exec_15y = float(dictbenchyield[date_sheet_select_after_exec].loc[benchmark_15y, date_after_exec])
            yield_before_exec_15y = float(dictbenchyield[date_sheet_select_before_exec].loc[benchmark_15y, date_before_exec])
            yield_after_exec_10y = float(dictbenchyield[date_sheet_select_after_exec].loc[benchmark_10y, date_after_exec])
            yield_before_exec_10y = float(dictbenchyield[date_sheet_select_before_exec].loc[benchmark_10y, date_before_exec])
            yield_15y=yield_after_exec_15y-yield_before_exec_15y
            yield_10y=yield_after_exec_10y-yield_before_exec_10y
            curve_benchmark_dates = yield_15y - yield_10y
        elif maturity in [15,30]:
            benchmark_30y = listbench.loc[listbench.index == f"{country} Sovereign Curve  -  30Y", date_normalized][0]
            benchmark_10y = listbench.loc[listbench.index == f"{country} Sovereign Curve  -  10Y", date_normalized][0]
            yield_after_exec_30y = float(dictbenchyield[date_sheet_select_after_exec].loc[benchmark_30y, date_after_exec])
            yield_before_exec_30y = float(dictbenchyield[date_sheet_select_before_exec].loc[benchmark_30y, date_before_exec])
            yield_after_exec_10y = float(dictbenchyield[date_sheet_select_after_exec].loc[benchmark_10y, date_after_exec])
            yield_before_exec_10y = float(dictbenchyield[date_sheet_select_before_exec].loc[benchmark_10y, date_before_exec])
            yield_30y=yield_after_exec_30y-yield_before_exec_30y
            yield_10y=yield_after_exec_10y-yield_before_exec_10y
            curve_benchmark_dates = yield_30y - yield_10y
        else:
            benchmark_30y = listbench.loc[listbench.index == f"{country} Sovereign Curve  -  30Y", date_normalized][0]
            yield_after_exec_30y = float(dictbenchyield[date_sheet_select_after_exec].loc[benchmark_30y, date_after_exec])
            yield_before_exec_30y = float(dictbenchyield[date_sheet_select_before_exec].loc[benchmark_30y, date_before_exec])
            yield_30y=yield_after_exec_30y-yield_before_exec_30y
            curve_benchmark_dates=yield_30y
        return curve_benchmark_dates

    def get_country_yields(self, listbench, dictbenchyield, country_code, maturity, date_normalized,
                           date_sheet_select, date):
        country = self.dict_country_code[country_code]
        # Find the correct benchmarks based on maturity
        if maturity in [2, 5, 10, 15, 30]:
            benchmark = listbench.loc[listbench.index == f"{country} Sovereign Curve  -  " \
                                                         f"{self.maturity_benchmarks[maturity]}", date_normalized][0]
            yields = [float(dictbenchyield[date_sheet_select].loc[benchmark, date])]
        elif maturity > 30:
            benchmark = listbench.loc[
                listbench.index == f"{country} Sovereign Curve  -  30Y", date_normalized][0]
            yields = [float(dictbenchyield[date_sheet_select].loc[benchmark, date])]
        else:
            for key, value in self.maturity_benchmarks.items():
                if isinstance(key, tuple):
                    lower, upper = key
                    if lower < maturity < upper:
                        # Handle maturity ranges and extract bench
                        lower_benchmark_isin = listbench.loc[
                            listbench.index == f"{country} Sovereign Curve  -  {value[0]}", date_normalized][0]
                        upper_benchmark_isin = listbench.loc[
                            listbench.index == f"{country} Sovereign Curve  -  {value[1]}", date_normalized][0]
                        yields = [float(dictbenchyield[date_sheet_select].loc[lower_benchmark_isin, date]),
                                  float(dictbenchyield[date_sheet_select].loc[upper_benchmark_isin, date])]

        return yields

    def get_curve_exact_dates(self, country_code, listbench, dictbenchyield, date_normalized,
                              maturity, date_sheet_select_before_exec, date_before_exec,
                              date_sheet_select_after_exec, date_after_exec, outright_rate):
        country = self.dict_country_code[country_code]
        benchmark_10y = listbench.loc[
            listbench.index == f"{country} Sovereign Curve  -  10Y", date_normalized][0]
        benchmark_30y=listbench.loc[listbench.index == f"{country} Sovereign Curve  -  30Y", date_normalized][0]
        # Classify according to DE poc table
        if maturity in [2,5] or maturity in [5,10]:
            yield_after_exec = float(dictbenchyield[date_sheet_select_after_exec].loc[benchmark_10y, date_after_exec])
            yield_before_exec = float(dictbenchyield[date_sheet_select_before_exec].loc[benchmark_10y, date_before_exec])
            bench_yield_select=yield_after_exec-yield_before_exec
            curve_exact_dates=bench_yield_select-outright_rate

        elif maturity in [10,15] or maturity in [15,30]:
            yield_after_exec = float(dictbenchyield[date_sheet_select_after_exec].loc[benchmark_10y, date_after_exec])
            yield_before_exec = float(dictbenchyield[date_sheet_select_before_exec].loc[benchmark_10y, date_before_exec])
            bench_yield_select=yield_after_exec-yield_before_exec
            curve_exact_dates=outright_rate-bench_yield_select
        else:
            yield_after_exec = float(dictbenchyield[date_sheet_select_after_exec].loc[benchmark_30y, date_after_exec])
            yield_before_exec = float(dictbenchyield[date_sheet_select_before_exec].loc[benchmark_30y, date_before_exec])
            bench_yield_select=yield_after_exec-yield_before_exec
            curve_exact_dates=outright_rate-bench_yield_select
        return curve_exact_dates

    def match_isin_yield(self, dictbenchyield, listbench, data_bench_isin):
        data_isin_yield = data_bench_isin.copy()
        dict_time_impact = defaultdict(lambda: defaultdict(dict))
        for row in range(0, len(data_isin_yield)):
            country_code = data_isin_yield.loc[row, 'Issuing_Country']
            maturity = data_isin_yield.loc[row, 'Weighted_Maturity']
            bounds = self.identify_matu_bounds(maturity)
            weights = self.equivalent_benchmark_yield(maturity, bounds)
            date_before_exec=(data_isin_yield.loc[row, 'Datetime']- datetime.timedelta(minutes=1)).\
                        to_pydatetime().replace(tzinfo=None, second=0, microsecond=0)
            date_sheet_select_before_exec = data_isin_yield.loc[row, 'Datetime'].strftime('%d%m')
            # Select sheet and date according to each timing
            for timing_name, timing_value in self.timings.items():
                if timing_name not in ['EOD_impact', 'next_EOD_impact']:
                    date_sheet_select_after_exec = data_isin_yield.loc[row, 'Datetime'].strftime('%d%m')
                    date_after_exec = (data_isin_yield.loc[row, 'Datetime']+ datetime.timedelta(minutes=timing_value)).\
                        to_pydatetime().replace(tzinfo=None, second=0, microsecond=0)
                    date_normalized = (data_bench_isin.loc[row, 'Datetime']+ datetime.timedelta(minutes=timing_value)).\
                        normalize().to_pydatetime().replace(tzinfo=None)
                elif timing_name=='EOD_impact':
                    date_sheet_select_after_exec = data_isin_yield.loc[row, 'Datetime'].strftime('%d%m')
                    date_after_exec=dictbenchyield[date_sheet_select_after_exec].columns[timing_value]
                    date_normalized = dictbenchyield[date_sheet_select_after_exec].columns[timing_value].normalize()
                else:
                    successful = False
                    next_available_day = 1
                    while not successful:
                        try:
                            # Calculate the next date after execution
                            date_sheet_select_after_exec = (
                                    data_isin_yield.loc[row, 'Datetime'] + datetime.timedelta(days=next_available_day)
                            ).strftime('%d%m')
                            # Attempt to access the desired data
                            date_after_exec = dictbenchyield[date_sheet_select_after_exec].columns[timing_value]
                            date_normalized = dictbenchyield[date_sheet_select_after_exec].columns[
                                timing_value].normalize()
                            successful = True  # If the above lines did not raise an exception, mark as successful
                        except (KeyError, IndexError):
                            # KeyError for missing date_sheet_select_after_exec,
                            # IndexError for invalid timing_value
                            next_available_day += 1


                # Linear combination of irregular maturity for outright rate
                if len(data_isin_yield.loc[row, 'Bench_Isin']) > 1:
                    try:
                        yield_lower_bench_after_exec = dictbenchyield[date_sheet_select_after_exec].\
                            loc[data_isin_yield.loc[row, 'Bench_Isin'][0], date_after_exec]
                        yield_upper_bench_after_exec = dictbenchyield[date_sheet_select_after_exec].\
                            loc[data_isin_yield.loc[row, 'Bench_Isin'][1], date_after_exec]
                        yield_lower_bench_before_exec = dictbenchyield[date_sheet_select_before_exec].\
                            loc[data_isin_yield.loc[row, 'Bench_Isin'][0], date_before_exec]
                        yield_upper_bench_before_exec = dictbenchyield[date_sheet_select_before_exec].\
                            loc[data_isin_yield.loc[row, 'Bench_Isin'][1], date_before_exec]
                        dict_time_impact[row][timing_name]['Outright_Rate'] =(weights[0] * yield_lower_bench_after_exec + weights[1]
                                                                              * yield_upper_bench_after_exec)-(weights[0] *
                                                                            yield_lower_bench_before_exec + weights[1] *
                                                                                        yield_upper_bench_before_exec)
                    # After exec not in market hours -> take eod
                    except Exception as e:
                        try:
                            date_after_exec = dictbenchyield[date_sheet_select_after_exec].columns[-1]
                            yield_lower_bench_after_exec = dictbenchyield[date_sheet_select_after_exec]. \
                                loc[data_isin_yield.loc[row, 'Bench_Isin'][0], date_after_exec]
                            yield_upper_bench_after_exec = dictbenchyield[date_sheet_select_after_exec]. \
                                loc[data_isin_yield.loc[row, 'Bench_Isin'][1], date_after_exec]
                            yield_lower_bench_before_exec = dictbenchyield[date_sheet_select_before_exec]. \
                                loc[data_isin_yield.loc[row, 'Bench_Isin'][0], date_before_exec]
                            yield_upper_bench_before_exec = dictbenchyield[date_sheet_select_before_exec]. \
                                loc[data_isin_yield.loc[row, 'Bench_Isin'][1], date_before_exec]
                            dict_time_impact[row][timing_name]['Outright_Rate'] = (weights[0] * yield_lower_bench_after_exec +
                                                                                   weights[1]* yield_upper_bench_after_exec) - (weights[0] *
                                                                                              yield_lower_bench_before_exec + weights[1] *
                                                                                              yield_upper_bench_before_exec)
                        except Exception as e:
                            dict_time_impact[row][timing_name]['Outright_Rate'] = 0


                # Regular maturity for outright rate
                else:
                    try:
                        dict_time_impact[row][timing_name]['Outright_Rate'] = (float(dictbenchyield[date_sheet_select_after_exec].loc
                                                                                [data_isin_yield.loc[row, 'Bench_Isin'], date_after_exec]))-\
                                                                                (float(dictbenchyield[date_sheet_select_before_exec].loc
                                                                                       [data_isin_yield.loc[row, 'Bench_Isin'], date_before_exec]))
                    # After exec not in market hours -> take eod
                    except Exception as e:
                        date_after_exec = dictbenchyield[date_sheet_select_after_exec].columns[-1]
                        dict_time_impact[row][timing_name]['Outright_Rate'] = (float(dictbenchyield[date_sheet_select_after_exec].loc
                                                                                [data_isin_yield.loc[row, 'Bench_Isin'], date_after_exec])) - \
                                                                              (float(dictbenchyield[ date_sheet_select_before_exec].loc
                                                                                     [data_isin_yield.loc[row, 'Bench_Isin'], date_before_exec]))


                # Country_spread
                if country_code=='DE':
                    try:
                        dict_time_impact[row][timing_name]['Curve_Exact_Dates'] =self.get_curve_exact_dates(country_code, listbench,
                                    dictbenchyield, date_normalized, maturity, date_sheet_select_before_exec, date_before_exec,
                                    date_sheet_select_after_exec, date_after_exec,dict_time_impact[row][timing_name]['Outright_Rate'])
                    # After exec not in market hours -> take eod
                    except Exception as e:
                        date_after_exec = dictbenchyield[date_sheet_select_after_exec].columns[-1]
                        date_normalized = dictbenchyield[date_sheet_select_after_exec].columns[-1].normalize()
                        dict_time_impact[row][timing_name]['Curve_Exact_Dates'] = self.get_curve_exact_dates(
                            country_code, listbench,dictbenchyield, date_normalized, maturity, date_sheet_select_before_exec, date_before_exec,
                            date_sheet_select_after_exec, date_after_exec,dict_time_impact[row][timing_name]['Outright_Rate'])

                else:
                    for country_spread in self.country_spreads[country_code]:
                        try:
                            yields_after_exec=self.get_country_yields(listbench, dictbenchyield,
                                            country_spread, maturity, date_normalized, date_sheet_select_after_exec, date_after_exec)
                            yields_before_exec=self.get_country_yields(listbench, dictbenchyield,
                                            country_spread, maturity, date_normalized, date_sheet_select_before_exec, date_before_exec)
                            if len(yields_after_exec)==1 and len(yields_before_exec)==1:
                                yield_actual_matu_other_country=yields_after_exec[0]-yields_before_exec[0]
                            else:
                                yield_actual_matu_other_country=(weights[0] * yields_after_exec[0] + weights[1]
                                                                 * yields_after_exec[1])-(weights[0] * yields_before_exec[0] +
                                                                                          weights[1] * yields_before_exec[1])
                            dict_time_impact[row][timing_name][f'Country_Spread_Diff_{country_spread}']=\
                                            dict_time_impact[row][timing_name]['Outright_Rate']-yield_actual_matu_other_country
                        # After exec not in market hours -> take eod
                        except Exception as e:
                            date_after_exec = dictbenchyield[date_sheet_select_after_exec].columns[-1]
                            date_normalized = dictbenchyield[date_sheet_select_after_exec].columns[-1].normalize()
                            yields_after_exec=self.get_country_yields(listbench, dictbenchyield,
                                            country_spread, maturity, date_normalized, date_sheet_select_after_exec, date_after_exec)
                            yields_before_exec=self.get_country_yields(listbench, dictbenchyield,
                                            country_spread, maturity, date_normalized, date_sheet_select_before_exec, date_before_exec)
                            if len(yields_after_exec)==1 and len(yields_before_exec)==1:
                                yield_actual_matu_other_country=yields_after_exec[0]-yields_before_exec[0]
                            else:
                                yield_actual_matu_other_country=(weights[0] * yields_after_exec[0] + weights[1]
                                                                 * yields_after_exec[1])-(weights[0] * yields_before_exec[0] +
                                                                                          weights[1] * yields_before_exec[1])
                            dict_time_impact[row][timing_name][f'Country_Spread_Diff_{country_spread}']=\
                                            dict_time_impact[row][timing_name]['Outright_Rate']-yield_actual_matu_other_country


                # Curve with benchmark dates
                try:
                    dict_time_impact[row][timing_name]['Curve_Benchmark_Dates'] = \
                        self.get_curve_benchmark_dates(maturity, country_code, listbench, dictbenchyield,date_normalized,
                        date_sheet_select_before_exec, date_before_exec, date_sheet_select_after_exec, date_after_exec)
                # After exec not in market hours -> take eod
                except Exception as e:
                    date_after_exec = dictbenchyield[date_sheet_select_after_exec].columns[-1]
                    date_normalized = dictbenchyield[date_sheet_select_after_exec].columns[-1].normalize()
                    dict_time_impact[row][timing_name]['Curve_Benchmark_Dates'] = \
                        self.get_curve_benchmark_dates(maturity, country_code, listbench, dictbenchyield,date_normalized,
                        date_sheet_select_before_exec, date_before_exec, date_sheet_select_after_exec, date_after_exec)

        df_output=pd.DataFrame(dict_time_impact).T
        for col in df_output.columns:
            dict_cols = df_output[col].apply(pd.Series)
            dict_cols.columns = [f"{col}_{sub_col}" for sub_col in dict_cols.columns]
            df_output = pd.concat([df_output, dict_cols], axis=1)
            df_output.drop(col, axis=1, inplace=True)
        df_output=pd.merge(data_isin_yield, df_output, left_index=True, right_index=True)
        df_output.to_csv('output_data.csv')
        return df_output


    def process_matching(self, entry_data: pd.DataFrame) -> pd.DataFrame:
        # Main method to process and match benchmarks
        # listbench, dictbenchyield=self.file_manager.read_bench_files()
        listbench = pd.read_pickle("listbenchmark_per_date.pkl")
        with open('dictbenchyield.json', 'r') as f:
            json_dictbenyield = json.load(f)
        dictbenchyield = {key: pd.read_json(value) for key, value in json_dictbenyield.items()}
        data_bond_bench = self.match_bond_bench(entry_data)
        data_bench_isin = self.match_bench_isin(listbench, data_bond_bench)
        return self.match_isin_yield(dictbenchyield, listbench, data_bench_isin)

class ResultInterpretor:
    def __init__(self, test_data, output_data):
        self.test_data=test_data
        self.output_data=output_data

    def linear_interpolation(self):
        # Filter rows matching the Issuing Country from test data
        df_output_country = self.output_data[self.output_data['Issuing_Country'] == self.test_data['Issuing_Country']]
        df_output_country = df_output_country.drop(columns=['Group_ID', 'Lower_Bond_Maturity', 'Upper_Bond_Maturity',
                                            'Total_Duration_Size', 'Benchmark_Curves', 'Bench_Isin'])
        #drop col when bench spread with one country not needed
        df_output_country = df_output_country.dropna(axis=1)
        # Create a new DataFrame for the input to be added
        new_row_df = pd.DataFrame([self.test_data])
        now = datetime.datetime.now(pytz.utc)
        new_row_df['Datetime'] = now
        df_output_country = pd.concat([df_output_country, new_row_df], ignore_index=True)
        df_output_country['Datetime'] = pd.to_datetime(df_output_country['Datetime'])
        # Calculate the time differences for all rows except the last (new) row
        df_output_country['Time_Diff'] = (now - df_output_country['Datetime']).dt.total_seconds()
        df_output_country_except_last = df_output_country.iloc[:-1]
        min_time_diff = df_output_country_except_last['Time_Diff'].min()
        # Calculate weights for interpolation
        df_output_country_except_last['Weight'] = min_time_diff / df_output_country_except_last['Time_Diff']
        df_output_country_except_last['Weight'] /= df_output_country_except_last['Weight'].sum()
        df_output_country['Weight'] = pd.concat(
            [df_output_country_except_last['Weight'], pd.Series([np.nan])]).reset_index(drop=True)
        # Interpolate the other column values using the weights
        df_output_country.to_csv('df_output_before_linear_interpol.csv')
        for col in df_output_country.columns:
            if col not in ['Datetime', 'Issuing_Country', 'Country', 'Time_Diff', 'Weight']:
                if pd.isnull(df_output_country.loc[df_output_country.index[-1], col]):
                    weighted_average = np.average(df_output_country_except_last[col].dropna(),
                                                  weights=df_output_country_except_last['Weight'].dropna())
                    df_output_country.loc[df_output_country.index[-1], col] = weighted_average
        return df_output_country

    def get_statistics_results(self):
        df_output_with_new_input=self.linear_interpolation()
        # Calculating statistics
        expected_range = df_output_with_new_input.select_dtypes(include=[np.number]).apply(lambda x: (x.min(), x.max()))
        mean = df_output_with_new_input.select_dtypes(include=[np.number]).mean()
        median = df_output_with_new_input.select_dtypes(include=[np.number]).median()
        std_deviation = df_output_with_new_input.select_dtypes(include=[np.number]).std()
        skewness = df_output_with_new_input.select_dtypes(include=[np.number]).skew()
        kurtosis = df_output_with_new_input.select_dtypes(include=[np.number]).kurtosis()

        # Print calculated statistics
        print("Expected Range:", expected_range)
        print("Mean:", mean)
        print("Median:", median)
        print("Standard Deviation:", std_deviation)
        print("Skewness:", skewness)
        print("Kurtosis:", kurtosis)

        # Visualizing the distribution of each numerical column
        for column in df_output_with_new_input.select_dtypes(include=[np.number]).columns:
            plt.figure(figsize=(10, 6))
            sns.histplot(df_output_with_new_input[column], kde=True)
            plt.title(f'Distribution of {column}')
            plt.xlabel(column)
            plt.ylabel('Frequency')
            plt.savefig(f"Distribution_of_{column}.png")
        df_output_with_new_input.to_csv('output_data_with_new_input.csv')


if __name__ == '__main__':
    # Setup file manager, data processor, and benchmark matcher
    file_manager = FileManager()
    dataprocessor = DataProcessor(file_manager)
    benchmark_matcher = BenchmarkMatcher(dataprocessor)

    # Process data and match benchmarks
    entry_data = dataprocessor.process_data()
    output_data = benchmark_matcher.process_matching(entry_data)

    # Perform result interpretation with a hypothetical test data
    test_data={'Issuing_Country':'IT', 'Order':'sell', 'Weighted_Maturity': 25, 'Total_Notional_Size': 50000000}
    result_interpretor = ResultInterpretor(test_data, output_data).get_statistics_results()
