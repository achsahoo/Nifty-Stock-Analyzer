import os
import pandas as pd

DTE_DIR = r"D:\NIFTY_MODEL\dtes_date"

class DTEManager:
    def __init__(self, dte_dir=DTE_DIR):
        self.dte_dir = dte_dir
        self.dte_dates = {}
        self._load_dates()
        
    def _load_dates(self):
        for i in range(5):
            fn = f"unique_{i}dte.csv"
            fp = os.path.join(self.dte_dir, fn)
            if os.path.exists(fp):
                df = pd.read_csv(fp)
                # Store set of string dates 'YYYY-MM-DD'
                dates = set(pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d'))
                self.dte_dates[f"{i} DTE"] = dates
            else:
                self.dte_dates[f"{i} DTE"] = set()
                
    def get_dates_for_dtes(self, selected_dtes):
        """Returns union of date strings for selected DTEs."""
        matched = set()
        for dte in selected_dtes:
            if dte in self.dte_dates:
                matched = matched.union(self.dte_dates[dte])
        return matched
        
    def get_dte_label_for_date(self, date_str):
        for dte_lbl, dates in self.dte_dates.items():
            if date_str in dates:
                return dte_lbl
        return "Unknown"

    def get_dte(self, date_str):
        for dte_lbl, dates in self.dte_dates.items():
            if date_str in dates:
                return dte_lbl
        return None
