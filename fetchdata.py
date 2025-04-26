import cbbpy.mens_scraper as scraper
import pandas as pd
import matplotlib.pyplot as plt
import concurrent.futures
class DataCollector:
    def __init__(self, filename="data.csv"):
        self.data_file = filename
    def scrape(self, seasons=[2025], info=True, box=True, pbp=False) -> None:
        all_data = []
        def fetch_data(season):
            return scraper.get_games_season(season, info=info, box=box, pbp=pbp)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = executor.map(fetch_data, seasons)
        for data in results:
            for element in data:
                if isinstance(element, pd.DataFrame):
                    all_data.append(element)

        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            combined_df.to_csv(self.data_file, index=False)
            print(f"Saved combined data to {self.data_file}")
        else:
            print("No data was collected.")
    def read_csv(self, number: int, fillna:any=0) -> pd.DataFrame:
        return pd.read_csv( self.get_filename(number) ).fillna(fillna)
    def get_filename(self, number: int) -> str:
        parts = self.data_file.split('.')
        parts[0] = parts[0] + str(number)
        return ".".join(parts)
    def print_df(self, number: int=0) -> None:
        print("----------",self.get_filename(number),"----------")
        print(self.read_csv(number))
        return



