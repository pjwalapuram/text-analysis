import sys
import requests
import pandas as pd
from bs4 import BeautifulSoup

def get_csv_data(filepath: str, columns: list = None) -> pd.DataFrame:
	rows = pd.read_csv(filepath)
	if columns:
		return rows[columns]
	else:
		return rows

def concatenate_columns(rows: list, merge_columns: list) -> pd.DataFrame:
	rows["merged_text"] = rows[merge_columns].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
	return rows


def get_html_data(urlpath: str):
	response = requests.get(urlpath)
	soup = BeautifulSoup(response.text, features='html.parser')
	for script in soup(["script", "style"]):
		script.extract()
	return soup.get_text()



	
