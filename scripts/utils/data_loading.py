import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime


def get_data_file_paths(data_root):
	data_paths = []
	tickers = os.listdir(data_root)
	tickers = [ticker for ticker in tickers if '.DS_Store' not in ticker]
	for ticker in tickers:
		ticker_path = os.path.join(data_root, ticker)
		for period_identifier in os.listdir(ticker_path):
			if '.DS_Store' in period_identifier:
				continue
			data_paths.append(os.path.join(ticker_path, period_identifier))
	data_paths = np.unique(['/'.join(path.split('/')[:-1]) for path in data_paths])
	return data_paths


def load_data(data_directory, start_time=None, end_time=None, max_files=None):
	"""Load parquet data from specified directory. By default loads all data within directory. However,
	if too much data, user can apply date filter to only load data within dates.


	PARAMETERS
	----------
	data_directory : str
		Path which specifies the location contained the (compressed) parquet files.
	start_time : datetime.datetime
		(Optional) Loads all files created after this date. Can also pass timestamp if preferred.
	end_time : datetime.datetime
		(Optional, required with start_time) Loads files created before this date. Can also pass timestamp if preferred.

	RETURNS
	-------
	pandas.DataFrame : A data frame containing the data in the parquet files.
	"""

	# select files within date inputs
	files = np.sort([f for f in os.listdir(data_directory) if os.path.isfile(os.path.join(data_directory, f))])

	# if desired, only select files within date range
	if (start_time is not None) and (end_time is not None):

		# reformat time inputs in case they are timestamps
		if not isinstance(start_time, datetime):
			start_time = datetime.fromtimestamp(int(start_time))
		if not isinstance(end_time, datetime):
			end_time = datetime.fromtimestamp(int(end_time))

		# apply date mask and select files
		dates = np.array([datetime.fromtimestamp(int(f.split('.')[0])) for f in files])
		file_mask = (start_time <= dates) & (dates < end_time)
		files = files[file_mask]

	# load selected data frames
	data_frame_list = []
	for file_count, file in tqdm(enumerate(files)):
		if file_count == max_files:
			break  # break and return file
		filepath = os.path.join(data_directory, file)
		df = pd.read_parquet(filepath)
		data_frame_list.append(df)

	return pd.concat(data_frame_list, sort=False)


if __name__ == '__main__':
	# example usage

	# parameters
	path_to_data = '/Users/maxlamberti/Desktop/XETHZUSD'
	start_timestamp = 1580109897
	end_timestamp = 1580115303
	start_time = datetime.fromtimestamp(start_timestamp)
	end_time = datetime.fromtimestamp(end_timestamp)

	# load data
	df = load_data(path_to_data, start_time, end_time)
	equivalent_df_using_timestamps = load_data(path_to_data, start_timestamp, end_timestamp)
# might_eat_your_ram_df = load_data(path_to_data)
