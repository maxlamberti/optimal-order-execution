import os
import feather
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime, time


def safe_mkdir(dir):
	sub_dir = os.path.join('/'.join(dir.split('/')[:-1]))
	if os.path.exists(sub_dir) and not os.path.exists(dir):
		os.mkdir(dir)
	elif not os.path.exists(sub_dir):
		safe_mkdir(sub_dir)
		safe_mkdir(dir)


def read_data(filename):

	if '.csv' in filename:
		df = pd.read_csv(filename)
	elif '.feather' in filename:
		df = feather.read_dataframe(filename)
	else:
		raise Exception("File format not understood: {}".format(filename))

	return df

if __name__ == '__main__':

	OB_FILE = 'ob.feather'
	TRADES_FILE = 'trades.feather'
	DATA_DIRECTORY_PATH = '../../data/onetick/raw/'
	SAVE_DIRECTORY_PATH = '../../data/onetick/processed_2/'
	DURATION = int(3 * 60 * 60)  # in seconds
	MIN_TRADES_IN_PERIOD = 1  # if there are less than this amount of trades, won't save a file
	NUM_DATA = 4 * 4 * 250  # four data points per day
	NUM_5SECS_IN_TRADING_DAY = int(6.5 * 60 * 12)
	DURATION_IN_5SEC_UNITS = int(DURATION / 5)
	MAX_5SEC = NUM_5SECS_IN_TRADING_DAY - DURATION_IN_5SEC_UNITS

	tickers = [ticker for ticker in os.listdir(DATA_DIRECTORY_PATH) if '.DS_Store' not in ticker]

	for ticker in tqdm(tickers):

		# load data
		ob = read_data(os.path.join(DATA_DIRECTORY_PATH, ticker, OB_FILE))
		trades = read_data(os.path.join(DATA_DIRECTORY_PATH, ticker, TRADES_FILE))
		ob = ob[['Time', 'ASK_PRICE', 'BID_PRICE', 'ASK_SIZE', 'BID_SIZE', 'LEVEL']]
		trades = trades[['Time', 'PRICE', 'SIZE', 'BUY_SELL_FLAG']]
		ob.set_index('Time', inplace=True, drop=False)
		trades.set_index('Time', inplace=True, drop=False)
		ob.sort_index(inplace=True)  # 2000x speed up on lookup
		trades.sort_index(inplace=True)

		# only during trading hours
		ob = ob[ob.Time.dt.time.between(time(9, 30, 0), time(16, 0, 0))]
		trades = trades[trades.Time.dt.time.between(time(9, 30, 0), time(16, 0, 0))]

		# generate time samples
		offsets = 5 * np.random.randint(0, MAX_5SEC, NUM_DATA)
		sample_dates = np.random.choice(trades.Time.dt.date.unique(), NUM_DATA, replace=True)
		sample_dates_start = pd.Series([datetime.combine(d, time(9, 30, 0)) for d in sample_dates]) + pd.to_timedelta(
			offsets, unit='s')
		sample_dates_end = pd.Series([datetime.combine(d, time(9, 30, 0)) for d in sample_dates]) + pd.to_timedelta(
			offsets, unit='s') + pd.to_timedelta(5 * DURATION_IN_5SEC_UNITS, unit='s')
		date_df = pd.DataFrame([sample_dates_start, sample_dates_end], index=['start', 'end']).T

		for sample in tqdm(date_df.itertuples()):

			# get trades data for respective period
			sub_trades = trades[sample.start:sample.end]
			if sub_trades.shape[0] < MIN_TRADES_IN_PERIOD:
				continue
			sub_ob = ob[sample.start:sample.end]

			# create save directory
			save_directory = os.path.join(SAVE_DIRECTORY_PATH, ticker, str(sample.start).replace('/', ':'))
			safe_mkdir(save_directory)

			# save data as feather
			ob_save_file = os.path.join(save_directory, 'ob.feather')
			trade_save_file = os.path.join(save_directory, 'trades.feather')
			feather.write_dataframe(sub_ob, ob_save_file)
			feather.write_dataframe(sub_trades, trade_save_file)
