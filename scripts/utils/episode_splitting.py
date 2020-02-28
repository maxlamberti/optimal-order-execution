import os
import feather
import pandas as pd
from tqdm import tqdm


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
	SAVE_DIRECTORY_PATH = '../../data/onetick/processed/'
	DURATION = int(6.5 * 60 * 60 / 2)  # in seconds
	MIN_TRADES_IN_PERIOD = 1  # if there are less than this amount of trades, won't save a file

	tickers = os.listdir(DATA_DIRECTORY_PATH)

	for ticker in tqdm(tickers):

		# load data
		ob = read_data(os.path.join(DATA_DIRECTORY_PATH, ticker, OB_FILE))
		trades = read_data(os.path.join(DATA_DIRECTORY_PATH, ticker, TRADES_FILE))
		ob['DateTime'] = pd.to_datetime(ob['Time'])
		trades['DateTime'] = pd.to_datetime(trades['Time'])
		# trades.sort_values('DateTime', inplace=True)  # doesn't improve speed, maybe use DateTime as index?

		for duration_id, sub_ob in tqdm(ob.resample('{}s'.format(DURATION), base=0, label='left', on='DateTime')):

			# get trades data for respective period
			sub_trades = trades[trades.DateTime.between(duration_id, duration_id + pd.to_timedelta(DURATION, unit='s'))]
			if sub_trades.shape[0] < MIN_TRADES_IN_PERIOD:
				continue

			# create save directory
			save_directory = os.path.join(SAVE_DIRECTORY_PATH, ticker, str(duration_id).replace('/', ':'))
			safe_mkdir(save_directory)

			# save data as feather
			ob_save_file = os.path.join(save_directory, 'ob.feather')
			trade_save_file = os.path.join(save_directory, 'trades.feather')
			feather.write_dataframe(sub_ob, ob_save_file)
			feather.write_dataframe(sub_trades, trade_save_file)
