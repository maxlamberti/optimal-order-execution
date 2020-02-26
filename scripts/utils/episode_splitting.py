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


if __name__ == '__main__':


	DATA_DIRECTORY_PATH = '../../data/onetick'
	SAVE_DIRECTORY_PATH = '../../data/feather_onetick/'
	DURATION = int(60 * 60 * 2)  # in seconds
	MIN_TRADES_IN_PERIOD = 1  # if there are less than this amount of trades, won't save a file

	files = os.listdir(DATA_DIRECTORY_PATH)
	files = [file for file in files if 'ob' in file]

	for file in tqdm(files):

		ticker = file.split('_')[0]
		trades_file = ticker + '_trades.csv'

		# load data
		ob = pd.read_csv(os.path.join(DATA_DIRECTORY_PATH, file))
		trades = pd.read_csv(os.path.join(DATA_DIRECTORY_PATH, trades_file))
		ob['DateTime'] = pd.to_datetime(ob['Time'])
		trades['DateTime'] = pd.to_datetime(trades['Time'])
		# trades.sort_values('DateTime', inplace=True)  # doesn't improve speed, maybe use DateTime as index?

		for duration_id, sub_ob in ob.resample('{}s'.format(DURATION), base=0, label='left', on='DateTime'):

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