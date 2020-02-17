import os
import numpy as np
import pandas as pd
import multiprocessing as mp
from functools import partial


def univariate_lob_aggregator(func, data_directory, verbose=False, **kwargs):
	"""Calculate a univariate data series from the limit order book.

	PARAMS
	------
	func : function
		Function object which takes as argument a pandas data frame of a single period LOB snapshot.
		Function performs some computation on the data frame and returns a single float or int value.
	data_directory : str
		String specifying the path to the LOB data files. E.g. "/path/to/limit-order-book/XXBTZEUR"
	verbose : bool
		When True, each successful completion of a multithread process will print a confirmation.
	kwargs (optional)
		Additional key word arguments that get passed to func.

	RETURNS
	-------
	pandas.Series : Univariate data series containing the aggregated data. Index is a timestamp.
	"""

	# select LOB files
	files = np.sort([f for f in os.listdir(data_directory) if os.path.isfile(os.path.join(data_directory, f))])

	# aggregate LOB data into univariate series
	with mp.Pool(max(mp.cpu_count() - 1, 1)) as pool:
		mp_result = pool.map(
			partial(aggregation_wrapper_func, data_directory=data_directory, func=func, verbose=verbose, **kwargs),
			files
		)

	return pd.concat(mp_result)


def aggregation_wrapper_func(file, func, data_directory, verbose=False, **kwargs):
	filepath = os.path.join(data_directory, file)
	lob_ts_df = pd.read_parquet(filepath)
	agg_result = lob_ts_df.groupby('query_time').apply(func, **kwargs)
	if verbose:
		print("Successfully completed process.")
	return agg_result


def calc_midprice(lob):
	minask = lob.loc[lob['type'] == 'a', 'price'].min()
	maxbid = lob.loc[lob['type'] == 'b', 'price'].max()
	midprice = (minask + maxbid) / 2
	return midprice


def calc_ask_price(lob):
	return lob.loc[lob['type'] == 'a', 'price'].min()


def calc_bid_price(lob):
	return lob.loc[lob['type'] == 'b', 'price'].max()


def calc_spread(lob):
	minask = lob.loc[lob['type'] == 'a', 'price'].min()
	maxbid = lob.loc[lob['type'] == 'b', 'price'].max()
	return minask - maxbid


if __name__ == '__main__':

	DATA_DIRECTORY = '/data/limit-order-book/XXBTZUSD'

	# example use case calculating the midprice
	# to calculate other results replace calc_midprice with one of the above or a custom function
	midprice_series = univariate_lob_aggregator(calc_midprice, DATA_DIRECTORY)
