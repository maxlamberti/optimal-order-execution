import logging
import numpy as np
import pandas as pd


logging.basicConfig(format='[%(levelname)s] | %(asctime)s | %(message)s', level=logging.DEBUG)


class SimulatorError(Exception):
	pass


class MarketSimulator:

	def __init__(self, order_book_file, trades_file, impact_param):

		logging.debug("Creating MarketSimulator object.")
		self.order_book_file = order_book_file
		self.trades_file = trades_file
		self.impact_param = impact_param
		self.trades_df = self._load_csv_data(trades_file)
		self.ob_df = self._load_csv_data(order_book_file)
		self.ob_iterator = iter(self.ob_df.groupby('DateTime'))
		self.time_index = self.ob_df.DateTime.unique()
		self.freq = int((self.time_index[1] - self.time_index[0]) / np.timedelta64(1, 's'))  # freq in seconds
		self.trade_iterator = iter(self.trades_df.resample('{}s'.format(self.freq), base=0, label='right', on='DateTime'))
		self.new_market_order = {}
		self.limit_orders = []

		# discard data until trading starts
		num_discard = int(np.ceil((self.trades_df.DateTime.min() - self.time_index[0]) / np.timedelta64(self.freq, 's')))
		for _ in range(num_discard):
			_, _ = next(self.ob_iterator)

	def _load_csv_data(self, path):
		df = pd.read_csv(path)
		df['DateTime'] = pd.to_datetime(df['Time'])
		return df


	def iterate(self):
		"""Take one step forward in time and return market data."""

		# TODO: limit order cancellations
		#place market order
		if self.new_market_order:
			vwap, ob, trds, sim_time = self.execute_market_order(self.new_market_order)
			self.new_market_order['execution_time'] = sim_time
			self.new_market_order['price'] = vwap
			# TODO: save order details / or return
			self.new_market_order = {}  # reset
		else:  # if no new market order, advance state
			ob, trds, sim_time = self.get_next_market_state()

		# place new limit orders
		for order in self.limit_orders:
			self.execute_limit_order(order)

		# TODO: check and update existing limit orders

		logging.debug("Current simulator state: %s", sim_time)

		return ob, trds


	def get_next_market_state(self):

		ob_time, ob = next(self.ob_iterator)
		trade_interval, trds = next(self.trade_iterator)

		if ob_time != trade_interval:
			raise SimulatorError("Order book and trade history are out of sync.")

		return ob, trds, trade_interval


	def execute_market_order(self, order):

		logging.debug("Executing trade in market: %s", order)

		ob, trds, sim_time = self.get_next_market_state()

		size = 'ASK_SIZE' if order['is_buy'] else 'BID_SIZE'
		top_tick_vol = ob.loc[ob['LEVEL'] == 1, [size]].values[-1, -1]
		vol_threshold = top_tick_vol / order['volume']

		if vol_threshold > self.impact_param:  # small trade, no impact
			logging.debug("Minor aggressive trade: executing trade with no impact on LOB.")
			vwap = self._simulate_trade_vwap_from_order_book(ob, order)  # calculate execution price from order book
		else:  # large trade, assume impact and jump to next aggressive sell order
			logging.debug("Major aggressive trade: executing trade on historical trade data.")
			while not (order['is_buy'] == trds['BUY_SELL_FLAG']).any():  # while there is no matching order skip period
				logging.debug("Skipping period because of no historic trades to execute on.")
				ob, trds, sim_time = self.get_next_market_state()
			vwap = self._simulate_trade_vwap_from_historic_trades(trds, order)

		return vwap, ob, trds, sim_time


	def execute_limit_order(self, ob, order):
		return


	def _simulate_trade_vwap_from_order_book(self, ob, order):

		unaccounted_volume, vwap = order['volume'], 0
		for lvl in ob.itertuples():  # assumes ob is ascendingly ordered by levels

			# get price and order depth at level
			lvl_size = lvl.ASK_SIZE if order['is_buy'] else lvl.BID_SIZE
			lvl_price = lvl.ASK_PRICE if order['is_buy'] else lvl.BID_PRICE

			# calculate execution price at level
			executed_volume = min(unaccounted_volume, lvl_size)
			unaccounted_volume = max(0, unaccounted_volume - executed_volume)
			vwap += executed_volume * lvl_price
			logging.debug("Executing volume of %s at level %s and price %s.", executed_volume, lvl.LEVEL, lvl_price)
			logging.debug("Unaccounted volume: %s.", unaccounted_volume)

			# break if order fully executed
			if unaccounted_volume == 0:
				break

		# if order book insufficient to fill order, complete order at worst price
		vwap = (vwap + unaccounted_volume * lvl_price) / order['volume']  # TODO: higher penalty for larger trades
		logging.debug("Final trade VWAP: %s", vwap)

		return vwap


	def _simulate_trade_vwap_from_historic_trades(self, trades, order):

		unaccounted_volume, vwap = order['volume'], 0
		relevant_trades = trades[trades['BUY_SELL_FLAG'] == order['is_buy']].sample(frac=1, replace=False)
		for trd in relevant_trades.itertuples():  # get execution price from historical trades, random order

			executed_volume = min(unaccounted_volume, trd.SIZE)
			unaccounted_volume = max(0, unaccounted_volume - executed_volume)
			vwap += trd.PRICE * executed_volume
			logging.debug("Executing against historic trade with volume %s and price %s", executed_volume, trd.PRICE)

			if unaccounted_volume == 0:
				break

		# if historic trades have insufficient size compared to desired order volume, ignore excess amount
		vwap = vwap / (order['volume'] - unaccounted_volume)
		logging.debug("Final unaccounted volume is %s and trade VWAP %s", unaccounted_volume, vwap)

		return vwap


	def place_market_sell_order(self, volume):
		logging.info("Registering market sell order with volume %s.", volume)
		self.new_market_order = {'type': 'market', 'volume': volume, 'is_buy': False}

	def place_market_buy_order(self, volume):
		logging.info("Registering market buy order with volume %s.", volume)
		self.new_market_order = {'type': 'market', 'volume': volume, 'is_buy': True}

	def place_limit_sell_order(self, volume, price):
		logging.info("Registering limit sell order with volume %s.", volume)
		limit_order = {'type': 'limit', 'volume': volume, 'is_buy': False, 'price': price}
		self.limit_orders.append(limit_order)


	def cancel_limit_order(self):
		return


	def get_market_state(self):
		return


if __name__ == '__main__':

	OB_DATA_PATH = '../data/onetick/cat_ob_5sec.csv'
	TRADES_DATA_PATH = '../data/onetick/cat_trades.csv'

	MarketSim = MarketSimulator(OB_DATA_PATH, TRADES_DATA_PATH, 2)

	MarketSim.iterate()
	MarketSim.place_market_sell_order(1000)
	MarketSim.iterate()
	MarketSim.place_market_sell_order(10)
	MarketSim.iterate()
	MarketSim.iterate()
	MarketSim.iterate()
	MarketSim.place_market_buy_order(50)
	MarketSim.iterate()
	MarketSim.iterate()
	MarketSim.place_market_buy_order(500)
	MarketSim.iterate()
