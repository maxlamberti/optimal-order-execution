import logging
import feather
import numpy as np
import pandas as pd
from operator import attrgetter
from collections import namedtuple
from simulator.limit_queue import LimitQueue


logging.basicConfig(format='[%(levelname)s] | %(asctime)s | %(message)s')


#TODO: if trades df has no trades or runs out of trades throw error

class OrderBookSimulator:

	def __init__(self, order_book_file, trades_file, impact_param):

		logging.debug("Creating MarketSimulator object.")
		self.order_book_file = order_book_file
		self.trades_file = trades_file
		self.impact_param = impact_param
		self.trades_df = self._load_data(trades_file)
		self.ob_df = self._load_data(order_book_file)
		self.ob_iterator = iter(self.ob_df.groupby('DateTime'))
		self.time_index = self.ob_df.DateTime.unique()
		self.freq = int((self.time_index[1] - self.time_index[0]) / np.timedelta64(1, 's'))  # freq in seconds
		self.trade_iterator = iter(
			self.trades_df.resample('{}s'.format(self.freq), base=0, label='right', on='DateTime'))
		self.new_market_order = {}
		self.limit_orders = []
		self.delete_limit_orders = []
		self.queue_tracker = {'BID': {}, 'ASK': {}}

		# discard data until trading starts
		num_discard = int(
			np.ceil((self.trades_df.DateTime.min() - self.time_index[0]) / np.timedelta64(self.freq, 's')))
		for _ in range(num_discard):
			_, _ = next(self.ob_iterator)

		# one period burn in, needed for limit order handling
		_, self.prev_period_ob = next(self.ob_iterator)
		_, _ = next(self.trade_iterator)

	@staticmethod
	def _load_data(path):
		if '.csv' in path:
			df = pd.read_csv(path)
		elif '.feather' in path:
			df = feather.read_dataframe(path)
		else:
			raise Exception("Not sure if data is of type feather or csv.")

		if 'DateTime' not in df.columns:
			df['DateTime'] = pd.to_datetime(df['Time'])

		return df

	def iterate(self, force=False):
		"""Take one step forward in time and return market data."""

		executed_orders = []  # agent orders executed in the market

		# place market order
		if self.new_market_order:
			vwap, ob, trds, sim_time = self._execute_market_order(self.new_market_order, force)
			self.new_market_order['execution_time'] = sim_time
			self.new_market_order['price'] = vwap
			executed_orders.append(self.new_market_order)
			self.new_market_order = {}  # reset
		else:  # if no new market order, advance state
			ob, trds, sim_time = self._get_next_market_state(force)

		# place new limit orders
		for order in self.limit_orders:
			self._execute_limit_order(order)
		self.limit_orders = []  # reset

		# update existing limit orders, need to be applied to previous period order book
		deletion_queues = {'BID': [], 'ASK': []}
		for side in ['BID', 'ASK']:

			if self.queue_tracker[side]:  # estimate executed volume per tick level
				level_to_trade_volume_map = self._get_level_to_trade_volume_mapping(side, self.prev_period_ob, trds)

			# update queue with new data
			for price_level, queue in self.queue_tracker[side].items():
				if queue.has_unfilled_agent_order():

					tick_depth = self._get_tick_depth(ob, side, price_level)
					is_buy = True if (side == 'BID') else False

					# execution by trades: get approximate market vol executed at this price level and update queue
					trade_volume_at_tick = level_to_trade_volume_map.get(price_level, 0)
					executed_agent_volume = queue.update(tick_depth, trade_volume_at_tick)
					if executed_agent_volume > 0:
						executed_orders.append(
							{'type': 'limit', 'is_buy': is_buy, 'volume': executed_agent_volume, 'price': price_level})

					# execute by bid and ask meeting: if our limit order wanders onto the opposite side, start executing
					if side == 'BID':
						ask_bid_volume_overlap = ob.loc[ob['ASK_PRICE'] <= price_level, ['ASK_SIZE']].sum().values[-1]
					else:
						ask_bid_volume_overlap = ob.loc[ob['BID_PRICE'] >= price_level, ['BID_SIZE']].sum().values[-1]
					if ask_bid_volume_overlap > 0: # order crosses opposite side:
						remaining_agent_volume = queue.get_agent_volume()
						executed_agent_volume = queue.update(0, min(ask_bid_volume_overlap, remaining_agent_volume))
						if executed_agent_volume > 0:
							executed_orders.append(
								{'type': 'limit', 'is_buy': is_buy, 'volume': executed_agent_volume, 'price': price_level})


					# check if price level has gone out of bounds, if yes delete queue
					if (not queue.has_unfilled_agent_order()) or (not self._check_if_valid_limit_order_position(side, price_level, ob)):
						deletion_queues[side].append(price_level)  # queue moved out of bounds

				else:
					deletion_queues[side].append(price_level)

			# retire old queues
			for price_level in deletion_queues[side]:
				del self.queue_tracker[side][price_level]

		# delete old limit orders
		for price_level in self.delete_limit_orders:
			logging.debug("Deleting limit order tracking queue at price %s", price_level)
			if price_level in self.queue_tracker['ASK']:
				del self.queue_tracker['ASK'][price_level]
			if price_level in self.queue_tracker['BID']:
				del self.queue_tracker['BID'][price_level]
		self.delete_limit_orders = []  # reset delete list

		# get active limit order levels to return as state to user
		active_limit_order_levels = {
			'ASK': list(self.queue_tracker['ASK'].keys()),
			'BID': list(self.queue_tracker['BID'].keys())
		}

		logging.debug("Current simulator state: %s", sim_time)

		self.prev_period_ob = ob  # get ready for next period

		return ob, trds, executed_orders, active_limit_order_levels

	def _get_level_to_trade_volume_mapping(self, side, ob, trds):

		use_buy_orders = True if (side == 'ASK') else False
		price_lvl_to_volume_mapping = {}
		remaining_trade_volume = trds[trds['BUY_SELL_FLAG'] == use_buy_orders].SIZE.sum()

		# add queues which aren't in order book, example use cases: order placed in spread
		ob_levels = list(ob[[side + '_PRICE', side + '_SIZE']].itertuples())
		lvl_tuple = namedtuple('Pandas', [side + '_PRICE', side + '_SIZE'])
		for price_level, queue in self.queue_tracker[side].items():
			if (price_level == ob[side + '_PRICE']).any():
				continue  # don't add levels which are already accounted for
			additional_level = lvl_tuple(price_level, 0)
			ob_levels.append(additional_level)
		ob_levels = sorted(ob_levels, key=attrgetter(side + '_PRICE'))

		# in price priority order, cycle through OB levels and calculate trade volume to apply
		for ob_level in ob_levels:  # assumes sorting by levels
			if 'BID_PRICE' in ob_level._fields:
				price = ob_level.BID_PRICE
				size = ob_level.BID_SIZE
				agent_volume = self.queue_tracker['BID'].get(price, LimitQueue(0)).get_agent_volume()
			else:
				price = ob_level.ASK_PRICE
				size = ob_level.ASK_SIZE
				agent_volume = self.queue_tracker['ASK'].get(price, LimitQueue(0)).get_agent_volume()

			trade_volume = min(size + agent_volume, remaining_trade_volume)
			remaining_trade_volume -= trade_volume
			price_lvl_to_volume_mapping[price] = trade_volume

		logging.debug("Calculated price level to volume mapping: %s", price_lvl_to_volume_mapping)

		return price_lvl_to_volume_mapping

	def _get_next_market_state(self, force=False):

		ob_time, ob = next(self.ob_iterator)
		trade_interval, trds = next(self.trade_iterator)

		if (not force) and (ob_time != trade_interval):
			raise Exception("Order book and trade history are out of sync.")

		return ob, trds, trade_interval

	def _execute_market_order(self, order, force=False):

		logging.debug("Executing trade in market: %s", order)

		ob, trds, sim_time = self._get_next_market_state(force)

		size = 'ASK_SIZE' if order['is_buy'] else 'BID_SIZE'
		top_tick_vol = ob.loc[ob['LEVEL'] == 1, [size]].values[-1, -1]
		vol_threshold = top_tick_vol / order['volume']

		if vol_threshold > self.impact_param:  # small trade, no impact
			logging.debug("Minor aggressive trade: executing trade with no impact on LOB.")
			vwap = self._simulate_trade_vwap_from_order_book(ob, order)  # calculate execution price from order book
		else:  # large trade, assume impact and jump to next aggressive sell order
			vwap = self._simulate_trade_vwap_from_order_book(ob, order)  # calculate execution price from order book
			logging.debug("Major aggressive trade: executing trade on historical trade data.")
			while not (order['is_buy'] == trds['BUY_SELL_FLAG']).any():  # while there is no matching order skip period
				logging.debug("Skipping period because of no historic trades to execute on.")
				ob, trds, sim_time = self._get_next_market_state(force)
			# vwap = self._simulate_trade_vwap_from_historic_trades(trds, order)

		return vwap, ob, trds, sim_time

	def _execute_limit_order(self, order):

		# make sure tick size is 1 cent
		if 'price' in order:
			order['price'] = round(order['price'], 2)
		else:
			prev_midprice = (self.prev_period_ob.BID_PRICE.max() + self.prev_period_ob.ASK_PRICE.min()) / 2
			prev_midprice = round(prev_midprice, 2)
			order['price'] = prev_midprice + order['tick'] * 0.01
			order['price'] = round(order['price'], 2)

		# valid order can only be placed within spread or on correct side of LOB
		side = 'BID' if order['is_buy'] else 'ASK'
		is_valid_order = self._check_if_valid_limit_order(side, order['price'], self.prev_period_ob)
		is_new_queue = order['price'] not in self.queue_tracker[side]
		if is_valid_order and is_new_queue:  # only place new order if dont already have an order in the queue
			# create queue at price level, place order at the end of the queue
			tick_depth = self._get_tick_depth(self.prev_period_ob, side, order['price'])
			self.queue_tracker[side][order['price']] = LimitQueue(tick_depth)
			self.queue_tracker[side][order['price']].add_agent_order(order['volume'])
		else:
			logging.warning("Invalid limit order placed, order is not accepted.")

	@staticmethod
	def _check_if_valid_limit_order(side, price, ob):
		# valid order can only be placed within spread or on correct side of LOB
		opposite_side = 'ASK' if (side == 'BID') else 'BID'
		if side == 'BID':  # valid order can only be placed within spread or on correct side of LOB
			is_valid_order = (price < ob[opposite_side + '_PRICE'].min()) and (price >= ob[side + '_PRICE'].min())
		else:
			is_valid_order = (price > ob[opposite_side + '_PRICE'].max()) and (price <= ob[side + '_PRICE'].max())
		return is_valid_order

	@staticmethod
	def _check_if_valid_limit_order_position(side, price, ob):
		# valid order must not be smaller than our furthest empirical LOB quote
		if side == 'BID':
			is_valid_order = price >= ob[side + '_PRICE'].min()
		else:
			is_valid_order = price <= ob[side + '_PRICE'].max()
		return is_valid_order

	def _get_tick_depth(self, ob, side, price):
		tick_depth = ob.loc[ob[side + '_PRICE'] == price, [side + '_SIZE']].values
		tick_depth = 0 if len(tick_depth) == 0 else tick_depth[-1, -1]
		return tick_depth

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
		vwap = (vwap + unaccounted_volume * lvl_price) / order['volume']  # TODO: consider higher penalty for larger trades
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
		logging.info("Registering limit sell order with volume %s at price %s.", volume, price)
		limit_order = {'type': 'limit', 'volume': volume, 'is_buy': False, 'price': price}
		self.limit_orders.append(limit_order)

	def place_limit_sell_order_at_tick(self, volume, tick):
		logging.info("Registering limit sell order with volume %s at rel. tick to midprice %s.", volume, tick)
		limit_order = {'type': 'limit', 'volume': volume, 'is_buy': False, 'tick': abs(tick)}
		self.limit_orders.append(limit_order)

	def place_limit_buy_order(self, volume, price):
		logging.info("Registering limit buy order with volume %s at price %s.", volume, price)
		limit_order = {'type': 'limit', 'volume': volume, 'is_buy': True, 'price': price}
		self.limit_orders.append(limit_order)

	def place_limit_buy_order_at_tick(self, volume, tick):
		logging.info("Registering limit buy order with volume %s at rel. tick to midprice %s.", volume, tick)
		limit_order = {'type': 'limit', 'volume': volume, 'is_buy': True, 'tick': -abs(tick)}
		self.limit_orders.append(limit_order)

	def cancel_limit_order(self, price_level):
		self.delete_limit_orders.append(price_level)


if __name__ == '__main__':
	OB_DATA_PATH = '../data/onetick/cat_ob_5sec.csv'
	TRADES_DATA_PATH = '../data/onetick/cat_trades.csv'

	MarketSim = OrderBookSimulator(OB_DATA_PATH, TRADES_DATA_PATH, 2)
	ob, trds, executed_orders, active_limit_order_levels = MarketSim.iterate()
	MarketSim.place_market_sell_order(1000)
	ob, trds, executed_orders, active_limit_order_levels = MarketSim.iterate()
	print(executed_orders, active_limit_order_levels)
	MarketSim.place_market_sell_order(10)
	ob, trds, executed_orders, active_limit_order_levels = MarketSim.iterate()
	print(executed_orders, active_limit_order_levels)
	ob, trds, executed_orders, active_limit_order_levels = MarketSim.iterate()
	MarketSim.place_limit_buy_order(100, 83.31)
	MarketSim.place_limit_sell_order(100, 83.34)
	MarketSim.place_limit_sell_order(100, 83.35)
	ob, trds, executed_orders, active_limit_order_levels = MarketSim.iterate()
	print(executed_orders, active_limit_order_levels)
	MarketSim.place_market_buy_order(50)
	ob, trds, executed_orders, active_limit_order_levels = MarketSim.iterate()
	print(executed_orders, active_limit_order_levels)
	MarketSim.place_limit_sell_order(100, 83.37)
	ob, trds, executed_orders, active_limit_order_levels = MarketSim.iterate()
	print(executed_orders, active_limit_order_levels)
	MarketSim.place_market_buy_order(500)
	ob, trds, executed_orders, active_limit_order_levels = MarketSim.iterate()
	MarketSim.place_limit_sell_order_at_tick(100, 2)
	MarketSim.place_limit_buy_order_at_tick(100, -2)
	print(executed_orders, active_limit_order_levels)
	ob, trds, executed_orders, active_limit_order_levels = MarketSim.iterate()
	print(executed_orders, active_limit_order_levels)
