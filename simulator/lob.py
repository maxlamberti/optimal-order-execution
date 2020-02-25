import logging
import numpy as np
import pandas as pd

logging.basicConfig(format='[%(levelname)s] | %(asctime)s | %(message)s', level=logging.DEBUG)


class SimulatorError(Exception):
	pass


class Queue:

	def __init__(self, depth):
		self.non_agent_depth = depth
		self.queue = []
		if depth > 0:
			self.queue.append({'is_agent_order': False, 'volume': depth, 'executed_volume': 0})

	def add_agent_order(self, volume):
		logging.debug("Adding agent order with volume: %s", volume)
		self.queue.append({'is_agent_order': True, 'volume': volume, 'executed_volume': 0})

	def add_non_agent_order(self, volume):
		logging.debug("Adding non-agent order with volume: %s", volume)
		self.queue.append({'is_agent_order': False, 'volume': volume, 'executed_volume': 0})
		self.non_agent_depth += volume

	def execute_volume(self, volume):

		# execute volume as waterfall through queue
		logging.debug("Executing volume %s in queue.", volume)
		remaining_execution_volume, executed_agent_volume = volume, 0
		for order_num, order in enumerate(self.queue):
			executed_volume = min(order['volume'], remaining_execution_volume)
			logging.debug("Executing volume %s against order with volume %s", remaining_execution_volume,
						  order['volume'])
			remaining_execution_volume = max(0, remaining_execution_volume - order['volume'])
			order['volume'] -= executed_volume
			order['executed_volume'] += executed_volume
			if order['is_agent_order']:
				executed_agent_volume += executed_volume
			logging.debug("Remaining execution volume is %s, order executed volume is %s, order remaining volume is %s",
						  remaining_execution_volume, order['executed_volume'], order['volume'])
			if order['volume'] > 0:
				logging.debug("Fully executed trade, stopping queue execution.")
				break
		logging.debug("Remaining execution volume is: %s", remaining_execution_volume)

		# delete fully executed orders
		self.queue = [order for order in self.queue if order['volume'] > 0]
		self.non_agent_depth = sum([order['volume'] for order in self.queue if not order['is_agent_order']])
		logging.debug("After execution queue has %s non-agent depth and structure: %s", self.non_agent_depth,
					  self.queue)

		return executed_agent_volume, remaining_execution_volume

	def update(self, new_non_agent_depth, executed_trade_volume):

		# calculate net depth change
		net_depth_change = new_non_agent_depth - self.non_agent_depth
		logging.debug("Queue has %s non-agent depth and structure: %s", self.non_agent_depth, self.queue)

		# simulate trade execution in the queue
		if executed_trade_volume > 0:
			executed_agent_volume, remaining_execution_volume = self.execute_volume(executed_trade_volume)
		else:
			executed_agent_volume, remaining_execution_volume = 0, 0

		# break down depth changes
		new_addition_or_cancellation_volume = net_depth_change + executed_trade_volume - executed_agent_volume - \
											  remaining_execution_volume
		logging.debug("Net change in tick depth %s, %s is new additions or cancellations.",
					  net_depth_change, new_addition_or_cancellation_volume)

		# cancel orders or update orders
		if new_addition_or_cancellation_volume > 0:  # new orders, place at end of queue
			self.add_non_agent_order(new_addition_or_cancellation_volume)
		elif new_addition_or_cancellation_volume < 0:
			self.cancel_non_agent_orders(abs(new_addition_or_cancellation_volume))

		return executed_agent_volume

	def cancel_non_agent_orders(self, volume):

		# index: volume
		u = np.random.uniform(0, 1)
		running_vol_sum = 0
		logging.debug("Cancelling order at volume position %s.", u)
		for order_idx, order in enumerate(self.queue):

			if order['is_agent_order']:  # only cancel non-agent orders
				continue

			running_vol_sum += order['volume'] / self.non_agent_depth

			if running_vol_sum > u:  # break if order is in uniform interval
				logging.debug("Found order to cancel at queue index %s and running_vol_sum %s.",
							  order_idx, running_vol_sum)
				break

		# cancel orders and remove from queue
		if order['volume'] < volume:
			self.queue.pop(order_idx)
			remaining_volume = volume - order['volume']
			self.non_agent_depth -= order['volume']
			logging.debug("Removing order at queue index %s, canceling remaining vol %s.",
						  order_idx, remaining_volume)
			self.cancel_non_agent_orders(remaining_volume)  # cancel more orders on remaining volume
		elif order['volume'] > volume:
			logging.debug("Removing volume but not removing order from queue.")
			order['volume'] -= volume
			self.non_agent_depth -= volume
		else:
			logging.debug("Removing order with index %s from queue.", order_idx)
			self.queue.pop(order_idx)
			self.non_agent_depth -= order['volume']

	def has_unfilled_agent_order(self):
		return len([True for order in self.queue if (order['is_agent_order'] and (order['volume'] > 0))]) > 0

	def get_agent_volume(self):
		return sum([order['volume'] for order in self.queue if order['is_agent_order']])


class OrderBookSimulator:

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
		self.prev_period_ob = next(self.ob_iterator)
		_, _ = next(self.trade_iterator)

	@staticmethod
	def _load_csv_data(path):
		df = pd.read_csv(path)
		df['DateTime'] = pd.to_datetime(df['Time'])
		return df

	def iterate(self):
		"""Take one step forward in time and return market data."""

		executed_orders = []

		# place market order
		if self.new_market_order:
			vwap, ob, trds, sim_time = self.execute_market_order(self.new_market_order)
			self.new_market_order['execution_time'] = sim_time
			self.new_market_order['price'] = vwap
			executed_orders.append(self.new_market_order)
			self.new_market_order = {}  # reset
		else:  # if no new market order, advance state
			ob, trds, sim_time = self.get_next_market_state()

		# update existing limit orders, need to be applied to previous period order book
		deletion_queues = {'BID': [], 'ASK': []}
		for side in ['BID', 'ASK']:

			if self.queue_tracker[side]:  # estimate executed volume per tick level
				level_to_trade_volume_map = self._get_level_to_trade_volume_mapping(side, self.prev_period_ob, trds)

			# update queue with new data
			for price_level, queue in self.queue_tracker[side].items():
				if queue.has_unfilled_agent_order():

					prev_tick_depth = self._get_tick_depth(self.prev_period_ob, side, price_level)

					# get approximate market vol executed at this price level and update queue
					trade_volume_at_tick = level_to_trade_volume_map.get(price_level, 0)
					executed_agent_volume = queue.update(prev_tick_depth, trade_volume_at_tick)
					if executed_agent_volume > 0:
						is_buy = True if (side == 'BID') else False
						executed_orders.append(
							{'type': 'limit', 'is_buy': is_buy, 'volume': executed_agent_volume, 'price': price_level})

					# check if price level has gone out of bounds, if yes delete queue
					if not self._check_if_valid_limit_order(side, price_level, ob):
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

		# place new limit orders
		for order in self.limit_orders:
			self.execute_limit_order(ob, order)
		self.limit_orders = []  # reset

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

		for ob_level in ob[[side + '_PRICE', side + '_SIZE']].itertuples():  # assumes sorting by levels

			if 'BID_PRICE' in ob_level._fields:
				price = ob_level.BID_PRICE
				size = ob_level.BID_SIZE
				agent_volume = self.queue_tracker['BID'].get(ob_level, Queue(0)).get_agent_volume()
			else:
				price = ob_level.ASK_PRICE
				size = ob_level.ASK_SIZE
				agent_volume = self.queue_tracker['ASK'].get(ob_level, Queue(0)).get_agent_volume()

			trade_volume = min(size + agent_volume, remaining_trade_volume)
			remaining_trade_volume -= trade_volume
			price_lvl_to_volume_mapping[price] = trade_volume

		logging.debug("Calculated price level to volume mapping: %s", price_lvl_to_volume_mapping)

		return price_lvl_to_volume_mapping

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
		is_valid_order = self._check_if_valid_limit_order(side, order['price'], ob)
		if is_valid_order:
			# create queue at price level, place order at the end of the queue
			tick_depth = self._get_tick_depth(ob, side, order['price'])
			self.queue_tracker[side][order['price']] = Queue(tick_depth)
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
		limit_order = {'type': 'limit', 'volume': volume, 'is_buy': False, 'tick': tick}
		self.limit_orders.append(limit_order)

	def place_limit_buy_order(self, volume, price):
		logging.info("Registering limit buy order with volume %s at price %s.", volume, price)
		limit_order = {'type': 'limit', 'volume': volume, 'is_buy': True, 'price': price}
		self.limit_orders.append(limit_order)

	def place_limit_buy_order_at_tick(self, volume, tick):
		logging.info("Registering limit buy order with volume %s at rel. tick to midprice %s.", volume, tick)
		limit_order = {'type': 'limit', 'volume': volume, 'is_buy': True, 'tick': tick}
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

	Q = Queue(500)
	Q.add_agent_order(100)
	Q.update(600, 150)
	Q.update(600, 0)
	Q.update(600, 0)
	Q.update(600, 0)
	Q.update(600, 400)
	Q.update(600, 0)
	Q.update(600, 200)
	Q.update(600, 100)
	Q.update(600, 50)
	Q.update(600, 250)
	Q.update(450, 0)
	Q.update(400, 25)
	Q.update(400, 0)
	Q.update(400, 450)
	Q.update(50, 600)
	Q.update(0, 0)
	Q.update(400, 10)
	Q.update(400, 10)
