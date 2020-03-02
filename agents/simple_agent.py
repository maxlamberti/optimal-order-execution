import os
import gym
import numpy as np
from datetime import datetime, date, time, timedelta
from simulator.lob import OrderBookSimulator
from scripts.utils.data_loading import get_data_file_paths


class DiscreteTrader(gym.Env):

	def __init__(self, inventory, target_inventory, trade_window, impact_param, data_path, limit_order_level=2, is_buy_agent=False, sampling_freq=5):

		self.metadata = None

		# Simulation parameters
		self.period = 0  # in index units, ie. period=0 is t=0secs, period=1 is t=5secs
		self.time = 0  # in seconds
		self.trade_window = trade_window  # trade needs to be completed within this time frame
		self.impact_param = impact_param  # parameter for the LOB sim
		self.sampling_freq = sampling_freq  # units of one period, fixed at 5 seconds for us
		self.data_path = data_path  # path to lob and trade data samples
		self.num_periods = int(trade_window / sampling_freq)
		self.simulation_ids = get_data_file_paths(data_path)

		# Agent parameters
		self.is_buy_agent = is_buy_agent
		self.initial_inventory = inventory
		self.current_inventory = inventory
		self.target_inventory = target_inventory
		self.limit_order_level = limit_order_level
		self.order_execution_history = []

		# Set up initial LOB simulator
		self.observation_space = gym.spaces.Box(
			low=np.array([0, -1, 0, 0]),
			high=np.array([1, 1, 1, np.inf]),
			dtype=np.float32
		)

		self.current_sim_id = np.random.choice(self.simulation_ids, 1)[-1]
		lob_file = os.path.join(self.current_sim_id, 'ob.feather')
		trades_file = os.path.join(self.current_sim_id, 'trades.feather')
		self.LOB_SIM = OrderBookSimulator(lob_file, trades_file, impact_param)
		ob, trds, executed_orders, active_limit_order_levels = self.LOB_SIM.iterate(force=True)
		self.initial_price = (ob.BID_PRICE.max() + ob.ASK_PRICE.min()) / 2
		self.state = self.calculate_state(ob, trds, executed_orders, active_limit_order_levels)

		# Define Action Space
		# 0: LO at tick level 2, 1: MO of size 100
		self.action_space = gym.spaces.Discrete(2)  # number of discrete action bins

	def step(self, action):

		# Perform agent action
		if action == 0:  # LO at tick level
			inventory_delta = abs(self.target_inventory - self.current_inventory)
			volume = min(inventory_delta, 100.0)
			if self.is_buy_agent:
				self.LOB_SIM.place_limit_buy_order_at_tick(volume, self.limit_order_level)
			else:
				self.LOB_SIM.place_limit_sell_order_at_tick(volume, self.limit_order_level)
		elif action == 1:  # MO of size 100
			inventory_delta = abs(self.target_inventory - self.current_inventory)
			volume = min(inventory_delta, 100.0)
			if self.is_buy_agent:
				self.LOB_SIM.place_market_buy_order(volume)
			else:
				self.LOB_SIM.place_market_sell_order(volume)

		# Update market environment
		try:
			ob, trds, executed_orders, active_limit_order_levels = self.LOB_SIM.iterate()
		except:
			return self.state, 0, True, {}
		self.order_execution_history += executed_orders

		# Check executed orders and update inventory
		price_weighted_volume, total_executed_volume = 0, 0
		for order in executed_orders:
			price_weighted_volume += order['volume'] * order['price']
			total_executed_volume += order['volume']
			if order['is_buy']:
				self.current_inventory = min(self.target_inventory, self.current_inventory + order['volume'])
			else:
				self.current_inventory = max(self.target_inventory, self.current_inventory - order['volume'])

		# Calculate implementation shortfall
		if total_executed_volume > 0:
			executed_vwap = price_weighted_volume / total_executed_volume
			if self.is_buy_agent:
				shortfall = (self.initial_price - executed_vwap) / self.initial_price
			else:
				shortfall = (executed_vwap - self.initial_price) / self.initial_price
		else:
			shortfall = 0

		# Do time accounting
		self.period += 1
		self.time += self.sampling_freq

		# Check if target inventory achieved
		reached_target_position = self.current_inventory == self.target_inventory  # early stop condition
		about_to_run_out_of_time = (self.time + self.sampling_freq) >= self.trade_window
		ran_out_of_time = self.time >= self.trade_window
		is_done = reached_target_position or ran_out_of_time

		if about_to_run_out_of_time:  # liquidate
			inventory_delta = abs(self.target_inventory - self.current_inventory)
			volume = inventory_delta
			if self.is_buy_agent:
				self.LOB_SIM.place_market_buy_order(volume)
			else:
				self.LOB_SIM.place_market_sell_order(volume)

		# Calculate reward
		reward = self.calculate_reward(shortfall)

		# Update agent state
		self.state = self.calculate_state(ob, trds, executed_orders, active_limit_order_levels)

		return self.state, reward, is_done, {}

	def reset(self):

		# Simulation parameters
		self.period = 0  # in index units, ie. period=0 is t=0secs, period=1 is t=5secs
		self.time = 0  # in seconds

		# Reset agent parameters
		self.current_inventory = self.initial_inventory

		# Reset LOB simulator
		self.current_sim_id = np.random.choice(self.simulation_ids, 1)[-1]
		lob_file = os.path.join(self.current_sim_id, 'ob.feather')
		trades_file = os.path.join(self.current_sim_id, 'trades.feather')
		self.LOB_SIM = OrderBookSimulator(lob_file, trades_file, self.impact_param)
		ob, trds, executed_orders, active_limit_order_levels = self.LOB_SIM.iterate(force=True)
		self.initial_price = (ob.BID_PRICE.max() + ob.ASK_PRICE.min()) / 2
		self.state = self.calculate_state(ob, trds, executed_orders, active_limit_order_levels)
		self.order_execution_history = []

		return self.state

	def calculate_reward(self, shortfall, gamma=1):

		# remaining_periods = self.num_periods - self.period
		# if (self.current_inventory / 100) > gamma * remaining_periods:
		# 	inventory_penalty = (gamma * remaining_periods - (self.current_inventory / 100)) * 0.01
		# else:
		# 	inventory_penalty = 0

		return shortfall #+ inventory_penalty

	def calculate_state(self, ob, trds, executed_orders, active_limit_order_levels):

		side = 'ASK' if self.is_buy_agent else 'BID'
		# opposite_side = 'BID' if (side == 'ASK') else 'ASK'
		best_tick_volume = ob.loc[ob['LEVEL'] == 1, [side + '_SIZE']].values[-1, -1] / 100
		# second_best_tick_volume = ob.loc[ob['LEVEL'] == 2, [side + '_SIZE']].values[-1, -1] / 100
		# trading_day_progression = (1.0 / (6.5 * 60 * 60)) * ((datetime.combine(date.today(), ob.Time.dt.time.values[
		# 	-1]) - datetime.combine(date.today(), time(9, 30, 0))) / timedelta(seconds=1))
		inventory_delta = abs(self.current_inventory - self.target_inventory) / max(abs(self.initial_inventory),
																					abs(self.target_inventory))
		pct_diff_from_initial_price = (ob.loc[ob['LEVEL'] == 1, [side + '_PRICE']].values[
										   -1, -1] - self.initial_price) / self.initial_price
		# gross_last_period_trade_volume = trds.SIZE.sum() / 100
		# net_last_period_trade_volume = (trds[trds['BUY_SELL_FLAG'] == 1].SIZE.sum() - trds[
		# 	trds['BUY_SELL_FLAG'] == 0].SIZE.sum()) / 100
		# spread = 10 * (ob.ASK_PRICE.min() - ob.BID_PRICE.max())
		pct_trade_window_progression = self.time / self.trade_window
		# num_open_lob_levels = (len(active_limit_order_levels['ASK']) + len(active_limit_order_levels['BID'])) / 10.0
		# has_limit_order_at_tick_2 = ob.loc[ob['LEVEL'] == 2, [opposite_side + '_PRICE']].values[-1, -1] in \
		# 							active_limit_order_levels[opposite_side]

		state = np.array([
			inventory_delta,
			pct_diff_from_initial_price,
			pct_trade_window_progression,
			best_tick_volume
		])

		return state

	def render(self, mode='human'):
		return

	def close(self):
		return