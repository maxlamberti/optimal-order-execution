import os
import gym
import numpy as np
from scripts.utils.data_loading import get_data_file_paths
from simulator.lob import OrderBookSimulator


class DiscreteTrader:

	def __init__(self, inventory, target_inventory, trade_window, impact_param, data_path, limit_order_level=2, is_buy_agent=False, sampling_freq=5):

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

		# Set up initial LOB simulator
		self.current_sim_id = np.random.choice(self.simulation_ids, 1)[-1]
		lob_file = os.path.join(self.current_sim_id, 'ob.feather')
		trades_file = os.path.join(self.current_sim_id, 'trades.feather')
		self.LOB_SIM = OrderBookSimulator(lob_file, trades_file, impact_param)
		ob, trds, executed_orders, active_limit_order_levels = self.LOB_SIM.iterate()
		self.initial_midprice = (ob.BID_PRICE.max() + ob.ASK_PRICE.min()) / 2
		self.state = self.calculate_state(ob, trds, executed_orders, active_limit_order_levels)

		# Define Action Space
		# 0: do nothing, 1: LO at tick level 2, 2: MO of size 100, 3: MO of size 200
		self.action_space = gym.spaces.Discrete(4)  # number of discrete action bins

	def step(self, action):

		# Perform agent action
		if action == 1:  # LO at tick level
			inventory_delta = abs(self.target_inventory - self.current_inventory)
			volume = np.min(inventory_delta, 100.0)
			if self.is_buy_agent:
				self.LOB_SIM.place_limit_buy_order_at_tick(volume, self.limit_order_level)
			else:
				self.LOB_SIM.place_limit_sell_order_at_tick(volume, self.limit_order_level)
		elif action == 2:  # MO of size 100
			inventory_delta = abs(self.target_inventory - self.current_inventory)
			volume = np.min(inventory_delta, 100.0)
			if self.is_buy_agent:
				self.LOB_SIM.place_market_buy_order(volume)
			else:
				self.LOB_SIM.place_market_sell_order(volume)
		elif action == 3:  # MO of size 200
			inventory_delta = abs(self.target_inventory - self.current_inventory)
			volume = np.min(inventory_delta, 200.0)
			if self.is_buy_agent:
				self.LOB_SIM.place_market_buy_order(volume)
			else:
				self.LOB_SIM.place_market_sell_order(volume)

		# Update market environment
		ob, trds, executed_orders, active_limit_order_levels = self.LOB_SIM.iterate()

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
				shortfall = (self.initial_midprice - executed_vwap) / self.initial_midprice
			else:
				shortfall = (executed_vwap - self.initial_midprice) / self.initial_midprice
		else:
			shortfall = 0

		# Do time accounting
		self.period += 1
		self.time += self.sampling_freq

		# Check if target inventory achieved
		reached_target_position = self.current_inventory == self.target_inventory  # early stop condition
		ran_out_of_time = self.time >= self.trade_window
		is_done = reached_target_position or ran_out_of_time  # TODO: Morgan, is this only for early stopping or also for running out of time

		# Calculate reward
		reward = self.calculate_reward(shortfall, self.time)

		# Update agent state
		self.state = self.calculate_state(ob, trds, executed_orders, active_limit_order_levels)

		return self.state, reward, is_done

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
		ob, trds, executed_orders, active_limit_order_levels = self.LOB_SIM.iterate()
		self.initial_midprice = (ob.BID_PRICE.max() + ob.ASK_PRICE.min()) / 2
		self.state = self.calculate_state(ob, trds, executed_orders, active_limit_order_levels)

		return self.state  # TODO: does the state need to be returned here?

	def calculate_reward(self, shortfall, time):
		if time >= self.trade_window:
			time_penalty = -self.current_inventory
		else:
			time_penalty = 0
		reward = shortfall + time_penalty  # TODO: finalize reward, time penalty weighted by remaining inventory?
		return reward

	def calculate_state(self, ob, trds, executed_orders, active_limit_order_levels):
		return np.array([])  # TODO: calculate state
