import logging
import numpy as np


logging.basicConfig(format='[%(levelname)s] | %(asctime)s | %(message)s')


class LimitQueue:

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


if __name__ == '__main__':

	Q = LimitQueue(500)
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
