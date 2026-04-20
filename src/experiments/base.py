from abc import ABC, abstractmethod

class Experiment:
	@abstractmethod
	def run(self, cfg):
		raise NotImplementedError

