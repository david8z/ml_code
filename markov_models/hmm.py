"""
Implementation of the Hidden Markov Models.
Elements of the HMM:
- N = Number of states.
- M = Number of symbols.
- A = Transition Matrix between states [NxN] ([NxN+1] if there is final state).
- B = Emision Matrix of a symbol on a state [NxM]
- pi = Initial distribution proabbility o a state of being initial [N]
"""
import numpy as np
from typing import Tuple, List

class Hmm:
	def __init__(self, A:np.ndarray, B:np.ndarray, pi:np.ndarray)->None:
		self.N, self.M = B.shape
		self.A = A 
		self.B = B
		self.pi = pi 

	def forward(self, observations:np.ndarray)->float:
		"""
		Returns the probability of obtaining the observations stablished on the parameter
		---
		Inner variables:
			alfa: matrix that contains the probabilty of reachng the observations.
		"""
		alfa = np.zeros((self.N,observations.size))
		for s in range(self.N):
			alfa[s,0] = pi[s] * self.B[s,observations[0]]
		for t in range(1,observations.size):
			for s in range(self.N):
				for s_i in range(self.N):
					alfa[s,t] += alfa[s_i,t-1]*self.A[s_i,s]* self.B[s,observations[t]]
		forward_prob=0
		for s in range(self.N):
			#If there is final state
			forward_prob += alfa[s,observations.size-1]*self.A[s,self.N]
			#If there is no final state
			#forward_prob += alfa[s,observations.size-1]

		return forward_prob
	
	def viterbi(self, observations:np.ndarray)->(float,List[int]):
		"""
		Returns the probability of obtaining the selected states that fullfill the 
		observation along to the state chain that better fullfills the observations.
		---
		Inner variables:
			vit: matrix that contains the maximum probabilty of reachng the observations.
			bp: matrix that contains the state index from where that state was reached
			last_state: Contains the last state reached before the final.
		"""

		vit = np.zeros((self.N,observations.size))
		bp = np.zeros((2,3))
		for s in range(self.N):
			vit[s,0] = pi[s] * self.B[s,observations[0]]
		for t in range(1,observations.size):
			for s in range(self.N):
				for s_i in range(self.N):
					aux_value = vit[s_i,t-1]*self.A[s_i,s]* self.B[s,observations[t]]
					if aux_value > vit[s,t]:
						vit[s,t]=aux_value 
						bp[s,t]=s_i
		viterbi_prob = 0
		last_state = 0
		viterbi_path = []
		for s in range(self.N):
			#If there is final state
			aux_value = vit[s,observations.size-1]*self.A[s,self.N]
			if aux_value>viterbi_prob:
				viterbi_prob = aux_value 
				last_state = s
		for index in reversed(range(observations.size)):
			viterbi_path.insert(0,last_state)
			last_state = int(bp[last_state,index])


		return viterbi_prob, viterbi_path


A = np.array(([0.3,0.7,0],[0,0.6,0.4]))
B = np.array(([0.5,0.5],[0.2,0.8]))
pi= np.array(([0.6,0.4]))

x = Hmm(A,B,pi)

observations = np.array(([1,0,0]))
print(x.viterbi(observations))
