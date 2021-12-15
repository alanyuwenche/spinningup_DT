# 5 actions - 0, 1(Buy), 2(Sell) 
import numpy as np
import math
import random
import enum
import gym

#import policyopt
#from policyopt import util

#MAX_pIndex = 100.

# returns the sigmoid
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

class Actions(enum.Enum):
    Skip = 0
    Buy = 1
    Sell = 2



class daytradeEnv(gym.Env):
  """Custom Environment that follows gym interface"""
  #metadata = {'render.modes': ['human']}
  
  def __init__(self, stateN, prInd,priS, p, comm=47):
      self.observation_space = gym.spaces.Box(low=-5., high=5., shape=(stateN+2,), dtype=np.float64)#20210605
      self.action_space = gym.spaces.Discrete(n=len(Actions))
      self.stateN = stateN
      self.pIndex = prInd
      self.price_std = priS
      self.price = p
      self.commission = comm
      self.position = np.array([0.])
      self.inventory = []
      self.d = 0 # day
      self.t = 0 # time
      self.done = False
      


  def getStateTv(self):
      aa = self.pIndex[self.t-4:self.t , self.d]
      pri_s = np.array([self.price_std[self.t-1, self.d]])
      aa = np.concatenate((aa, pri_s, self.position), axis=0)
      return aa
  
  def _take_action(self, action):
      reward = 0
      if action == 1:
          if int(self.position[0]) == 0:
              self.position = np.array([1.])
              self.inventory.append(self.price[self.d][0].iloc[self.t-1,1])

          if int(self.position[0]) == -1:
              sold_price = self.inventory.pop(0)
              reward = 50*(sold_price - self.price[self.d][0].iloc[self.t-1,1])-2*self.commission
              self.done = True
              self.position = np.array([0.])

 
              
      elif action == 2:
          if int(self.position[0]) == 0:
              self.position = np.array([-1.])
              self.inventory.append(self.price[self.d][0].iloc[self.t-1,1])

          if int(self.position[0]) == 1:
              bought_price = self.inventory.pop(0)
              reward = 50*(self.price[self.d][0].iloc[self.t-1,1] - bought_price)-2*self.commission
              self.done = True
              self.position = np.array([0.])
              
              
      return reward, self.position

              


      
  def step(self, action):
      reward, self.position = self._take_action(action)
      self.t += 1
      #observation = self.getState()
      observation = self.getStateTv()
      if self.t == 284:
          self.done = True
          if len(self.inventory) > 0:
              if int(self.position[0]) == 1:
                  bought_price = self.inventory.pop(0)
                  reward = 50*(self.price[self.d][0].iloc[self.t-1,1] - bought_price)-2*self.commission
                  observation[self.stateN+1] = np.array([0.])
              elif int(self.position[0]) == -1:
                  sold_price = self.inventory.pop(0)
                  reward = 50*(sold_price - self.price[self.d][0].iloc[self.t-1,1])-2*self.commission
                  observation[self.stateN+1] = np.array([0.])
                  
      return observation, reward, self.done

  

  def reset(self):
      self.position = np.array([0.])
      self.inventory = []
      dth = random.randint(0, self.pIndex.shape[1]-1)
      self.d = dth # day
      self.t = 4 # time
      self.done = False
      #return self.getState()  # reward, done, info can't be included
      return self.getStateTv()  #
  
  def render(self, mode='human', close=False):
      print(f'Day: {self.d} ( t: {self.t})')
      print(f'done: {self.done}')          
      
  
  def close (self):
      pass


    

      


  

