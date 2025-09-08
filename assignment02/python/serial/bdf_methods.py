from utils_tools import prod, div

def BDF2(Un1,Un,z):
  num = [1,0]
  den = [3,0]-2*z
  Un2 = prod(div(num,den),4*Un1 - Un)
  return Un2

def BDF3(Un2, Un1, Un, z):
  num1 = [18,0]
  num2 = [9,0]
  num3 = [2,0]
  den = [11,0] - 6*z
  Un3 = prod(div([18,0],den),Un2) - prod(div([9,0],den),Un1) + prod(div([2,0],den),Un)
  return Un3

def BDF4(Un3, Un2, Un1, Un, z):
  num1 = [48,0]
  num2 = [36,0]
  num3 = [16,0]
  num4 = [3,0]
  den = [25,0] - 12*z
  Un4 = prod(div([48,0],den),Un3) - prod(div([36,0],den),Un2) \
        + prod(div([16,0],den),Un1) - prod(div([3,0],den),Un)
  return Un4

def BDF5(Un4, Un3, Un2, Un1, Un, z):
  num1 = [300,0]
  num2 = [300,0]
  num3 = [200,0]
  num4 = [75,0]
  num5 = [12,0]
  den  = [137,0] - 60*z
  Un5 = prod(div(num1,den),Un4) - prod(div(num2,den),Un3) \
        + prod(div(num3,den),Un2) - prod(div(num4,den),Un1) \
        + prod(div(num5,den),Un)
  return Un5

def BDF6(Un5, Un4, Un3, Un2, Un1, Un, z):
  num1 = [360, 0]
  num2 = [450, 0]
  num3 = [400, 0]
  num4 = [225, 0]
  num5 = [72 , 0]
  num6 = [10 , 0]
  den  = [147, 0] - 60*z
  Un6  =  prod(div(num1, den), Un5) - prod(div(num2, den), Un4)\
	+ prod(div(num3, den), Un3) - prod(div(num4, den), Un2)\
	+ prod(div(num5, den), Un1) - prod(div(num6, den), Un)
  return Un6
