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



############################## MÉTODOS TR_BDF'S#########################
def TR_BDF2(Un, z):
    h = 1.0
    # First stage (TR)
    den_tr = [1, 0] - div(prod([h, 0], z), [4, 0])  # 1 - h*z/4
    num_tr = [1, 0] + div(prod([h, 0], z), [4, 0])  # 1 + h*z/4
    Y2 = prod(div(num_tr, den_tr), Un)  # Y₂ = Uⁿ * (1 + h*z/4)/(1 - h*z/4)

    # Second stage (BDF2)
    den_bdf = [3, 0] - prod([h, 0], z)  # 3 - h*z
    num_bdf = [4 * Y2[0] - Un[0], 4 * Y2[1] - Un[1]]  # 4Y₂ - Uⁿ
    Un1 = div(num_bdf, den_bdf)  # Uⁿ⁺¹ = (4Y₂ - Uⁿ)/(3 - h*z)
    return Un1


def TR_BDF2_explicit(Un, z):
    """
    TR-BDF2 with explicit coefficient calculation
    """
    h = 1.0

    # Calculate coefficients explicitly
    # Denominator: (3 - h*z) * (1 - h*z/4)
    hz = prod([h, 0], z)  # h*z

    term1 = div(hz, [4, 0])  # h*z/4
    den_part1 = [1, 0]  # 1
    den_part1 = [den_part1[0] - term1[0], den_part1[1] - term1[1]]  # 1 - h*z/4

    den_part2 = [3, 0]  # 3
    den_part2 = [den_part2[0] - hz[0], den_part2[1] - hz[1]]  # 3 - h*z

    den = prod(den_part1, den_part2)  # (1 - h*z/4)*(3 - h*z)

    # Numerator: (4 + h*z)
    num = [4, 0]  # 4
    num = [num[0] + hz[0], num[1] + hz[1]]  # 4 + h*z

    Un1 = prod(div(num, den), Un)
    return Un1


def TR_BDF2_v1 (Un,z):
  num   = [1,0] + z/4
  den   = [1,0] - z/4
  Ustar = prod(div(num,den),Un)
  num = [1,0]
  den = [3,0]-z
  Un1 = prod(div(num,den),4*Ustar - Un)
  return Un1

