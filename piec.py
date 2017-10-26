from miningprob import miningprob

Pweight = 0.01 # 001
Iweight = 0.003
Dweight = 0.0 #005 # 0.1
Expected = 600

Imax = 1000
Imin = -1000

def bits_to_target(bits):
    size = bits >> 24
    assert size <= 0x1d

    word = bits & 0x00ffffff
    assert 0x8000 <= word <= 0x7fffff

    if size <= 3:
        return word >> (8 * (3 - size))
    else:
        return word << (8 * (size - 3))

def miningVariance(x):
    #return Expected - x
    if 0:
      error = Expected - x
      sign = -1 if error < 0 else 1
      # log(sign * error*error / 100.0)
      return sign * error*error / 10.0
    else:
      if x >= len(miningprob):
        decay = miningprob[-1]
      elif x < 0:
        decay = miningprob[0]
      else:
        decay = miningprob[x]

      error = (1.0 - 400.0*decay)*(100+Expected-x)
      return error

def error_sum(states):
    ret = 0
    times = [state.timestamp for state in states]
    i = 1
    while i<len(times):
        ret +=  miningVariance(times[i]-times[i-1])
        i+=1
    return ret

def compute_piec_target(states):
    History=50
    # pdb.set_trace()
    # If the error is zero all of these terms == 0
    P = miningVariance(int((states[-1].timestamp - states[-3].timestamp)/2.0))

    In = 0.0
    #for i in range(1,History):
    #    In += miningVariance(states[-i].timestamp - states[-i-1].timestamp)
    In = 600 - int((states[-1].timestamp - states[-History-1].timestamp)/History)
    # log(In)
    if In > Imax: In=Imax
    if In < Imin: In=Imin
    D = (error_sum(states[-40:-20]) - error_sum(states[-20:-1]))
    # log(D)
#    if I > 10000:
#        pdb.set_trace()
    adj = float(P)*Pweight + float(In)*Iweight - float(D)*Dweight
    adjFrac = 1.0 + adj/600.0
    # print("P %f I %f D %f adjFrac %d" % (P, I, D, adjFrac))

    if adjFrac > 1.05:
        adjFrac = 1.05
    if adjFrac < .95:
        adjFrac = .95

    target = bits_to_target(states[-1].bits)
    target /= adjFrac  # The lower the target, the more difficult
    return int(target)

