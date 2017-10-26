#!/usr/bin/env python2
from __future__ import print_function
import sys
import pdb
import argparse
import datetime
import math
import random
try:
  import statistics
except:
  print("for python2, run 'sudo pip install statistics'")
import sys
import time
from collections import namedtuple
from functools import partial
from operator import attrgetter
from functools import partial

from piec import *
squareWave = False
log = partial(print, file=sys.stderr)

useTailRemoval = False

PREFIX_BLOCKS = 2020

SIM_LENGTH = 10000

TARGET_1 = bits_to_target(486604799)

INITIAL_BCC_BITS = 403458999
INITIAL_SWC_BITS = 402734313
INITIAL_FX = 0.18
INITIAL_TIMESTAMP = 1503430225
INITIAL_HASHRATE = 500    # In PH/s.
INITIAL_HEIGHT = 481824

# Steady hashrate mines the BCC chain all the time.  In PH/s.
STEADY_HASHRATE = 300

# simulate a single hashrate drop
SQUARE_HASHRATE = STEADY_HASHRATE*10

# Variable hash is split across both chains according to relative
# revenue.  If the revenue ratio for either chain is at least 15%
# higher, everything switches.  Otherwise the proportion mining the
# chain is linear between +- 15%.
VARIABLE_HASHRATE = 2000   # In PH/s.
VARIABLE_PCT = 15   # 85% to 115%
VARIABLE_WINDOW = 6  # No of blocks averaged to determine revenue ratio

# Greedy hashrate switches chain if that chain is more profitable for
# GREEDY_WINDOW BCC blocks.  It will only bother to switch if it has
# consistently been GREEDY_PCT more profitable.
GREEDY_HASHRATE = 1000     # In PH/s.
GREEDY_PCT = 10
GREEDY_WINDOW = 6

State = namedtuple('State', 'height wall_time timestamp bits chainwork fx '
                   'hashrate rev_ratio greedy_frac msg')

states = []

def bits_to_target(bits):
    size = bits >> 24
    assert size <= 0x1d

    word = bits & 0x00ffffff
    assert 0x8000 <= word <= 0x7fffff

    if size <= 3:
        return word >> (8 * (3 - size))
    else:
        return word << (8 * (size - 3))

MAX_BITS = 0x1d00ffff
MAX_TARGET = bits_to_target(MAX_BITS)

def target_to_bits(target):
    assert target > 0
    if target > MAX_TARGET:
        print('Warning: target went above maximum ({} > {})'
              .format(target, MAX_TARGET), file=sys.stderr)
        target = MAX_TARGET
    size = (target.bit_length() + 7) // 8
    mask64 = 0xffffffffffffffff
    if size <= 3:
        compact = (target & mask64) << (8 * (3 - size))
    else:
        compact = (target >> (8 * (size - 3))) & mask64

    if compact & 0x00800000:
        compact >>= 8
        size += 1

    assert compact == (compact & 0x007fffff)
    assert size < 256
    return compact | size << 24

def bits_to_work(bits):
    return (2 << 255) // (bits_to_target(bits) + 1)

INITIAL_SINGLE_WORK = bits_to_work(INITIAL_BCC_BITS)

def target_to_hex(target):
    h = hex(target)[2:]
    return '0' * (64 - len(h)) + h

def next_bits_flat(msg):
    interval_target = bits_to_target(states[-1].bits)
    return target_to_bits(interval_target)

def next_bits_piec(msg):
    interval_target = compute_piec_target(states)
    if interval_target <= 0:
        interval_target = 1
    return target_to_bits(interval_target)

def compute_dyn_target(states):
    P = 600 - int((states[-1].timestamp - states[-11].timestamp)/10.0)

    adj = float(P)*0.01
    adjFrac = 1.0 + adj/600.0

    if adjFrac > 1.05:
        adjFrac = 1.05
    if adjFrac < .95:
        adjFrac = .95

    target = 0
    for state in states[-11:-1]:
        target += bits_to_target(state.bits)
    target /= 10
    target /= adjFrac  # The lower the target, the more difficult
#    pdb.set_trace()
    return int(target)


def next_bits_dyn(msg):
    interval_target = compute_dyn_target(states)
    if interval_target <= 0:
        interval_target = 1
    return target_to_bits(interval_target)

def print_headers(fil=sys.stdout):
    fil.write(', '.join(['Height', 'FX', 'Block Time', 'Unix', 'Timestamp',
                     'Difficulty (bn)', 'Implied Difficulty (bn)',
                         'Hashrate (PH/s)', 'Rev Ratio', 'Greedy?', 'Comments']) + "\n")

def print_state(state=None, priorState=None, fil=sys.stdout):
    if state is None:
        state = states[-1]
        priorState = states[-2]
    block_time = state.timestamp - priorState.timestamp
    if state.timestamp < 10000000000:
      t = datetime.datetime.fromtimestamp(state.timestamp)
    t = None
    difficulty = TARGET_1 / bits_to_target(state.bits)
    implied_diff = TARGET_1 / ((2 << 255) / (state.hashrate * 1e15 * 600))
    fil.write(', '.join(['{:d}'.format(state.height),
                     '{:.8f}'.format(state.fx),
                     '{:d}'.format(block_time),
                     '{:d}'.format(state.timestamp),
                     '{:%Y-%m-%d %H:%M:%S}'.format(t) if t else "",
                     '{:.2f}'.format(difficulty / 1e9),
                     '{:.2f}'.format(implied_diff / 1e9),
                     '{:.0f}'.format(state.hashrate),
                     '{:.3f}'.format(state.rev_ratio),
                     'Yes' if state.greedy_frac == 1.0 else 'No',
                          state.msg]) + "\n")

def revenue_ratio(fx, BCC_target):
    '''Returns the instantaneous SWC revenue rate divided by the
    instantaneous BCC revenue rate.  A value less than 1.0 makes it
    attractive to mine BCC.  Greater than 1.0, SWC.'''
    SWC_fees = 0.25 + 2.0 * random.random()
    SWC_revenue = 12.5 + SWC_fees
    SWC_target = bits_to_target(INITIAL_SWC_BITS)

    BCC_fees = 0.2 * random.random()
    BCC_revenue = (12.5 + BCC_fees) * fx

    SWC_difficulty_ratio = BCC_target / SWC_target
    return SWC_revenue / SWC_difficulty_ratio / BCC_revenue

def median_time_past(states):
    times = [state.timestamp for state in states]
    return sorted(times)[len(times) // 2]

def next_bits_k(msg, mtp_window, high_barrier, target_raise_frac,
                low_barrier, target_drop_frac, fast_blocks_pct):
    # Calculate N-block MTP diff
    MTP_0 = median_time_past(states[-11:])
    MTP_N = median_time_past(states[-11-mtp_window:-mtp_window])
    MTP_diff = MTP_0 - MTP_N
    bits = states[-1].bits
    target = bits_to_target(bits)

    # Long term block production time stabiliser
    t = states[-1].timestamp - states[-2017].timestamp
    if t < 600 * 2016 * fast_blocks_pct // 100:
        msg.append("2016 block time difficulty raise")
        target -= target // target_drop_frac

    if MTP_diff > high_barrier:
        target += target // target_raise_frac
        msg.append("Difficulty drop {}".format(MTP_diff))
    elif MTP_diff < low_barrier:
        target -= target // target_drop_frac
        msg.append("Difficulty raise {}".format(MTP_diff))
    else:
        msg.append("Difficulty held {}".format(MTP_diff))

    return target_to_bits(target)

def suitable_block_index(index):
    assert index >= 3
    indices = [index - 2, index - 1, index]

    if states[indices[0]].timestamp > states[indices[2]].timestamp:
        indices[0], indices[2] = indices[2], indices[0]

    if states[indices[0]].timestamp > states[indices[1]].timestamp:
        indices[0], indices[1] = indices[1], indices[0]

    if states[indices[1]].timestamp > states[indices[2]].timestamp:
        indices[1], indices[2] = indices[2], indices[1]

    return indices[1]

def compute_index_fast(index_last):
    for candidate in range(index_last - 3, 0, -1):
        index_fast = suitable_block_index(candidate)
        if index_last - index_fast < 5:
            continue
        if (states[index_last].timestamp - states[index_fast].timestamp
            >= 13 * 600):
            return index_fast
    raise AssertionError('should not happen')

def compute_target(first_index, last_index):
    work = states[last_index].chainwork - states[first_index].chainwork
    work *= 600
    work //= states[last_index].timestamp - states[first_index].timestamp
    return (2 << 255) // work - 1

def next_bits_d(msg):
    N = len(states) - 1
    index_last = suitable_block_index(N)
    index_first = suitable_block_index(N - 2016)
    interval_target = compute_target(index_first, index_last)
    index_fast = compute_index_fast(index_last)
    fast_target = compute_target(index_fast, index_last)

    next_target = interval_target
    if (fast_target < interval_target - (interval_target >> 2) or
        fast_target > interval_target + (interval_target >> 2)):
        msg.append("fast target")
        next_target = fast_target
    else:
        msg.append("interval target")

    prev_target = bits_to_target(states[-1].bits)
    min_target = prev_target - (prev_target >> 3)
    if next_target < min_target:
        msg.append("min target")
        return target_to_bits(min_target)

    max_target = prev_target + (prev_target >> 3)
    if next_target > max_target:
        msg.append("max target")
        return target_to_bits(max_target)

    return target_to_bits(next_target)

def compute_cw_target(block_count):
    first, last  = -1-block_count, -1
    timespan = states[last].timestamp - states[first].timestamp
    timespan = max(block_count * 600 // 2, min(block_count * 2 * 600, timespan))
    work = (states[last].chainwork - states[first].chainwork) * 600 // timespan
    return (2 << 255) // work - 1

def next_bits_cw(msg, block_count):
    interval_target = compute_cw_target(block_count)
    return target_to_bits(interval_target)

def next_bits_wt(msg, block_count):
    first, last  = -1-block_count, -1
    last_target = bits_to_target(states[last].bits)
    timespan = 0
    prior_timestamp = states[first].timestamp
    for i in range(first + 1, last + 1):
        target_i = bits_to_target(states[i].bits)
        # Prevent negative time_i values
        timestamp = max(states[i].timestamp, prior_timestamp)
        time_i = timestamp - prior_timestamp
        prior_timestamp = timestamp
        adj_time_i = time_i * target_i // last_target # Difficulty weight
        timespan += adj_time_i * (i - first) # Recency weight
    timespan = timespan * 2 // (block_count + 1) # Normalize recency weight
    target = last_target * timespan # Standard retarget
    target //= 600 * block_count
    return target_to_bits(target)

def next_bits_dgw3(msg, block_count):
    ''' Dark Gravity Wave v3 from Dash '''
    block_reading = -1 # dito
    counted_blocks = 0
    last_block_time = 0
    actual_time_span = 0
    past_difficulty_avg = 0
    past_difficulty_avg_prev = 0
    i = 1
    while states[block_reading].height > 0:
        if i > block_count:
            break
        counted_blocks += 1
        if counted_blocks <= block_count:
            if counted_blocks == 1:
                past_difficulty_avg = bits_to_target(states[block_reading].bits)
            else:
                past_difficulty_avg = ((past_difficulty_avg_prev * counted_blocks) + bits_to_target(states[block_reading].bits)) // ( counted_blocks + 1 )
        past_difficulty_avg_prev = past_difficulty_avg
        if last_block_time > 0:
            diff = last_block_time - states[block_reading].timestamp
            actual_time_span += diff
        last_block_time = states[block_reading].timestamp
        block_reading -= 1
        i += 1
    target_time_span = counted_blocks * 600
    target = past_difficulty_avg
    if actual_time_span < (target_time_span // 3):
        actual_time_span = target_time_span // 3
    if actual_time_span > (target_time_span * 3):
        actual_time_span = target_time_span * 3
    target = target // target_time_span
    target *= actual_time_span
    if target > MAX_TARGET:
        return MAX_BITS
    else:
        return target_to_bits(int(target))

def next_bits_m2(msg, window_1, window_2):
    interval_target = compute_target(-1 - window_1, -1)
    interval_target += compute_target(-2 - window_2, -2)
    return target_to_bits(interval_target >> 1)

def next_bits_m4(msg, window_1, window_2, window_3, window_4):
    interval_target = compute_target(-1 - window_1, -1)
    interval_target += compute_target(-2 - window_2, -2)
    interval_target += compute_target(-3 - window_3, -3)
    interval_target += compute_target(-4 - window_4, -4)
    return target_to_bits(interval_target >> 2)

def block_time(mean_time):
    # Sample the exponential distn
    sample = random.random()
    lmbda = 1 / mean_time
    return math.log(1 - sample) / -lmbda

def next_fx_random(r):
    return states[-1].fx * (1.0 + (r - 0.5) / 200)

def next_fx_static(r):
    return states[-1].fx

def next_fx_square(r):
    global squareWave
    squareWave = True
    return states[-1].fx

def next_fx_ramp(r):
    return states[-1].fx * 1.00017149454

DDBEGIN=600*2
DECLINE_TIME=600*2
def tailremoval(hashrate, target, seconds):
    # reduce difficulty based on time to mine
    if seconds > DDBEGIN:
        difficulty = TARGET_1 / target
        decline = difficulty/DECLINE_TIME
        difficulty = difficulty - (seconds-DDBEGIN)*decline
        if difficulty < 1:
            difficulty = 1
        target = TARGET_1 / difficulty
    # simulate 1 second of mining
    mean_hashes = pow(2, 256) // target
    mean_time = mean_hashes / hashrate
    chance = 1.0/mean_time
    num = random.random()
    if num < chance:
        return int(target)
    return 0

def next_step(algo, scenario, fx_jump_factor, count, timestamp):
    # First figure out our hashrate
    msg = []
    high = 1.0 + VARIABLE_PCT / 100
    scale_fac = 50 / VARIABLE_PCT
    N = VARIABLE_WINDOW
    mean_rev_ratio = sum(state.rev_ratio for state in states[-N:]) / N
    var_fraction = max(0, min(1, (high - mean_rev_ratio) * scale_fac))

    N = GREEDY_WINDOW
    gready_rev_ratio = sum(state.rev_ratio for state in states[-N:]) / N
    greedy_frac = states[-1].greedy_frac
    if mean_rev_ratio >= 1 + GREEDY_PCT / 100:
        if greedy_frac != 0.0:
            msg.append("Greedy miners left")
        greedy_frac = 0.0
    elif mean_rev_ratio <= 1 - GREEDY_PCT / 100:
        if greedy_frac != 1.0:
            msg.append("Greedy miners joined")
        greedy_frac = 1.0

    hashrate = (STEADY_HASHRATE + scenario.dr_hashrate
                + VARIABLE_HASHRATE * var_fraction
                + GREEDY_HASHRATE * greedy_frac)

    if squareWave:
      hashrate = STEADY_HASHRATE  # + scenario.dr_hashrate
      if SQUARE_HASHRATE and timestamp > INITIAL_TIMESTAMP + 1000*600:  # count > SIM_LENGTH/4:
        hashrate = SQUARE_HASHRATE
      if SQUARE_HASHRATE and timestamp > INITIAL_TIMESTAMP + 4000*600:  # count > 3*SIM_LENGTH/4:
        hashrate = STEADY_HASHRATE
      if SQUARE_HASHRATE and timestamp > INITIAL_TIMESTAMP + 8000*600:  # count > 3*SIM_LENGTH/4:
        hashrate = SQUARE_HASHRATE

    # Calculate our dynamic difficulty
    bits = algo.next_bits(msg, **algo.params)
    target = bits_to_target(bits)
    # See how long we take to mine a block
#   pdb.set_trace()

    if useTailRemoval:
        seconds = 0
        while 1:
          newtarget = tailremoval(hashrate * 1e15, target, seconds)
          if newtarget: break
          seconds += 1
        wall_time = states[-1].wall_time + seconds
        # set to make the block reflect the new target
        # bits = target_to_bits(newtarget)
    else:
        mean_hashes = pow(2, 256) // target
        mean_time = mean_hashes / (hashrate * 1e15)
        time = int(block_time(mean_time) + 0.5)
        wall_time = states[-1].wall_time + time

    # Did the difficulty ramp hashrate get the block?
    if random.random() < (scenario.dr_hashrate / hashrate):
        timestamp = median_time_past(states[-11:]) + 1
    else:
        timestamp = wall_time
    # Get a new FX rate
    rand = random.random()
    fx = scenario.next_fx(rand, **scenario.params)
    if fx_jump_factor != 1.0:
        msg.append('FX jumped by factor {:.2f}'.format(fx_jump_factor))
        fx *= fx_jump_factor
    rev_ratio = revenue_ratio(fx, target)

    chainwork = states[-1].chainwork + bits_to_work(bits)

    # add a state
    states.append(State(states[-1].height + 1, wall_time, timestamp,
                        bits, chainwork, fx, hashrate, rev_ratio,
                        greedy_frac, ' / '.join(msg)))

Algo = namedtuple('Algo', 'next_bits params')

Algos = {
    "flat": Algo(next_bits_flat,{}),
    "piec" : Algo(next_bits_piec, {}),
    "dyn" : Algo(next_bits_dyn, {}),
    'k-1' : Algo(next_bits_k, {
        'mtp_window': 6,
        'high_barrier': 60 * 128,
        'target_raise_frac': 64,   # Reduce difficulty ~ 1.6%
        'low_barrier': 60 * 30,
        'target_drop_frac': 256,   # Raise difficulty ~ 0.4%
        'fast_blocks_pct': 95,
    }),
    'k-2' : Algo(next_bits_k, {
        'mtp_window': 4,
        'high_barrier': 60 * 55,
        'target_raise_frac': 100,   # Reduce difficulty ~ 1.0%
        'low_barrier': 60 * 36,
        'target_drop_frac': 256,   # Raise difficulty ~ 0.4%
        'fast_blocks_pct': 95,
    }),
    'd-1' : Algo(next_bits_d, {}),
    'cw-72' : Algo(next_bits_cw, {
        'block_count': 72,
    }),
    'cw-108' : Algo(next_bits_cw, {
        'block_count': 108,
    }),
    'cw-144' : Algo(next_bits_cw, {
        'block_count': 144,
    }),
    'cw-180' : Algo(next_bits_cw, {
        'block_count': 180,
    }),
    'wt-144' : Algo(next_bits_wt, {
        'block_count': 144,
    }),
    'dgw3-24' : Algo(next_bits_dgw3, { # 24-blocks, like Dash
        'block_count': 24,
    }),
    'dgw3-144' : Algo(next_bits_dgw3, { # 1 full day
        'block_count': 144,
    }),
    'meng-1' : Algo(next_bits_m2, { # mengerian_algo_1
        'window_1': 71,
        'window_2': 137,
    }),
    'meng-2' : Algo(next_bits_m4, { # mengerian_algo_2
        'window_1': 13,
        'window_2': 37,
        'window_3': 71,
        'window_4': 137,
    }),
}

Scenario = namedtuple('Scenario', 'next_fx params, dr_hashrate')

Scenarios = {
    'default' : Scenario(next_fx_random, {}, 0),
    'static' : Scenario(next_fx_static, {}, 0),
    'square' : Scenario(next_fx_square, {}, 0),
    'fxramp' : Scenario(next_fx_ramp, {}, 0),
    # Difficulty rampers with given PH/s
    'dr50' : Scenario(next_fx_random, {}, 50),
    'dr75' : Scenario(next_fx_random, {}, 75),
    'dr100' : Scenario(next_fx_random, {}, 100),
}

def run_one_simul(algo, scenario, print_it):
    global states
    if sys.hexversion < 0x3000000:
        states = []
    else:
        states.clear()

    # Initial state is after 2020 steady prefix blocks
    N = PREFIX_BLOCKS
    for n in range(-N, 0):
        state = State(INITIAL_HEIGHT + n, INITIAL_TIMESTAMP + n * 600,
                      INITIAL_TIMESTAMP + n * 600,
                      INITIAL_BCC_BITS, INITIAL_SINGLE_WORK * (n + N + 1),
                      INITIAL_FX, INITIAL_HASHRATE, 1.0, False, '')
        states.append(state)

    # Add 10 randomly-timed FX jumps (up or down 10 and 15 percent) to
    # see how algos recalibrate
    fx_jumps = {}
    factor_choices = [0.85, 0.9, 1.1, 1.15]
    for n in range(10):
        fx_jumps[random.randrange(10000)] = random.choice(factor_choices)

    # Run the simulation
    if print_it:
        print_headers()
    count=0
    for n in range(SIM_LENGTH):
        fx_jump_factor = fx_jumps.get(n, 1.0)
        timestamp = states[-1].timestamp
        next_step(algo, scenario, fx_jump_factor, count, timestamp)
        if print_it:
            print_state()
        count +=1

    # Drop the prefix blocks to be left with the simulation blocks
    simul = states[N:]

    block_times = [simul[n + 1].timestamp - simul[n].timestamp
                   for n in range(len(simul) - 1)]
    return block_times

def run(algo_name, scenario_name, seed, tailremoval, filename=None):
    global useTailRemoval
    algo = Algos.get(algo_name)
    scenario = Scenarios.get(scenario_name)
    if seed is None:
      seed = int(time.time())

    useTailRemoval = tailremoval

    block_times = run_one_simul(algo, scenario, False)
    if filename:
        with open(filename,"w") as f:
            print_headers(f)
            p = None
            for s in states[PREFIX_BLOCKS-1:]:
                if p: print_state(s,p,f)
                p = s

    mean = statistics.mean(block_times)
    std_dev  = statistics.stdev(block_times)
    median = sorted(block_times)[len(block_times) // 2]
    mx = max(block_times)
    log("%s.%s: mean: %0.1f median: %0.1f stdev: %0.1f max:%0.1f\n" % (algo_name, scenario_name, mean, std_dev, median, mx))


    return states

def main():
    '''Outputs CSV data to stdout.   Final stats to stderr.'''
    global useTailRemoval

    parser = argparse.ArgumentParser('Run a mining simulation')
    parser.add_argument('-a', '--algo', metavar='algo', type=str,
                        choices = list(Algos.keys()),
                        default = 'k-1', help='algorithm choice')
    parser.add_argument('-t', '--tailremoval', action="store_true")
    parser.add_argument('-s', '--scenario', metavar='scenario', type=str,
                        choices = list(Scenarios.keys()),
                        default = 'default', help='scenario choice')
    parser.add_argument('-r', '--seed', metavar='seed', type=int,
                        default = None, help='random seed')
    parser.add_argument('-n', '--count', metavar='count', type=int,
                        default = 1, help='count of simuls to run')
    args = parser.parse_args()

    useTailRemoval = args.tailremoval
    log("Tail Removal? ", useTailRemoval)
    count = max(1, args.count)
    algo = Algos.get(args.algo)
    scenario = Scenarios.get(args.scenario)
    seed = int(time.time()) if args.seed is None else args.seed

    to_stderr = partial(print, file=sys.stderr)
    to_stderr("Starting seed {} for {} simuls".format(seed, count))


    means = []
    std_devs = []
    medians = []
    maxs = []
    for loop in range(count):
        random.seed(seed)
        seed += 1
        block_times = run_one_simul(algo, scenario, count == 1)
        means.append(statistics.mean(block_times))
        std_devs.append(statistics.stdev(block_times))
        medians.append(sorted(block_times)[len(block_times) // 2])
        maxs.append(max(block_times))

    def stats(text, values):
        if count == 1:
            to_stderr('{} {}s'.format(text, values[0]))
        else:
            to_stderr('{}(s) Range {:0.1f}-{:0.1f} Mean {:0.1f} '
                      'Std Dev {:0.1f} Median {:0.1f}'
                      .format(text, min(values), max(values),
                              statistics.mean(values),
                              statistics.stdev(values),
                              sorted(values)[len(values) // 2]))

    stats("Mean   block time", means)
    stats("StdDev block time", std_devs)
    stats("Median block time", medians)
    stats("Max    block time", maxs)

if __name__ == '__main__':
    main()

def run4(scenario, filesfx = "", dyn=False):
    seed = 1234
    states = run("piec",scenario,seed,dyn, "piec%s.csv" % filesfx)
    states = run("wt-144",scenario,seed,dyn, "wt144%s.csv" % filesfx)
    states = run("k-1",scenario,seed,dyn, "k1%s.csv" % filesfx)
    states = run("cw-144",scenario,seed,dyn, "cw144%s.csv" % filesfx)

def Test():
    scenario = "square"
    run4(scenario, "", False)
    print("tail removal difficulty adjustment")
    run4(scenario, "_tr", True)

    #sys.argv = "./mining.py -r 110 -a dyn -s static -d".split()
    #main()
