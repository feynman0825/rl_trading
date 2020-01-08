# encoding: utf-8
# Related constant used in project

# Action constants
HOLD = 0
BUY = 1
SELL = 2

# Position constants
FLAT = 0
LONG = 1
SHORT = -1

STATE_MATHINE = {(FLAT, BUY): LONG,
                 (FLAT, SELL): SHORT,
                 (FLAT, HOLD): FLAT,
                 (LONG, BUY): LONG,
                 (LONG, SELL): FLAT,
                 (LONG, HOLD): LONG,
                 (SHORT, BUY): FLAT,
                 (SHORT, SELL): SHORT,
                 (SHORT, HOLD): SHORT}
