FEATURE_SIGMAS       = [0, 0.1, 0.2, 0.3, 0.5]
LABEL_FLIPS          = [0, 0.05, 0.15, 0.30]

# Extended ranges used for the standalone accuracy plots (NN only)
FEATURE_SIGMAS_plots = [0, 0.1, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
LABEL_FLIPS_plots    = [0, 0.05, 0.15, 0.30, 0.4, 0.5, 0.6, 0.63, 0.66, 0.7, 0.8, 0.9, 1]

COMBINED        = [(0.1, 0.05), (0.2, 0.15), (0.30, 0.15), (0.30, 0.30)]
EPOCHS   = 20
BATCH    = 64
LR       = 0.01
EPS_LOW  = 0.05
N_HIDDEN = 32
BASE_PORT = 6550   # incremented per PTAS run to avoid port collisions
N_RUNS = 5