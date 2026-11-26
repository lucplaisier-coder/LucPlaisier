"""
plotstyle.py: identical figures on every computer (step 15 reproducibility fix).

Matplotlib reads a machine-wide settings file (matplotlibrc) that can change
the font. The machine that made the original figures had one that selects the
"Inter" font; Colab has none and uses DejaVu Sans. Same numbers, different
pixels. Importing this module resets matplotlib to its own built-in defaults,
so every computer draws with DejaVu Sans, which ships inside matplotlib itself.
It changes how figures look, never what they show.
"""

import matplotlib

matplotlib.use("Agg")
matplotlib.rcdefaults()   # ignore any machine-wide matplotlibrc
