#!/usr/bin
convert $(ls figs/*.0.png | sort -n -t . -k 2) heatmap.0.gif
