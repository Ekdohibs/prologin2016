# -*- Makefile -*-

lib_TARGETS = champion

# Tu peux rajouter des fichiers source ici
champion-dists = prologin.py

# Evite de toucher a ce qui suit
champion-dists += api.py interface.hh
champion-srcs = interface.cc

champion-cxxflags = -fPIC $(shell python3-config --includes)
champion-ldflags = -s $(shell python3-config --ldflags)

STECHEC_LANG=python
include ../includes/rules.mk
