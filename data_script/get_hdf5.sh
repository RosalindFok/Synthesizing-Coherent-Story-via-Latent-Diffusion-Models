#!/bin/bash

rm -rf ../../dataset/newcaps
rm ../oxford_data/oxford.hdf5

python move

python oxford_hdf5.py