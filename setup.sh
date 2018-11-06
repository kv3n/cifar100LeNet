#!/bin/sh

# Make data Directory
if [ -d "data" ]; then
  rm -rf data
fi
mkdir data
cd data

# Fetch data
wget https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz
tar -C . -xvf cifar-100-python.tar.gz
rm -rf cifar-100-python.tar.gz
mv cifar-100-python/* .
rm -rf cifar-100-python/