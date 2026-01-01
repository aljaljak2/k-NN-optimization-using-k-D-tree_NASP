#!/bin/bash

# Script to download nanoflann.hpp

echo "Downloading nanoflann.hpp..."

cd include

# Try wget first
if command -v wget &> /dev/null; then
    wget https://raw.githubusercontent.com/jlblancoc/nanoflann/master/include/nanoflann.hpp
    echo "Downloaded using wget"
# Fall back to curl
elif command -v curl &> /dev/null; then
    curl -O https://raw.githubusercontent.com/jlblancoc/nanoflann/master/include/nanoflann.hpp
    echo "Downloaded using curl"
else
    echo "Error: Neither wget nor curl is available"
    echo "Please manually download nanoflann.hpp from:"
    echo "https://raw.githubusercontent.com/jlblancoc/nanoflann/master/include/nanoflann.hpp"
    echo "and place it in benchmarks/include/"
    exit 1
fi

cd ..

if [ -f "include/nanoflann.hpp" ]; then
    echo "✓ nanoflann.hpp successfully downloaded to benchmarks/include/"
    echo "You can now build the benchmark suite"
else
    echo "✗ Failed to download nanoflann.hpp"
    exit 1
fi
