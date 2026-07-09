#!/bin/bash
set -e
cd /home/itield7/TeraCyte_assignment
perl -0777 -pi -e 's/("output_type": "stream",\n)(          "text")/$1          "name": "stdout",\n$2/g' analysis.ipynb
echo "stdout count: $(grep -c stdout analysis.ipynb)"