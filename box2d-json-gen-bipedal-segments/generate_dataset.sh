#!/bin/bash

for i in `seq 12`; do
	 ~/anaconda3/bin/python generate_scripts/generate_bipedal.py --body-segments $i --no-rigid-spine --spine-motors --filename 'box2d-json-gen-bipedal-segments/GeneratedBipedalWalker-'$i'-Segments.json'
done
