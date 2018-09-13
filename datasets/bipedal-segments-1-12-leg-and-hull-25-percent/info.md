# bipedal-segments-1-12-leg-and-hull-25-percent

python3 generate_scripts/build_datasets.py BipedalWalker --num-bodies 10 --no-spine-motors --no-report-extra-segments --num-segments 1 12 --hull-width 48.0 80.0 --hull-height 12.0 20.0 --leg-width 6.0 10.0 --leg-height 25.5 42.5 --lower-width 4.8 8.0 --lower-height 25.5 42.5 --directory datasets/bipedal-segments-1-12-leg-and-hull-25-percent

+/- 25% tolerance on hull, leg, and lower width/height, and random number of segments between 1 and 12, with 6 segment numbers given to training, 3 given to validation, and 3 given to test

## train

| File name                     | Body segments      |
| ----------------------------- | ------------------ |
| GeneratedBipedalWalker0.json  | 8                  |
| GeneratedBipedalWalker1.json  | 6                  |
| GeneratedBipedalWalker2.json  | 1                  |
| GeneratedBipedalWalker3.json  | 10                 |
| GeneratedBipedalWalker4.json  | 11                 |
| GeneratedBipedalWalker5.json  | 9                  |
| GeneratedBipedalWalker6.json  | 11                 |
| GeneratedBipedalWalker7.json  | 3                  |
| GeneratedBipedalWalker8.json  | 11                 |
| GeneratedBipedalWalker9.json  | 11                 |

## valid

| File name                     | Body segments      |
| ----------------------------- | ------------------ |
| GeneratedBipedalWalker0.json  | 1                  |
| GeneratedBipedalWalker1.json  | 12                 |
| GeneratedBipedalWalker2.json  | 5                  |
| GeneratedBipedalWalker3.json  | 12                 |
| GeneratedBipedalWalker4.json  | 11                 |
| GeneratedBipedalWalker5.json  | 6                  |
| GeneratedBipedalWalker6.json  | 8                  |
| GeneratedBipedalWalker7.json  | 7                  |
| GeneratedBipedalWalker8.json  | 1                  |
| GeneratedBipedalWalker9.json  | 10                 |

## test

| File name                     | Body segments      |
| ----------------------------- | ------------------ |
| GeneratedBipedalWalker0.json  | 6                  |
| GeneratedBipedalWalker1.json  | 7                  |
| GeneratedBipedalWalker2.json  | 2                  |
| GeneratedBipedalWalker3.json  | 6                  |
| GeneratedBipedalWalker4.json  | 11                 |
| GeneratedBipedalWalker5.json  | 5                  |
| GeneratedBipedalWalker6.json  | 4                  |
| GeneratedBipedalWalker7.json  | 7                  |
| GeneratedBipedalWalker8.json  | 10                 |
| GeneratedBipedalWalker9.json  | 9                  |

