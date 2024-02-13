#!/bin/bash
# To run, please have an initial train log file. Change the sequence
#   to reflect the number of epochs

# --lossfun can be specified as crossentropy for my loss, or laploss (default)

for i in `seq 15 15 300`
do
  echo "i: $i"
  cp -r train_log train_log_epoch=$i
  python train.py --mps=True --epoch=15 --model=train_log_epoch=$i --lossfun=crossentropy
done

