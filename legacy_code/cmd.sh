#!/bin/bash
for i in {1..20}
do
    python3.7 Carlo.py &
    sleep 1s
    python3.7 P0standard.py &
    sleep 1s
    python3.7 P1standard.py &
    sleep 1s
    python3.7 distribute_data.py 4 8 &
    sleep 1s
    echo 'Wait 4mins...'
    sleep 4m
done

Sun Mar  3 17:45:36 2019
Sun Mar  3 17:47:45 2019