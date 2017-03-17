#!/usr/bin/env python
# -*- coding: utf-8 -*-

x = [1, 8, 14, 23, 44, 55, 67, 88, 103, 146]
print x
print "探したい数字を入力"
i = int(raw_input())

high = len(x)
low = 0
center = (high + low) / 2

while (low<=high):
    if i>=x[center]:
        low = center+1
    elif i<=x[center]:
        high = center-1
    elif i==x[center]:
        break

    center = (high + low) / 2

if (i==x[center]):
    print str(center+1) + "番目にあります"
else:
    print "そんな番号ねーよ、ばかちんが"
