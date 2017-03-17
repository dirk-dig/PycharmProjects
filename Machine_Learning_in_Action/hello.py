# coding: UTF-8
#コメント

score = 40
if score > 60 :
    print "OK"
elif  score > 40:
    print "soso"
else:
    print "NG"

sales = [3, 5, 11, 23, 17]
sum = 0
for sale in sales :
    sum += sale
else:
    print sum

for i in range(10):
    if i == 3:
        break
    print i

n = 0
while n < 10:
    if n == 3:
        n += 1
        continue
    print n
    n += 1