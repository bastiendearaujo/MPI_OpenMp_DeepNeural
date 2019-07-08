import urllib, json, time
sumAll = 0
url = "http://admin:CART@pies!284879@186.248.79.60:8099/sistema.cgi?lermc=0,26"
for x in range(0, 10):
	response = urllib.urlopen(url)
	data = json.loads(response.read())
	sumAll += data[3]
	print data[3]
	time.sleep(2)

print "moyenne : %f " % (sumAll/3)