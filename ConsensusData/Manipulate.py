import csv

Area = dict()
AreaNew = dict()
import pprint
with open('Area14Heatmap.tsv','rb') as tsvin, open('Area14FinalHeatmap.tsv', 'w') as csvout:
	tsvin = csv.reader(tsvin, delimiter='\t')
	# csvout = csv.writer(csvout)

	prev = 0

	firstline = True
	for row in tsvin:
		if firstline:
			firstline = False
			continue
		if row[0] == prev: 
			AreaNew[int(row[1])] = float(row[2])
		else: 
			Area[int(prev)] = AreaNew
			AreaNew = dict()
			AreaNew[int(row[1])] = float(row[2])
			prev=row[0]
	pprint.pprint(Area)

	# computation
	kValues = 'day' #K-values
	timeSteps = 'hour' #hour
	value = 'value'
	csvout.write("{}\t{}\t{}\n".format(timeSteps,kValues,value))

	for i, j in Area.iteritems():
		print i,j
		Min = min(j.values())
		MaX = max(j.values())
		for k,l in j.iteritems():
			try: 
				value = float(l-Min)/(MaX-Min)
			except ZeroDivisionError:
				value = 0
			csvout.write("{}\t{}\t{}\n".format(i,k,value))

# pprint.pprint(AreaNew)
        # if count > 0:
        #     csvout.writerows([row[2:4] for _ in xrange(count)])