

def getdata(fname):
  f = open(fname, 'r')
  dataraw = f.readlines()
  del dataraw[0]  # header
  datas = [ zzz.split(",") for zzz in dataraw]
  data = []
  for l in datas:
    row = [int(l[0]), float(l[1]), int(l[2]), int(l[3]), float(l[5]),float(l[6]),float(l[7])]
    data.append(row)
  return data
# height, FX, block time, unix ts, difficulty, implied difficulty,hashrate

def diffbytime(data):
    basetime = data[0][3]
    diff_data = [ (d[3] - basetime,d[4]) for d in data]
    return diff_data

def hashbytime(data,scale=1.0):
    basetime = data[0][3]
    diff_data = [ (d[3] - basetime,d[6]*scale) for d in data]
    return diff_data

def blktimebytime(data):
    basetime = data[0][3]
    diff_data = [ (d[3] - basetime,d[2]) for d in data]
    return diff_data

cw144style = {"facecolor":"lightgreen", "edgecolor":"green","markersize":1}
cw144ddstyle = {"facecolor":"lightgreen", "edgecolor":"red","markersize":1}
k1style = {"facecolor":"lightblue", "edgecolor":"blue","markersize":1}
wt144style = {"facecolor":"lavender", "edgecolor":"purple","markersize":1}
piecstyle = {"facecolor":"pink", "edgecolor":"red","markersize":1}

cw144style_tr = {"facecolor":"lightgreen", "edgecolor":"lightgreen","markersize":1}
k1style_tr = {"facecolor":"lightblue", "edgecolor":"lightblue","markersize":1}
wt144style_tr = {"facecolor":"purple", "edgecolor":"lavender","markersize":1}
piecstyle_tr = {"facecolor":"pink", "edgecolor":"pink","markersize":1}


algonames  = ["cw-144", "k-1", "wt-144", "piec"]
algostyles = [cw144style, k1style, wt144style, piecstyle]
algocolors = [ x["edgecolor"] for x in algostyles]
