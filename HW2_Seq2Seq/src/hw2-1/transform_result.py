import sys
rf = open(sys.argv[1], "r")
wf = open(sys.argv[2], "w")
for line in rf.readlines():
    tmp = line.strip('\n')
    begin = tmp.find('<SOS>') + 5
    end = tmp.find('<EOS>')
    sentence = '<SOS> '
    for c in tmp[begin: end]:
        sentence += (c + ' ')
    sentence += '<EOS>'
    wf.write(sentence + '\n')
