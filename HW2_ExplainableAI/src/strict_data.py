import sys
rf = open(sys.argv[1], "r")
wf = open(sys.argv[2], "w")
for line in rf.readlines():
    tmp = line.strip('\n').split(' ')
    begin = tmp.index('<SOS>')
    end = tmp.index('<EOS>')
    sentence = ''
    if end - begin - 1 > 32:
        for i in range(begin, begin + 32 + 1):
            sentence += (tmp[i] + ' ')
        for i in range(end, len(tmp)):
            sentence += (tmp[i] + ' ')
        sentence = sentence[:-1]
    else:
        sentence = ' '.join(tmp)
    wf.write(sentence + '\n')
