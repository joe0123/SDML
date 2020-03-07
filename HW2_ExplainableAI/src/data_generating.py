import random
random.seed(871024)

import sys

rf = open("corpus.txt", "r")
wf = open("train_original.txt", "w")
append = False
sentence = ""
for line in rf.readlines():
    tmp = line.strip('\n')
    if append == True:
        case = 0
        if len(tmp) > 1:
            case = random.choice(range(2))
        if case == 0:
            control = random.sample(range(len(tmp)), 1)[0]
            wf.write(sentence + " " + str(control + 1) + ' ' + tmp[control] + '\n')
        else:
            controls = sorted(random.sample(range(len(tmp)), 2))
            wf.write(sentence + " " + str(controls[0] + 1) + ' ' + tmp[controls[0]] + ' ' + str(controls[1] + 1) + ' ' + tmp[controls[1]] + '\n')
    else:
        append = True
    sentence = "<SOS> "
    for c in tmp:
        sentence += c + ' '
    sentence += "<EOS>"



