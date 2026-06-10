import random
import numpy as np
def cosine(a,b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))
def nvoc(lass):
    nass = []
    while len(nass) < lass:
        l = random.choice(vocab)
        l2 = random.choice(vocab)
        k = (l,l2)
        if k not in ass and l != l2 and k not in nass: 
            nass.append(k)
    return nass  
def norm(v):
    n = sum(x*x for x in v)**0.5
    return [x/n for x in v]
def update(a,b,tar):
    pre = cosine(a,b)
    error = tar-pre
    for k in range(len(a)):
        ak = a[k]
        a[k]+=lr*error*b[k]
        b[k]+=lr*error*ak
    return error
def predict(word,vectors,vocab):
    v = vectors[word]
    bestw = None
    bests = -1
    for w in vocab:
        if w == word:
            continue
        score = cosine(v,vectors[w])
        if score > bests:
            bests = score
            bestw = w
    return bestw
lr = 0.1
train = "Bro had one job"
ind = 1
vocab = []
ivoc = {}
voci = {}
vectors = {}
data = train.lower().split()
ass = []
for i in data:
    ass.append((data[ind-1],data[ind]))
    ind+=1
    if ind == len(data):
        break
for k in ass:
    if k[0] not in vocab:
        vocab.append(k[0])
    if k[1] not in vocab:
        vocab.append(k[1])
nass = nvoc(len(ass))
vocab = set(vocab)
ind = 0
for i in vocab:
    ivoc[ind] = i
    voci[i] = ind
    ind+=1
for i in vocab:
    vectors[i] = [random.uniform(-1,1) for _ in range(50)]
for e in range(1000):
    ter = 0
    i = 0
    for _ in ass:
        w1,w2 = ass[i]
        a = vectors[w1]
        b = vectors[w2]
        error = update(a,b,1)
        ter+=abs(error)
        vectors[w1] = norm(a)
        vectors[w2] = norm(b)
        i+=1
    i = 0
    for _ in nass:
        w1,w2 = nass[i]
        a = vectors[w1]
        b = vectors[w2]
        pre = cosine(a,b)
        error = update(a,b,0)
        ter+=abs(error)
        vectors[w1] = norm(a)
        vectors[w2] = norm(b)
        i+=1
    if e == 0:
        fter = ter
print(ter)
fir = "bro "
sent = f"{fir}"
for i in vocab:
    fo = predict(i,vectors,vocab)
    sent+=f"{fo} " 
print(sent)