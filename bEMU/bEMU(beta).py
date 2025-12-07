import P4E
da = []
db = []
train = "The cat sat on the mat while the dog watched then the dog sat on the mat while the cat watched and both the cat and the dog liked to rest together on the warm mat in the sun"
ass = {} # EX: CAT+DOG, DOG+MAT,etc.
data = train.split()# training data
print(len(data))
vectors = {} # the values and shit
similarity = {} # to store all of the similarity values
pvals = {} # to store all the outputs of the sigmoid
loss = [] # to store all the loss values of the P values (AKA pvals)
tosum = [] # so we can sum all of the vectors when multiplied with their pair. EX: cat = [0.29,0.51,-0.32] dog = [0.42,0.12,0.87] 0.29*0.42+0.51*0.12+-0.32*0.87
ID = 0 # just to number the ass dict keys
ID2 = 0 # just to number the [cat,dog] raw similarity values
ID3 = 0 # just to number all the pvals in the dict
ind = 1 # just to be ahead of the current letter when forming the (letter,letter_after) thing
for a in data:
    vectors[a] = [P4E.uni(),P4E.uni(),P4E.uni(),] # Pull some vectors out of its ass
#print(vectors)
for i in data: # form ass and [cat,dog]
    ass[ID] = [i,data[ind]] 
    ind+=1
    ID+=1
    if ind > len(data)-1:
        break
similarity = P4E.similarity(ass,vectors)
print(similarity) 
for s in similarity:
    sim = similarity[s]
    P = P4E.sigmoid(sim)
    pvals[ID3] = P
    ID3+=1
for p in pvals:
    P = pvals[p]
    loss.append(P4E.log(True, P))
lx = sum(loss)/5
print(lx)
print(vectors)

for ioe in ass:
    liss = ass[ioe]
    dot = (vectors[liss[0]][0]*vectors[liss[1]][0]) + (vectors[liss[0]][1]*vectors[liss[1]][1]) + (vectors[liss[0]][2]*vectors[liss[1]][2])
    norma = ((vectors[liss[0]][0]**2) + (vectors[liss[0]][1]**0.5) + (vectors[liss[0]][2]**2)) ** 0.5
    normb = ((vectors[liss[1]][0]**2) + (vectors[liss[1]][1]**0.5) + (vectors[liss[1]][2]**2)) ** 0.5
    for i in range(3):
        (vectors[liss[1]][i]/(norma*normb))
        
        
        

