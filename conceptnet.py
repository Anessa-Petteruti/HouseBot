import requests
import numpy as np

#TODO: refine these verbs here
ai2thor_verbs = ["toggle","break","fill_with_liquid","dirty","use_up","cook","heat_up","make_cold","slice","open","pick_up","move"]

basic_url = 'http://api.conceptnet.io'
relatedness1 = '/relatedness?node1=/c/en/'
relatedness2 = '&node2=/c/en/'
query_url1 = 'http://api.conceptnet.io/query?start=/c/en/'
query_url2 = '&rel=/r/CapableOf&limit=20'

#Calculates probablity of using ai2thor verb by similarity between object noun and ai2thor. 
#In terms of knowledge representation, this may be cheating because it assumes that the robot
#already understands the relationship between a new object and the ai2thor verbs.
def calculateProbBySimilarity(word):
    similarity = []
    for verb in ai2thor_verbs:
        relatedness = requests.get(basic_url + relatedness1 + word + relatedness2 + verb).json()
        similarity.append(relatedness['value'])
    similarity = normalizeBetween01(similarity)
    similarity = [num/sum(similarity) for num in similarity]
    return similarity


def calculateProbBySimilarVerbs(word):
    relatedVerbs = requests.get(query_url1 + word + query_url2).json()
    verbs = []
    for edge in relatedVerbs['edges']:
        verbs.append(edge['end']['label'])
    similarity = [0]*len(ai2thor_verbs)
    verbs = formatCorrectly(verbs)
    print(verbs)
    for i in range(len(ai2thor_verbs)):
        for verb2 in verbs:
            verb1 = ai2thor_verbs[i]
            #print(verb1,verb2)
            #TODO: Add error checking for if this request does not exist
            relatedness = requests.get(basic_url + relatedness1 + verb2 + relatedness2 + verb1).json()
            similarity[i] += relatedness['value']
    similarity = [num/len(verbs) for num in similarity]
    similarity = normalizeBetween01(similarity)
    similarity = [num/sum(similarity) for num in similarity]
    return similarity

def normalizeBetween01(verbList):
    minVal = min(verbList)
    denom = max(verbList) - minVal
    verbList = [(val - minVal)/denom for val in verbList]
    return verbList

def formatCorrectly(verbs):
    return [verb.replace(" ","_") for verb in verbs]

sim = calculateProbBySimilarVerbs("oven")
for i in range(len(sim)):
    print(ai2thor_verbs[i] + ": " + str(sim[i]))

'''
potato UseFor relation:
power special clock
eating 

potato CapableOf relation:
taste good

oven CapableOf relation:
brown chicken
warm a meal
roast
cool its temperature
warm a pie
heat a meal
bake

oven UsedFor relation:
cooking
bake cookies
baking food
bake a cake
roasting
cood food
reheat food
burn
bake things, like food or ceramics
baking, broiling, and heating food
roast a turkey
baking
get rid of story witches
preparing food
grill vegetables

Oven - calculate prob by similar verbs - capable of:
toggle: 0.024320457796852647
break: 0.06994118582101415
fill_with_liquid: 0.05960896518836433
dirty: 0.025274201239866474
use_up: 0.0
cook: 0.2541726275631854
heat_up: 0.3109203624225083
make_cold: 0.04879987283420759
slice: 0.07248450166905102
open: 0.03671912255603243
pick_up: 0.0683516134159911
move: 0.029407089492926395

Over - calculate prob by similarity:
toggle: 0.061006289308176094
break: 0.027672955974842765
fill_with_liquid: 0.07672955974842767
dirty: 0.0220125786163522
use_up: 0.04591194968553459
cook: 0.359748427672956
heat_up: 0.17106918238993712
make_cold: 0.035849056603773584
slice: 0.12515723270440252
open: 0.06352201257861635
pick_up: 0.011320754716981131
move: 0.0

ALSO test with usedFor relation
'''