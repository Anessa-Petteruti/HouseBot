# HouseBot
Data/Charts:
* aithornoun_toProperty_relatedness.csv: maps each aithor noun to a similarity score for each actionable property based on the conceptnet similarity between that noun and the property
* aithornouns_capableOf_relatedness.csv: maps each aithor noun to a related verb obtained from concept net using the "capable of" relation, and a similarity score to each actionable property as the similarity of that related verb to the property. This list includes the first 5 returned conceptnet related verbs as well as the average scores of the top 5.
* aithornouns_capableof.csv: maps each aithor noun to a list of verbs which are the related verbs obtained from conceptnet using the "capable of" relation
* aithornouns_usedFor.csv: maps each aithor noun to a list of verbs which are the related verbs obtained from conceptnet using the "used for" relation
* aithornouns_usedFor_relatedness.csv: maps each aithor noun to a related verb obtained from concept net using the "used For" relation, and a similarity score to each actionable property as the similarity of that related verb to the property. This list includes the first 5 returned conceptnet related verbs as well as the average scores of the top 5.
* conceptnetverbs.csv : to be deleted
* conceptnetverbs_capableOf.csv: to be deleted
* conceptnetverbs_usedFor.csv: to be deleted
* detection_nouns_consistent.csv: maps all detectron nouns to their "ground truth" actionable properties using a criteria of consistency with known objects
* detectron_toProperty_relatedness.csv: maps each detectron noun to a similarity score for each actionable property based on the conceptnet similarity between that noun and the property
* detectronnouns_capableOf_relatedness.csv: maps each detectron noun to a related verb obtained from concept net using the "capable of" relation, and a similarity score to each actionable property as the similarity of that related verb to the property. This list includes the first 5 returned conceptnet related verbs as well as the average scores of the top 5.
* detectronnouns_capableof.csv: maps each detectron noun to a list of verbs which are the related verbs obtained from conceptnet using the "capable of" relation
* detectronnouns_usedFor_relatedness.csv: maps each detectron noun to a related verb obtained from concept net using the "used For" relation, and a similarity score to each actionable property as the similarity of that related verb to the property. This list includes the first 5 returned conceptnet related verbs as well as the average scores of the top 5.
* detectronnouns_usedFor.csv: maps each detectron noun to a list of verbs which are the related verbs obtained from conceptnet using the "used For" relation
* final.csv: maps images to objects detected in them by detectron
* groundTruthLabels.csv: all nouns mapped to their actionable properties
* help_usedFor_relatedness.csv: to be deleted
* ithor.csv: maps each aithor noun to its actionable properties alongside other info
* ithor_reformatted.csv: the above chart but reformatted correctly and only the object type and actionable properties columns
* overallAverageCapable.csv: maps all nouns to average scores of similarity to properties and related verbs (top 5) using "capable of" relation
* overallAverageUsed.csv: maps all nouns to average scores of similarity to properties and related verbs (top 5) using "used for" relation
* relatedness.csv: to be deleted
* relatedness3.csv: to be deleted
* relatednessGood.csv: object 1 -> object 2 -> similarity b/n object 1 and object 2 for most nouns

aithorNouns = ["alarm_clock", "aluminum_foil", "apple", "armchair", "baseball_bat", "basketball", "bathtub",
"bathtub_basin", "bed", "blinds", "book", "boots", "bottle", "bowl", "box", "bread", "butter_knife", 
"cabinet", "candle", "cd", "cell_phone", "chair", "cloth", "coffee_machine", "coffee_table", "counter_top",
"credit_card", "cup", "curtains", "desk", "desk_lamp", "desktop", "dining_table", "dish_sponge", "dog_bed",
"drawer", "dresser", "dumbbell", "egg", "faucet", "floor", "floor_lamp", "footstool", "fork", 
"fridge", "garbage_bag", "garbage_can","hand_towel", "hand_towel_holder", "house_plant", "kettle", "key_chain", "knife", 
"ladle", "laptop", "laundry_hamper", "lettuce", "light_switch", "microwave", "mirror", "mug", "newspaper", 
"ottoman", "painting", "pan", "paper_towel_roll", "pen", "pencil", "pepper_shaker", "pillow", "plate", "plunger",
"poster", "pot", "potato", "remote_control", "room_decor", "safe", "salt_shaker", "scrub_brush", "shelf", "shelving_unit", 
"shower_curtain", "shower_door", "shower_glass", "shower_head", "side_table", "sink", "sink_basin", "soap_bar", 
"soap_bottle", "sofa", "spatula", "spoon", "spray_bottle", "statue", "stool", "stove_burner", "stove_knob","table_top_decor",
"target_circle", "teddy_bear", "television", "tennis_racket", "tissue_box", "toaster", "toilet", "toilet_paper", 
"toilet_paper_hanger", "tomato","towel", "towel_holder", "tv_stand", "vacuum_cleaner", "vase","watch", "watering_can", "window", "wine_bottle"]

detectronNouns = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck","boat", "traffic_light",
"fire_hydrant", "stop_sign", "parking_meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow","elephant", "bear", 
"zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports_ball", 
"kite","baseball_glove", "skateboard", "surfboard", "wine_glass", "banana", "sandwich", "orange", "broccoli", "carrot", 
"hot_dog", "pizza", "donut", "cake", "couch","potted_plant", "mouse", "keyboard", "oven","clock","scissors", "hair_drier", "tooth_brush"]

Code:
* closestSimilarity.py: contains functions that calculate scores based on known information of certain objects
* conceptnet.py: contains functions that use conceptnet incuding obtaining scores by similarity of related verbs as well as similarity between object and properties
* extractData.py: used to create various charts
* main.py: main overall functions/accuracy functions for testing found here
* iTHOR_api.py: code to obtain ithor data through various floor plans
* iTHOR_web_scrape.py: code to obtain actionable poperties about aithor objects using webscraping
* neural.py: functions to calculates scores using neural net



