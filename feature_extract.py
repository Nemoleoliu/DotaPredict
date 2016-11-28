from pymongo import MongoClient

'''
input: count
return: 216 * count int array, 1 * count int array
'''
def feature_vector_extract(count):
    #connecting to Local MongoDB
    client = MongoClient('localhost', 27017)
    db = client.dotabot
    matches = db.matches
    datavol = matches.count()

    #if required item num exceed current data volume, return None
    if count > datavol:
        return None

    #read slice with length 'count' starting form line 1
    player_vector = []
    result_vector = []

    for match in matches.find().skip(1).limit(count): 
        hero_vec, match_result = read_match(match)
        player_vector.append(hero_vec)
        result_vector.append(match_result)

    return player_vector, result_vector

def read_match(match):
    hero_vec = []
    match_result = 1 if match["radiant_win"] else 0
    players = match["players"]
    pp = [p["hero_id"] for p in players]
    for x in range(108):
        if x in pp:
            hero_vec.append(1)
        else:
            hero_vec.append(0)

    return hero_vec, match_result


if __name__ == '__main__':
    feature_vector_extract(10)