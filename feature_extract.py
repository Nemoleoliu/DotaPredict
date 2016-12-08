from pymongo import MongoClient
import role
'''
input: count
return: 216 * count int array, 1 * count int array
'''
def feature_vector_extract(count, asRole=False):
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
        hero_vec, match_result = read_match(match, asRole)
        player_vector.append(hero_vec)
        result_vector.append(match_result)

    return player_vector, result_vector

def read_match(match, asRole=False):
    if not asRole:
        hero_vec = []
        match_result = 1 if match["radiant_win"] else 0
        players = match["players"]
        pp = [p["hero_id"] for p in players]
        #build radiant players hero vector

        for x in range(108):
            if x in pp[0:5]:
                hero_vec.append(1)
            else:
                hero_vec.append(0)
        ##dire players hero vector
        for x in range(108):
            if x in pp[5:10]:
                hero_vec.append(1)
            else:
                hero_vec.append(0)   

        return hero_vec, match_result
    else :
        role_dict = role. get_hero_role()
        roles = [
            'Carry',
            'Nuker',
            'Initiator',
            'Disabler',
            'Durable',
            'Escape',
            'Support',
            'Pusher',
            'Jungler'
        ]
        role_vec = [0]* (len(roles) * 2)
        match_result = 1 if match["radiant_win"] else 0
        players = match["players"]
        pp = [p["hero_id"] for p in players]
        for hero in pp[0:5]:
            hero_roles = role_dict[hero]
            for i in range(len(roles)):
                if hero_roles.has_key(roles[i]):
                    role_vec[i] = max(role_vec[i], hero_roles[roles[i]])

        for hero in pp[5:10]:
            hero_roles =  role_dict[hero]
            for i in range(len(roles)):
                if hero_roles.has_key(roles[i]):
                    role_vec[i+len(roles)] = max(role_vec[i], hero_roles[roles[i]])
        return role_vec, match_result


if __name__ == '__main__':
    print feature_vector_extract(10, True)