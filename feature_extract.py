from pymongo import MongoClient
import role
'''
input: count
return: 216 * count int array, 1 * count int array
'''

class FeatureExtractor:
    def __init__(self, mode):
        self.hero_role_dict = role.get_hero_role()
        self.mode = mode
        self.roles = [
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

    def feature_vector_extract(self, count, func= lambda x, y: max(x, y)):
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
            hero_vec, match_result = self.read_match(match, func)
            player_vector.append(hero_vec)
            result_vector.append(match_result)

        return player_vector, result_vector

    def read_match(self, match, func):
        if self.mode == 0:
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
        elif self.mode == 1:
            role_vec = [0]* (len(self.roles) * 2)
            match_result = 1 if match["radiant_win"] else 0
            players = match["players"]
            pp = [p["hero_id"] for p in players]
            for hero in pp[0:5]:
                hero_roles = self.hero_role_dict[hero]
                for i in range(len(self.roles)):
                    if hero_roles.has_key(self.roles[i]):
                        role_vec[i] = func(role_vec[i], hero_roles[self.roles[i]])

            for hero in pp[5:10]:
                hero_roles =  self.hero_role_dict[hero]
                for i in range(len(self.roles)):
                    if hero_roles.has_key(self.roles[i]):
                        role_vec[i+len(self.roles)] = func(role_vec[i], hero_roles[self.roles[i]])
            return role_vec, match_result
        elif self.mode == 2:
            role_vec = []
            match_result = 1 if match["radiant_win"] else 0
            players = match["players"]
            pp = [p["hero_id"] for p in players]
            for hero in pp:
                hero_roles =  self.hero_role_dict[hero]
                for i in range(len(self.roles)):
                    if hero_roles.has_key(self.roles[i]):
                        role_vec.append(hero_roles[self.roles[i]])
                    else:
                        role_vec.append(0)
            return role_vec, match_result
        elif self.mode == 3:
            role_vec = []
            match_result = 1 if match["radiant_win"] else 0
            players = match["players"]
            pp = [p["hero_id"] for p in players]
            for hero in pp:
                hero_roles =  self.hero_role_dict[hero]
                for i in range(len(self.roles)):
                    if hero_roles.has_key(self.roles[i]):
                        for j in range(3):
                            if hero_roles[self.roles[i]] == (j+1):
                                role_vec.append(1)
                            else:
                                role_vec.append(0)
                    else:
                        role_vec.append(0)
                        role_vec.append(0)
                        role_vec.append(0)
            return role_vec, match_result

if __name__ == '__main__':
    print FeatureExtractor(2).feature_vector_extract(10)