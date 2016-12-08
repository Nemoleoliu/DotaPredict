from urllib2 import urlopen
from bs4 import BeautifulSoup

url = 'http://wiki.teamliquid.net/dota2/Hero_Roles'

soup = BeautifulSoup(urlopen(url), 'lxml')
tables = soup.find_all('table')

role_dict = dict()
roles = [
    'Carry',
    'Nuker',
    'Initiator',
    'Disabler',
    'Durable',
    'Escape',
    'Support',
    'Pusher',
    'Jungler',
    'Hero'
]
table_index = 0
for table in tables:
    if 'class' in table.attrs:
        child_index = -1
        for child in table.children:
            if child.name == 'tr':
                child_index+=1
                #  print table_index, child_index
                if child_index % 2 == 0:
                    continue
                else :
                    for h in child.td.children:
                        if h.name =='a' and 'title' in h.attrs:
                            # print h.attrs['title']
                            hero_name = h.attrs['title']
                            if not role_dict.has_key(hero_name):
                                role_dict[hero_name] = dict()
                            role_dict[hero_name][roles[table_index]] = 4 - (child_index + 1)/2
        table_index+=1

print role_dict
