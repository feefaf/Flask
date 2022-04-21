from xml.etree import ElementTree

import json

js = open('variables.js','r')


euh = js.read()



'''
tree = ElementTree.fromstring(euh).getroot() #get the root
#use etree.find or whatever to find the text you need in your html file
script_text = tree.text.strip()
'''

#extract json string
#you could use the re module if the string extraction is complex
json_string = euh.split('var jason =')[1]
#note that this will work only for the example you have given.
try:
    data = json.loads(json_string)
except ValueError:
    print("invalid json", json_string)
else:
    value = data

print(json_string)