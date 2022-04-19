import csv
import sqlite3

connection = sqlite3.connect('database.db')

with open('schema.sql') as f:
    connection.executescript(f.read())

cur = connection.cursor()

with open('criminel.csv', 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='"')
    for line in reader:
        print(line)
        if line[0] != 'nom':
            cur.execute("INSERT INTO posts (nom, meurtre, pays) VALUES (?, ?, ?)",
                        (str(line[0]), int(line[1]), str(line[2])))

connection.commit()
connection.close()
