import sqlite3

# Connect to sqlite
connection = sqlite3.connect("student.db")

# Create a cursor object to insert record, create table, retrieve
cursor = connection.cursor()

# Create the student table
table_info = """
CREATE TABLE student (
name VARCHAR(25),
class VARCHAR(25),
section VARCHAR(25),
marks INT
)
"""
cursor.execute(table_info)

# insert sample records
cursor.execute("INSERT INTO student VALUES('Krish','Data Science','A',90)")
cursor.execute("INSERT INTO student VALUES('Sudhir','Arts','B',80)")
cursor.execute("INSERT INTO student VALUES('Ganesh','Commerce','A',95)")
cursor.execute("INSERT INTO student VALUES('Ramesh','Science','B',85)")
cursor.execute("INSERT INTO student VALUES('Suresh','Arts','A',75)")

connection.commit()

# Display all the records
print( "The inserted records are")

data = cursor.execute('''SELECT * FROM student''')

for row in data:
    print(row)

# Close the connection
connection.close()