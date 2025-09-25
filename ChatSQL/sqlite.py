import sqlite3 

# connect to sqlite
connection = sqlite3.connect("student.db")

# create a cursor object to insert record,create table
cursor = connection.cursor()

#create the table
table_info = """
create table STUDENT(NAME VARCHAR(25),CLASS VARCHAR(25),SECTION VARCHAR(25),MARKS INT)
"""
cursor.execute(table_info) # to execute query table_info

#insert some more records
cursor.execute('''Insert Into STUDENT values('Kirti','Data Science','A',90)''')
cursor.execute('''Insert Into STUDENT values('Kunj','Data Science','B',80)''')
cursor.execute('''Insert Into STUDENT values('Vayu','Data Science','A',95)''')
cursor.execute('''Insert Into STUDENT values('Lakshit','DEVOPS','C',75)''')
cursor.execute('''Insert Into STUDENT values('Shaurya','DEVOPS','B',85)''')

# Display all the records
print("The inserted records are:")
data=cursor.execute('''Select * from STUDENT''')
for row in data:
    print(row)

#commit changes in database
connection.commit()
connection.close()
