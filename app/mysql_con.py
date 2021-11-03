import os
from sqlalchemy import create_engine  
from sqlalchemy import Column, String, Integer, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base  
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import func

# Create the MySQL connection string
db_string = f"mysql://{os.environ['MYSQL_USER']}:{os.environ['MYSQL_PASSWORD']}@{os.environ['MYSQL_HOST']}:3306/{os.environ['MYSQL_DB']}"

# Establish the connection
db = create_engine(db_string)
base = declarative_base()
session = sessionmaker(db)()

# Create a table for storing the predictions if it doesn't exist yet
db.execute('''
    CREATE TABLE IF NOT EXISTS drybeans (
	id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)
    ''')

class DryBeansSQL(base): 
    ''' Defining new entries '''
    __tablename__ = 'drybeans'
    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(DateTime, default=func.now())
    name = Column(String)

def push_to_sql(array):
    '''  Pushing new data to storage '''
    for i in array:
        drybean = DryBeansSQL(name = i)
        session.add(drybean) 
        
    session.commit()
    print('Data upload complete!')