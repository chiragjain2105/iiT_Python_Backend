import databases
import sqlalchemy
from databases import Database
from sqlalchemy import Table, Column, Integer, String

DATABASE_URL = "mysql://root:root@localhost:3306/iiT_db"
database = Database(DATABASE_URL)
metadata = sqlalchemy.MetaData()

engine = sqlalchemy.create_engine(DATABASE_URL)
metadata.create_all(engine)



users = Table(
    "users",
    metadata,
    Column("id", Integer, primary_key=True, index=True),
    Column("username", String, unique=True, index=True),
    Column("password", String),
)


