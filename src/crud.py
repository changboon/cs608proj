import sqlite3
from sqlite3 import Error
import pandas as pd


def create_connection(db_file):
    """ create a database connection to the SQLite database
        specified by the db_file
    :param db_file: database file
    :return: Connection object or None
    """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
    except Error as e:
        print(e)

    return conn


def get_tables(conn):
    table = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table';", conn)
    return table


def get_records(conn, table_name, limit=None):
    query = "SELECT * FROM " + table_name
    if limit!=None:
        query += " LIMIT " + str(limit)
    table = pd.read_sql_query(query, conn)
    return table