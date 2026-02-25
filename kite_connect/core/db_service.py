"""
Shared database utilities for the kite_connect package.

Provides a single connection factory used by all modules that need
PostgreSQL access, eliminating duplicated psycopg2.connect() calls.
"""

import sys
import os
import psycopg2

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from core.config import DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME


def get_connection(dbname=None):
    """
    Return a new psycopg2 connection.

    Parameters
    ----------
    dbname : str, optional
        Database to connect to.  Defaults to the application DB
        (livestocks_ind).  Pass ``"postgres"`` when creating the
        application DB itself.
    """
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        dbname=dbname or DB_NAME,
    )
