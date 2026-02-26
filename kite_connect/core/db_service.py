"""
Shared database utilities for the kite_connect package.

Provides a single connection factory used by all modules that need
PostgreSQL access, eliminating duplicated psycopg2.connect() calls.
"""

import os
import sys

import psycopg2

# Append kite_connect to path (not insert) to avoid shadowing top-level packages
_kite_root = os.path.dirname(os.path.dirname(__file__))
if _kite_root not in sys.path:
    sys.path.append(_kite_root)

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
