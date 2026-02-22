"""
Creates the 'livestocks_ind' PostgreSQL database and its tables.

Database: livestocks_ind

Tables:
  stocks        - Live stock data (name, high, low, volume, ltp, change)
  index_groups  - Index definitions (e.g. NIFTY50, NIFTYBANK, etc.)
  index_stocks  - Maps stocks to their index group
"""

import sys
import os
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from core.config import DB_NAME, TABLE_NAME, INDEX_GROUPS
from core.db_service import get_connection


def create_database():
    """Create the livestocks_ind database if it doesn't exist."""
    conn = get_connection(dbname="postgres")
    conn.autocommit = True
    cur = conn.cursor()

    # Check if database already exists
    cur.execute("SELECT 1 FROM pg_database WHERE datname = %s;", (DB_NAME,))
    if cur.fetchone() is None:
        cur.execute(f'CREATE DATABASE "{DB_NAME}";')
        print(f"[OK] Database '{DB_NAME}' created.")
    else:
        print(f"[OK] Database '{DB_NAME}' already exists.")

    cur.close()
    conn.close()


def create_table():
    """Create the stocks, index_groups, and index_stocks tables."""
    conn = get_connection()
    conn.autocommit = True
    cur = conn.cursor()

    # Main stocks table
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
            id         SERIAL PRIMARY KEY,
            name       VARCHAR(255) NOT NULL,
            high       NUMERIC(18, 4),
            low        NUMERIC(18, 4),
            volume     BIGINT,
            ltp        NUMERIC(18, 4),
            change     NUMERIC(18, 4),
            updated_at TIMESTAMP DEFAULT NOW()
        );
    """)
    print(f"[OK] Table '{TABLE_NAME}' created (or already exists).")

    # Index groups table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS index_groups (
            id         SERIAL PRIMARY KEY,
            index_name VARCHAR(100) NOT NULL UNIQUE,
            created_at TIMESTAMP DEFAULT NOW()
        );
    """)
    print("[OK] Table 'index_groups' created (or already exists).")

    # Mapping table: stocks belonging to each index
    cur.execute("""
        CREATE TABLE IF NOT EXISTS index_stocks (
            id             SERIAL PRIMARY KEY,
            index_group_id INTEGER NOT NULL REFERENCES index_groups(id) ON DELETE CASCADE,
            stock_name     VARCHAR(255) NOT NULL,
            UNIQUE (index_group_id, stock_name)
        );
    """)
    print("[OK] Table 'index_stocks' created (or already exists).")

    cur.close()
    conn.close()


def populate_index_groups():
    """Insert index groups and assign random stocks equally to each."""
    conn = get_connection()
    conn.autocommit = True
    cur = conn.cursor()

    # Insert index groups
    for idx_name in INDEX_GROUPS:
        cur.execute(
            "INSERT INTO index_groups (index_name) VALUES (%s) ON CONFLICT (index_name) DO NOTHING;",
            (idx_name,),
        )
    print(f"[OK] Index groups inserted: {INDEX_GROUPS}")

    # Get all stock names (exclude the NIFTY 50 summary row)
    cur.execute("SELECT name FROM stocks WHERE name NOT LIKE 'NIFTY%%' ORDER BY name;")
    all_stocks = [r[0] for r in cur.fetchall()]

    if not all_stocks:
        print("[!] No stocks found in stocks table. Load CSV data first.")
        cur.close()
        conn.close()
        return

    # Shuffle and split equally across index groups
    random.shuffle(all_stocks)
    num_groups = len(INDEX_GROUPS)
    per_group = len(all_stocks) // num_groups
    remainder = len(all_stocks) % num_groups

    # Clear existing mappings
    cur.execute("TRUNCATE TABLE index_stocks RESTART IDENTITY;")

    # Get index group IDs
    cur.execute("SELECT id, index_name FROM index_groups ORDER BY id;")
    groups = cur.fetchall()

    offset = 0
    for i, (group_id, group_name) in enumerate(groups):
        # Distribute remainder stocks to the first few groups
        count = per_group + (1 if i < remainder else 0)
        group_stocks = all_stocks[offset : offset + count]
        offset += count

        for stock_name in group_stocks:
            cur.execute(
                "INSERT INTO index_stocks (index_group_id, stock_name) VALUES (%s, %s) ON CONFLICT DO NOTHING;",
                (group_id, stock_name),
            )

        print(f"  {group_name}: {count} stocks -> {sorted(group_stocks)}")

    cur.close()
    conn.close()
    print(f"\n[OK] {len(all_stocks)} stocks distributed across {num_groups} index groups.")


if __name__ == "__main__":
    create_database()
    create_table()
    populate_index_groups()
