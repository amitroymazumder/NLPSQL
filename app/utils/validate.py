# utils/validate.py

import re

def is_safe_sql(sql):
    """Allow only SELECT queries"""
    return bool(re.match(r"^\s*SELECT", sql.strip(), re.IGNORECASE))
