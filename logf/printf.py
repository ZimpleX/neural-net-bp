"""
print formatted string to console / terminal
"""

from logf.stringf import stringf

def printf(string, *reflex, type='INFO', separator='default'):
    """
    for usage, check out logf.stringf
    """
    print(stringf(string, reflex, type, separator))
