"""
====================================================
Utility Functions
====================================================

Common helper functions
"""

import datetime


def print_divider():
    """
    Console ma divider print garne
    """

    print("=" * 60)


def print_title(title):
    """
    Section title print garne
    """

    print_divider()
    print(title)
    print_divider()


def print_status(message):
    """
    Progress message print garne
    """

    current_time = datetime.datetime.now().strftime("%H:%M:%S")

    print(f"[{current_time}] {message}")