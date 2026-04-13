"""Knowledge-base / cache layer for SWIFTbot.

A single-file SQLite DB that stores groups, Sawicki universality results, and
Q_T measurements with provenance. Content-addressable: inputs with the same
matrix contents produce the same key so nothing is ever re-computed.
"""
