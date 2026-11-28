"""TileLang wrappers over ``csrc/streaming_load.h``; a caller's program pre-includes the header.

Each wrapper moves the 16-byte vector of *src* at *at* to *dst* at *slot*.
"""

import tilelang.language as T

__all__ = ["cp_async16", "load16", "load16_evict_first", "load16_evict_last"]


@T.macro
def load16(dst, slot, src, at):
    """Read into local *dst* with the default cache policy."""
    T.call_extern("handle", "tileops::load16", T.address_of(dst[slot]), T.address_of(src[at]))


@T.macro
def load16_evict_first(dst, slot, src, at):
    """Read into local *dst*, evict-first in L1 and L2: for data nothing reads again."""
    T.call_extern(
        "handle", "tileops::load16_evict_first", T.address_of(dst[slot]), T.address_of(src[at])
    )


@T.macro
def load16_evict_last(dst, slot, src, at):
    """Read into local *dst*, evict-last in L2."""
    T.call_extern(
        "handle", "tileops::load16_evict_last", T.address_of(dst[slot]), T.address_of(src[at])
    )


@T.macro
def cp_async16(dst, slot, src, at):
    """Start a ``cp.async`` into shared *dst*; the caller commits and waits for the group."""
    T.call_extern("handle", "tileops::cp_async16", T.address_of(dst[slot]), T.address_of(src[at]))
