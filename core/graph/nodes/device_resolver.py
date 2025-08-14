# -*- coding: utf-8 -*-
"""
@author: LIU Ziyi
@email: lavandejoey@outlook.com
@date: 2025/08/14
@version: 0.11.0
"""

from core.config.devices import resolve_devices


def device_resolver_node(state):
    """
    Resolves the devices to use for the different models.
    """
    devices = resolve_devices()
    return {"devices": devices}
