# -*- coding: utf-8 -*-
"""
@file: core/graph/nodes/device_resolver.py
@author: LIU Ziyi
@email: lavandejoey@outlook.com
@date: 2025/08/14
@version: 0.11.0
"""


# from core.config.devices import resolve_devices


def device_resolver_node(state, resolved_devices):
    """
    Resolves the devices to use for the different models.
    """
    return {"devices": resolved_devices}
