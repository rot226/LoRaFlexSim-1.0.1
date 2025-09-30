_node_counter = 0
_gateway_counter = 0


def next_node_id() -> int:
    """Return a new unique node identifier."""
    global _node_counter
    _node_counter += 1
    return _node_counter


def next_gateway_id() -> int:
    """Return a new unique gateway identifier."""
    global _gateway_counter
    _gateway_counter += 1
    return _gateway_counter


def reset() -> None:
    """Reset both node and gateway counters."""
    global _node_counter, _gateway_counter
    _node_counter = 0
    _gateway_counter = 0
