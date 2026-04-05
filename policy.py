from interface import ActionCode, HomeObservation


def choose_action(obs: HomeObservation) -> int:
    """Baseline energy-aware policy shared by inference and grader."""
    if obs.is_human_home:
        if obs.current_temp > 23:
            return int(ActionCode.COOL)
        if obs.current_temp < 21:
            return int(ActionCode.HEAT)
        return int(ActionCode.STAY)

    # When no one is home, allow wider comfort band to reduce cost.
    if obs.current_temp > 27:
        return int(ActionCode.COOL)
    if obs.current_temp < 18:
        return int(ActionCode.HEAT)
    return int(ActionCode.STAY)
