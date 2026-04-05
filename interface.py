from enum import IntEnum

from pydantic import BaseModel, field_validator


class ActionCode(IntEnum):
    STAY = 0
    COOL = 1
    HEAT = 2

class HomeObservation(BaseModel):
    current_temp: float
    is_human_home: bool
    energy_price: float


class HomeAction(BaseModel):
    # 0: Stay, 1: Cool, 2: Heat
    action_code: int

    @field_validator("action_code")
    @classmethod
    def validate_action_code(cls, value: int) -> int:
        if value not in (ActionCode.STAY, ActionCode.COOL, ActionCode.HEAT):
            raise ValueError("action_code must be 0 (stay), 1 (cool), or 2 (heat)")
        return int(value)

    @property
    def code(self) -> ActionCode:
        return ActionCode(self.action_code)