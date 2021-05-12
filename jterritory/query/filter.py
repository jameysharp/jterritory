from __future__ import annotations

from sqlalchemy import and_, not_, or_
from sqlalchemy.sql import ClauseElement
from typing import Generic, List, Literal, TypeVar, Union
from ..exceptions import method
from ..types import BaseModel, GenericModel


class FilterCondition(BaseModel):
    def compile(self) -> ClauseElement:
        raise method.UnsupportedFilter().exception()


FilterImpl = TypeVar("FilterImpl", bound=FilterCondition)


class FilterOperator(GenericModel, Generic[FilterImpl]):
    operator: Literal["AND", "OR", "NOT"]
    conditions: List[Union[FilterOperator[FilterImpl], FilterImpl]]

    def compile(self) -> ClauseElement:
        clauses = [condition.compile() for condition in self.conditions]
        if self.operator == "AND":
            return and_(*clauses)
        if self.operator == "OR":
            return or_(*clauses)
        if self.operator == "NOT":
            return and_(*map(not_, clauses))
        raise method.UnsupportedFilter().exception()
