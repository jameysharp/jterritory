from sqlalchemy import func, select
from sqlalchemy.future import Connection
from typing import NamedTuple, Optional, TypedDict
from . import models
from .exceptions import method


class ObjectId(int):
    def __str__(self) -> str:
        return "o" + super().__str__()


class AddedItem(TypedDict):
    index: int
    id: str


class ObjectPosition(NamedTuple):
    position: int
    objectId: ObjectId

    def offset(self, offset: int) -> "ObjectPosition":
        if not offset:
            return self
        return self._replace(position=self.position + offset)

    def as_dict(self) -> AddedItem:
        return {
            "index": self.position,
            "id": str(self.objectId),
        }


class Account(NamedTuple):
    id: int
    account: str
    details: dict


class Datatype:
    def __init__(self, connection: Connection, accountId: str, datatype: int) -> None:
        account: Optional[Account] = connection.execute(
            models.accounts.select().where(models.accounts.c.account == accountId)
        ).first()
        if account is None:
            raise method.AccountNotFound()

        self.connection = connection
        self.account = account
        self.datatype = datatype

    def __str__(self) -> str:
        return f"{self.account.account}/{self.datatype}"

    @property
    def accountId(self) -> str:
        return self.account.account

    @property
    def id(self) -> int:
        return self.datatype

    def lastChanged(self) -> int:
        result = self.connection.scalar(
            select(func.max(models.objects.c.changed))
            .where(models.objects.c.account == self.account.id)
            .where(models.objects.c.datatype == self.id)
        )
        return result or 0
