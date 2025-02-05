import logging

from peewee import Model
from playhouse.sqlite_ext import SqliteExtDatabase

logger = logging.getLogger(__name__)

# database = SqliteExtDatabase(None)
database = SqliteExtDatabase("C:/Users/keyang/Desktop/yan0/Agent/cb2/follower_bots/pretraining_data/cb2-data-base/train/game_data.db")

class BaseModel(Model):
    class Meta:
        database = database  # Use proxy for our DB.


def SetDatabaseByPath(path):
    database.init(
        path,
        pragmas=[
            ("journal_mode", "wal"),
            ("cache_size", -1024 * 64),  # 64MB
            ("foreign_keys", 1),
            ("ignore_check_constraints", 0),
            ("synchronous", 1),
        ],
    )


def SetDatabase(config):
    logger.info(f"Pragmas: {config.sqlite_pragmas}")
    # Configure our proxy to use the db we specified in config.
    SetDataConfig(config.data_config())


def SetDataConfig(data_config):
    database.init(
        data_config.sqlite_db_path, # follower_bots\pretraining_data\cb2-data-base\human_model\game_data.db
        pragmas=data_config.sqlite_pragmas,
    )


def SetDatabaseForTesting():
    database.init(":memory:")


def ConnectDatabase():
    database.connect()


def GetDatabase():
    return database


def CloseDatabase():
    database.close()


def CreateTablesIfNotExists(tables):
    # Peewee injects an IF NOT EXISTS check in their create_tables command.
    # It's good to create a function name that explicitly mentions this.
    database.create_tables(tables)
