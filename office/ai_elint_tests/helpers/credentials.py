import functools
import os

import allure

@functools.lru_cache(maxsize=None)
def get_activities_db_database_name() -> str:
    """
    Provides the database name for Activities DB used in ELINT

    Yields:
        str: The URI for the Direction Finder interface from environment variables.
    """
    uri = os.environ["ACTIVITIES_DB_NAME"]
    allure.attach(uri, "Activities DB name", allure.attachment_type.TEXT)
    return uri


@functools.lru_cache(maxsize=None)
def get_activities_db_username() -> str:
    """
    Provides the USERNAME for Activities DB used in ELINT

    Yields:
        str: The URI for the Direction Finder interface from environment variables.
    """
    uri = os.environ["ACTIVITIES_DB_USERNAME"]
    allure.attach(uri, "Activities DB name", allure.attachment_type.TEXT)
    return uri


@functools.lru_cache(maxsize=None)
def get_activities_db_hostname() -> str:
    """
    Provides the IP for Activities DB used in ELINT

    Yields:
        str: The URI for the Direction Finder interface from environment variables.
    """
    uri = os.environ["ACTIVITIES_DB_IP"]
    allure.attach(uri, "Activities DB IP", allure.attachment_type.TEXT)
    return uri


@functools.lru_cache(maxsize=None)
def get_activities_db_port() -> str:
    """
    Provides the IP for Activities DB used in ELINT

    Yields:
        str: The URI for the Direction Finder interface from environment variables.
    """
    uri = os.environ["ACTIVITIES_DB_PORT"]
    allure.attach(uri, "Activities DB PORT", allure.attachment_type.TEXT)
    return uri


@functools.lru_cache(maxsize=None)
def get_intelligence_db_database_name() -> str:
    """
    Provides the database name for Intelligence DB used in ELINT

    Yields:
        str: The URI for the Direction Finder interface from environment variables.
    """
    uri = os.environ["INTELLIGENCE_DB_NAME"]
    allure.attach(uri, "Intelligence DB name", allure.attachment_type.TEXT)
    return uri


@functools.lru_cache(maxsize=None)
def get_intelligence_db_username() -> str:
    """
    Provides the USERNAME for Intelligence DB used in ELINT

    Yields:
        str: The URI for the Direction Finder interface from environment variables.
    """
    uri = os.environ["INTELLIGENCE_DB_USERNAME"]
    allure.attach(uri, "Intelligence DB name", allure.attachment_type.TEXT)
    return uri


@functools.lru_cache(maxsize=None)
def get_intelligence_db_hostname() -> str:
    """
    Provides the IP for Intelligence DB used in ELINT

    Yields:
        str: The URI for the Direction Finder interface from environment variables.
    """
    uri = os.environ["INTELLIGENCE_DB_IP"]
    allure.attach(uri, "Intelligence DB IP", allure.attachment_type.TEXT)
    return uri


@functools.lru_cache(maxsize=None)
def get_intelligence_db_port() -> str:
    """
    Provides the IP for Activities DB used in ELINT

    Yields:
        str: The URI for the Direction Finder interface from environment variables.
    """
    uri = os.environ["INTELLIGENCE_DB_PORT"]
    allure.attach(uri, "Intelligence DB PORT", allure.attachment_type.TEXT)
    return uri
