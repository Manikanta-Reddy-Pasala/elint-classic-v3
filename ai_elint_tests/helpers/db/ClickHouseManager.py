import logging
from typing import Any, Sequence

from clickhouse_driver import Client
from tenacity import (
    RetryCallState,
    retry,
    retry_if_result,
    stop_after_delay,
    wait_fixed,
)
from helpers.exceptions.exceptions import ResponseNotReceivedException
import io
from typing import Any, Optional
import allure
import pandas as pd


class ClickHouseNativeManager:
    """Framework manager for ClickHouse using the Native TCP driver (Python 3.14+ compatible)."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 9000,
        user: str = "default",
        password: str = "",
        database: str = "default",
    ):
        self.config = {
            "host": host,
            "port": port,
            "user": user,
            "password": password,
            "database": database,
            "compression": False,
        }
        self.client: Client = None

    def __enter__(self) -> "ClickHouseNativeManager":
        try:
            self.client = Client(**self.config)
            self.client.execute("SELECT 1")
            logging.info(f"Connected to ClickHouse (Native) at {self.config['host']}")
            return self
        except Exception as e:
            logging.error(f"Native connection failed: {e}")
            raise

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self.client:
            self.client.disconnect()
            logging.info("ClickHouse native connection closed")

    def execute_command(self, sql: str) -> Any:
        if not self.client:
            raise RuntimeError("Client not initialized. Use 'with' statement.")
        return self.client.execute(sql)

    def select_query(self, sql: str, parameters: dict[str, Any] | None = None) -> list[Sequence[Any]]:
        """Executes SELECT and returns rows as a list of tuples."""
        if not self.client:
            raise RuntimeError("Client not initialized. Use 'with' statement.")
        result = self.client.execute(sql, params=parameters)
        return [row for row in result]

    def select_column(self, sql: str, parameters: dict[str, Any] | None = None) -> list[Any]:
        """Returns a flat list of values for a single-column query."""
        rows = self.select_query(sql, parameters)
        return [row[0] for row in rows]

    def poll_until_different(
        self,
        sql: str,
        current_value: Any,
        parameters: dict[str, Any] | None = None,
        timeout: int = 60,
        interval: int = 2,
    ) -> Any:
        """
        Polls the database until the first column of the query result
        is different from current_value.
        """

        def raise_assertion_error(retry_state: RetryCallState) -> None:
            if retry_state.outcome is None:
                raise ResponseNotReceivedException(f"Query result remained '{current_value}' after {timeout}s. ")
            last_val: Any = retry_state.outcome.result() if retry_state.outcome else "No result"
            raise ResponseNotReceivedException(
                f"Query result remained '{current_value}' after {timeout}s. " f"Last value seen: '{last_val}'"
            )

        @retry(
            retry=retry_if_result(lambda x: x == current_value),
            stop=stop_after_delay(timeout),
            wait=wait_fixed(interval),
            retry_error_callback=raise_assertion_error,
        )
        def _execute_with_retry() -> Any:
            result = self.select_column(sql, parameters)
            # Safely extract the value; assuming select_column returns a list
            val = result[0] if isinstance(result, list) and result else result
            logging.info(f"Polling for change: current value is '{val}' (Waiting for change from '{current_value}')")
            return val

        return _execute_with_retry()

    def poll_until_count_exceeds_specified_threshold(
        self,
        table_name: str,
        threshold: int,
        where_clause: str = "1=1",
        timeout: int = 60,
        interval: int = 2,
    ) -> int:
        """
        Polls until the row count of a table exceeds the specified threshold.
        """
        sql = f"SELECT count() FROM {table_name} WHERE {where_clause}"

        def raise_assertion_error(retry_state: RetryCallState) -> None:
            if retry_state.outcome is None:
                raise ResponseNotReceivedException(
                    f"Table '{table_name}' timed out after {timeout}s with no result recorded."
                )
            last_count = retry_state.outcome.result()
            raise ResponseNotReceivedException(
                f"Table '{table_name}' failed to exceed threshold {threshold} "
                f"after {timeout}s. Last count was {last_count}."
            )

        @retry(
            retry=retry_if_result(lambda count: count <= threshold),
            stop=stop_after_delay(timeout),
            wait=wait_fixed(interval),
            retry_error_callback=raise_assertion_error,
        )
        def _poll() -> int:
            val = self.select_column(sql)
            current_count = val[0] if val else 0
            logging.info(f"Polling {table_name}: {current_count} rows (Target: >{threshold})")
            return current_count

        return _poll()

    def fetch_all_results_as_df(
        self,
        query: str,
        params: Optional[Any] = None,
        attach_to_allure: bool = False,
        attachment_name: str = "Query Results"
    ) -> pd.DataFrame:
        """
        Executes a SELECT query on ClickHouse and returns the data as a Pandas DataFrame.
        Optionally attaches the results to an Allure test report as a CSV layout.
        """
        # Ensure the client context is opened before executing
        if not self.client:
            raise RuntimeError("Client not initialized. Use 'with' statement.")

        try:
            # 1. Fix: Use 'self.client' instead of the undefined local 'client' variable
            results, column_types = self.client.execute(
                query,
                params,
                with_column_types=True
            )

            # 2. Fix: clickhouse-driver returns metadata tuples where index 0 is the column name
            columns = [col[0] for col in column_types] if column_types else []

            df = pd.DataFrame(results, columns=columns)

            if attach_to_allure and not df.empty:
                csv_data = df.to_csv(index=False)
                allure.attach(
                    body=csv_data,
                    name=attachment_name,
                    attachment_type=allure.attachment_type.CSV
                )

            return df

        except Exception as e:
            logging.error(f"Error fetching multiple results from ClickHouse to DataFrame: {e}")
            return pd.DataFrame()
