from types import TracebackType
from typing import Any, Optional, Self
import pandas as pd
import psycopg2
from psycopg2.extensions import connection as _connection
from psycopg2.extras import RealDictCursor
from tenacity import retry, retry_if_result, stop_after_delay, wait_fixed


class DBUtils:
    def __init__(self, db_config: dict[str, Any]) -> None:
        """
        Initializes DBUtils with database configuration.
        :param db_config: A dictionary containing connection parameters like host, dbname, etc.
        """
        self.config: dict[str, Any] = db_config
        self.connection: _connection | None = None

    def __enter__(self) -> Self:
        self.connect()
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: TracebackType | None
    ) -> None:
        self.close()

    def connect(self) -> _connection:
        """Establishes and returns a connection to the PostgreSQL database."""
        if not self.connection or self.connection.closed:
            self.connection = psycopg2.connect(**self.config)
        return self.connection

    def execute_query(self, query: str, params: Optional[Any] = None) -> Optional[list[dict[str, Any]]]:
        """
        Executes a query and returns results as a list of dictionaries for SELECTs.
        """
        with self.connect() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(query, params)

                if cursor.description:
                    return [dict(row) for row in cursor.fetchall()]

                return None

    def populate_table_from_csv(self, file_path: str, table_name: str) -> None:
        """
        Populates a database table from a local CSV file using copy_expert.
        Assumes the first row of the CSV contains the matching column names.

        :param file_path: Path to the CSV file on your local machine.
        :param table_name: Name of the target table in the database.
        """
        # Open connection and file
        conn = self.connect()
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                # Dynamically extract headers to specify exact target columns
                header_line = f.readline().strip()

                # Rewind the file back to the start so copy_expert can read the header too
                f.seek(0)

                # Build the SQL string using target columns from CSV header
                sql = f"COPY {table_name} ({header_line}) FROM STDIN WITH CSV HEADER"

                with conn.cursor() as cursor:
                    cursor.copy_expert(sql, f)

            # Save changes to the database
            conn.commit()

        except Exception as e:
            # Cancel changes if anything breaks
            conn.rollback()
            raise e

    def fetch_scalar(self, query: str, params: Optional[Any] = None) -> Any:
        with self.connect() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, params)
                row = cursor.fetchone()
                return row[0] if row else None

    def fetch_column(self, query: str, params: Optional[Any] = None) -> list[Any]:
        """
        Executes a query and returns the first column of all rows as a flat list.
        """
        conn = self.connect()
        try:

            with conn.cursor() as cursor:
                cursor.execute(query, params)
                results = cursor.fetchall()
                return [row[0] for row in results] if results else []
        except Exception as e:
            print(f"Error fetching column: {e}")
            return []

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
        sql = f"SELECT count(*) FROM {table_name} WHERE {where_clause}"

        @retry(
            retry=retry_if_result(lambda count: count <= threshold),
            stop=stop_after_delay(timeout),
            wait=wait_fixed(interval),
            reraise=True,
        )
        def _poll() -> int:
            current_count = self.fetch_scalar(sql) or 0
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
        Executes a SELECT query and returns the data as a Pandas DataFrame.
        Optionally attaches the results to an Allure test report as a CSV or text layout.
        """
        conn = self.connect()
        try:
            with conn.cursor() as cursor:
                cursor.execute(query, params)
                results = cursor.fetchall()

                columns = [desc[0] for desc in cursor.description] if cursor.description else []

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
            print(f"Error fetching multiple results: {e}")
            return pd.DataFrame()

    def close(self) -> None:
        if self.connection and not self.connection.closed:
            self.connection.close()
            self.connection = None
