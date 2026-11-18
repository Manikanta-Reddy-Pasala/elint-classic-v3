import os
from pathlib import Path
from typing import Any, Generator, Iterator, List, Tuple
import gc
import pytest
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass
try:
    import h5py
except ImportError:
    h5py = None
import numpy as np
import onnxruntime as ort
from collections import Counter
import allure
import pandas as pd
from onnxruntime import InferenceSession
from sklearn.metrics import classification_report
from typing import Any, Dict, List, Tuple
from helpers.db.ClickHouseManager import ClickHouseNativeManager
from helpers.db.PostgreSQLManager import DBUtils
import pytest
from _pytest.fixtures import SubRequest
import allure
import time
import helpers.config as config
from pathlib import Path
import psycopg2
import threading
import sys
import os
import sqlite3
import os
import time
import gc
import psycopg2
from psycopg2.extras import execute_values
import runpy
import subprocess
from datetime import datetime, timezone
from urllib.error import HTTPError
import requests
import allure
import tempfile
import csv
from helpers.utils.pdf_reporter import ParameterizedSuiteReporter
from typing import List, Tuple, Dict, Generator, Any, Optional
from _pytest.config import Config
from _pytest.config.argparsing import Parser
from _pytest.reports import TestReport
from _pytest.nodes import Item
from _pytest.runner import CallInfo
from _pytest.main import Session
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
import time
from datetime import datetime


SUMMARY_CSV = "test_results_summary.csv"


def _db_unreachable(kind: str, host: str, port: int, timeout: float = 2.0) -> bool:
    """Best-effort pre-check: skip early when the DB port is not reachable."""
    import socket
    try:
        if kind == "tcp":
            with socket.create_connection((host, port), timeout=timeout):
                return False
        else:
            import clickhouse_driver
            client = clickhouse_driver.Client(host=host, port=port, timeout=int(timeout + 1))
            client.execute("SELECT 1")
            return False
    except Exception:
        return True
    return False

def pytest_sessionstart(session: pytest.Session) -> None:
    """Runs exactly once before any tests are collected or executed.
    Safely deletes the old summary report file if it exists."""
    Path(SUMMARY_CSV).unlink(missing_ok=True)
    print(f"\n--- Cleaned up old summary file: {SUMMARY_CSV} ---")


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo[None]):
    outcome = yield
    rep = outcome.get_result()

    if rep.when == "call":
        test_name = item.obj.__name__
        input_csv_path = item.funcargs.get("csv_file_path", "N/A")
        input_csv_path = Path(input_csv_path).name
        param_str = item.name.split('[')[-1].replace(']', '') if '[' in item.name else "None"
        status = rep.outcome  # 'passed' or 'failed'

        # --- EXTRACT ASSERTION ERROR ---
        error_msg = ""
        if rep.failed and call.excinfo is not None:
            exc_value = call.excinfo.value
            error_msg = str(exc_value).strip().splitlines()[-1]
            exc_type_name = call.excinfo.typename
            if exc_type_name and not error_msg.startswith(exc_type_name):
                error_msg = f"{exc_type_name}: {error_msg}"

        # --- NEW LOGIC: EXTRACT TEST WARNINGS (CRASH-PROOF) ---
        warnings_found = []

        # A. Check for your custom database condition message first
        if hasattr(item, "stash") and "custom_db_warning" in item.stash:
            warnings_found.append(item.stash["custom_db_warning"])

        # B. Fallback/Append modern pytest Stash system warnings (pytest 7.0+)
        if hasattr(item, "stash") and hasattr(item.stash, "_storage"):
            for key in item.stash._storage.keys():
                if "Warning" in type(key).__name__ or "warning" in str(key).lower():
                    warnings_list = item.stash.get(key, [])
                    if isinstance(warnings_list, list):
                        for w in warnings_list:
                            msg = getattr(w, "message", w)
                            warnings_found.append(str(msg))
                    break

        # C. Fallback for older pytest versions
        if not warnings_found and hasattr(item, "_warnings"):
            old_warnings = getattr(item, "_warnings", [])
            if isinstance(old_warnings, list):
                for w in old_warnings:
                    msg = getattr(w, "message", w)
                    warnings_found.append(str(msg))

        # Join everything nicely into your column string
        warnings_str = "; ".join(warnings_found) if warnings_found else "None"

        # Check if the summary file is brand new or empty
        file_is_empty = not os.path.exists(SUMMARY_CSV) or os.stat(SUMMARY_CSV).st_size == 0

        # Open the summary tracker in append mode
        with open(SUMMARY_CSV, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)

            # Write headers including the new Warnings column
            if file_is_empty:
                writer.writerow(
                    ["Test Name", "Input CSV Path", "Parameter Group", "Status", "Error Message", "Warnings"])

            # Append the completed test metrics
            writer.writerow([
                test_name,
                input_csv_path,
                param_str,
                status,
                error_msg.strip(),
                warnings_str.strip()  # <-- Added here
            ])

        if os.path.exists(SUMMARY_CSV) and os.stat(SUMMARY_CSV).st_size > 0:
            with open(SUMMARY_CSV, mode='r', encoding='utf-8') as f:
                csv_content = f.read()

            allure.attach(
                body=csv_content,
                name="Final Test Execution Status Summary",
                attachment_type=allure.attachment_type.CSV
            )


def get_test_path() -> Path:
    return Path(__file__).parent

@pytest.fixture(scope="module")
@allure.title("Calculator Backend Client on DSP server")
def activities_db_connector(
) -> Generator[ClickHouseNativeManager, Any, None]:
    """
    Provides a configured Calculator Backend HGLB client running on DSP server for test functions.

    Args:
        calculator_backend_dsp_hglb_interface_uri:
        The URI for connecting to the Calculator
        Backend HGLB service running on DSP server
        zmq_context: The ZeroMQ context for socket operations
        zmq_identity: The unique identity for the ZeroMQ client

    Yields:
        CalculatorBackendClient: A configured client instance ready for communication
    """
    with allure.step("Initialize Activities DB connector"):
        host = config.CLICKHOUSE_CONFIG['host']
        port = int(os.getenv('CH_NATIVE_PORT') or config.CLICKHOUSE_CONFIG['port'])
        if host == "127.0.0.1" and not os.environ.get("AIF_VM_HOST") and _db_unreachable("tcp", host, port, 2):
            pytest.skip(f"DB {host}:{port} not reachable from this host - run the harness on the VM (10.10.120.13) or start socat forwards there")
        with ClickHouseNativeManager(
            host=host,
            port=port,
            user=config.CLICKHOUSE_CONFIG['user'],
            password=config.CLICKHOUSE_CONFIG['password'],
            database=config.CLICKHOUSE_CONFIG['database'],
        ) as client:
            yield client


@pytest.fixture(scope="module")
def intelligence_db_connector(
) -> Generator[DBUtils, Any, None]:
    config_pg = {
        "host": config.PG_CONFIG['host'],
        "database": config.PG_CONFIG['database'],
        "user": config.PG_CONFIG['user'],
        "port": int(os.getenv('PG_NATIVE_PORT') or config.PG_CONFIG['port']),
        "password": config.PG_CONFIG['password'],
    }
    with DBUtils(config_pg) as db:
        yield db


@pytest.fixture(scope="function")
def create_local_database():
    """Connects to default postgres db to create the new target database."""
    print(f"Connecting to default database to create '{NEW_DB_NAME}'...")

    conn = psycopg.connect(
        host=config.PG_CONFIG['host'],
        user=config.PG_CONFIG['user'],
        password="postgres",
        port=config.PG_CONFIG['port'],
        dbname=config.PG_CONFIG['database'],
        autocommit=True  # Required for CREATE DATABASE
    )

    with conn.cursor() as cur:
        cur.execute("SELECT 1 FROM pg_database WHERE datname = %s;", (config.PG_CONFIG['database'],))
        exists = cur.fetchone()

        if not exists:
            cur.execute(f"CREATE DATABASE {config.PG_CONFIG['database']};")
            print(f"🎉 Database '{config.PG_CONFIG['database']}' created successfully.")
        else:
            print(f"ℹ️ Database '{config.PG_CONFIG['database']}' already exists.")

    conn.close()

@pytest.fixture(scope="function")
def create_csv_input_table(
    activities_db_connector: ClickHouseNativeManager,
) -> None:

    if os.environ.get("AI_ENGINE") == "rust":
        # Rust engine wipes + loads the CSV into ClickHouse itself; the
        # csv_input table is created by the engine, not the harness.
        return
    with allure.step(
        "Creating table to store CSV input in Activities DB(Clickhouse)"
    ):
        drop_query = 'DROP TABLE IF EXISTS csv_input'

        query = (
            """
            CREATE TABLE IF NOT EXISTS signals.csv_input (
            event_id	UInt64,					
	        power_dbm	Float64,					
	        center_frequency_hz	Float64,					
	        bandwidth_hz	Float64,					
	        duration_ns	Float64,					
	        elevation_deg	Float32,					
	        pulse_repetition_interval_ns	Float64,					
	        emitter_id	UInt32,					
	        source	String,					
	        dtoa_ns	Int64,					
	        snr_db	Float64,					
	        sidelobe_blanking_flag_as_int8	Int64,					
	        time_of_arrival_ns	Int64,					
	        direction_of_arrival_deg	Float64,					
	        type	String,					
	        id	Int64
            )
            ENGINE = MergeTree()
            ORDER BY id
            """
        )
        activities_db_connector.execute_command(drop_query)

        activities_db_connector.execute_command(query)


@pytest.fixture(scope="function")
def truncate_activities_db_tables(
    activities_db_connector: ClickHouseNativeManager,
) -> None:

    with allure.step(
        "Clearing tables from Activities DB(Clickhouse)"
    ):
        queries = ['truncate table "signals"."clusters"', 'truncate table "signals"."csv_input"']

        for query in queries:
            activities_db_connector.execute_command(query)


@pytest.fixture(scope="function")
def truncate_intelligence_db_tables(
    intelligence_db_connector: DBUtils,
) -> None:

    intelligence_db_connector.execute_query('DELETE FROM clusters CASCADE')
    intelligence_db_connector.execute_query('DELETE FROM signals CASCADE')
    intelligence_db_connector.execute_query('DELETE FROM emitters CASCADE')


@pytest.fixture(scope="function")
def populate_signatures_table(
    intelligence_db_connector: DBUtils,
) -> None:

    if os.environ.get("AI_ENGINE") == "rust":
        # Rust engine populates the signatures table itself.
        return

    # Rust engine uses the signatures table but the CSV has extra columns not in PG.
    # Only insert the columns that exist in the table, using pandas to handle the mismatch.
    import pandas as pd

    signatures_table_content_csv_path = 'helpers/utils/signatures_table_content.csv'
    df = pd.read_csv(signatures_table_content_csv_path)

    # Get existing columns in the signatures table
    with intelligence_db_connector.connect() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT column_name FROM information_schema.columns WHERE table_name='signatures'"
            )
            db_cols = {row[0] for row in cursor.fetchall()}

    # Keep only columns that exist in the DB
    df = df[[c for c in df.columns if c in db_cols]]

    # Insert row by row (safe, small table)
    with intelligence_db_connector.connect() as conn:
        with conn.cursor() as cursor:
            cols = list(df.columns)
            placeholders = ', '.join(['%s'] * len(cols))
            col_list = ', '.join(cols)
            insert_sql = f"INSERT INTO signatures ({col_list}) VALUES ({placeholders})"
            for _, row in df.iterrows():
                cursor.execute(insert_sql, tuple(row))
            conn.commit()

@pytest.fixture(scope="function")
def csv_file_path(request: SubRequest) -> str:

    input_csv_filename = request.param

    return input_csv_filename

@pytest.fixture(scope="session")
def shared_confusion_matrix():
    data = [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ]

    col_names = ["PSR1", "PSR2", "PSR3", "PSR4", "PSR5m1", "PSR5m2", "MF03", "MF04", "unclustered", "unknownai"]
    row_names = ["PSR1", "PSR2", "PSR3", "PSR4", "PSR5m1", "PSR5m2", "MF03", "MF04"]

    state = {
        "confusion_matrix": pd.DataFrame(data, columns=col_names, index=row_names)
    }
    yield state

@pytest.fixture(scope="session")
def shared_stats_dictionary():

    dictionary_stats = {}

    state = {
        "dictionary_stats": dictionary_stats
    }
    yield state

@pytest.fixture(scope="function")
def reset_shared_stats(shared_stats_dictionary):
    shared_stats_dictionary["dictionary_stats"].clear()
    yield

@pytest.fixture(scope="function")
def get_existing_emitters(intelligence_db_connector) -> str:

    query = "select name from emitters;"
    emitters = intelligence_db_connector.fetch_column(query)
    yield emitters


@pytest.fixture(scope="function")
def populate_csv_input_table(
    activities_db_connector: ClickHouseNativeManager,
    csv_file_path: str
) -> None:

    if os.environ.get("AI_ENGINE") == "rust":
        # Rust engine loads the CSV into ClickHouse csv_input itself.
        return

    with allure.step(
        "Populating CSV input table from Activities DB(Clickhouse)"
    ):

        file_path = csv_file_path
        df_full = pd.read_csv(file_path)
        df_full["id"] = range(1, len(df_full) + 1)

        df_full.to_csv(file_path, index=False)
        insert_query = ('INSERT INTO csv_input (power_dbm, center_frequency_hz, bandwidth_hz, duration_ns, elevation_deg, pulse_repetition_interval_ns, emitter_id,'
                        'source, dtoa_ns, snr_db, sidelobe_blanking_flag_as_int8, time_of_arrival_ns, direction_of_arrival_deg, type, id) VALUES')
        # Fall back to a standard dictionary-based insertion syntax
        activities_db_connector.client.execute(insert_query, df_full.to_dict('records'))

@pytest.fixture(scope="function")
def delete_known_clusters_centers() -> None:

    file = [
        "known_cluster_centers.json",
        "helpers/app/known_cluster_centers.json",
        "helpers/app_latest/known_cluster_centers.json",
        "helpers/app_source_omni/known_cluster_centers.json",
        "helpers/app_main_lobe/known_cluster_centers.json",
    ]

    for f in file:
        if os.path.exists(f):
            os.remove(f)

@pytest.fixture(scope="function")
def delete_emitter_signal_map() -> None:

    file = [
        "emitter_signal_map.npy",
        "helpers/app/emitter_signal_map.npy",
        "helpers/app_latest/emitter_signal_map.npy",
        "helpers/app_source_omni/emitter_signal_map.npy",
        "helpers/app_main_lobe/emitter_signal_map.npy"

    ]

    for f in file:
        if os.path.exists(f):
            os.remove(f)

@pytest.fixture(scope="function")
def delete_known_medoids() -> None:
    file = [
        "known_medoids.npy",
        "helpers/app/known_medoids.npy",
        "helpers/app_latest/known_medoids.npy",
        "helpers/app_source_omni/known_medoids.npy",
        "helpers/app_main_lobe/known_medoids.npy"

    ]

    for f in file:
        if os.path.exists(f):
            os.remove(f)

@pytest.fixture(scope="function")
def delete_matched_metrics() -> None:
    file = [
        "matched_metrics.csv",
        "helpers/app/matched_metrics.csv",
        "helpers/app_latest/matched_metrics.csv",
        "helpers/app_source_omni/matched_metrics.csv",
        "helpers/app_main_lobe/matched_metrics.csv"

    ]

    for f in file:
        if os.path.exists(f):
            os.remove(f)


@pytest.fixture(scope="function")  # Or "session" depending on how long you want to keep it
def onnx_session_2() -> Generator[ort.InferenceSession, None, None]:
    onnx_session = init_onnx_session(ONNX_MODEL_PATH)

    yield onnx_session  # This provides the session to your tests

    # 2. Teardown: Forcefully close and free memory after tests finish
    del onnx_session
    gc.collect()

@pytest.fixture(scope="function")
def connect_postgres():
    """Establishes a connection to the Postgres database."""
    conn = psycopg2.connect(
        host=PG_CONFIG['host'],
        port=PG_CONFIG['port'],
        database=PG_CONFIG['database'],
        user=PG_CONFIG['user'],
        password=PG_CONFIG['password']
    )
    conn.autocommit = False  # Ensures transactions commit automatically
    print("Connected successfully to Postgres database.")
    yield conn
    conn.close()


@pytest.fixture(scope="function")
def run_ai_model_fixture_main_lobe(csv_file_path: str) -> None:

    dev_folder_path = "helpers/app_main_lobe"
    start_time = time.perf_counter()
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print(f"current_execution_time: {current_time}")
    try:
        run_ai_model(csv_file_path,600, dev_folder_path)
    finally:
        end_time = time.perf_counter()
        execution_time = end_time - start_time

        allure.attach(
            f"Test execution start time {current_time}",
            name="Test execution start time",
            attachment_type=allure.attachment_type.TEXT
        )
        allure.attach(
            f"Total execution time for AI module is {execution_time}",
            name="Time Execution for AI module",
            attachment_type=allure.attachment_type.TEXT
        )
        gc.collect()
        time.sleep(0.05)

@pytest.fixture(scope="function")
def run_ai_model_fixture_source_omni(csv_file_path: str) -> None:

    dev_folder_path = "helpers/app_source_omni"
    start_time = time.perf_counter()
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")

    if os.environ.get("AI_ENGINE") == "rust":
        # Rust engine loads the CSV into ClickHouse csv_input itself;
        # skip the Python-side CSV population fixtures.
        from tests.conftest_rust import _run_rust_engine
        _run_rust_engine(str(csv_file_path), source="omni", timeout=600)
        return
    try:
        run_ai_model(csv_file_path,600,  dev_folder_path)
    finally:
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        allure.attach(
            f"Test execution start time {current_time}",
            name="Test execution start time",
            attachment_type=allure.attachment_type.TEXT
        )
        allure.attach(
            f"Total execution time for AI module is {execution_time}",
            name="Time Execution for AI module",
            attachment_type=allure.attachment_type.TEXT
        )
        gc.collect()
        time.sleep(0.05)

def run_ai_model(csv_file_path: str, timeout: int = 60, dev_folder_path: str = "helpers/app") -> None:
    if os.environ.get("AI_ENGINE") == "rust":
        from tests.conftest_rust import _run_rust_engine
        source = "omni" if "omni" in dev_folder_path else "lobe"
        _run_rust_engine(
            str(csv_file_path),
            source=source,
            timeout=timeout,
        )
        return
    app_file_path = os.path.join(dev_folder_path, "app.py")
    env = os.environ.copy()
    env['INPUT_CSV_PATH'] = str(csv_file_path)
    env['PYTHONPATH'] = f"{dev_folder_path}{os.pathsep}{env.get('PYTHONPATH', '')}"

    # Force single-threaded execution to prevent deadlocks in CI
    env['OMP_NUM_THREADS'] = '1'
    env['MKL_NUM_THREADS'] = '1'
    folder_name = os.path.basename(os.path.dirname(csv_file_path))

    with tempfile.TemporaryFile(mode="w+") as stdout_file, \
            tempfile.TemporaryFile(mode="w+") as stderr_file:

        try:
            result = subprocess.run(
                [sys.executable, app_file_path],
                env=env,
                stdout=stdout_file,
                stderr=stderr_file,
                text=True,
                check=True,
                timeout=timeout
            )

            stdout_file.seek(0)
            stdout_text = stdout_file.read()
            stderr_file.seek(0)
            stderr_text = stderr_file.read()

            # --- Console ---
            print(f"=== STDOUT ({folder_name}) ===")
            print(stdout_text)
            if stderr_text:
                print(f"=== STDERR ({folder_name}) ===")
                print(stderr_text)

            # --- Allure ---
            if stdout_text:
                allure.attach(
                    body=stdout_text,
                    name=f"Logs - Success ({folder_name})",
                    attachment_type=allure.attachment_type.TEXT
                )

        except subprocess.CalledProcessError as e:
            stdout_file.seek(0)
            stderr_file.seek(0)
            stdout_text = stdout_file.read()
            stderr_text = stderr_file.read()

            error_msg = f"STDOUT:\n{stdout_text}\n\nSTDERR:\n{stderr_text}"

            # --- Console ---
            print(f"=== SUBPROCESS FAILED (exit {e.returncode}) ({folder_name}) ===")
            print(error_msg, file=sys.stderr)  # send failures to stderr so CI highlights them

            # --- Allure ---
            allure.attach(
                body=error_msg,
                name=f"Logs - CRASHED ({folder_name})",
                attachment_type=allure.attachment_type.TEXT
            )
            raise e

        except subprocess.TimeoutExpired as e:
            stdout_file.seek(0)
            stderr_file.seek(0)
            stdout_text = stdout_file.read()
            stderr_text = stderr_file.read()

            hang_msg = (
                f"PROCESS HANGED! Captured before timeout:\n\n"
                f"STDOUT:\n{stdout_text}\n\nSTDERR:\n{stderr_text}"
            )

            # --- Console ---
            print(f"=== SUBPROCESS TIMED OUT after {timeout}s ({folder_name}) ===")
            print(hang_msg, file=sys.stderr)

            # --- Allure ---
            allure.attach(
                body=hang_msg,
                name=f"Logs - TIMEOUT HANG ({folder_name})",
                attachment_type=allure.attachment_type.TEXT
            )
            raise RuntimeError(f"Subprocess timed out after {timeout} on file: {csv_file_path}") from e



def _cell_color(pct: float, threshold: float = 95.0) -> str:
    """Green if above threshold, red otherwise."""
    return "#2ecc71" if pct >= threshold else "#e74c3c"


def attach_pdw_accuracy_report(
    emitter_1_name: str,
    emitter_2_name: str,
    pdw_emitter_1_correctly: float,
    pdw_emitter_2_correctly: float,
    pdw_unknown_emitter: float,
    threshold: float = 95.0,
) -> None:
    """Attach a formatted PDW clustering accuracy report to the current Allure step."""

    html = f"""
    <div style="font-family: Arial, Helvetica, sans-serif; padding: 8px;">
        <h3 style="margin-bottom: 12px;">PDW Clustering Accuracy Report</h3>
        <table border="1" cellspacing="0" cellpadding="8"
               style="border-collapse: collapse; width: 100%; font-size: 14px;">
            <thead>
                <tr style="background-color: #f2f2f2;">
                    <th align="left" style="padding: 6px;">Emitter</th>
                    <th align="right" style="padding: 6px;">% PDWs correctly clustered</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td style="padding: 6px;">{emitter_1_name}</td>
                    <td align="right" style="padding: 6px; font-weight: bold;
                        color: {_cell_color(pdw_emitter_1_correctly, threshold)};">
                        {pdw_emitter_1_correctly:.2f}%
                    </td>
                </tr>
                <tr>
                    <td style="padding: 6px;">{emitter_2_name}</td>
                    <td align="right" style="padding: 6px; font-weight: bold;
                        color: {_cell_color(pdw_emitter_2_correctly, threshold)};">
                        {pdw_emitter_2_correctly:.2f}%
                    </td>
                </tr>
                <tr>
                    <td style="padding: 6px;">Unknown_emitter</td>
                    <td align="right" style="padding: 6px; font-weight: bold;
                        color: {_cell_color(pdw_unknown_emitter, threshold)};">
                        {pdw_unknown_emitter:.2f}%
                    </td>
                </tr>
            </tbody>
        </table>
    </div>
    """

    with allure.step(
        f"Generating test status report for {emitter_1_name} and {emitter_2_name}"
    ):
        allure.attach(
            html,
            name="PDW Clustering Accuracy",
            attachment_type=allure.attachment_type.HTML,
        )


def attach_pdw_report2(emitters_data, threshold):
    """
    Attaches a dynamic PDW clustering report to Allure.

    :param emitters_data: A list of dictionaries, e.g.,
                          [{"name": "Emitter Alpha", "pct": 95.5}, {"name": "Emitter Beta", "pct": 82.1}]
    :param unknown_pdw_pct: Float percentage for the unknown emitter
    :param threshold: The threshold value used by your _cell_color helper
    """
    # 1. Build the dynamic rows for known emitters using a loop
    table_rows = ""
    for emitter in emitters_data:
        color = _cell_color(emitter['pct'], threshold)
        table_rows += f"""
        <tr>
            <td style="padding: 6px;">{emitter['name']}</td>
            <td align="right" style="padding: 6px; font-weight: bold; color: {color};">
                {emitter['pct']:.2f}%
            </td>
        </tr>
        """

    # 3. Insert all rows into the main HTML template
    html = f"""
    <div style="font-family: Arial, Helvetica, sans-serif; padding: 8px;">
        <h3 style="margin-bottom: 12px;">PDW Clustering Accuracy Report</h3>
        <table border="1" cellspacing="0" cellpadding="8"
               style="border-collapse: collapse; width: 100%; font-size: 14px;">
            <thead>
                <tr style="background-color: #f2f2f2;">
                    <th align="left" style="padding: 6px;">Emitter</th>
                    <th align="right" style="padding: 6px;">% PDWs correctly clustered</th>
                </tr>
            </thead>
            <tbody>
                {table_rows}
            </tbody>
        </table>
    </div>
    """

    # 4. Attach to the Allure report
    allure.attach(
        html,
        name="PDW Clustering Accuracy Report",
        attachment_type=allure.attachment_type.HTML
    )


def attach_emitter_statistics_report(emitters_data):
    """
    Attaches a formatted emitter average values statistics report to Allure.

    :param emitters_data: A dictionary containing emitter statistics, e.g.,
                          {
                              "Emitter 1": {'PW': 37.7510, 'PRI': 1190.7985, 'FREQ': Decimal('2749990288.00')},
                              "Emitter 2": {'PW': 45.2100, 'PRI': 1050.1234, 'FREQ': Decimal('1200000000.00')}
                          }
    """
    table_rows = ""

    # Loop through each emitter and its metrics
    for name, metrics in emitters_data.items():
        # Safeguard missing keys with defaults, convert Decimals/Floats safely
        pri_value = metrics.get('PRI')
        pri = float(pri_value) if pri_value is not None else 0.0

        pw_value = metrics.get('PW')
        pw = float(pw_value) if pw_value is not None else 0.0

        freq_value = metrics.get('FREQ')
        freq = float(freq_value) if freq_value is not None else 0.0

        # Build the dynamic HTML row for this emitter
        table_rows += f"""
        <tr>
            <td style="padding: 10px; border-bottom: 1px solid #ddd; font-weight: bold; color: #333;">{name}</td>
            <td align="right" style="padding: 10px; border-bottom: 1px solid #ddd;">{pri:,.2f}</td>
            <td align="right" style="padding: 10px; border-bottom: 1px solid #ddd;">{pw:,.2f}</td>
            <td align="right" style="padding: 10px; border-bottom: 1px solid #ddd; color: #2c3e50; font-weight: 500;">{freq:,.2f}</td>
        </tr>
        """

    # Create the complete HTML wrapper with modern styling
    html = f"""
    <div style="font-family: 'Segoe UI', Arial, sans-serif; padding: 15px; background-color: #fafafa; border-radius: 6px;">
        <h3 style="margin-top: 0; margin-bottom: 15px; color: #2c3e50; border-bottom: 2px solid #34495e; padding-bottom: 8px;">
            📡 Emitter Average Values Statistics Report
        </h3>
        <table cellspacing="0" cellpadding="0" style="border-collapse: collapse; width: 100%; font-size: 14px; background: white; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
            <thead>
                <tr style="background-color: #34495e; color: white;">
                    <th align="left" style="padding: 12px 10px;">Emitter Name</th>
                    <th align="right" style="padding: 12px 10px;">PRI (Pulse Repetition Interval)</th>
                    <th align="right" style="padding: 12px 10px;">PW (Pulse Width)</th>
                    <th align="right" style="padding: 12px 10px;">FREQ (Frequency)</th>
                </tr>
            </thead>
            <tbody>
                {table_rows}
            </tbody>
        </table>
    </div>
    """

    # Attach the built HTML template to Allure
    allure.attach(
        html,
        name="Emitter Average Values Statistics",
        attachment_type=allure.attachment_type.HTML
    )


@pytest.fixture(scope="session")
def artifactory_user() -> str:
    return os.getenv("ARTIFACTORY_USER")

@pytest.fixture(scope="session")
def artifactory_password() -> str:
    return os.getenv("ARTIFACTORY_TOKEN") or os.getenv("ARTIFACTORY_PASSWORD")


def upload_file_to_artifactory(local_file_path, artifactory_user, artifactory_password):
    """
    Uploads a local file to JFrog Artifactory using a PUT request.
    """
    # 1. Turn the string path into a Path object
    file_path = Path(local_file_path)

    # 2. Check if the file actually exists on your computer before uploading
    if not file_path.is_file():
        raise FileNotFoundError(f"The local file was not found: {file_path}")

    filename = file_path.name

    base_url = "https://artifactory.internal:443/artifactory/elint/ai_module_test_files/unknown_emitters"

    # 3. Create the full Artifactory URL where the file will live
    # Example: "https://my-artifactory.com"
    artifactory_url = f"{base_url.rstrip('/')}/{filename}"

    print(f"Uploading {filename} to {artifactory_url}...")

    # 4. Open the local file and send it to Artifactory
    with open(file_path, "rb") as file_data:
        response = requests.put(
            artifactory_url,
            auth=(artifactory_user, artifactory_password),
            data=file_data,
            verify=False  # Skips SSL verification just like your download script
        )

        # 5. Check if the upload was successful (throws an error if it failed)
        response.raise_for_status()

    print("Upload completed successfully!")
    return response


@pytest.fixture(scope="function")
def download_file_artifactory_known_emitters(request, csv_file_path: str, artifactory_password, artifactory_user: str):

    file_path = Path(csv_file_path)
    downloaded = not file_path.is_file()
    if downloaded:
        folder_path = file_path.parent
        filename = file_path.name

        artifactory_url = "https://artifactory.internal:443/artifactory/elint/ai_module_test_files/known_emitters/" + filename
        save_path = file_path

        os.makedirs(folder_path, exist_ok=True)

        with requests.get(artifactory_url, auth=(artifactory_user, artifactory_password), stream=True, verify=False) as response:
            response.raise_for_status()
            with open(save_path, "wb") as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)


    yield

    # Only remove the file if this fixture downloaded it; never delete a
    # CSV that was already on disk (local test-data copies).
    if downloaded and os.path.exists(file_path):
        os.remove(file_path)

@pytest.fixture(scope="function")
def download_file_artifactory_unknown_emitters(request, csv_file_path: str, artifactory_password, artifactory_user: str):

    file_path = Path(csv_file_path)
    downloaded = not file_path.is_file()
    if downloaded:
        folder_path = file_path.parent
        filename = file_path.name

        artifactory_url = "https://artifactory.internal:443/artifactory/elint/ai_module_test_files/unknown_emitters/" + filename
        save_path = file_path

        os.makedirs(folder_path, exist_ok=True)

        with requests.get(artifactory_url, auth=(artifactory_user, artifactory_password), stream=True, verify=False) as response:
            response.raise_for_status()
            with open(save_path, "wb") as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)


    yield

    # Only remove the file if this fixture downloaded it; never delete a
    # CSV that was already on disk (local test-data copies).
    if downloaded and os.path.exists(file_path):
        os.remove(file_path)


@pytest.fixture(scope="function")
def pdf_suite_report(request: pytest.FixtureRequest) -> Generator[Dict[str, Any], None, None]:
    """Manages tracking bounds using an explicit incremental counter context.

    Attaches the final report directly to the main body log steps of the last parameter loop.
    """
    session = request.session

    # Initialize in-memory counter repositories on the global pytest session objects
    if not hasattr(session, "_suite_reporters"):
        session._suite_reporters = {}
    if not hasattr(session, "_suite_run_counters"):
        session._suite_run_counters = {}

    base_test_name = request.node.originalname if request.node.originalname else request.node.name

    # Instantiate the data structures if this is the first parameter running
    if base_test_name not in session._suite_reporters:
        session._suite_reporters[base_test_name] = ParameterizedSuiteReporter(base_test_name)
    if base_test_name not in session._suite_run_counters:
        session._suite_run_counters[base_test_name] = 0

    reporter: ParameterizedSuiteReporter = session._suite_reporters[base_test_name]

    # Calculate exactly how many total parameters are expected to run for this specific test case definition
    all_collected_items = list(session.items) if hasattr(session, "items") else []
    same_test_items = [
        item for item in all_collected_items
        if (item.originalname if item.originalname else item.name) == base_test_name
    ]
    total_expected_runs = len(same_test_items)

    # Increment our tracking counter index for the current running thread loop
    session._suite_run_counters[base_test_name] += 1
    current_run_number = session._suite_run_counters[base_test_name]

    runtime_payload: Dict[str, Any] = {
        "is_last_test": (current_run_number == total_expected_runs),
        "confusion_matrix": None,
        "report_title": None,
        "report_description": None
    }

    # ---------------------------------------------------------------------
    # 1. CORE TEST EXECUTION PHASE FLOWS HERE
    # ---------------------------------------------------------------------
    yield runtime_payload

    # ---------------------------------------------------------------------
    # 2. RUNTIME COMPILATION & ACTIVE CONTEXT ATTACHMENT
    # ---------------------------------------------------------------------
    if runtime_payload["report_title"] is not None:
        reporter.custom_title = runtime_payload["report_title"]

    if runtime_payload["report_description"] is not None:
        reporter.custom_description = runtime_payload["report_description"]

    if runtime_payload["confusion_matrix"] is not None:
        reporter.attach_final_dataframe(runtime_payload["confusion_matrix"])

    # If this loop is determined to be the final expected item run, compile and dump right here
    if runtime_payload["is_last_test"]:
        pdf_path = reporter.generate_pdf()

        if pdf_path and os.path.exists(pdf_path):
            with open(pdf_path, "rb") as pdf_file:
                # Triggers the upload BEFORE the fixture terminates to drop it inside the Main Body log step
                allure.attach(
                    body=pdf_file.read(),
                    name=f"{reporter.custom_title} Summary (PDF)" if reporter.custom_title else "Visual Summary Report (PDF)",
                    attachment_type=allure.attachment_type.PDF
                )


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item: Item, call: CallInfo[Any]) -> Generator[None, None, None]:
    """Unified pytest hook that records structured telemetry to a central CSV tracking sheet,

    extracts assertpy's custom validation descriptions for the PDF summary generator,
    and captures environment runtime warnings safely.
    """
    # Execute the primary standard test validation phase wrapper
    outcome = yield
    rep: TestReport = outcome.get_result()

    if rep.when == "call":
        # Extract metadata identifiers for the current running argument execution context
        test_name: str = item.obj.__name__ if hasattr(item, "obj") and item.obj else item.name
        input_csv_path_raw = item.funcargs.get("csv_file_path", "N/A")
        input_csv_path: str = Path(input_csv_path_raw).name
        param_str: str = item.name.split('[')[-1].replace(']', '') if '[' in item.name else "None"
        status: str = rep.outcome

        # --- EXTRACT STANDARD TRACEBACK ---
        error_msg: str = ""
        if rep.failed and call.excinfo is not None:
            exc_value = call.excinfo.value
            error_msg = str(exc_value).strip().splitlines()[-1]
            exc_type_name = call.excinfo.typename
            if exc_type_name and not error_msg.startswith(exc_type_name):
                error_msg = f"{exc_type_name}: {error_msg}"

        # --- EXTRACT ASSERTPY DESCRIBED_AS DETAILS FOR THE PDF ---
        custom_description: str = "Assertion failed without descriptive details."
        if rep.failed:
            raw_text: str = ""
            if hasattr(rep, "longrepr") and rep.longrepr is not None:
                if hasattr(rep.longrepr, "reprcrash") and rep.longrepr.reprcrash is not None:
                    raw_text = str(rep.longrepr.reprcrash.message)
                else:
                    raw_text = str(rep.longrepr)
            else:
                raw_text = rep.longreprtext if rep.longreprtext else ""

            if raw_text:
                lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
                assertpy_match = None
                for line in lines:
                    # Target bracket logs formatted by assertpy: "[Your custom error string text] Expected..."
                    if line.startswith("[") and "]" in line:
                        end_bracket_idx = line.find("]")
                        assertpy_match = line[1:end_bracket_idx].strip()
                        break
                    elif "AssertionError: [" in line and "]" in line:
                        start_idx = line.find("[") + 1
                        end_idx = line.find("]")
                        assertpy_match = line[start_idx:end_idx].strip()
                        break

                if assertpy_match:
                    custom_description = assertpy_match
                else:
                    custom_description = error_msg if error_msg else (lines[-1] if lines else "Unknown Exception")

            # Attach the cleaned string back to the item context so the teardown manager can harvest it
            item._custom_failed_text = custom_description

        # --- ROUTE RUN DATA DIRECTLY BACK INTO ACTIVE SESSION REPORTERS ---
        base_test_name = item.originalname if item.originalname else item.name
        session = item.session
        if hasattr(session, "_suite_reporters") and base_test_name in session._suite_reporters:
            reporter = session._suite_reporters[base_test_name]
            passed_bool = (status == "passed")

            # Combine the hard failure (if any) with the non-blocking DB warnings
            # so both show up in the PDF's Functional Error Classification Summary.
            error_details = getattr(item, "_custom_failed_text", None)
            db_warning = item.stash.get("custom_db_warning", None) if hasattr(item, "stash") else None
            # NEW: also harvest the "total number of emitters" warning into its own key
            db_warning_total_count = (
                item.stash.get("custom_db_warning_total_count", None) if hasattr(item, "stash") else None
            )

            combined: List[str] = []
            if error_details:
                combined.append(error_details)
            if db_warning:
                combined.append(db_warning)
            if db_warning_total_count:  # NEW
                combined.append(db_warning_total_count)  # NEW

            reporter.record_run(passed=passed_bool, error_details=combined if combined else None)

        # --- EXTRACT TEST WARNINGS (CRASH-PROOF) ---
        warnings_found = []
        if hasattr(item, "stash") and "custom_db_warning" in item.stash:
            warnings_found.append(item.stash["custom_db_warning"])
        if hasattr(item, "stash") and hasattr(item.stash, "_storage"):
            for key in item.stash._storage.keys():
                if "Warning" in type(key).__name__ or "warning" in str(key).lower():
                    warnings_list = item.stash.get(key, [])
                    if isinstance(warnings_list, list):
                        for w in warnings_list:
                            msg = getattr(w, "message", w)
                            warnings_found.append(str(msg))
                    break
        if not warnings_found and hasattr(item, "_warnings"):
            old_warnings = getattr(item, "_warnings", [])
            if isinstance(old_warnings, list):
                for w in old_warnings:
                    msg = getattr(w, "message", w)
                    warnings_found.append(str(msg))

        warnings_str: str = "; ".join(warnings_found) if warnings_found else "None"

        # --- WRITE DATA MATRIX TO SUMMARY_CSV FILE ---
        file_is_empty = not os.path.exists(SUMMARY_CSV) or os.stat(SUMMARY_CSV).st_size == 0
        with open(SUMMARY_CSV, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if file_is_empty:
                writer.writerow(
                    ["Test Name", "Input CSV Path", "Parameter Group", "Status", "Error Message", "Warnings"])
            writer.writerow([test_name, input_csv_path, param_str, status, error_msg.strip(), warnings_str.strip()])

        # --- ALLURE ENGINE MATRIX SHEET ATTACHMENT ---
        if os.path.exists(SUMMARY_CSV) and os.stat(SUMMARY_CSV).st_size > 0:
            with open(SUMMARY_CSV, mode='r', encoding='utf-8') as f:
                csv_content = f.read()
            allure.attach(body=csv_content, name="Final Test Execution Status Summary",
                          attachment_type=allure.attachment_type.CSV)


def pytest_sessionfinish(session: pytest.Session, exitstatus: Any) -> None:
    """Fires at absolute suite teardown finalization bounds to make the PDF

    blazingly visible on the main Allure Home/Overview dashboard page.
    """
    if not hasattr(session, "_suite_reporters"):
        return

    # Extract the allure command-line results path target (defaults to "allure-results")
    allure_dir = session.config.getoption("--alluredir", "allure-results")

    for base_name, reporter in session._suite_reporters.items():
        if reporter.total_runs > 0:
            # 1. Compile the clean summary PDF file
            pdf_path = reporter.generate_pdf()

            if pdf_path and allure_dir and os.path.exists(pdf_path) and os.path.isdir(allure_dir):
                # 2. Copy the PDF file straight into the Allure raw results folder
                destination_pdf = os.path.join(allure_dir, os.path.basename(pdf_path))
                import shutil
                shutil.copy(pdf_path, destination_pdf)

                # 3. Create/Append to the properties file that Allure reads to render the HOME DASHBOARD PANEL
                properties_path = os.path.join(allure_dir, "environment.properties")

                # Generate clean dynamic status metrics for the main screen dashboard panel view
                status_symbol = "🔴 FAILED" if reporter.failed > 0 else "🟢 PASSED"

                with open(properties_path, "a", encoding="utf-8") as env_file:
                    env_file.write(f"\n\n# === {reporter.custom_title or base_name} ===\n")
                    env_file.write(f"Global_Execution_Status = {status_symbol}\n")
                    env_file.write(f"Total_CSV_Scenarios_Audited = {reporter.total_runs}\n")
                    env_file.write(f"Passed_Scenarios = {reporter.passed}\n")
                    env_file.write(f"Failed_Scenarios = {reporter.failed}\n")
                    env_file.write(f"Generated_PDF_Summary_File = {os.path.basename(pdf_path)}\n")