from pathlib import Path
from time import sleep
import sys
import allure
import pytest
import os
import allure
import pytest
from assertpy import assert_that, soft_assertions

from helpers.db import ClickHouseManager
from helpers.db.PostgreSQLManager import DBUtils
from tests.conftest import get_test_path, attach_pdw_accuracy_report, attach_pdw_report2, attach_emitter_statistics_report, run_ai_model
from helpers.misc import read_test_parameters
from datetime import datetime, timezone
import os
import runpy
import threading
from unittest.mock import patch
import pytest
import pandas as pd
from helpers.emitter_specifications.unknown_emitters import *
from tests.conftest import upload_file_to_artifactory
from helpers.types import RadarMapper
import pytest_check as check


@allure.suite("Emitter detection known emitters")
class TestPDWGeneratorFiles:


    @allure.sub_suite("Duration is incremented for known emitters")
    @allure.title("Test duration is incremented for detected known emitters")
    @allure.description("""
                Validate that duration_ms field from signals table is incremented after signal is detected
                
              Test Steps:
                   1. Extract features from CSV file
                   2. Connect to AI model via ONNX interface
                   3. Upload features to model
                   4. Run Model 
                   4. Write results to ELINT Intelligence db
                   7. Query DB and assert desired signal was detected 
                   9. Run AI model again
                   10 Verify duration field was increased
        """)
    @pytest.mark.parametrize(
        [
            "csv_file_path",
            "emitter_id_1",
            "emitter_1_id_csv",
            "emitter_1_name",
        ],
        read_test_parameters(
            get_test_path() / "test_data",
            "known_emitters_duration.csv",
        ),
        indirect=["csv_file_path"]
    )
    @pytest.mark.skip(reason="Test is not relevant for the moment")
    def test_known_emitters_emitters_duration(
            self,
            csv_file_path: str,
            download_file_artifactory_known_emitters: None,
            create_csv_input_table: None,
            truncate_activities_db_tables: None,
            populate_csv_input_table: None,
            truncate_intelligence_db_tables: None,
            populate_signatures_table: None,
            delete_known_clusters_centers: None,
            delete_emitter_signal_map: None,
            run_ai_model_fixture_main_lobe: None,
            intelligence_db_connector: DBUtils,
            emitter_id_1: str,
            emitter_1_id_csv: str,
            emitter_1_name: str,
    ) -> None:

        with allure.step(f"Assert {emitter_1_name} was detected by AI model"):
            query = f"select count(*) from emitters where name = '{emitter_id_1}'"
            count_query_result = intelligence_db_connector.fetch_scalar(query)
            assert_that(count_query_result, description=query).is_greater_than(0)

            duration_ms_value1 = intelligence_db_connector.fetch_scalar(f"select s.duration_ms from signals s join emitters e on s.emitter_id = e.id and e.name = '{emitter_id_1}'")

        with allure.step(f"Re-run AI model and verify duration was increased"):

            run_ai_model(csv_file_path)

            duration_ms_value2 = intelligence_db_connector.fetch_scalar(f"select s.duration_ms from signals s join emitters e on s.emitter_id = e.id and e.name = '{emitter_id_1}'")

            assert_that(duration_ms_value2).is_greater_than(duration_ms_value1)


    @allure.sub_suite("Known emitter detection main lobe")
    @allure.title("Test AI model is able to detect known emitters")
    @allure.description("""
            Validate that AI classifier is able to classify known radars from data
            generated using PDW generator. Model will read a CSV file containing
            a pair of each of the following radar types:
            - PSR1
            - PSR2
            - PSR3
            - PSR4
            - PSR5 mod 1 and mod 2
            - MF-03
            - MF-04
            - MF-05
            - MF-06

          Test Steps:
               1. Extract features from CSV file
               2. Connect to AI model via ONNX interface
               3. Upload features to model
               4. Run Model 
               4. Write results to ELINT Intelligence db
               7. Query Emitters table and assert each emitter was detected by AI model
               9. 
    """)
    @pytest.mark.parametrize(
        [
            "csv_file_path",
            "emitter_id_1",
            "emitter_id_2",
            "emitter_1_id_csv",
            "emitter_2_id_csv",
            "emitter_1_name",
            "emitter_2_name",
            "number_emitters",
            "std_no_emitters"
        ],
        read_test_parameters(
            get_test_path() / "test_data",
            "known_emitters_db_emitter_level.csv",
        ),
        indirect=["csv_file_path"]
    )
    @pytest.mark.skip(reason="Testing only with source OMNI")
    def test_known_emitters_detection(
            self,
            csv_file_path: str,
            download_file_artifactory_known_emitters: None,
            create_csv_input_table: None,
            truncate_activities_db_tables: None,
            populate_csv_input_table: None,
            truncate_intelligence_db_tables: None,
            populate_signatures_table: None,
            delete_known_clusters_centers: None,
            delete_emitter_signal_map: None,
            run_ai_model_fixture_main_lobe: None,
            intelligence_db_connector: DBUtils,
            activities_db_connector: ClickHouseManager,
            emitter_id_1: str,
            emitter_id_2: str,
            emitter_1_id_csv: str,
            emitter_2_id_csv: str,
            emitter_1_name: str,
            emitter_2_name: str,
            number_emitters: int,
            std_no_emitters: int,
            shared_stats_dictionary,
            shared_confusion_matrix,
            request
    ) -> None:

        emitter_stats = []
        stats_dict = shared_stats_dictionary["dictionary_stats"]

        df = shared_confusion_matrix["confusion_matrix"]

        print(f"\n[TEST PROGRESS] Running iteration on: {csv_file_path}", flush=True)

        with allure.step(f"Query total number of PDWs in main lobe from input CSV file and attach to report"):

            query = "SELECT emitter_id, count(emitter_id)  FROM signals.csv_input where sidelobe_blanking_flag_as_int8 = 1 group by emitter_id order by emitter_id asc"
            activities_db_connector.fetch_all_results_as_df(
                query=query,
                attach_to_allure=True,
                attachment_name=f"Number of PDWs in main lobe from input CSV file"
            )

        with allure.step(f"Assert {emitter_1_name} was detected by AI model"):
            query = f"select count(*) from emitters where name = '{emitter_id_1}'"
            count_query_result = intelligence_db_connector.fetch_scalar(query)
            assert_that(count_query_result, description=query).is_greater_than(0)

        with allure.step(f"Assert {emitter_2_name} was detected by AI model"):

            query = f"select count(*) from emitters where name = '{emitter_id_2}'"
            count_query_result = intelligence_db_connector.fetch_scalar(query)
            assert_that(count_query_result, description=query).is_greater_than(0)

        with allure.step(f"Assert that each cluster was assigned to a Signal"):

            query_clusters_not_assigned_signal = "select count(*) from clusters where signal_id is null"
            count_query_result = intelligence_db_connector.fetch_scalar(query_clusters_not_assigned_signal)
            assert_that(count_query_result).is_equal_to(0)

        with allure.step(f"Get all PDW labeled as {emitter_1_name} and {emitter_2_name} from input CSV file"):

            query_pdw_emitter_1_ids_count = f"""
                   SELECT
                       count(*)
                       FROM signals.csv_input
                       WHERE emitter_id = {emitter_1_id_csv}
                       AND sidelobe_blanking_flag_as_int8 = 1
               """
            no_emitter_1_pdws_id = activities_db_connector.select_column(
                query_pdw_emitter_1_ids_count
            )

            if f"{emitter_1_name}_total_input" in stats_dict:
                stats_dict[f"{emitter_1_name}_total_input"] += no_emitter_1_pdws_id[0]
            else:
                stats_dict[f"{emitter_1_name}_total_input"] = no_emitter_1_pdws_id[0]

            query_pdw_emitter_2_ids_count = f"""
                              SELECT
                                  count(*)
                                  FROM signals.csv_input
                                  WHERE emitter_id = {emitter_2_id_csv}
                                  AND sidelobe_blanking_flag_as_int8 = 1
                          """

            no_emitter_2_pdws_id = activities_db_connector.select_column(
                query_pdw_emitter_2_ids_count,
            )

            if f"{emitter_2_name}_total_input" in stats_dict:
                stats_dict[f"{emitter_2_name}_total_input"] += no_emitter_2_pdws_id[0]
            else:
                stats_dict[f"{emitter_2_name}_total_input"] = no_emitter_2_pdws_id[0]

        with allure.step(f"Get number of PDWs correctly labeled as {emitter_1_name}"):

            query_cluster_id_emitter_1 = f"""
                    select c.id
                    from signals s join emitters e on s.emitter_id = e.id 
                    join clusters c on c.signal_id = s.id
                    where e.name = '{emitter_id_1}'
            """

            clusters_emitter_ids = intelligence_db_connector.fetch_column(
                query_cluster_id_emitter_1
            )

            clusters_emitter_ids_as_string = ", ".join(map(str, clusters_emitter_ids)) or "NULL"
            query_pdws_count_emitter_1 = f"""
                select count(*) from signals.csv_input where id in (               
                SELECT     
                    arrayJoin(extracted_ids)  from (
                    select 
                    arrayMap(
                    x -> toUInt64(extract(toString(x), '([0-9]+)$')), 
                    activity_ids
                ) AS extracted_ids
                FROM signals.clusters
                    WHERE cluster_id IN ({clusters_emitter_ids_as_string}))) and  emitter_id = {emitter_1_id_csv}
            """

            no_pdws_emitter_1_correctly_labeled = activities_db_connector.select_query(query_pdws_count_emitter_1)

            if f"{emitter_1_name}_{emitter_1_name}" in stats_dict:
                stats_dict[f"{emitter_1_name}_{emitter_1_name}"] += no_pdws_emitter_1_correctly_labeled[0][0]
            else:
                stats_dict[f"{emitter_1_name}_{emitter_1_name}"] = no_pdws_emitter_1_correctly_labeled[0][0]

            emitter_1_labeled_correctly = no_pdws_emitter_1_correctly_labeled[0][0] / no_emitter_1_pdws_id[0] * 100

            emitter_stats.append({"name": f"{emitter_1_name} labeled correctly", "pct": emitter_1_labeled_correctly})

        with allure.step(f"Get number of PDWs incorrectly labeled as {emitter_1_name} that were actually {emitter_2_name}"):

            query_cluster_id_emitter_1 = f"""
                    select c.id
                    from signals s join emitters e on s.emitter_id = e.id 
                    join clusters c on c.signal_id = s.id
                    where e.name = '{emitter_id_1}'
            """

            clusters_emitter_1_ids = intelligence_db_connector.fetch_column(
                query_cluster_id_emitter_1
            )

            clusters_emitter_1_ids_as_string = ", ".join(map(str, clusters_emitter_1_ids)) or "NULL"
            query_pdws_count_emitter_1_incorrect_labeled = f"""
                select count(*) from signals.csv_input where id in (               
                SELECT     
                    arrayJoin(extracted_ids)  from (
                    select 
                    arrayMap(
                    x -> toUInt64(extract(toString(x), '([0-9]+)$')), 
                    activity_ids
                ) AS extracted_ids
                FROM signals.clusters

                    WHERE cluster_id IN ({clusters_emitter_1_ids_as_string}))) and  emitter_id = {emitter_2_id_csv}
            """

            no_pdws_emitter_1_labeled_incorrectly = activities_db_connector.select_query(query_pdws_count_emitter_1_incorrect_labeled)

            emitter_1_labeled_incorrectly = no_pdws_emitter_1_labeled_incorrectly[0][0] / no_emitter_1_pdws_id[0] * 100

            emitter_stats.append({"name": f"{emitter_2_name} mislabeled as {emitter_1_name}", "pct": emitter_1_labeled_incorrectly})


            if f"{emitter_2_name}_{emitter_1_name}" in stats_dict:
                stats_dict[f"{emitter_2_name}_{emitter_1_name}"] += no_pdws_emitter_1_labeled_incorrectly[0][0]
            else:
                stats_dict[f"{emitter_2_name}_{emitter_1_name}"]= no_pdws_emitter_1_labeled_incorrectly[0][0]

        with allure.step(f"Get number of PDWs incorrectly labeled as unknown_ai that were actually {emitter_1_name}"):

            query_cluster_id_unknown_ai = f"""
                    select c.id
                    from signals s join emitters e on s.emitter_id = e.id 
                    join clusters c on c.signal_id = s.id
                    where e.name = 'unknown_ai'
            """

            cluster_id_unknown_ai = intelligence_db_connector.fetch_column(
                query_cluster_id_unknown_ai
            )

            cluster_id_unknown_ai_as_string = ", ".join(map(str, cluster_id_unknown_ai)) or "NULL"
            query_pdws_count_emitter_1_incorrect_labeled_unknown_ai = f"""
                select count(*) from signals.csv_input where id in (               
                SELECT     
                    arrayJoin(extracted_ids)  from (
                    select 
                    arrayMap(
                    x -> toUInt64(extract(toString(x), '([0-9]+)$')), 
                    activity_ids
                ) AS extracted_ids
                FROM signals.clusters

                    WHERE cluster_id IN ({cluster_id_unknown_ai_as_string}))) and  emitter_id = {emitter_1_id_csv}
            """

            no_pdws_emitter_1_labeled_incorrectly_unknown_ai = activities_db_connector.select_query(query_pdws_count_emitter_1_incorrect_labeled_unknown_ai)

            emitter_1_labeled_incorrectly_unknown_ai = no_pdws_emitter_1_labeled_incorrectly_unknown_ai[0][0] / no_emitter_1_pdws_id[0] * 100

            emitter_stats.append({"name": f"{emitter_1_name} mislabeled as unknown_ai", "pct": emitter_1_labeled_incorrectly_unknown_ai})


            if f"{emitter_1_name}_unknownai" in stats_dict:
                stats_dict[f"{emitter_1_name}_unknownai"] += no_pdws_emitter_1_labeled_incorrectly_unknown_ai[0][0]
            else:
                stats_dict[f"{emitter_1_name}_unknownai"]= no_pdws_emitter_1_labeled_incorrectly_unknown_ai[0][0]

        with allure.step(f"Get number of PDWs labeled as {emitter_1_name} that were unclustered"):

            query_emitter_1_pdws_unclustered = f"""
                SELECT count(*) FROM signals.csv_input
                WHERE id NOT IN (
                SELECT arrayJoin(extracted_ids)
                FROM (
                    SELECT
                        arrayMap(
                            x -> toUInt64(extract(toString(x), '([0-9]+)$')),
                            activity_ids
                    ) AS extracted_ids
                FROM signals.clusters
                    )
                )
                AND emitter_id = {emitter_1_id_csv} AND sidelobe_blanking_flag_as_int8 = 1
            """
            no_pdws_emitter_1_unclustered = activities_db_connector.select_query(query_emitter_1_pdws_unclustered)

            emitter_1_unclustered = no_pdws_emitter_1_unclustered[0][0] / no_emitter_1_pdws_id[0] * 100

            emitter_stats.append({"name": f"{emitter_1_name} unclustered", "pct": emitter_1_unclustered})

            if f"{emitter_1_name}_unclustered" in stats_dict:
                stats_dict[f"{emitter_1_name}_unclustered"] += no_pdws_emitter_1_unclustered[0][0]
            else:
                stats_dict[f"{emitter_1_name}_unclustered"]= no_pdws_emitter_1_unclustered[0][0]

        with allure.step(f"Get number of PDWs correctly labeled as {emitter_2_name}"):

            query_cluster_id_emitter_2 = f"""
                    select c.id
                    from signals s join emitters e on s.emitter_id = e.id 
                    join clusters c on c.signal_id = s.id
                    where e.name = '{emitter_id_2}'
            """

            clusters_emitter_ids = intelligence_db_connector.fetch_column(
                query_cluster_id_emitter_2
            )

            clusters_emitter_ids_as_string = ", ".join(map(str, clusters_emitter_ids)) or "NULL"
            query_pdws_count_emitter_2 = f"""
                select count(*) from signals.csv_input where id in (               
                SELECT     
                    arrayJoin(extracted_ids)  from (
                    select 
                    arrayMap(
                    x -> toUInt64(extract(toString(x), '([0-9]+)$')), 
                    activity_ids
                ) AS extracted_ids
                FROM signals.clusters

                    WHERE cluster_id IN ({clusters_emitter_ids_as_string}))) and  emitter_id = {emitter_2_id_csv}
            """

            no_pdws_emitter_2_labeled = activities_db_connector.select_query(query_pdws_count_emitter_2)

            emitter_2_labeled_correctly = no_pdws_emitter_2_labeled[0][0] / no_emitter_2_pdws_id[0] * 100

            emitter_stats.append({"name": f"{emitter_2_name} labeled correctly", "pct": emitter_2_labeled_correctly})

            if f"{emitter_2_name}_{emitter_2_name}" in stats_dict:
                stats_dict[f"{emitter_2_name}_{emitter_2_name}"] += no_pdws_emitter_2_labeled[0][0]
            else:
                stats_dict[f"{emitter_2_name}_{emitter_2_name}"] = no_pdws_emitter_2_labeled[0][0]

        with allure.step(f"Get number of PDWs incorrectly labeled as {emitter_2_name} that were actually {emitter_1_name}"):

            query_cluster_id_emitter_2 = f"""
                    select c.id
                    from signals s join emitters e on s.emitter_id = e.id 
                    join clusters c on c.signal_id = s.id
                    where e.name = '{emitter_id_2}'
            """

            clusters_emitter_ids = intelligence_db_connector.fetch_column(
                query_cluster_id_emitter_2,
            )

            clusters_emitter_ids_as_string = ", ".join(map(str, clusters_emitter_ids)) or "NULL"

            query_pdws_count_emitter_2_incorrect_labeled = f"""
                select count(*) from signals.csv_input where id in (               
                SELECT     
                    arrayJoin(extracted_ids)  from (
                    select 
                    arrayMap(
                    x -> toUInt64(extract(toString(x), '([0-9]+)$')), 
                    activity_ids
                ) AS extracted_ids
                FROM signals.clusters

                    WHERE cluster_id IN ({clusters_emitter_ids_as_string}))) and  emitter_id = {emitter_1_id_csv}
            """

            no_pdws_emitter_2_labeled_incorrectly = activities_db_connector.select_query(query_pdws_count_emitter_2_incorrect_labeled)

            if f"{emitter_1_name}_{emitter_2_name}" in stats_dict:
                stats_dict[f"{emitter_1_name}_{emitter_2_name}"] += no_pdws_emitter_2_labeled_incorrectly[0][0]
            else:
                stats_dict[f"{emitter_1_name}_{emitter_2_name}"] = no_pdws_emitter_2_labeled_incorrectly[0][0]

            emitter_2_labeled_incorrectly = no_pdws_emitter_2_labeled_incorrectly[0][0] / no_emitter_2_pdws_id[0] * 100

            emitter_stats.append({"name": f"{emitter_1_name} mislabeled as {emitter_2_name}", "pct": emitter_2_labeled_incorrectly})

        with allure.step(f"Get number of PDWs incorrectly labeled as unknown_ai that were actually {emitter_2_name}"):

            query_cluster_id_unknown_ai = f"""
                    select c.id
                    from signals s join emitters e on s.emitter_id = e.id 
                    join clusters c on c.signal_id = s.id
                    where e.name = 'unknown_ai'
            """

            cluster_id_unknown_ai = intelligence_db_connector.fetch_column(
                query_cluster_id_unknown_ai
            )

            cluster_id_unknown_ai_as_string = ", ".join(map(str, cluster_id_unknown_ai)) or "NULL"
            query_pdws_count_emitter_2_incorrect_labeled_unknown_ai = f"""
                select count(*) from signals.csv_input where id in (               
                SELECT     
                    arrayJoin(extracted_ids)  from (
                    select 
                    arrayMap(
                    x -> toUInt64(extract(toString(x), '([0-9]+)$')), 
                    activity_ids
                ) AS extracted_ids
                FROM signals.clusters

                    WHERE cluster_id IN ({cluster_id_unknown_ai_as_string}))) and  emitter_id = {emitter_2_id_csv}
            """

            no_pdws_emitter_2_labeled_incorrectly_unknown_ai = activities_db_connector.select_query(query_pdws_count_emitter_2_incorrect_labeled_unknown_ai)

            emitter_2_labeled_incorrectly_unknown_ai = no_pdws_emitter_2_labeled_incorrectly_unknown_ai[0][0] / no_emitter_2_pdws_id[0] * 100

            emitter_stats.append({"name": f"{emitter_2_name} mislabeled as unknown_ai", "pct": emitter_2_labeled_incorrectly_unknown_ai})

            if f"{emitter_2_name}_unknownai" in stats_dict:
                stats_dict[f"{emitter_2_name}_unknownai"] += no_pdws_emitter_2_labeled_incorrectly_unknown_ai[0][0]
            else:
                stats_dict[f"{emitter_2_name}_unknownai"]= no_pdws_emitter_2_labeled_incorrectly_unknown_ai[0][0]

        with allure.step(f"Get number of PDWs labeled as {emitter_2_name} that were unclustered"):

            query_emitter_2_pdws_unclustered = f"""
                SELECT count(*) FROM signals.csv_input
                WHERE id NOT IN (
                SELECT arrayJoin(extracted_ids)
                FROM (
                    SELECT
                        arrayMap(
                            x -> toUInt64(extract(toString(x), '([0-9]+)$')),
                            activity_ids
                    ) AS extracted_ids
                FROM signals.clusters
                    )
                )
                AND emitter_id = {emitter_2_id_csv} AND sidelobe_blanking_flag_as_int8 = 1
            """
            no_pdws_emitter_2_unclustered = activities_db_connector.select_query(query_emitter_2_pdws_unclustered)

            emitter_2_unclustered = no_pdws_emitter_2_unclustered[0][0] / no_emitter_2_pdws_id[0] * 100
            emitter_stats.append({"name": f"{emitter_2_name} unclustered", "pct": emitter_2_unclustered})

            if f"{emitter_2_name}_unclustered" in stats_dict:
                stats_dict[f"{emitter_2_name}_unclustered"] += no_pdws_emitter_2_unclustered[0][0]
            else:
                stats_dict[f"{emitter_2_name}_unclustered"] = no_pdws_emitter_2_unclustered[0][0]

        with allure.step(f"Generating test status report for {emitter_1_name} and {emitter_2_name}"):

            attach_pdw_report2(emitters_data=emitter_stats, threshold=90.0)

            with allure.step(
                    f"Verify if system detected additional emitter except {emitter_1_name} or {emitter_2_name}"
                    f""):

                query_no_other_emitters = f"select count(*) from emitters where name not in ('{emitter_id_1}', '{emitter_id_2}', 'unknown_ai')"

                no_other_emitters_detected = intelligence_db_connector.fetch_scalar(query_no_other_emitters)

                if (no_other_emitters_detected > 0):
                    with allure.step(f"⚠️ WARNING: System has detected other emitters except {emitter_1_name} or {emitter_2_name} either from known or unknown controlled emitters"):
                        other_emitters_name_query = f"select name from emitters where name not in ('{emitter_id_1}', '{emitter_id_2}', 'unknown_ai')"

                        other_emitters_name = intelligence_db_connector.fetch_column(other_emitters_name_query)
                        allure.attach(
                            body=' '.join(other_emitters_name),
                            name=f"Other emitters detected except {emitter_1_name} or {emitter_2_name} or 'unknown_ai'",
                            attachment_type=allure.attachment_type.TEXT
                        )

                        for emitter in other_emitters_name:
                            emitter_name = RadarMapper(emitter).name if emitter in RadarMapper else emitter

                            custom_msg = (f"Warning: System detected other known emitters except {emitter_1_name}, {emitter_2_name} or 'unknown_ai' "
                                          f"System detected {emitter_name}")
                            request.node.stash["custom_db_warning"] = custom_msg
                            allure.attach(f"Warning, system detected {emitter_name}, except expected {emitter_1_name} or {emitter_2_name}",
                                          name="Warning details",
                                          attachment_type=allure.attachment_type.TEXT)


                            with allure.step(
                                    f"Calculate the percentage of PDWs marked as {emitter_name} that were labeled as {emitter_1_name}"):

                                query_cluster_id_emitter_other = f"""
                                               select c.id
                                               from signals s join emitters e on s.emitter_id = e.id 
                                               join clusters c on c.signal_id = s.id
                                               where e.name = '{emitter}'
                                       """

                                clusters_emitter_ids = intelligence_db_connector.fetch_column(
                                    query_cluster_id_emitter_other,
                                )

                                clusters_emitter_ids_as_string = ", ".join(map(str, clusters_emitter_ids)) or "NULL"
                                query_pdws_other_emitters_labeled_emitter_1_count = f"""
                                           select count(*) from signals.csv_input where id in (               
                                           SELECT     
                                               arrayJoin(extracted_ids)  from (
                                               select 
                                               arrayMap(
                                               x -> toUInt64(extract(toString(x), '([0-9]+)$')), 
                                               activity_ids
                                           ) AS extracted_ids
                                           FROM signals.clusters
                                               WHERE cluster_id IN ({clusters_emitter_ids_as_string}))) and  emitter_id = {emitter_1_id_csv}
                                       """

                                no_pdws_other_emitter_labeled_emitter_1 = activities_db_connector.select_query(
                                    query_pdws_other_emitters_labeled_emitter_1_count)

                                if f"{emitter_1_name}_{emitter_name}" in stats_dict:
                                    stats_dict[f"{emitter_1_name}_{emitter_name}"] += \
                                    no_pdws_other_emitter_labeled_emitter_1[0][0]
                                else:
                                    stats_dict[f"{emitter_1_name}_{emitter_name}"] = \
                                    no_pdws_other_emitter_labeled_emitter_1[0][0]

                                other_emitters_detected_percentage = no_pdws_other_emitter_labeled_emitter_1[0][0] / \
                                                                     no_emitter_1_pdws_id[0] * 100

                                emitter_stats.append({
                                                         "name": f"Other radar with id {emitter_name} was detected, Percentage of PDWs marked as {emitter_name} that were labeled as {emitter_1_name}",
                                                         "pct": other_emitters_detected_percentage})

                            with allure.step(
                                    f"Calculate the percentage of PDWs marked as {emitter_name} that were labeled as {emitter_2_name}"):

                                query_cluster_id_emitter_other = f"""
                                               select c.id
                                               from signals s join emitters e on s.emitter_id = e.id 
                                               join clusters c on c.signal_id = s.id
                                               where e.name = '{emitter}'
                                       """

                                clusters_emitter_ids = intelligence_db_connector.fetch_column(
                                    query_cluster_id_emitter_other,
                                )

                                clusters_emitter_ids_as_string = ", ".join(map(str, clusters_emitter_ids)) or "NULL"
                                query_pdws_other_emitters__labeled_emitter_1_count = f"""
                                           select count(*) from signals.csv_input where id in (               
                                           SELECT     
                                               arrayJoin(extracted_ids)  from (
                                               select 
                                               arrayMap(
                                               x -> toUInt64(extract(toString(x), '([0-9]+)$')), 
                                               activity_ids
                                           ) AS extracted_ids
                                           FROM signals.clusters
                                               WHERE cluster_id IN ({clusters_emitter_ids_as_string}))) and  emitter_id = {emitter_2_id_csv}
                                       """

                                no_pdws_other_emitter_labeled_emitter_2 = activities_db_connector.select_query(
                                    query_pdws_other_emitters__labeled_emitter_1_count)

                                if f"{emitter_2_name}_{emitter_name}" in stats_dict:
                                    stats_dict[f"{emitter_2_name}_{emitter_name}"] += \
                                    no_pdws_other_emitter_labeled_emitter_2[0][0]
                                else:
                                    stats_dict[f"{emitter_2_name}_{emitter_name}"] = \
                                    no_pdws_other_emitter_labeled_emitter_2[0][0]

                                other_emitters_detected_percentage = no_pdws_other_emitter_labeled_emitter_2[0][0] / \
                                                                     no_emitter_2_pdws_id[0] * 100
                                emitter_stats.append({
                                    "name": f"Other radar with id {emitter_name} was detected, Percentage of PDWs marked as {emitter_name} that were labeled as {emitter_2_name}",
                                    "pct": other_emitters_detected_percentage})

                            with allure.step(
                                    f"Attaching PDWs that generated AI model identification of a known emitter"):
                                query_pdws_other_emitters_identified = f"""
                                                              select * from signals.csv_input where id in (               
                                                              SELECT     
                                                                  arrayJoin(extracted_ids)  from (
                                                                  select 
                                                                  arrayMap(
                                                                  x -> toUInt64(extract(toString(x), '([0-9]+)$')), 
                                                                  activity_ids
                                                              ) AS extracted_ids
                                                              FROM signals.clusters
                                                                  WHERE cluster_id IN ({clusters_emitter_ids_as_string}))) 
                                                          """

                                activities_db_connector.fetch_all_results_as_df(
                                    query=query_pdws_other_emitters_identified,
                                    attach_to_allure=True,
                                    attachment_name=f"Other emitters value for {emitter_name}"
                                )

        with allure.step(f"Attaching updated confusion matrix for {emitter_1_name} and {emitter_2_name}"):

            for key, value in stats_dict.items():
                df.loc[f"{key.split("_")[0]}", f"{key.split("_")[1]}"] = value / stats_dict[f"{key.split("_")[0]}_total_input"] * 100.00

            allure.attach(
                df.to_markdown(),
                name="Undated confusion matrix",
                attachment_type=allure.attachment_type.TEXT
            )

        with allure.step(f"Get total number of detected emitters"):

            query_detected_emitters = "select count(*) from emitters"
            count_detected_emitters = intelligence_db_connector.fetch_scalar(query_detected_emitters)
            assert_that(count_detected_emitters) \
                .described_as("Detected emitter count is outside the expected standard deviation range. "
                             f"Expected {number_emitters} with  standard deviation of +- {std_no_emitters} emitters "
                             f"got only {count_detected_emitters}") \
                .is_between(number_emitters - std_no_emitters, number_emitters + std_no_emitters)

    @allure.sub_suite("Unknown emitter detection")
    @allure.title("Test AI model is able to detect unknown emitters")
    @allure.description("""
               Validate that AI classifier is able to classify unknown radars from data
               generated using PDW generator. Model will read a CSV file containing
               a pair of each of the following radar types:
               - CTA1
               - NAV1
               - NAV2
               - NAV3

             Test Steps:
                  1. Extract features from CSV file
                  2. Connect to AI model via ONNX interface
                  3. Upload features to model
                  4. Run Model 
                  4. Write results to ELINT Intelligence db
                  7. Query Emitters table and assert each emitter was detected by AI model
                  9. 
       """)
    @pytest.mark.parametrize(
        [
            "csv_file_path",
            "emitter_1_id_csv",
            "emitter_2_id_csv",
            "emitter_1_name",
            "emitter_2_name",
            "number_emitters",
            "std_no_emitters"

        ],
        read_test_parameters(
            get_test_path() / "test_data",
            "unknown_emitters_db.csv",
        ),
        indirect=["csv_file_path"]
    )
    def test_unknown_emitter_detection_main_lobe(
            self,
            csv_file_path: str,
            download_file_artifactory_unknown_emitters: None,
            create_csv_input_table: None,
            truncate_activities_db_tables: None,
            populate_csv_input_table: None,
            truncate_intelligence_db_tables: None,
            delete_known_clusters_centers: None,
            delete_emitter_signal_map: None,
            delete_known_medoids: None,
            run_ai_model_fixture_main_lobe: None,
            intelligence_db_connector: DBUtils,
            emitter_1_id_csv: str,
            emitter_2_id_csv: str,
            emitter_1_name: str,
            emitter_2_name: str,
            number_emitters: int,
            std_no_emitters: int,
            pdf_suite_report,
            request,
    ) -> None:

        pdf_suite_report["report_title"] = f"Test summary Unknown(controlled) emitters detection"

        pdf_suite_report["report_description"] = (
            "Validates the AI model's ability to detect and identify controlled emitters (CTA1, CTA2, NAV1, NAV2, NAV3) from a PDW generator CSV file containing injected unknown emitters as noise (0, 2, 3, or 5). "
            "Success Criteria: "
            "All controlled emitters correctly detected at every noise level. "
            "Unknown emitters not misclassified as controlled. "
            "Per-emitter accuracy reported across all noise variations. "
        )

        print(f"\n[TEST PROGRESS] Running iteration on: {csv_file_path}", flush=True)

        with allure.step(f"Get number of emitters"):
            query = 'SELECT COUNT(*) FROM emitters'

            no_emitters = intelligence_db_connector.fetch_scalar(query)

        with allure.step(f"Get all emitter IDs"):

            query = 'SELECT id FROM emitters'
            emitter_ids = intelligence_db_connector.fetch_column(query)

        with allure.step(f"Assert that AI model detected minimum number of emitters"):

            assert_that(len(emitter_ids)) \
                .described_as("AI model failed to detected minimum number of emitters from testcase"
                              "Expected 2 emitters, detected only 1 emitter") \
                .is_greater_than_or_equal_to(2)

        with allure.step(f"Assert that average values for frequency, PRI and PW falls between specified characteristics"):
            emitters_average_values_list = []

            for id in emitter_ids:

                emitter_average_value_dict = {}
                params = {
                    'emitter_id': id
                }

                query_emitter_pri_average_value = """
                                  select avg(pulse_repetition_interval_us) as PRI
                                  from signals s join emitters e on s.emitter_id = e.id
                                  join clusters c on c.signal_id = s.id
                                  where emitter_id = '%(emitter_id)s'
                                  group by(e.id)
                              """

                emitter_pri = intelligence_db_connector.fetch_scalar(query_emitter_pri_average_value, params)

                emitter_average_value_dict["PRI"] = emitter_pri

                query_emitter_pw_average_value = """
                                              select avg(pulse_width_us) as PW
                                              from signals s join emitters e on s.emitter_id = e.id
                                              join clusters c on c.signal_id = s.id
                                              where emitter_id = '%(emitter_id)s'
                                              group by(e.id)
                                          """

                emitter_pw = intelligence_db_connector.fetch_scalar(query_emitter_pw_average_value, params)

                emitter_average_value_dict["PW"] = emitter_pw

                query_emitter_1_freq_average_value = """
                                              select avg(c.frequency_hz) as FREQ
                                              from signals s join emitters e on s.emitter_id = e.id
                                              join clusters c on c.signal_id = s.id
                                              where emitter_id = '%(emitter_id)s'
                                              group by(e.id)
                                          """

                emitter_freq = intelligence_db_connector.fetch_scalar(query_emitter_1_freq_average_value, params)

                emitter_average_value_dict["FREQ"] = emitter_freq


                emitters_average_values_list.append(emitter_average_value_dict)

        with allure.step("Create report"):
            test_results = {}

            for x in  range(len(emitters_average_values_list)):
                test_results[f"Emitter{x+1}"] = emitters_average_values_list[x]

            attach_emitter_statistics_report(test_results)

        with allure.step("Assert unknown controlled emitters were detected by AI model"):

            emitter_names = []
            emitter_names.append(emitter_1_name)
            emitter_names.append(emitter_2_name)

            for emitter_name in emitter_names:

                if emitter_name.lower() == "cta1":


                    query_emitter_1_count = """
                                select count(*) from (
                                    select s.emitter_id ,avg(pulse_repetition_interval_us) as PRI, avg(pulse_width_us) as PW, avg(c.frequency_hz) as FREQ
                                    from signals s join emitters e on s.emitter_id = e.id
                                    join clusters c on c.signal_id = s.id
                                    group by(s.emitter_id)
                                    order by s.emitter_id asc
                                    ) where PW between %(pw_start)s and %(pw_end)s
                                and FREQ between %(freq_start)s and %(freq_end)s
                                and PRI between %(pri_start)s and %(pri_end)s
                                """

                    params = {
                        'pw_start': cta1["pulse_width_micro_seconds"][0],
                        'pw_end': cta1["pulse_width_micro_seconds"][1],
                        'freq_start': cta1["frequency"][0],
                        'freq_end': cta1["frequency"][1],
                        'pri_start': cta1["pulse_repetition_interval"][0],
                        'pri_end': cta1["pulse_repetition_interval"][3],
                    }

                    emitter_1_count = intelligence_db_connector.fetch_scalar(query_emitter_1_count, params)

                    assert_that(emitter_1_count).described_as(f"Emitter: {emitter_name} with this specifications: "
                                                              f"Pulse Width: {params['pw_start']} - {params['pw_end']} "
                                                              f"PRI: {params['pri_start']} - {params['pri_end']}"
                                                              f"Frequency: {params['freq_start']} - {params['freq_end']} "
                                                              f"was not detected by AI model").is_greater_than_or_equal_to(1)

                elif emitter_name.lower() == "cta2":


                    query_emitter_1_count = """
                                select count(*) from (
                                    select s.emitter_id ,avg(pulse_repetition_interval_us) as PRI, avg(pulse_width_us) as PW, avg(c.frequency_hz) as FREQ
                                    from signals s join emitters e on s.emitter_id = e.id
                                    join clusters c on c.signal_id = s.id
                                    group by(s.emitter_id)
                                    order by s.emitter_id asc
                                    ) where PW between %(pw_start)s and %(pw_end)s
                                and FREQ between %(freq_start)s and %(freq_end)s
                                and PRI between %(pri_start)s and %(pri_end)s
                                """

                    params = {
                        'pw_start': cta2["pulse_width_micro_seconds"][0],
                        'pw_end': cta2["pulse_width_micro_seconds"][1],
                        'freq_start': cta2["frequency"][0],
                        'freq_end': cta2["frequency"][1],
                        'pri_start': cta2["pulse_repetition_interval"][0],
                        'pri_end': cta2["pulse_repetition_interval"][3],
                    }

                    emitter_1_count = intelligence_db_connector.fetch_scalar(query_emitter_1_count, params)

                    assert_that(emitter_1_count).described_as(f"Emitter: {emitter_name} with this specifications: "
                                                              f"Pulse Width: {params['pw_start']} - {params['pw_end']} "
                                                              f"PRI: {params['pri_start']} - {params['pri_end']} "
                                                              f"Frequency: {params['freq_start']} - {params['freq_end']} "
                                                              f"was not detected by AI model").is_greater_than_or_equal_to(1)


                elif emitter_name.lower() == "nav1":


                    query_emitter_1_count = """
                                select count(*) from (
                                    select s.emitter_id ,avg(pulse_repetition_interval_us) as PRI, avg(pulse_width_us) as PW, avg(c.frequency_hz) as FREQ
                                    from signals s join emitters e on s.emitter_id = e.id
                                    join clusters c on c.signal_id = s.id
                                    group by(s.emitter_id)
                                    order by s.emitter_id asc
                                    ) where PW between %(pw_start)s and %(pw_end)s
                                and FREQ between %(freq_start)s and %(freq_end)s
                                and PRI between %(pri_start)s and %(pri_end)s
                                """

                    params = {
                        'pw_start': nav1["pulse_width_micro_seconds"][0],
                        'pw_end': nav1["pulse_width_micro_seconds"][1],
                        'freq_start': nav1["frequency"][0],
                        'freq_end': nav1["frequency"][1],
                        'pri_start': nav1["pulse_repetition_interval"][0],
                        'pri_end': nav1["pulse_repetition_interval"][1],
                    }

                    emitter_1_count = intelligence_db_connector.fetch_scalar(query_emitter_1_count, params)

                    assert_that(emitter_1_count).described_as(f"Emitter: {emitter_name} with this specifications: "
                                                              f"Pulse Width: {params['pw_start']} - {params['pw_end']} "
                                                              f"PRI: {params['pri_start']} - {params['pri_end']} "
                                                              f"Frequency: {params['freq_start']} - {params['freq_end']} "
                                                              f"was not detected by AI model").is_greater_than_or_equal_to(1)

                elif emitter_name.lower() == "nav2":


                    query_emitter_1_count = """
                                select count(*) from (
                                    select s.emitter_id ,avg(pulse_repetition_interval_us) as PRI, avg(pulse_width_us) as PW, avg(c.frequency_hz) as FREQ
                                    from signals s join emitters e on s.emitter_id = e.id
                                    join clusters c on c.signal_id = s.id
                                    group by(s.emitter_id)
                                    order by s.emitter_id asc
                                    ) where PW between %(pw_start)s and %(pw_end)s
                                and FREQ between %(freq_start)s and %(freq_end)s
                                and PRI between %(pri_start)s and %(pri_end)s
                                """
                    params = {
                        'pw_start': nav2["pulse_width_micro_seconds"][0],
                        'pw_end': nav2["pulse_width_micro_seconds"][1],
                        'freq_start': nav2["frequency"][0],
                        'freq_end': nav2["frequency"][1],
                        'pri_start': nav2["pulse_repetition_interval"][0],
                        'pri_end': nav2["pulse_repetition_interval"][1],
                    }

                    emitter_1_count = intelligence_db_connector.fetch_scalar(query_emitter_1_count, params)

                    assert_that(emitter_1_count).described_as(f"Emitter: {emitter_name} with this specifications: "
                                                              f"Pulse Width: {params['pw_start']} - {params['pw_end']} "
                                                              f"PRI: {params['pri_start']} - {params['pri_end']} "
                                                              f"Frequency: {params['freq_start']} - {params['freq_end']} "
                                                              f"was not detected by AI model").is_greater_than_or_equal_to(1)

                elif emitter_name.lower() == "nav3":


                    query_emitter_1_count = """
                                select count(*) from (
                                    select s.emitter_id ,avg(pulse_repetition_interval_us) as PRI, avg(pulse_width_us) as PW, avg(c.frequency_hz) as FREQ
                                    from signals s join emitters e on s.emitter_id = e.id
                                    join clusters c on c.signal_id = s.id
                                    group by(s.emitter_id)
                                    order by s.emitter_id asc
                                    ) where PW between %(pw_start)s and %(pw_end)s
                                and FREQ between %(freq_start)s and %(freq_end)s
                                and PRI between %(pri_start)s and %(pri_end)s
                                """

                    params = {
                        'pw_start': nav3["pulse_width_micro_seconds"][0],
                        'pw_end': nav3["pulse_width_micro_seconds"][1],
                        'freq_start': nav3["frequency"][0],
                        'freq_end': nav3["frequency"][1],
                        'pri_start': nav3["pulse_repetition_interval"][0],
                        'pri_end': nav3["pulse_repetition_interval"][1],
                    }

                    emitter_1_count = intelligence_db_connector.fetch_scalar(query_emitter_1_count, params)

                    assert_that(emitter_1_count).described_as(f"Emitter: {emitter_name} with this specifications: "
                                                              f"Pulse Width: {params['pw_start']} - {params['pw_end']} "
                                                              f"PRI: {params['pri_start']} - {params['pri_end']} "
                                                              f"Frequency: {params['freq_start']} - {params['freq_end']} "
                                                              f"was not detected by AI model").is_greater_than_or_equal_to(1)

        with allure.step(f"Get total number of detected emitters"):
            query_detected_emitters = "select count(*) from emitters"
            count_detected_emitters = intelligence_db_connector.fetch_scalar(query_detected_emitters)
            custom_msg = (
                "AI model failed to detect total number of emitters(controlled + unknown, random emitters) "
                f"Expected {number_emitters} with standard deviation of +- {std_no_emitters} emitters "
                f"Got only {count_detected_emitters}"
            )

            existing_total_count = request.node.stash.get("custom_db_warning_total_count", "")
            request.node.stash["custom_db_warning_total_count"] = (
                    (existing_total_count + "; " if existing_total_count else "") + custom_msg
            )
            try:
                assert_that(count_detected_emitters) \
                    .described_as(
                    "AI model failed to detect total number of emitters(controlled + unknown, random emitters) "
                    f"Expected {number_emitters} with standard deviation of +- {std_no_emitters} emitters "
                    f"Got only {count_detected_emitters}") \
                    .is_between(number_emitters - std_no_emitters, number_emitters + std_no_emitters)
            except AssertionError:
                allure.attach(
                    custom_msg,
                    name="Warning details",
                    attachment_type=allure.attachment_type.TEXT)


    @allure.sub_suite("Known emitter detection source omni")
    @allure.title("Test AI model is able to detect known emitters")
    @allure.description("""
                Validate that AI classifier is able to classify known radars from data
                generated using PDW generator. Model will read a CSV file containing
                a pair of each of the following radar types:
                - PSR1
                - PSR2
                - PSR3
                - PSR4
                - PSR5 mod 1 and mod 2
                - MF-03
                - MF-04
                - MF-05
                - MF-06

              Test Steps:
                   1. Extract features from CSV file
                   2. Connect to AI model via ONNX interface
                   3. Upload features to model
                   4. Run Model 
                   4. Write results to ELINT Intelligence db
                   7. Query Emitters table and assert each emitter was detected by AI model
                   9. 
        """)
    @pytest.mark.parametrize(
        [
            "csv_file_path",
            "emitter_id_1",
            "emitter_id_2",
            "emitter_1_id_csv",
            "emitter_2_id_csv",
            "emitter_1_name",
            "emitter_2_name",
            "number_emitters",
            "std_no_emitters"
        ],
        read_test_parameters(
            get_test_path() / "test_data",
            "known_emitters_db_emitter_level.csv",
        ),
        indirect=["csv_file_path"]
    )
    def test_known_emitters_detection_source_omni(
            self,
            pdf_suite_report,
            csv_file_path: str,
            download_file_artifactory_known_emitters: None,
            create_csv_input_table: None,
            truncate_activities_db_tables: None,
            populate_csv_input_table: None,
            truncate_intelligence_db_tables: None,
            populate_signatures_table: None,
            delete_known_clusters_centers: None,
            delete_emitter_signal_map: None,
            run_ai_model_fixture_source_omni: None,
            intelligence_db_connector: DBUtils,
            activities_db_connector: ClickHouseManager,
            emitter_id_1: str,
            emitter_id_2: str,
            emitter_1_id_csv: str,
            emitter_2_id_csv: str,
            emitter_1_name: str,
            emitter_2_name: str,
            number_emitters: int,
            std_no_emitters: int,
            reset_shared_stats: None,
            shared_stats_dictionary,
            shared_confusion_matrix,
            request,
    ) -> None:

        emitter_stats = []
        stats_dict = shared_stats_dictionary["dictionary_stats"]

        df = shared_confusion_matrix["confusion_matrix"]

        print(f"\n[TEST PROGRESS] Running iteration on: {csv_file_path}", flush=True)

        pdf_suite_report["report_title"] = f"Test summary Known emitters detection"

        pdf_suite_report["report_description"] = (
            "Validates the AI model's ability to accurately classify and identify known radar emitters from CSV files containing PDWs generated by the PDW generator. Test coverage includes 8 radar types: "
            "PSR1 – PSR5 (with PSR5 tested in both Mod 1 and Mod 2 variants) "
            "MF-03, MF-04, MF-05 "
            "Each test case verifies correct radar type identification from the provided PDW data. The suite also builds a confusion matrix to calculate and report per-radar-type accuracy, "
            "providing insight into individual classification performance and cross-type misidentification patterns."
        )

        with allure.step(f"Query total number of PDWs in main lobe from input CSV file and attach to report"):

            query = "SELECT emitter_id, count(emitter_id)  FROM signals.csv_input where source = 'omni' group by emitter_id order by emitter_id asc"
            activities_db_connector.fetch_all_results_as_df(
                query=query,
                attach_to_allure=True,
                attachment_name=f"Number of PDWs in main lobe from input CSV file"
            )

        with (allure.step(f"Assert {emitter_1_name} was detected by AI model")):

            query = f"select count(*) from emitters where name = '{emitter_id_1}'"
            count_query_result = intelligence_db_connector.fetch_scalar(query)
            assert_that(count_query_result) \
                .described_as(f"AI model failed to detect known emitter {emitter_1_name}") \
                .is_greater_than(0)

        with allure.step(f"Assert {emitter_2_name} was detected by AI model"):

            query = f"select count(*) from emitters where name = '{emitter_id_2}'"
            count_query_result = intelligence_db_connector.fetch_scalar(query)
            assert_that(count_query_result) \
                .described_as(f"AI model failed to detect known emitter {emitter_2_name}") \
                .is_greater_than(0)

        with allure.step(f"Assert that each cluster was assigned to a Signal"):

            query_clusters_not_assigned_signal = "select count(*) from clusters where signal_id is null"
            count_query_result = intelligence_db_connector.fetch_scalar(query_clusters_not_assigned_signal)
            assert_that(count_query_result).described_as("There are clusters were signal_id is null").is_equal_to(0)

        with allure.step(f"Get all PDW labeled as {emitter_1_name} and {emitter_2_name} from input CSV file"):

            query_pdw_emitter_1_ids_count = f"""
                       SELECT
                           count(*)
                           FROM signals.csv_input
                           WHERE emitter_id = {emitter_1_id_csv}
                           AND source = 'omni'
                   """
            no_emitter_1_pdws_id = activities_db_connector.select_column(
                query_pdw_emitter_1_ids_count
            )

            if f"{emitter_1_name}_total_input" in stats_dict:
                stats_dict[f"{emitter_1_name}_total_input"] += no_emitter_1_pdws_id[0]
            else:
                stats_dict[f"{emitter_1_name}_total_input"] = no_emitter_1_pdws_id[0]

            query_pdw_emitter_2_ids_count = f"""
                              SELECT
                                  count(*)
                                  FROM signals.csv_input
                                  WHERE emitter_id = {emitter_2_id_csv}
                                  AND source = 'omni'
                          """

            no_emitter_2_pdws_id = activities_db_connector.select_column(
                query_pdw_emitter_2_ids_count,
            )

            if f"{emitter_2_name}_total_input" in stats_dict:
                stats_dict[f"{emitter_2_name}_total_input"] += no_emitter_2_pdws_id[0]
            else:
                stats_dict[f"{emitter_2_name}_total_input"] = no_emitter_2_pdws_id[0]

        with allure.step(f"Get number of PDWs correctly labeled as {emitter_1_name}"):

            query_cluster_id_emitter_1 = f"""
                        select c.id
                        from signals s join emitters e on s.emitter_id = e.id
                        join clusters c on c.signal_id = s.id
                        where e.name = '{emitter_id_1}'
                """

            clusters_emitter_ids = intelligence_db_connector.fetch_column(
                query_cluster_id_emitter_1
            )

            clusters_emitter_ids_as_string = ", ".join(map(str, clusters_emitter_ids)) or "NULL"
            query_pdws_count_emitter_1 = f"""
                    select count(*) from signals.csv_input where id in (
                    SELECT
                        arrayJoin(extracted_ids)  from (
                        select
                        arrayMap(
                        x -> toUInt64(extract(toString(x), '([0-9]+)$')),
                        activity_ids
                    ) AS extracted_ids
                    FROM signals.clusters
                        WHERE cluster_id IN ({clusters_emitter_ids_as_string}))) and  emitter_id = {emitter_1_id_csv}
                """

            no_pdws_emitter_1_correctly_labeled = activities_db_connector.select_query(query_pdws_count_emitter_1)

            if f"{emitter_1_name}_{emitter_1_name}" in stats_dict:
                stats_dict[f"{emitter_1_name}_{emitter_1_name}"] += no_pdws_emitter_1_correctly_labeled[0][0]
            else:
                stats_dict[f"{emitter_1_name}_{emitter_1_name}"] = no_pdws_emitter_1_correctly_labeled[0][0]

            emitter_1_labeled_correctly = no_pdws_emitter_1_correctly_labeled[0][0] / no_emitter_1_pdws_id[0] * 100

            emitter_stats.append({"name": f"{emitter_1_name} labeled correctly", "pct": emitter_1_labeled_correctly})

        with allure.step(
                f"Get number of PDWs incorrectly labeled as {emitter_1_name} that were actually {emitter_2_name}"):

            query_cluster_id_emitter_1 = f"""
                        select c.id
                        from signals s join emitters e on s.emitter_id = e.id
                        join clusters c on c.signal_id = s.id
                        where e.name = '{emitter_id_1}'
                """

            clusters_emitter_1_ids = intelligence_db_connector.fetch_column(
                query_cluster_id_emitter_1
            )

            clusters_emitter_1_ids_as_string = ", ".join(map(str, clusters_emitter_1_ids)) or "NULL"
            query_pdws_count_emitter_1_incorrect_labeled = f"""
                    select count(*) from signals.csv_input where id in (
                    SELECT
                        arrayJoin(extracted_ids)  from (
                        select
                        arrayMap(
                        x -> toUInt64(extract(toString(x), '([0-9]+)$')),
                        activity_ids
                    ) AS extracted_ids
                    FROM signals.clusters

                        WHERE cluster_id IN ({clusters_emitter_1_ids_as_string}))) and  emitter_id = {emitter_2_id_csv}
                """

            no_pdws_emitter_1_labeled_incorrectly = activities_db_connector.select_query(
                query_pdws_count_emitter_1_incorrect_labeled)

            emitter_1_labeled_incorrectly = no_pdws_emitter_1_labeled_incorrectly[0][0] / no_emitter_1_pdws_id[0] * 100

            emitter_stats.append(
                {"name": f"{emitter_2_name} mislabeled as {emitter_1_name}", "pct": emitter_1_labeled_incorrectly})

            if f"{emitter_2_name}_{emitter_1_name}" in stats_dict:
                stats_dict[f"{emitter_2_name}_{emitter_1_name}"] += no_pdws_emitter_1_labeled_incorrectly[0][0]
            else:
                stats_dict[f"{emitter_2_name}_{emitter_1_name}"] = no_pdws_emitter_1_labeled_incorrectly[0][0]

        with allure.step(f"Get number of PDWs incorrectly labeled as unknown_ai that were actually {emitter_1_name}"):

            query_cluster_id_unknown_ai = f"""
                        select c.id
                        from signals s join emitters e on s.emitter_id = e.id
                        join clusters c on c.signal_id = s.id
                        where e.name = 'unknown_ai'
                """

            cluster_id_unknown_ai = intelligence_db_connector.fetch_column(
                query_cluster_id_unknown_ai
            )

            cluster_id_unknown_ai_as_string = ", ".join(map(str, cluster_id_unknown_ai)) or "NULL"
            query_pdws_count_emitter_1_incorrect_labeled_unknown_ai = f"""
                    select count(*) from signals.csv_input where id in (
                    SELECT
                        arrayJoin(extracted_ids)  from (
                        select
                        arrayMap(
                        x -> toUInt64(extract(toString(x), '([0-9]+)$')),
                        activity_ids
                    ) AS extracted_ids
                    FROM signals.clusters

                        WHERE cluster_id IN ({cluster_id_unknown_ai_as_string}))) and  emitter_id = {emitter_1_id_csv}
                """

            no_pdws_emitter_1_labeled_incorrectly_unknown_ai = activities_db_connector.select_query(
                query_pdws_count_emitter_1_incorrect_labeled_unknown_ai)

            emitter_1_labeled_incorrectly_unknown_ai = no_pdws_emitter_1_labeled_incorrectly_unknown_ai[0][0] / \
                                                       no_emitter_1_pdws_id[0] * 100

            emitter_stats.append(
                {"name": f"{emitter_1_name} mislabeled as unknown_ai", "pct": emitter_1_labeled_incorrectly_unknown_ai})

            if f"{emitter_1_name}_unknownai" in stats_dict:
                stats_dict[f"{emitter_1_name}_unknownai"] += no_pdws_emitter_1_labeled_incorrectly_unknown_ai[0][0]
            else:
                stats_dict[f"{emitter_1_name}_unknownai"] = no_pdws_emitter_1_labeled_incorrectly_unknown_ai[0][0]

        with allure.step(f"Get number of PDWs labeled as {emitter_1_name} that were unclustered"):

            query_emitter_1_pdws_unclustered = f"""
                    SELECT count(*) FROM signals.csv_input
                    WHERE id NOT IN (
                    SELECT arrayJoin(extracted_ids)
                    FROM (
                        SELECT
                            arrayMap(
                                x -> toUInt64(extract(toString(x), '([0-9]+)$')),
                                activity_ids
                        ) AS extracted_ids
                    FROM signals.clusters
                        )
                    )
                    AND emitter_id = {emitter_1_id_csv} AND source = 'omni'
                """
            no_pdws_emitter_1_unclustered = activities_db_connector.select_query(query_emitter_1_pdws_unclustered)

            emitter_1_unclustered = no_pdws_emitter_1_unclustered[0][0] / no_emitter_1_pdws_id[0] * 100

            emitter_stats.append({"name": f"{emitter_1_name} unclustered", "pct": emitter_1_unclustered})

            if f"{emitter_1_name}_unclustered" in stats_dict:
                stats_dict[f"{emitter_1_name}_unclustered"] += no_pdws_emitter_1_unclustered[0][0]
            else:
                stats_dict[f"{emitter_1_name}_unclustered"] = no_pdws_emitter_1_unclustered[0][0]

        with allure.step(f"Get number of PDWs correctly labeled as {emitter_2_name}"):

            query_cluster_id_emitter_2 = f"""
                        select c.id
                        from signals s join emitters e on s.emitter_id = e.id
                        join clusters c on c.signal_id = s.id
                        where e.name = '{emitter_id_2}'
                """

            clusters_emitter_ids = intelligence_db_connector.fetch_column(
                query_cluster_id_emitter_2
            )

            clusters_emitter_ids_as_string = ", ".join(map(str, clusters_emitter_ids)) or "NULL"
            query_pdws_count_emitter_2 = f"""
                    select count(*) from signals.csv_input where id in (
                    SELECT
                        arrayJoin(extracted_ids)  from (
                        select
                        arrayMap(
                        x -> toUInt64(extract(toString(x), '([0-9]+)$')),
                        activity_ids
                    ) AS extracted_ids
                    FROM signals.clusters

                        WHERE cluster_id IN ({clusters_emitter_ids_as_string}))) and  emitter_id = {emitter_2_id_csv}
                """

            no_pdws_emitter_2_labeled = activities_db_connector.select_query(query_pdws_count_emitter_2)

            emitter_2_labeled_correctly = no_pdws_emitter_2_labeled[0][0] / no_emitter_2_pdws_id[0] * 100

            emitter_stats.append({"name": f"{emitter_2_name} labeled correctly", "pct": emitter_2_labeled_correctly})

            if f"{emitter_2_name}_{emitter_2_name}" in stats_dict:
                stats_dict[f"{emitter_2_name}_{emitter_2_name}"] += no_pdws_emitter_2_labeled[0][0]
            else:
                stats_dict[f"{emitter_2_name}_{emitter_2_name}"] = no_pdws_emitter_2_labeled[0][0]

        with allure.step(
                f"Get number of PDWs incorrectly labeled as {emitter_2_name} that were actually {emitter_1_name}"):

            query_cluster_id_emitter_2 = f"""
                        select c.id
                        from signals s join emitters e on s.emitter_id = e.id
                        join clusters c on c.signal_id = s.id
                        where e.name = '{emitter_id_2}'
                """

            clusters_emitter_ids = intelligence_db_connector.fetch_column(
                query_cluster_id_emitter_2,
            )

            clusters_emitter_ids_as_string = ", ".join(map(str, clusters_emitter_ids)) or "NULL"

            query_pdws_count_emitter_2_incorrect_labeled = f"""
                    select count(*) from signals.csv_input where id in (
                    SELECT
                        arrayJoin(extracted_ids)  from (
                        select
                        arrayMap(
                        x -> toUInt64(extract(toString(x), '([0-9]+)$')),
                        activity_ids
                    ) AS extracted_ids
                    FROM signals.clusters

                        WHERE cluster_id IN ({clusters_emitter_ids_as_string}))) and  emitter_id = {emitter_1_id_csv}
                """

            no_pdws_emitter_2_labeled_incorrectly = activities_db_connector.select_query(
                query_pdws_count_emitter_2_incorrect_labeled)

            if f"{emitter_1_name}_{emitter_2_name}" in stats_dict:
                stats_dict[f"{emitter_1_name}_{emitter_2_name}"] += no_pdws_emitter_2_labeled_incorrectly[0][0]
            else:
                stats_dict[f"{emitter_1_name}_{emitter_2_name}"] = no_pdws_emitter_2_labeled_incorrectly[0][0]

            emitter_2_labeled_incorrectly = no_pdws_emitter_2_labeled_incorrectly[0][0] / no_emitter_2_pdws_id[0] * 100

            emitter_stats.append(
                {"name": f"{emitter_1_name} mislabeled as {emitter_2_name}", "pct": emitter_2_labeled_incorrectly})

        with allure.step(f"Get number of PDWs incorrectly labeled as unknown_ai that were actually {emitter_2_name}"):

            query_cluster_id_unknown_ai = f"""
                        select c.id
                        from signals s join emitters e on s.emitter_id = e.id
                        join clusters c on c.signal_id = s.id
                        where e.name = 'unknown_ai'
                """

            cluster_id_unknown_ai = intelligence_db_connector.fetch_column(
                query_cluster_id_unknown_ai
            )

            cluster_id_unknown_ai_as_string = ", ".join(map(str, cluster_id_unknown_ai)) or "NULL"
            query_pdws_count_emitter_2_incorrect_labeled_unknown_ai = f"""
                    select count(*) from signals.csv_input where id in (
                    SELECT
                        arrayJoin(extracted_ids)  from (
                        select
                        arrayMap(
                        x -> toUInt64(extract(toString(x), '([0-9]+)$')),
                        activity_ids
                    ) AS extracted_ids
                    FROM signals.clusters

                        WHERE cluster_id IN ({cluster_id_unknown_ai_as_string}))) and  emitter_id = {emitter_2_id_csv}
                """

            no_pdws_emitter_2_labeled_incorrectly_unknown_ai = activities_db_connector.select_query(
                query_pdws_count_emitter_2_incorrect_labeled_unknown_ai)

            emitter_2_labeled_incorrectly_unknown_ai = no_pdws_emitter_2_labeled_incorrectly_unknown_ai[0][0] / \
                                                       no_emitter_2_pdws_id[0] * 100

            emitter_stats.append(
                {"name": f"{emitter_2_name} mislabeled as unknown_ai", "pct": emitter_2_labeled_incorrectly_unknown_ai})

            if f"{emitter_2_name}_unknownai" in stats_dict:
                stats_dict[f"{emitter_2_name}_unknownai"] += no_pdws_emitter_2_labeled_incorrectly_unknown_ai[0][0]
            else:
                stats_dict[f"{emitter_2_name}_unknownai"] = no_pdws_emitter_2_labeled_incorrectly_unknown_ai[0][0]

        with allure.step(f"Get number of PDWs labeled as {emitter_2_name} that were unclustered"):

            query_emitter_2_pdws_unclustered = f"""
                    SELECT count(*) FROM signals.csv_input
                    WHERE id NOT IN (
                    SELECT arrayJoin(extracted_ids)
                    FROM (
                        SELECT
                            arrayMap(
                                x -> toUInt64(extract(toString(x), '([0-9]+)$')),
                                activity_ids
                        ) AS extracted_ids
                    FROM signals.clusters
                        )
                    )
                    AND emitter_id = {emitter_2_id_csv} and source = 'omni'
                """
            no_pdws_emitter_2_unclustered = activities_db_connector.select_query(query_emitter_2_pdws_unclustered)

            emitter_2_unclustered = no_pdws_emitter_2_unclustered[0][0] / no_emitter_2_pdws_id[0] * 100
            emitter_stats.append({"name": f"{emitter_2_name} unclustered", "pct": emitter_2_unclustered})

            if f"{emitter_2_name}_unclustered" in stats_dict:
                stats_dict[f"{emitter_2_name}_unclustered"] += no_pdws_emitter_2_unclustered[0][0]
            else:
                stats_dict[f"{emitter_2_name}_unclustered"] = no_pdws_emitter_2_unclustered[0][0]

        with allure.step(f"Generating test status report for {emitter_1_name} and {emitter_2_name}"):

            attach_pdw_report2(emitters_data=emitter_stats, threshold=90.0)

            with allure.step(
                    f"Verify if system detected additional emitter except {emitter_1_name} or {emitter_2_name}"
                    f""):

                query_no_other_emitters = f"select count(*) from emitters where name not in ('{emitter_id_1}', '{emitter_id_2}', 'unknown_ai')"

                no_other_emitters_detected = intelligence_db_connector.fetch_scalar(query_no_other_emitters)

                if (no_other_emitters_detected > 0):
                    with allure.step(
                            f"⚠️ WARNING: System has detected other emitters except {emitter_1_name} or {emitter_2_name} either from known or unknown controlled emitters"):
                        other_emitters_name_query = f"select name from emitters where name not in ('{emitter_id_1}', '{emitter_id_2}', 'unknown_ai')"

                        other_emitters_name = intelligence_db_connector.fetch_column(other_emitters_name_query)
                        allure.attach(
                            body=' '.join(other_emitters_name),
                            name=f"Other emitters detected except {emitter_1_name} or {emitter_2_name} or 'unknown_ai'",
                            attachment_type=allure.attachment_type.TEXT
                        )

                        for emitter in other_emitters_name:
                            emitter_name = RadarMapper(emitter).name if emitter in RadarMapper else emitter

                            custom_msg = (
                                f"Warning: System detected other known emitters except {emitter_1_name}, {emitter_2_name} or 'unknown_ai' "
                                f"System detected {emitter_name}")
                            existing = request.node.stash.get("custom_db_warning", "")
                            request.node.stash["custom_db_warning"] = (existing + "; " if existing else "") + custom_msg
                            allure.attach(
                                f"Warning, system detected {emitter_name}, except expected {emitter_1_name} or {emitter_2_name}",
                                name="Warning details",
                                attachment_type=allure.attachment_type.TEXT)

                            with allure.step(
                                    f"Calculate the percentage of PDWs marked as {emitter_name} that were labeled as {emitter_1_name}"):

                                query_cluster_id_emitter_other = f"""
                                                   select c.id
                                                   from signals s join emitters e on s.emitter_id = e.id
                                                   join clusters c on c.signal_id = s.id
                                                   where e.name = '{emitter}'
                                           """

                                clusters_emitter_ids = intelligence_db_connector.fetch_column(
                                    query_cluster_id_emitter_other,
                                )

                                clusters_emitter_ids_as_string = ", ".join(map(str, clusters_emitter_ids)) or "NULL"
                                query_pdws_other_emitters_labeled_emitter_1_count = f"""
                                               select count(*) from signals.csv_input where id in (
                                               SELECT
                                                   arrayJoin(extracted_ids)  from (
                                                   select
                                                   arrayMap(
                                                   x -> toUInt64(extract(toString(x), '([0-9]+)$')),
                                                   activity_ids
                                               ) AS extracted_ids
                                               FROM signals.clusters
                                                   WHERE cluster_id IN ({clusters_emitter_ids_as_string}))) and  emitter_id = {emitter_1_id_csv}
                                           """

                                no_pdws_other_emitter_labeled_emitter_1 = activities_db_connector.select_query(
                                    query_pdws_other_emitters_labeled_emitter_1_count)

                                if f"{emitter_1_name}_{emitter_name}" in stats_dict:
                                    stats_dict[f"{emitter_1_name}_{emitter_name}"] += \
                                        no_pdws_other_emitter_labeled_emitter_1[0][0]
                                else:
                                    stats_dict[f"{emitter_1_name}_{emitter_name}"] = \
                                        no_pdws_other_emitter_labeled_emitter_1[0][0]

                                other_emitters_detected_percentage = no_pdws_other_emitter_labeled_emitter_1[0][0] / \
                                                                     no_emitter_1_pdws_id[0] * 100

                                emitter_stats.append({
                                    "name": f"Other radar with id {emitter_name} was detected, Percentage of PDWs marked as {emitter_name} that were labeled as {emitter_1_name}",
                                    "pct": other_emitters_detected_percentage})

                            with allure.step(
                                    f"Calculate the percentage of PDWs marked as {emitter_name} that were labeled as {emitter_2_name}"):

                                query_cluster_id_emitter_other = f"""
                                                   select c.id
                                                   from signals s join emitters e on s.emitter_id = e.id
                                                   join clusters c on c.signal_id = s.id
                                                   where e.name = '{emitter}'
                                           """

                                clusters_emitter_ids = intelligence_db_connector.fetch_column(
                                    query_cluster_id_emitter_other,
                                )

                                clusters_emitter_ids_as_string = ", ".join(map(str, clusters_emitter_ids)) or "NULL"
                                query_pdws_other_emitters__labeled_emitter_1_count = f"""
                                               select count(*) from signals.csv_input where id in (
                                               SELECT
                                                   arrayJoin(extracted_ids)  from (
                                                   select
                                                   arrayMap(
                                                   x -> toUInt64(extract(toString(x), '([0-9]+)$')),
                                                   activity_ids
                                               ) AS extracted_ids
                                               FROM signals.clusters
                                                   WHERE cluster_id IN ({clusters_emitter_ids_as_string}))) and emitter_id = {emitter_2_id_csv}
                                           """

                                no_pdws_other_emitter_labeled_emitter_2 = activities_db_connector.select_query(
                                    query_pdws_other_emitters__labeled_emitter_1_count)

                                if f"{emitter_2_name}_{emitter_name}" in stats_dict:
                                    stats_dict[f"{emitter_2_name}_{emitter_name}"] += \
                                        no_pdws_other_emitter_labeled_emitter_2[0][0]
                                else:
                                    stats_dict[f"{emitter_2_name}_{emitter_name}"] = \
                                        no_pdws_other_emitter_labeled_emitter_2[0][0]

                                other_emitters_detected_percentage = no_pdws_other_emitter_labeled_emitter_2[0][0] / \
                                                                     no_emitter_2_pdws_id[0] * 100
                                emitter_stats.append({
                                    "name": f"Other radar with id {emitter_name} was detected, Percentage of PDWs marked as {emitter_name} that were labeled as {emitter_2_name}",
                                    "pct": other_emitters_detected_percentage})

                            with allure.step(
                                    f"Attaching PDWs that generated AI model identification of a known emitter"):
                                query_pdws_other_emitters_identified = f"""
                                                                  select * from signals.csv_input where id in (
                                                                  SELECT
                                                                      arrayJoin(extracted_ids)  from (
                                                                      select
                                                                      arrayMap(
                                                                      x -> toUInt64(extract(toString(x), '([0-9]+)$')),
                                                                      activity_ids
                                                                  ) AS extracted_ids
                                                                  FROM signals.clusters
                                                                      WHERE cluster_id IN ({clusters_emitter_ids_as_string})))
                                                              """

                                activities_db_connector.fetch_all_results_as_df(
                                    query=query_pdws_other_emitters_identified,
                                    attach_to_allure=True,
                                    attachment_name=f"Other emitters value for {emitter_name}"
                                )

        with allure.step(f"Attaching updated confusion matrix for {emitter_1_name} and {emitter_2_name}"):


            for key, value in stats_dict.items():
                df.loc[f"{key.split("_")[0]}", f"{key.split("_")[1]}"] = value / stats_dict[
                    f"{key.split("_")[0]}_total_input"] * 100.00

            allure.attach(
                df.to_markdown(),
                name="Undated confusion matrix",
                attachment_type=allure.attachment_type.TEXT
            )

            pdf_suite_report["confusion_matrix"] = df

        with allure.step(f"Get total number of detected emitters"):
            query_detected_emitters = "select count(*) from emitters"
            count_detected_emitters = intelligence_db_connector.fetch_scalar(query_detected_emitters)
            custom_msg = (
                "AI model failed to detect total number of emitters(controlled + unknown, random emitters) "
                f"Expected {number_emitters} with standard deviation of +- {std_no_emitters} emitters "
                f"Got only {count_detected_emitters}"
            )

            existing_total_count = request.node.stash.get("custom_db_warning_total_count", "")
            request.node.stash["custom_db_warning_total_count"] = (
                    (existing_total_count + "; " if existing_total_count else "") + custom_msg
            )
            try:
                assert_that(count_detected_emitters) \
                    .described_as(
                    "AI model failed to detect total number of emitters(controlled + unknown, random emitters) "
                    f"Expected {number_emitters} with standard deviation of +- {std_no_emitters} emitters "
                    f"Got only {count_detected_emitters}") \
                    .is_between(number_emitters - std_no_emitters, number_emitters + std_no_emitters)
            except AssertionError:
                allure.attach(
                    custom_msg,
                    name="Warning details",
                    attachment_type=allure.attachment_type.TEXT)


    @allure.sub_suite("Unknown emitter detection from source OMNI")
    @allure.title("Test AI model is able to detect unknown emitters from source OMNI")
    @allure.description("""
               Validate that AI classifier is able to classify unknown radars from data
               generated using PDW generator. Model will read a CSV file containing
               a pair of each of the following radar types:
               - CTA1
               - NAV1
               - NAV2
               - NAV3

             Test Steps:
                  1. Extract features from CSV file
                  2. Connect to AI model via ONNX interface
                  3. Upload features to model
                  4. Run Model 
                  4. Write results to ELINT Intelligence db
                  7. Query Emitters table and assert each emitter was detected by AI model
                  9. 
       """)
    @pytest.mark.parametrize(
        [
            "csv_file_path",
            "emitter_1_id_csv",
            "emitter_2_id_csv",
            "emitter_1_name",
            "emitter_2_name",
            "number_emitters",
            "std_no_emitters"

        ],
        read_test_parameters(
            get_test_path() / "test_data",
            "unknown_emitters_db.csv",
        ),
        indirect=["csv_file_path"]
    )
    @pytest.mark.skip(reason="For unknown emitters we test only with main lobe")
    def test_unknown_emitter_detection_source_omni(
            self,
            csv_file_path: str,
            download_file_artifactory_unknown_emitters: None,
            create_csv_input_table: None,
            truncate_activities_db_tables: None,
            populate_csv_input_table: None,
            truncate_intelligence_db_tables: None,
            delete_known_clusters_centers: None,
            delete_emitter_signal_map: None,
            run_ai_model_fixture_source_omni: None,
            intelligence_db_connector: DBUtils,
            emitter_1_id_csv: str,
            emitter_2_id_csv: str,
            emitter_1_name: str,
            emitter_2_name: str,
            number_emitters: int,
            std_no_emitters: int,
    ) -> None:

        with allure.step(f"Get number of emitters"):
            query = 'SELECT COUNT(*) FROM emitters'

            no_emitters = intelligence_db_connector.fetch_scalar(query)

        with allure.step(f"Get all emitter IDs"):

            query = 'SELECT id FROM emitters'
            emitter_ids = intelligence_db_connector.fetch_column(query)

        with allure.step(f"Assert that average values for frequency, PRI and PW falls between specified characteristics"):
            emitters_average_values_list = []

            for id in emitter_ids:

                emitter_average_value_dict = {}
                params = {
                    'emitter_id': id
                }

                query_emitter_pri_average_value = """
                                  select avg(pulse_repetition_interval_us) as PRI
                                  from signals s join emitters e on s.emitter_id = e.id
                                  join clusters c on c.signal_id = s.id
                                  where emitter_id = '%(emitter_id)s'
                                  group by(e.id)
                              """

                emitter_pri = intelligence_db_connector.fetch_scalar(query_emitter_pri_average_value, params)

                emitter_average_value_dict["PRI"] = emitter_pri

                query_emitter_pw_average_value = """
                                              select avg(pulse_width_us) as PW
                                              from signals s join emitters e on s.emitter_id = e.id
                                              join clusters c on c.signal_id = s.id
                                              where emitter_id = '%(emitter_id)s'
                                              group by(e.id)
                                          """

                emitter_pw = intelligence_db_connector.fetch_scalar(query_emitter_pw_average_value, params)

                emitter_average_value_dict["PW"] = emitter_pw

                query_emitter_1_freq_average_value = """
                                              select avg(c.frequency_hz) as FREQ
                                              from signals s join emitters e on s.emitter_id = e.id
                                              join clusters c on c.signal_id = s.id
                                              where emitter_id = '%(emitter_id)s'
                                              group by(e.id)
                                          """

                emitter_freq = intelligence_db_connector.fetch_scalar(query_emitter_1_freq_average_value, params)

                emitter_average_value_dict["FREQ"] = emitter_freq


                emitters_average_values_list.append(emitter_average_value_dict)

        with allure.step("Create report"):
            test_results = {}

            for x in  range(len(emitters_average_values_list)):
                test_results[f"Emitter{x+1}"] = emitters_average_values_list[x]

            attach_emitter_statistics_report(test_results)

        with allure.step("Assert unknown controlled emitters were detected by AI model"):

            emitter_names = []
            emitter_names.append(emitter_1_name)
            emitter_names.append(emitter_2_name)

            for emitter_name in emitter_names:

                if emitter_name.lower() == "cta1":


                    query_emitter_1_count = """
                                select count(*) from ( 
                                    select s.emitter_id ,avg(pulse_repetition_interval_us) as PRI, avg(pulse_width_us) as PW, avg(c.frequency_hz) as FREQ 
                                    from signals s join emitters e on s.emitter_id = e.id 
                                    join clusters c on c.signal_id = s.id 
                                    group by(s.emitter_id) 
                                    order by s.emitter_id asc 
                                    ) where PW between %(pw_start)s and %(pw_end)s
                                and FREQ between %(freq_start)s and %(freq_end)s
                                and PRI between %(pri_start)s and %(pri_end)s 
                                """

                    params = {
                        'pw_start': cta1["pulse_width_micro_seconds"] - 0.1,
                        'pw_end': cta1["pulse_width_micro_seconds"] + 0.1,
                        'freq_start': cta1["frequency"][0],
                        'freq_end': cta1["frequency"][1],
                        'pri_start': cta1["pulse_repetition_interval"][0],
                        'pri_end': cta1["pulse_repetition_interval"][3],
                    }

                    emitter_1_count = intelligence_db_connector.fetch_scalar(query_emitter_1_count, params)

                    assert_that(emitter_1_count).described_as(f"Emitter: {emitter_name} with this specifications: "
                                                              f"Pulse Width: {params['pw_start']} - {params['pw_end']} "
                                                              f"PRI: {params['pri_start']} - {params['pri_end']} "
                                                              f"Frequency: {params['freq_start']} - {params['freq_end']} "
                                                              f"was not detected by AI model").is_greater_than_or_equal_to(1)

                elif emitter_name.lower() == "cta2":


                    query_emitter_1_count = """
                                select count(*) from ( 
                                    select s.emitter_id ,avg(pulse_repetition_interval_us) as PRI, avg(pulse_width_us) as PW, avg(c.frequency_hz) as FREQ 
                                    from signals s join emitters e on s.emitter_id = e.id 
                                    join clusters c on c.signal_id = s.id 
                                    group by(s.emitter_id) 
                                    order by s.emitter_id asc 
                                    ) where PW between %(pw_start)s and %(pw_end)s
                                and FREQ between %(freq_start)s and %(freq_end)s
                                and PRI between %(pri_start)s and %(pri_end)s 
                                """

                    params = {
                        'pw_start': cta2["pulse_width_micro_seconds"] - 0.1,
                        'pw_end': cta2["pulse_width_micro_seconds"] + 0.1,
                        'freq_start': cta2["frequency"][0],
                        'freq_end': cta2["frequency"][1],
                        'pri_start': cta2["pulse_repetition_interval"][0],
                        'pri_end': cta2["pulse_repetition_interval"][3],
                    }

                    emitter_1_count = intelligence_db_connector.fetch_scalar(query_emitter_1_count, params)

                    assert_that(emitter_1_count).described_as(f"Emitter: {emitter_name} with this specifications: "
                                                              f"Pulse Width: {params['pw_start']} - {params['pw_end']} "
                                                              f"PRI: {params['pri_start']} - {params['pri_end']} "
                                                              f"Frequency: {params['freq_start']} - {params['freq_end']} "
                                                              f"was not detected by AI model").is_greater_than_or_equal_to(1)


                elif emitter_name.lower() == "nav1":


                    query_emitter_1_count = """
                                select count(*) from ( 
                                    select s.emitter_id ,avg(pulse_repetition_interval_us) as PRI, avg(pulse_width_us) as PW, avg(c.frequency_hz) as FREQ 
                                    from signals s join emitters e on s.emitter_id = e.id 
                                    join clusters c on c.signal_id = s.id 
                                    group by(s.emitter_id) 
                                    order by s.emitter_id asc 
                                    ) where PW between %(pw_start)s and %(pw_end)s
                                and FREQ between %(freq_start)s and %(freq_end)s
                                and PRI between %(pri_start)s and %(pri_end)s 
                                """

                    params = {
                        'pw_start': nav1["pulse_width_micro_seconds"] - 0.1,
                        'pw_end': nav1["pulse_width_micro_seconds"] + 0.1,
                        'freq_start': nav1["frequency"][0],
                        'freq_end': nav1["frequency"][1],
                        'pri_start': nav1["pulse_repetition_interval"][0],
                        'pri_end': nav1["pulse_repetition_interval"][1],
                    }

                    emitter_1_count = intelligence_db_connector.fetch_scalar(query_emitter_1_count, params)

                    assert_that(emitter_1_count).described_as(f"Emitter: {emitter_name} with this specifications: "
                                                              f"Pulse Width: {params['pw_start']} - {params['pw_end']} "
                                                              f"PRI: {params['pri_start']} - {params['pri_end']} "
                                                              f"Frequency: {params['freq_start']} - {params['freq_end']} "
                                                              f"was not detected by AI model").is_greater_than_or_equal_to(1)

                elif emitter_name.lower() == "nav2":


                    query_emitter_1_count = """
                                select count(*) from ( 
                                    select s.emitter_id ,avg(pulse_repetition_interval_us) as PRI, avg(pulse_width_us) as PW, avg(c.frequency_hz) as FREQ 
                                    from signals s join emitters e on s.emitter_id = e.id 
                                    join clusters c on c.signal_id = s.id 
                                    group by(s.emitter_id) 
                                    order by s.emitter_id asc 
                                    ) where PW between %(pw_start)s and %(pw_end)s
                                and FREQ between %(freq_start)s and %(freq_end)s
                                and PRI between %(pri_start)s and %(pri_end)s 
                                """
                    params = {
                        'pw_start': nav2["pulse_width_micro_seconds"] - 0.1,
                        'pw_end': nav2["pulse_width_micro_seconds"] + 0.1,
                        'freq_start': nav2["frequency"][0],
                        'freq_end': nav2["frequency"][1],
                        'pri_start': nav2["pulse_repetition_interval"][0],
                        'pri_end': nav2["pulse_repetition_interval"][1],
                    }

                    emitter_1_count = intelligence_db_connector.fetch_scalar(query_emitter_1_count, params)

                    assert_that(emitter_1_count).described_as(f"Emitter: {emitter_name} with this specifications: "
                                                              f"Pulse Width: {params['pw_start']} - {params['pw_end']} "
                                                              f"PRI: {params['pri_start']} - {params['pri_end']} "
                                                              f"Frequency: {params['freq_start']} - {params['freq_end']} "
                                                              f"was not detected by AI model").is_greater_than_or_equal_to(1)

                elif emitter_name.lower() == "nav3":


                    query_emitter_1_count = """
                                select count(*) from ( 
                                    select s.emitter_id ,avg(pulse_repetition_interval_us) as PRI, avg(pulse_width_us) as PW, avg(c.frequency_hz) as FREQ 
                                    from signals s join emitters e on s.emitter_id = e.id 
                                    join clusters c on c.signal_id = s.id 
                                    group by(s.emitter_id) 
                                    order by s.emitter_id asc 
                                    ) where PW between %(pw_start)s and %(pw_end)s
                                and FREQ between %(freq_start)s and %(freq_end)s
                                and PRI between %(pri_start)s and %(pri_end)s 
                                """

                    params = {
                        'pw_start': nav3["pulse_width_micro_seconds"] - 0.1,
                        'pw_end': nav3["pulse_width_micro_seconds"] + 0.1,
                        'freq_start': nav3["frequency"][0],
                        'freq_end': nav3["frequency"][1],
                        'pri_start': nav3["pulse_repetition_interval"][0],
                        'pri_end': nav3["pulse_repetition_interval"][1],
                    }

                    emitter_1_count = intelligence_db_connector.fetch_scalar(query_emitter_1_count, params)

                    assert_that(emitter_1_count).described_as(f"Emitter: {emitter_name} with this specifications: "
                                                              f"Pulse Width: {params['pw_start']} - {params['pw_end']} "
                                                              f"PRI: {params['pri_start']} - {params['pri_end']} "
                                                              f"Frequency: {params['freq_start']} - {params['freq_end']} "
                                                              f"was not detected by AI model").is_greater_than_or_equal_to(1)

        with (allure.step("Assert correct number of emitters was detected")):

            assert_that(no_emitters).described_as(f"Expected number of emitters to be between {number_emitters - std_no_emitters} and {number_emitters + std_no_emitters}, but got {no_emitters}"
                                              ).is_between(number_emitters - std_no_emitters, number_emitters + std_no_emitters)









