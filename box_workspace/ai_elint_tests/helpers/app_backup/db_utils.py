import pandas as pd

def truncate_tables(client, pg_conn):
    client.query('truncate table "signals"."clusters"')
    client.query('truncate table "signals"."csv_input"')

    execute_query_postgres(pg_conn, 'truncate table clusters cascade;')
    execute_query_postgres(pg_conn, 'truncate table emitter_scan_rate_estimation cascade;')
    execute_query_postgres(pg_conn, 'truncate table emitters cascade;')
    execute_query_postgres(pg_conn, 'truncate table signals cascade;')

def create_populate_csv_input_table(filename, client):
    df_full = pd.read_csv(filename)
    df_full["id"] = range(1, len(df_full) + 1)

    df_full.to_csv(filename, index=False)

    client.insert_df('csv_input', df_full)

def execute_query_postgres(pg_conn, query):
    cur = pg_conn.cursor()

    # 3. Execute the query using the cursor
    cur.execute(query)

    # # 4. Fetch the results
    # rows = cur.fetchall()
    # for row in rows:
    #     print(row)

    # 5. Clean up
    cur.close()

def create_csv_input_table(
    client
) -> None:


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
            """
        )

        client.query(query)