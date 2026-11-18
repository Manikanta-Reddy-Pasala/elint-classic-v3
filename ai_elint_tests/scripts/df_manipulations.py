import pandas as pd

# 1. Load your CSV
df = pd.read_csv("psr1_psr3_best-0_plus10_PDWTable.csv")

# 2. Define your baseline target date
base_time = pd.Timestamp("2026-10-24 02:06:06")

# 3. Calculate the running total (cumulative sum) of the nanoseconds
cumulative_ns = df["time_of_arrival_ns"].cumsum()

# 4. Add the cumulative nanoseconds to the baseline time
cumulative_datetimes = base_time + pd.to_timedelta(cumulative_ns, unit="ns")

# 5. Overwrite the original column with the formatted strings
df["time_of_arrival_ns"] = cumulative_datetimes.apply(
    lambda ts: f"{ts.strftime('%Y-%m-%d %H:%M:%S')}.{ts.nanosecond:09d}"
)

# 2. Define the new column name
col_name = "is_cw"

# 3. Check if the column already exists; if not, add it
if col_name not in df.columns:
  # Example: Add a new column with default values or calculations
  df[col_name] = 'false'

  # Save the updated DataFrame back to the CSV
  df.to_csv("updated_file.csv", index=False)
  print(f"Column '{col_name}' added successfully.")
else:
  print(f"Column '{col_name}' already exists. No changes made.")

# 2. Define the new column name
col_name = "quality_percents"

# 3. Check if the column already exists; if not, add it
if col_name not in df.columns:
  # Example: Add a new column with default values or calculations
  df[col_name] = 100

  # Save the updated DataFrame back to the CSV
  df.to_csv("updated_file.csv", index=False)
  print(f"Column '{col_name}' added successfully.")
else:
  print(f"Column '{col_name}' already exists. No changes made.")


# 2. Define the new column name
col_name = "modulation"

# 3. Check if the column already exists; if not, add it
if col_name not in df.columns:
  # Example: Add a new column with default values or calculations
  df[col_name] = ''

  # Save the updated DataFrame back to the CSV
  df.to_csv("updated_file.csv", index=False)
  print(f"Column '{col_name}' added successfully.")
else:
  print(f"Column '{col_name}' already exists. No changes made.")

df.rename(columns={"sidelobe_blanking_flag_as_int8": "sidelobe_blanking_flag"}, inplace=True)


sidelobe_blanking_flag_map = {
    0: "UNKNOWN",
    1: "MAINLOBE",
    2: "SIDELOBE",
    3: "MISSING_OMNI",
    4: "MISSING_SDF",
    5: "OMNI"
}

df["sidelobe_blanking_flag"] = df["sidelobe_blanking_flag"].map(sidelobe_blanking_flag_map)


# 6. Save the updated data
df.to_csv("updated_file.csv", index=False)
