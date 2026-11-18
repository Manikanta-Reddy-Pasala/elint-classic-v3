import pandas as pd

# Load the CSV files
df1 = pd.read_csv('psr1_psr2_best-0_plus0.csv')
df2 = pd.read_csv('psr3_psr4_best-0_plus0.csv')

df2['emitter_id'] = df2['emitter_id'] + 2


# Stack files vertically (Union)
combined_df = pd.concat([df1, df2], ignore_index=True)

# Sort by the numerical column
sorted_df = combined_df.sort_values(by='time_of_arrival_ns', ascending=True)

# Save to a new CSV file
sorted_df.to_csv('psr1_psr2_psr3_psr4_best_0_plus0.csv', index=False)