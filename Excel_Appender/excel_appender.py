import pandas as pd
import os

def append_excel_files(input_folder, output_file):
    # List to store data from each file
    data_frames = []

    # Read all Excel files in the specified folder
    for file in sorted(os.listdir(input_folder)):
        if file.endswith(".xlsx") or file.endswith(".xls"):
            file_path = os.path.join(input_folder, file)
            df = pd.read_excel(file_path)
            row_count = len(df)  # Count the number of rows in the file
            print(f"Reading file: {file} with {row_count} rows")
            data_frames.append(df)

    # Concatenate all data frames
    combined_data = pd.concat(data_frames, ignore_index=True)

    # Print the total number of rows in the combined file
    print(f"Total rows in combined data: {len(combined_data)}")

    # Write to a new Excel file
    combined_data.to_excel(output_file, index=False)
    print(f"Appended Excel file saved as: {output_file}")


input_path = "Excel_Files_To_Append"
output_path = "Output/combined.xlsx"
append_excel_files(input_path, output_path)