import csv
import os

file = "kit_audio_extract_2min_hf.rttm"

def convert_rttm_to_csv(rttm_file_path):
    # Define the output CSV file path
    csv_file_path = os.path.splitext(rttm_file_path)[0] + '.csv'
    
    with open(rttm_file_path, 'r') as rttm_file, open(csv_file_path, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        
        # Write header based on RTTM fields
        csv_writer.writerow(['Type', 'File ID', 'Channel ID', 'Turn Onset', 'Turn Duration', 
                             'Orthography', 'Speaker Type', 'Speaker Name', 'Confidence Score', 'Signal Lookahead Time'])
        
        for line in rttm_file:
            # Split each line by whitespace
            fields = line.strip().split()
            # Write fields to CSV
            csv_writer.writerow(fields)
    
    print(f"RTTM data has been converted to CSV and saved as {csv_file_path}")

# Example usage
convert_rttm_to_csv(file)

