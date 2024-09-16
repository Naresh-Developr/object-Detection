import os
from datetime import datetime

# Function to determine the session based on the current time
def get_session():
    current_hour = datetime.now().hour
    if current_hour < 12:
        return 'FN'
    elif 12 <= current_hour < 17:
        return 'AN'
    else:
        return 'Evening'

# Function to append data to a CSV file
def append_to_csv(room_number, df):
    # Ensure 'csv' directory exists
    if not os.path.exists("csv"):
        os.makedirs("csv")
    
    # File name format: {room_number}_report.csv
    file_path = f"csv/{room_number}_report.csv"
    
    # Add date and session to the DataFrame
    current_date = datetime.now().strftime("%Y-%m-%d")
    session = get_session()
    df['Date'] = current_date
    df['Session'] = session
    
    # Append to the CSV file, create it if it doesn't exist
    if not os.path.exists(file_path):
        df.to_csv(file_path, mode='w', index=False)
    else:
        df.to_csv(file_path, mode='a', header=False, index=False)
