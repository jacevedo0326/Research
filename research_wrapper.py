import Research
import time
import sys
import os
import multiprocessing

def run_research():
    # Assuming Research module has these global variables
    Research.failed_count = 0
    Research.max_failed_count = 10
    Research.file_count = 0
    Research.total_files = len(Research.selectedFiles)

    for fileName in Research.selectedFiles:
        try:
            # Process each file
            Research.process_file(fileName)
        except Exception as e:
            print(f"Error processing file: {str(e)}")
            Research.failed_count += 1
            if Research.failed_count >= Research.max_failed_count:
                print(f"Encountered {Research.max_failed_count} consecutive failures.")
                return False
    return True

def run_with_timeout(timeout):
    process = multiprocessing.Process(target=run_research)
    process.start()

    process.join(timeout)

    if process.is_alive():
        print(f"Research processing is taking too long. Terminating...")
        process.terminate()
        process.join()
        return False
    return True

def run_research_script():
    max_attempts = 3
    attempt = 0
    timeout = 3600  # 1 hour timeout, adjust as needed

    while True:
        try:
            print(f"Starting Research script - Attempt {attempt + 1}")
            
            # Run the research function with a timeout
            completed = run_with_timeout(timeout)

            if not completed:
                print("Research processing timed out.")
                attempt += 1
            elif not Research.selectedFiles:
                print('All files in the folder have been processed.')
                break
            else:
                # Reset attempt count if we've successfully processed some files
                attempt = 0
                continue

            if attempt >= max_attempts:
                print(f"Max attempts ({max_attempts}) reached. Exiting.")
                break

            time.sleep(5)  # Wait for 5 seconds before restarting

        except Exception as e:
            attempt += 1
            if attempt >= max_attempts:
                print(f"Max attempts ({max_attempts}) reached. Exiting.")
                break
            print(f"An error occurred: {str(e)}. Restarting...")
            time.sleep(5)  # Wait for 5 seconds before restarting

    print("Script execution completed.")

if __name__ == "__main__":
    run_research_script()