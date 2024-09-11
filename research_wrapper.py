import subprocess
import time
import sys

def run_research_script():
    max_retries = sys.maxsize
    retry_delay = 20

    for attempt in range(1, max_retries + 1):
        print(f"Attempt {attempt}")
        if run.status = 
        
        try:
            process = subprocess.Popen([sys.executable, "-u", "research.py"], )  
            
            for line in iter(process.stdout.readline, ''):
                print(line, end='', flush=True)
                
                if "run cancelled" in line.lower():
                    print("\nDetected 'run cancelled'. Restarting the script...")
                    process.terminate()
                    break
            
            process.wait()
            
            if process.returncode == 0:
                print("\nResearch script completed successfully.")
                break
            else:
                print(f"\nResearch script exited with code {process.returncode}")
        
        except Exception as e:
            print(f"\nAn unexpected error occurred: {str(e)}")
        
        print(f"Restarting in {retry_delay} seconds...")
        time.sleep(retry_delay)

    else:
        print(f"\nScript has run {max_retries} times. This should never be reached with sys.maxsize.")
        sys.exit(1)

if __name__ == "__main__":
    run_research_script()