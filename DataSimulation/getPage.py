import urllib.request
import schedule
import time

def page():
    p = urllib.request.urlopen('https://www.youtube.com/')
    print("Request done")

schedule.every(3).minutes.do(page) 

# Loop so that the scheduling task 
# keeps on running all time. 
while True: 
  
    # Checks whether a scheduled task  
    # is pending to run or not 
    schedule.run_pending() 
    time.sleep(1) 