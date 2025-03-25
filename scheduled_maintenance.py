# scheduled_maintenance.py
import os
import time
import schedule
import threading
import logging
from datetime import datetime
from token_maintenance import archive_old_transactions, generate_monthly_report

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('scheduled_maintenance.log', mode='a')
    ]
)
logger = logging.getLogger('scheduled_maintenance')

def run_daily_maintenance():
    """Run daily maintenance tasks."""
    logger.info("Running daily maintenance...")
    
    # Archive old transactions (keep 90 days in main table)
    success, message = archive_old_transactions(90)
    logger.info(f"Archive result: {message}")

def run_monthly_maintenance():
    """Run monthly maintenance tasks."""
    logger.info("Running monthly maintenance...")
    
    # Generate report for previous month
    now = datetime.now()
    month = now.month - 1
    year = now.year
    
    if month == 0:
        month = 12
        year -= 1
        
    report = generate_monthly_report(month, year)
    
    if "error" in report:
        logger.error(f"Error generating monthly report: {report['error']}")
    else:
        logger.info(f"Monthly report generated successfully")
        logger.info(f"Total tokens used: {report['summary']['total_tokens_used']}")
        logger.info(f"Total API calls: {report['summary']['total_api_calls']}")

def run_scheduled_tasks():
    """Run scheduled tasks in a separate thread."""
    while True:
        schedule.run_pending()
        time.sleep(60)

def start_scheduler():
    """Set up and start the scheduler."""
    # Schedule daily tasks (run at 2 AM)
    schedule.every().day.at("02:00").do(run_daily_maintenance)
    
    # Schedule monthly tasks (run on the 1st of each month at 3 AM)
    schedule.every().month.at("03:00").do(run_monthly_maintenance)
    
    # Log scheduled tasks
    logger.info("Scheduled maintenance tasks:")
    logger.info("- Daily maintenance: 2:00 AM")
    logger.info("- Monthly maintenance: 3:00 AM on the 1st of each month")
    
    # Start scheduler in a separate thread
    scheduler_thread = threading.Thread(target=run_scheduled_tasks, daemon=True)
    scheduler_thread.start()
    logger.info("Scheduler started")

if __name__ == "__main__":
    # Run as standalone script
    start_scheduler()
    try:
        # Keep main thread alive
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        logger.info("Scheduler stopped by user")