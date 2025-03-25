# token_maintenance.py
import os
import json
import argparse
from datetime import datetime, timedelta
from db_pool import DatabaseConnection
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('token_maintenance.log', mode='a')
    ]
)
logger = logging.getLogger('token_maintenance')

def archive_old_transactions(days=90):
    """
    Archive token transactions older than specified days.
    
    Args:
        days: Number of days to keep in main table
        
    Returns:
        tuple: (success, message)
    """
    try:
        with DatabaseConnection() as conn:
            cursor = conn.cursor()
            
            # Get DB_ENVIRONMENT from config
            db_env = os.environ.get("DB_ENVIRONMENT", "development")
            
            # Create archive table if it doesn't exist
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {db_env}.token_transactions_archive (
                    id INTEGER NOT NULL,
                    api_key_id INTEGER NOT NULL,
                    amount INTEGER NOT NULL,
                    transaction_type TEXT NOT NULL,
                    endpoint TEXT,
                    description TEXT,
                    created_at TIMESTAMP,
                    request_id TEXT,
                    client_ip TEXT,
                    metadata JSONB,
                    archived_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Archive old records
            cutoff_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
            
            cursor.execute(f"""
                INSERT INTO {db_env}.token_transactions_archive
                    (id, api_key_id, amount, transaction_type, endpoint, description, 
                     created_at, request_id, client_ip, metadata)
                SELECT 
                    id, api_key_id, amount, transaction_type, endpoint, description, 
                    created_at, request_id, client_ip, metadata
                FROM {db_env}.token_transactions
                WHERE created_at < %s
            """, (cutoff_date,))
            
            archived_count = cursor.rowcount
            
            # Delete archived records from main table
            cursor.execute(f"""
                DELETE FROM {db_env}.token_transactions
                WHERE created_at < %s
            """, (cutoff_date,))
            
            deleted_count = cursor.rowcount
            
            conn.commit()
            
            message = f"Archived {archived_count} records, deleted {deleted_count} records older than {days} days"
            logger.info(message)
            return True, message
    except Exception as e:
        logger.error(f"Error archiving old transactions: {e}")
        return False, f"Error: {str(e)}"

def generate_monthly_report(month=None, year=None):
    """
    Generate monthly token usage report.
    
    Args:
        month: Month number (1-12, default: previous month)
        year: Year (default: current year or previous year if month is 12)
        
    Returns:
        dict: Report data
    """
    now = datetime.now()
    
    # Default to previous month
    if month is None:
        month = now.month - 1
        if month == 0:
            month = 12
            year = now.year - 1
    
    if year is None:
        year = now.year
        if month == 12 and now.month == 1:
            year = now.year - 1
    
    try:
        with DatabaseConnection() as conn:
            cursor = conn.cursor()
            
            # Calculate date range
            start_date = datetime(year, month, 1).strftime('%Y-%m-%d')
            if month == 12:
                end_date = datetime(year + 1, 1, 1).strftime('%Y-%m-%d')
            else:
                end_date = datetime(year, month + 1, 1).strftime('%Y-%m-%d')
            
            # Get client usage statistics
            cursor.execute("""
                SELECT 
                    k.id as api_key_id,
                    k.client_name,
                    SUM(CASE WHEN t.transaction_type = 'purchase' THEN t.amount ELSE 0 END) as tokens_purchased,
                    SUM(CASE WHEN t.transaction_type = 'usage' THEN ABS(t.amount) ELSE 0 END) as tokens_used,
                    COUNT(CASE WHEN t.transaction_type = 'usage' THEN 1 ELSE NULL END) as api_calls
                FROM api_keys k
                LEFT JOIN token_transactions t ON k.id = t.api_key_id
                WHERE (t.created_at >= %s AND t.created_at < %s) OR t.created_at IS NULL
                GROUP BY k.id, k.client_name
                ORDER BY tokens_used DESC
            """, (start_date, end_date))
            
            client_usage = []
            for row in cursor.fetchall():
                client_usage.append({
                    "api_key_id": row[0],
                    "client_name": row[1],
                    "tokens_purchased": row[2] or 0,
                    "tokens_used": row[3] or 0,
                    "api_calls": row[4] or 0
                })
            
            # Get endpoint usage statistics
            cursor.execute("""
                SELECT 
                    endpoint,
                    SUM(ABS(amount)) as tokens_used,
                    COUNT(*) as call_count
                FROM token_transactions
                WHERE transaction_type = 'usage'
                AND created_at >= %s AND created_at < %s
                GROUP BY endpoint
                ORDER BY tokens_used DESC
            """, (start_date, end_date))
            
            endpoint_usage = []
            for row in cursor.fetchall():
                endpoint_usage.append({
                    "endpoint": row[0],
                    "tokens_used": row[1],
                    "call_count": row[2]
                })
            
            # Generate report
            report = {
                "period": {
                    "year": year,
                    "month": month,
                    "start_date": start_date,
                    "end_date": end_date
                },
                "summary": {
                    "total_clients": len(client_usage),
                    "total_tokens_purchased": sum(c["tokens_purchased"] for c in client_usage),
                    "total_tokens_used": sum(c["tokens_used"] for c in client_usage),
                    "total_api_calls": sum(c["api_calls"] for c in client_usage)
                },
                "client_usage": client_usage,
                "endpoint_usage": endpoint_usage,
                "generated_at": datetime.now().isoformat()
            }
            
            # Log the report was generated
            report_file = f"token_report_{year}_{month:02d}.json"
            with open(report_file, "w") as f:
                json.dump(report, f, indent=2)
                
            logger.info(f"Monthly report generated: {report_file}")
            return report
    except Exception as e:
        logger.error(f"Error generating monthly report: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Token system maintenance tasks")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Archive command
    archive_parser = subparsers.add_parser("archive", help="Archive old token transactions")
    archive_parser.add_argument("--days", type=int, default=90, help="Days to keep in main table")
    
    # Report command
    report_parser = subparsers.add_parser("report", help="Generate monthly usage report")
    report_parser.add_argument("--month", type=int, help="Month (1-12)")
    report_parser.add_argument("--year", type=int, help="Year")
    
    args = parser.parse_args()
    
    if args.command == "archive":
        success, message = archive_old_transactions(args.days)
        print(message)
    elif args.command == "report":
        report = generate_monthly_report(args.month, args.year)
        if "error" in report:
            print(f"Error generating report: {report['error']}")
        else:
            print(f"Report generated successfully")
            print(f"Total tokens used: {report['summary']['total_tokens_used']}")
            print(f"Total API calls: {report['summary']['total_api_calls']}")
    else:
        parser.print_help()