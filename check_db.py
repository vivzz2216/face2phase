import sqlite3
import os

db_path = 'storage/db/face2phrase.db'
print(f"Database path: {os.path.abspath(db_path)}")
print(f"Database exists: {os.path.exists(db_path)}")
print()

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Get all tables
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = [row[0] for row in cursor.fetchall()]
print("Tables:", tables)
print()

# Check users table
try:
    cursor.execute("SELECT id, username, email, created_at FROM users")
    users = cursor.fetchall()
    print(f"Total users: {len(users)}")
    for user in users:
        print(f"  - id={user[0]}, username={user[1]}, email={user[2]}, created={user[3]}")
except Exception as e:
    print(f"Error checking users: {e}")

print()

# Check analyses table
try:
    cursor.execute("SELECT COUNT(*) FROM analyses")
    analyses_count = cursor.fetchone()[0]
    print(f"Total analyses: {analyses_count}")
    
    if analyses_count > 0:
        cursor.execute("SELECT session_id, user_id, file_name, created_at FROM analyses ORDER BY created_at DESC LIMIT 5")
        print("\nRecent analyses:")
        for row in cursor.fetchall():
            print(f"  - session_id: {row[0][:16] if row[0] else None}..., user_id: {row[1]}, file: {row[2]}, created: {row[3]}")
except Exception as e:
    print(f"Error checking analyses: {e}")

print()

# Check session_summaries table
try:
    cursor.execute("SELECT COUNT(*) FROM session_summaries")
    sessions_count = cursor.fetchone()[0]
    print(f"Total session_summaries: {sessions_count}")
    
    if sessions_count > 0:
        cursor.execute("SELECT session_id, user_id, title, file_type, overall_score, created_at FROM session_summaries ORDER BY created_at DESC LIMIT 5")
        print("\nRecent session summaries:")
        for row in cursor.fetchall():
            print(f"  - session_id: {row[0][:16] if row[0] else None}..., user_id: {row[1]}, title: {row[2]}, type: {row[3]}, score: {row[4]}, created: {row[5]}")
except Exception as e:
    print(f"Error checking session_summaries: {e}")

print()

# Check for user_id=3 (the logged-in user)
try:
    cursor.execute("SELECT COUNT(*) FROM analyses WHERE user_id = 3")
    user_analyses = cursor.fetchone()[0]
    print(f"Analyses for user_id=3: {user_analyses}")
    
    cursor.execute("SELECT COUNT(*) FROM session_summaries WHERE user_id = 3")
    user_sessions = cursor.fetchone()[0]
    print(f"Session summaries for user_id=3: {user_sessions}")
except Exception as e:
    print(f"Error checking user data: {e}")

print()

# Check for orphaned sessions (no user_id)
try:
    cursor.execute("SELECT COUNT(*) FROM session_summaries WHERE user_id IS NULL")
    orphaned_sessions = cursor.fetchone()[0]
    print(f"Orphaned session_summaries (no user_id): {orphaned_sessions}")
    
    if orphaned_sessions > 0:
        cursor.execute("SELECT session_id, title, file_type, created_at FROM session_summaries WHERE user_id IS NULL LIMIT 5")
        print("\nOrphaned sessions (could be migrated):")
        for row in cursor.fetchall():
            print(f"  - session_id: {row[0][:16] if row[0] else None}..., title: {row[1]}, type: {row[2]}, created: {row[3]}")
except Exception as e:
    print(f"Error checking orphaned sessions: {e}")

print()

# Check reports directory for existing session reports
import pathlib
reports_dir = pathlib.Path('storage/reports')
if reports_dir.exists():
    json_files = list(reports_dir.glob('*.json'))
    print(f"Report files in storage/reports/: {len(json_files)}")
    for f in json_files[:10]:
        print(f"  - {f.name}")
else:
    print("Reports directory does not exist!")

conn.close()
