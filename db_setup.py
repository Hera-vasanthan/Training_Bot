import sqlite3
import os

# Absolute DB path (change if needed)
DB_PATH = r"C:\Users\Hera\Desktop\appservicefolder\Chat_bot\resources.db"

# Make sure folder exists
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

# Connect to database
conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

# Create resources table if not exists
cur.execute("""
CREATE TABLE IF NOT EXISTS resources (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    keyword TEXT NOT NULL,
    document_link TEXT,
    video_link TEXT
)
""")

# Clear old rows (optional, for fresh start)
cur.execute("DELETE FROM resources")

# Insert rows
resources_data = [
    # Both doc + video
    ("springboot", r"C:\Users\Hera\Desktop\appservicefolder\docs\springboot_intro.pdf",
     "https://youtu.be/springboot_tutorial"),

    # Only document
    ("java", r"C:\Users\Hera\Desktop\appservicefolder\docs\java_basics.pdf", None),

    # Only video
    ("docker", None, "https://youtu.be/docker_basics"),

    # HR Policy example
    ("hr policy", r"C:\Users\Hera\Desktop\appservicefolder\docs\hr_policy.pdf",
     "https://youtu.be/hr_policy_video"),
]

cur.executemany(
    "INSERT INTO resources (keyword, document_link, video_link) VALUES (?, ?, ?)",
    resources_data
)

# Commit changes
conn.commit()

# Verify insertion
cur.execute("SELECT * FROM resources")
rows = cur.fetchall()
print("Inserted rows:")
for row in rows:
    print(row)

# Close connection
conn.close()
