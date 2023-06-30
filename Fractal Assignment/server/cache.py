import sqlite3
conn = sqlite3.connect('cache.db')

conn.execute('''
create table if not exists 
section_summaries
(section_id integer primary key autoincrement,
section_summary text,
actual_text text)
''')

def get_all_summaries():
    cursor = conn.execute("SELECT section_summary FROM section_summaries")
    result = cursor.fetchall()
    section_summaries = [row[0] for row in result]
    joined_summaries = '\n'.join(section_summaries)
    
    return joined_summaries

if __name__ == '__main__':
    print(get_all_summaries())