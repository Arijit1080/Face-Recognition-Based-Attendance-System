from flask import Flask, render_template, request
import sqlite3
import os
from datetime import datetime
app = Flask(__name__)

@app.route('/')
def index():
    default_date = datetime.now().strftime('%Y-%m-%d')
    return render_template('home.html', default_date=default_date)

@app.route('/show_attendance', methods=['GET'])
def show_attendance():
    selected_date = request.args.get('date')
    selected_date = datetime.strptime(selected_date, '%Y-%m-%d').strftime('%d-%m-%Y')  # Convert to dd-mm-yyyy format

    db_filename = "attendance.db"
    conn = sqlite3.connect(db_filename)
    c = conn.cursor()

    table_name = f"Attendance_{selected_date.replace('-', '_')}"

    if os.path.isfile(db_filename):
        c.execute(f'''
            CREATE TABLE IF NOT EXISTS {table_name} (
                enrollment TEXT UNIQUE,
                time TEXT
            )
        ''')

        c.execute(f"SELECT * FROM {table_name}")
        attendance_data = c.fetchall()

        conn.close()

        return render_template('attendance.html', date=selected_date, attendance_data=attendance_data)
    else:
        return "Database file not found."


if __name__ == '__main__':
    app.run()
