from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib
import json
import os
import re
from assets_data_prep import prepare_data

app = Flask(__name__, template_folder=os.path.abspath('templates'))

try:
    model = joblib.load('trained_model.pkl')
    print("--- Model loaded successfully! ---")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No input data provided'}), 400

        # --- 1. בדיקות תקינות קלט ---
        required_fields = ['startYear', 'runtimeMinutes', 'budget', 'Language', 'Country', 'actor_1', 'writer_1', 'genre_1']
        for field in required_fields:
            if field not in data or data[field] == "":
                return jsonify({'error': f'Missing required field: {field}'}), 400 

        if not (1900 <= int(data['startYear']) <= 2026):
            return jsonify({'error': 'שנת יציאה חייבת להיות בין 1900 ל-2026'}), 400
        if not (1 <= int(data['runtimeMinutes']) <= 300):
            return jsonify({'error': 'אורך סרט חייב להיות בין 1 ל-300 דקות'}), 400

        def is_english_only(val):
            return bool(re.match(r'^[a-zA-Z\s]+$', str(val)))

        if not is_english_only(data['Language']) or not is_english_only(data['Country']):
            return jsonify({'error': 'שדה שפה ומדינה חייבים להכיל אנגלית בלבד'}), 400

        # פונקציית אימות משופרת: פורמט + אורך + כפילויות
        def validate_list_format(val_str, field_name, max_items, check_mn=False):
            try:
                items = json.loads(val_str)
                if not isinstance(items, list): return False, "חייב להיות רשימה"
                if len(items) > max_items: return False, f"ניתן להזין לכל היותר {max_items} ערכים"
                
                # בדיקת כפילויות
                if len(set(items)) != len(items):
                    return False, f"קיימים ערכים כפולים בשדה {field_name}"
                
                for item in items:
                    if check_mn and not re.match(r'^mn\d{7}$', str(item)):
                        return False, f"ערך '{item}' אינו תקין (נדרש mn + 7 ספרות)"
                    if not check_mn and not is_english_only(item):
                        return False, f"ערך '{item}' אינו תקין (נדרש אנגלית)"
                return True, ""
            except: return False, "פורמט JSON לא תקין"

        # אימות שדות
        ok, msg = validate_list_format(data['actor_1'], "שחקנים", 5, check_mn=True)
        if not ok: return jsonify({'error': msg}), 400
        ok, msg = validate_list_format(data['writer_1'], "כותבים", 2, check_mn=True)
        if not ok: return jsonify({'error': msg}), 400
        ok, msg = validate_list_format(data['genre_1'], "ז'אנרים", 3, check_mn=False)
        if not ok: return jsonify({'error': msg}), 400

        # --- 2. הכנת הנתונים ---
        parsed_data = {
            'startYear': data['startYear'], 'runtimeMinutes': data['runtimeMinutes'],
            'budget': data['budget'], 'Language': str(data['Language']),
            'Country': str(data['Country']), 'tconst': 'new_movie', 'plot': '', 'directors': ''
        }

        def parse_list_field(field_input, max_items):
            items = json.loads(field_input)
            return [items[i] if i < len(items) else "" for i in range(max_items)]

        for i, act in enumerate(parse_list_field(data['actor_1'], 5), 1): parsed_data[f'actor_{i}'] = act
        for i, writ in enumerate(parse_list_field(data['writer_1'], 2), 1): parsed_data[f'writer_{i}'] = writ
        for i, gen in enumerate(parse_list_field(data['genre_1'], 3), 1): parsed_data[f'genre_{i}'] = gen

        df_input = pd.DataFrame([parsed_data])
        for col in df_input.columns:
            if col not in ['startYear', 'runtimeMinutes', 'budget']:
                df_input[col] = df_input[col].fillna('').astype(str)

        df_processed = prepare_data(df_input)
        boruta_features = [
            'startYear', 'runtimeMinutes', 'budget', 'Language', 'Country',
            'actor_1', 'actor_2', 'actor_3', 'actor_4', 'actor_5', 
            'writer_1', 'writer_2', 'genre_1', 'genre_2', 'genre_3', 
            'my_features_1_ActorGenerInteraction', 'my_features_2_runtime_range', 
            'myfeatures_4_runtime_feat1', 'myfeatures_3_year_budget', 'my_features_5_no_leader_missing' 
        ]
        
        for col in boruta_features:
            if col not in df_processed.columns: df_processed[col] = 0 if 'my_features' in col else ''
        df_processed = df_processed[boruta_features]

        prediction = model.predict(df_processed)
        return jsonify({'predicted_rating': round(float(prediction[0]), 1)}), 200

    except Exception as e:
        return jsonify({'error': f'Internal Server Error: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)