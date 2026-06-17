import pandas as pd  
import numpy as np  
import re
import matplotlib.pyplot as plt  
import seaborn as sns  
import ppscore as pps  
import joblib
import ast
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV, KFold
from sklearn.pipeline import Pipeline  
from sklearn.impute import SimpleImputer  
from sklearn.preprocessing import StandardScaler, OneHotEncoder  
from sklearn.compose import ColumnTransformer  
from sklearn.linear_model import HuberRegressor
from sklearn.linear_model import ElasticNet   
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error, mean_absolute_error  
from sklearn.inspection import permutation_importance

#---------------------------------------------------------------------------------------פעולות----------------------------------------------------------------------
def prepare_data(df):
    df = df.copy()

    # === התאמה מיוחדת עבור חיזוי בזמן אמת מהאתר ===
    # אם הקלט מגיע מה-API (סרט בודד), נבנה את המבנה שהפונקציה מצפה לו ונמנע מקריאת ה-TSV
    is_live_prediction = ('tconst' in df.columns and df['tconst'].iloc[0] == 'new_movie') or (len(df) == 1)

    if is_live_prediction:
        # אם הגיעו שדות מפורקים מה-API, נוודא שהם קיימים בפורמט טקסט עבור המשך הפונקציות
        if 'writer_1' in df.columns:
            df['writers'] = df['writer_1'].fillna('')
        if 'genre_1' in df.columns:
            # במידה והגיעו ז'אנרים מפורקים, נחבר אותם זמנית כדי שפונקציית clean_genres_column תפרק מחדש בצורה אחידה
            genre_cols = [df[c].iloc[0] for c in ['genre_1', 'genre_2', 'genre_3'] if c in df.columns and pd.notna(df[c].iloc[0])]
            df['genres'] = ",".join([str(g) for g in genre_cols])
        if 'directors' not in df.columns:
            df['directors'] = None
        if 'plot' not in df.columns:
            df['plot'] = ''
    else:
        # הוספת עמודות שלא קיימות בדאטה שקיבלנו (מתבצע רק בזמן אימון על כל הדאטאסט)
        try:
            df_crew = pd.read_csv("title.crew.tsv.gz", sep='\t', usecols=['tconst', 'directors', 'writers'])
            df = pd.merge(df, df_crew, on='tconst', how='left')
        except FileNotFoundError:
            # הגנה במקרה וקובץ ה-TSV לא נמצא בסביבת הריצה של ה-API
            if 'directors' not in df.columns: df['directors'] = None
            if 'writers' not in df.columns: df['writers'] = None

    # פונקציית ניקוי התא מערכי טקסט לעמודת תקציב
    def clean_budget_advanced(value):
        if pd.isna(value) or str(value).lower() in ['nan', 'unknown', '']: 
            return 0.0 
       
        val_str = str(value).lower() 
        rates = {'£': 1.3, '€': 1.1, '₹': 0.012, 'rs': 0.012, '¥': 0.007, 'hk$': 0.13, 'a$': 0.65, 'ca$': 0.73, 'sek': 0.09} 
        multipliers = {'million': 1000000, 'billion': 1000000000, 'crore': 10000000, 'lakh': 100000, 'mio': 1000000} 
       
        current_rate = 1.0 
        for symbol, rate in rates.items(): 
            if symbol in val_str: 
                current_rate = rate 
                break
               
        current_multiplier = 1.0 
        for word, mult in multipliers.items(): 
            if word in val_str: 
                current_multiplier = mult 
                break
               
        numbers = re.findall(r"[-+]?\d*\.\d+|\d+", val_str.replace(',', '')) 
       
        try:
            if numbers:
                return float(numbers[0]) * current_rate * current_multiplier 
            return 0.0 
        except:
            return 0.0 

    # ניקוי כל הדאטה פריים מתווים שהם לא אנגלית או מספר
    def clean_non_latin_chars(df_to_clean, columns):
        df_cleaned = df_to_clean.copy()
        for col in columns:
            if col in df_cleaned.columns:            
                df_cleaned[col] = df_cleaned[col].astype(str).str.replace(r'[^a-zA-Z0-9\s]', '', regex=True)
                df_cleaned[col] = df_cleaned[col].str.strip()
                df_cleaned[col] = df_cleaned[col].str.replace(r'\s+', ' ', regex=True)
        return df_cleaned

    # טקסט קטן
    def lower(df_to_lower):
        columns_to_lower = ['primaryTitle', 'genres', 'Language', 'Country', 'plot','directors', 'writers'] 
        for col in columns_to_lower: 
             if col in df_to_lower.columns:  
                    df_to_lower[col] = df_to_lower[col].astype(str).str.lower()
        return df_to_lower

    # עמודת תקציר החלפה לנאן
    def replace_with_nan(df_to_mod, column_name, values_to_replace):
        df_mod = df_to_mod.copy()
        if column_name in df_mod.columns:
            df_mod.loc[df_mod[column_name].isin(values_to_replace), column_name] = np.nan    
        return df_mod 

    # עמודת שנת הסרט עד 1900 יחשב כ-נאל
    def start_year(df_to_clean, column_name):
        df_cleaned = df_to_clean.copy()
        if column_name in df_cleaned.columns:
            # המרה למספר בבטחה
            df_cleaned[column_name] = pd.to_numeric(df_cleaned[column_name], errors='coerce')
            df_cleaned.loc[df_cleaned[column_name] < 1900, column_name] = np.nan  
        return df_cleaned

    # עמודת שחקנים המרת השמות לעמודות נפרדות
    def process_actors(val):
        if pd.isna(val) or val is None:
            return [None] * 5
        try:
            if isinstance(val, str):
                val = val.strip()
                if val.startswith('['):
                    val = ast.literal_eval(val)
                else:
                    val = [val]
            if isinstance(val, list):
                return (val[:5] + [None] * 5)[:5]
        except:
            pass
        return [None] * 5

    # עמודת כותבים חלוקה ל-2 עמודות
    def split_string_column(dataframe, column_name, prefix, max_cols=2):
        def process_row(val):
            if pd.isna(val) or val == '':
                return [None] * max_cols
            if isinstance(val, list):
                parts = val
            else:
                parts = [item.strip() for item in str(val).split(',') if item.strip()]
            return (parts[:max_cols] + [None] * max_cols)[:max_cols]

        cols_to_drop = [c for c in dataframe.columns if c.startswith(f"{prefix}_") and c not in ['writer_1', 'writer_2']]
        dataframe = dataframe.drop(columns=cols_to_drop, errors='ignore')
        
        if column_name in dataframe.columns:
            new_data = dataframe[column_name].apply(process_row).tolist()
        else:
            new_data = [[dataframe[f'{prefix}_1'].iloc[0] if f'{prefix}_1' in dataframe.columns else None, 
                         dataframe[f'{prefix}_2'].iloc[0] if f'{prefix}_2' in dataframe.columns else None]]

        new_cols_names = [f"{prefix}_{i+1}" for i in range(max_cols)]
        expanded_df = pd.DataFrame(new_data, columns=new_cols_names, index=dataframe.index)
        
        # דריסת העמודות הישנות עם המעובדות
        for col in new_cols_names:
            if col in dataframe.columns: dataframe = dataframe.drop(columns=[col])
            
        return pd.concat([dataframe, expanded_df], axis=1)

    # עמודת סוגות - סוגה בכל עמודה
    def clean_genres_column(df_genres):
        if 'genres' in df_genres.columns:
            df_genres['genres'] = df_genres['genres'].astype(str).str.replace(r"[\[\]\"']", '', regex=True)
            split_genres = df_genres['genres'].str.split(',', n=3, expand=True)
            max_columns = min(split_genres.shape[1], 3)
        
            for i in range(3):
                if i < max_columns:
                    df_genres[f'genre_{i+1}'] = split_genres[i].str.strip()
                else:
                    if f'genre_{i+1}' not in df_genres.columns:
                        df_genres[f'genre_{i+1}'] = None
            df_genres = df_genres.drop(columns=['genres'], errors='ignore') 
        return df_genres

    # ניקוי תווים בלתי נראים
    def clean_text_for_ml(df_text):
        df_text = df_text.copy()
        text_cols = df_text.select_dtypes(include=['object', 'string']).columns
        for col in text_cols:
            df_text[col] = df_text[col].fillna('').astype(str)
            df_text[col] = df_text[col].str.replace(r'[\x00-\x1F\x7F-\x9F\u200b-\u200d\ufeff]', '', regex=True)
            df_text[col] = df_text[col].str.replace(r'\s+', ' ', regex=True)
            df_text[col] = df_text[col].str.strip()
            df_text[col] = df_text[col].str.lower()
        return df_text

    # ערכי נאל זהיים
    def standardize_missing_values(df_miss):
        df_miss = df_miss.copy()  
        for col in df_miss.select_dtypes(include=["object", "string"]).columns:  
            df_miss[col] = (
                df_miss[col]
                .astype(str)  
                .str.replace(r"\n", " ", regex=True)  
                .str.replace(r"\\n", " ", regex=True)  
                .str.replace(r"\\", " ", regex=True)  
                .str.strip() 
            )

        missing_variants = ["n/", "nan", "Nan", "NAN", "none", "None", "null", "Null", "NULL", "", " ", "N/A", "n/a", "NA", "na", "-"]
        df_miss = df_miss.replace(missing_variants, np.nan)  
        return df_miss  

    # הוספת הפיצ'רים המותאמים (Custom Features)
    def add_my_custom_features(df_custom):
        df_custom = df_custom.copy()      
        actor_cols = [c for c in df_custom.columns if c.startswith("actor_")]  
        genre_cols = [c for c in df_custom.columns if c.startswith("genre_")]  

        def is_missing(val):
            val_str = str(val).lower().strip()  
            return val_str in ["nan", "none", "null", "", "n/a", "-", "nat"]

        # חישוב זמני של כמות שחקנים וז'אנרים
        act_count = df_custom[actor_cols].map(is_missing).eq(False).sum(axis=1)
        gen_count = df_custom[genre_cols].map(is_missing).eq(False).sum(axis=1)

        # F1 - אינטראקציה בין שחקנים לז'אנרים
        df_custom["my_features_1_ActorGenerInteraction"] = act_count * gen_count

        # F2 - פיצ'ר בינארי לטווח אורך סרט אופטימלי (120 עד 140 דקות) 
        df_custom["runtimeMinutes"] = pd.to_numeric(df_custom["runtimeMinutes"], errors='coerce').fillna(0)
        df_custom["my_features_2_runtime_range"] = df_custom["runtimeMinutes"].between(120, 140).astype(int)

        # F3 - תקציב ושנת יציאה בינארי 
        df_custom["budget"] = pd.to_numeric(df_custom["budget"], errors='coerce').fillna(0)
        df_custom["startYear"] = pd.to_numeric(df_custom["startYear"], errors='coerce').fillna(0)
        df_custom["myfeatures_3_year_budget"] = ((df_custom["budget"] > 5000000) & (df_custom["startYear"] > 1970)).astype(int)

        # F4 - יחס זמן ריצה ופיצר 1
        df_custom["myfeatures_4_runtime_feat1"] = np.where(
            df_custom["my_features_1_ActorGenerInteraction"] == 0, 
            0, 
            df_custom["runtimeMinutes"] / df_custom["my_features_1_ActorGenerInteraction"]
        ) 

        # F5 - בדיקה האם גם הכותב הראשי וגם השחקן הראשי חסרים
        w1_miss = df_custom["writer_1"].map(is_missing) if "writer_1" in df_custom.columns else pd.Series([True]*len(df_custom))
        a1_miss = df_custom["actor_1"].map(is_missing) if "actor_1" in df_custom.columns else pd.Series([True]*len(df_custom))
        df_custom["my_features_5_no_leader_missing"] = (w1_miss & a1_miss).astype(int)

        return df_custom 

    #---------------------------------------------------------------------------------------קוד הפעלת הפעולות----------------------------------------------------------------------
    
    # הסרת עמודות שאינן ידועות לפני יציאת הסרט
    df = df.drop(['numVotes', 'BoxOffice'], axis=1, errors='ignore') 
    df = df.drop_duplicates()
   
    # תקציב
    df['budget'] = df['budget'].apply(clean_budget_advanced) 
    
    # ניקוי עמודות טקסט
    columns_to_fix = ['tconst', 'primaryTitle' ,'Language', 'Country' ,'plot']
    df = clean_non_latin_chars(df, columns_to_fix) 
 
    # אותיות קטנות
    df = lower(df)
   
    # עמודת תקציר החלפת ערכים ל-NaN
    text_replacements = ["no plot found" , "plot description missing"]
    df = replace_with_nan(df, 'plot', text_replacements)
    
    # עמודת שנת התחלה גדולה מ-1900
    df = start_year(df, 'startYear')
        
    # עיבוד עמודות שחקנים (במידה והקלט הגיע מאימון ולא מהאתר)
    if not is_live_prediction and 'lead_actors_ids' in df.columns:
        cols_to_drop = [c for c in df.columns if c.startswith('actor_') and c != 'lead_actors_ids']
        df = df.drop(columns=cols_to_drop, errors='ignore')
        new_data = df['lead_actors_ids'].apply(process_actors).tolist()
        actors_df = pd.DataFrame(new_data, columns=[f'actor_{i+1}' for i in range(5)], index=df.index)
        df = pd.concat([df, actors_df], axis=1)
        df = df.drop(['lead_actors_ids'] , axis=1, errors='ignore')

    # כותבים
    df = split_string_column(df, 'writers', prefix='writer', max_cols=2)
    df = df.drop(['writers'] , axis=1, errors='ignore')
    
    # במאים
    if 'directors' in df.columns:
        df['directors'] = df['directors'].astype(str).str.split(',').str[0].str.strip()
      
    # הרצת ג'אנרים
    df = clean_genres_column(df)

    # תווים בלתי נראים
    df = clean_text_for_ml(df)
      
    # ערכים חסרים
    df = standardize_missing_values(df)
       
    # הוספת הפיצ'רים שלי
    df = add_my_custom_features(df)
    
    return df