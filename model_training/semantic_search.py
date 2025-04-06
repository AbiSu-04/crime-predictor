import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch

# Load sentence transformer model
model = SentenceTransformer('all-mpnet-base-v2')

CSV_FILE_PATH = "model_training/crime-data-from-2010-to-present.csv"

def load_and_preprocess_data(n_rows=10000):
    df = pd.read_csv(CSV_FILE_PATH, nrows=n_rows)
    df.columns = [col.lower().replace(' ', '_') for col in df.columns]

    column_mapping = {
        'dr_no': 'event_id',
        'date_occ': 'event_time',
        'area_name': 'event_location',
        'crm_cd_desc': 'event_text',
        'vict_descent': 'event_person'
    }

    alternatives = {
        'dr_no': ['dr_number', 'dr', 'report_no'],
        'date_occ': ['date_occurred', 'date', 'occurred_date', 'date_reported'],
        'area_name': ['area', 'location', 'district'],
        'crm_cd_desc': ['crime_code_description', 'crime_desc', 'crime_description'],
        'vict_descent': ['victim_descent', 'victim_race', 'vict_race']
    }

    for expected_col in column_mapping:
        if expected_col not in df.columns:
            for alt in alternatives.get(expected_col, []):
                if alt in df.columns:
                    column_mapping[expected_col] = alt
                    break

    def create_enhanced_description(row):
        base_text = str(row.get(column_mapping.get('crm_cd_desc', 'crime_desc'), ''))
        location = str(row.get(column_mapping.get('area_name', 'area'), ''))
        weapon = str(row.get('weapon_desc', '')) if 'weapon_desc' in row else ''
        premise = str(row.get('premis_desc', '')) if 'premis_desc' in row else ''
        enhanced = f"{base_text} in {location}"
        if weapon and weapon.lower() not in ['none', 'nan', '']:
            enhanced += f" involving {weapon}"
        if premise and premise.lower() not in ['none', 'nan', '']:
            enhanced += f" at {premise}"
        return enhanced

    df['enhanced_text'] = df.apply(create_enhanced_description, axis=1)

    for old_col, new_col in column_mapping.items():
        if old_col in df.columns:
            df[new_col] = df[old_col]

    required_columns = ['event_id', 'event_time', 'event_location', 'event_text', 'event_person', 'enhanced_text']
    for col in required_columns:
        if col not in df.columns:
            df[col] = "Unknown"

    df["event_id"] = df["event_id"].astype(str)
    df["enhanced_text"] = df["enhanced_text"].fillna(df["event_text"])

    return df[required_columns]

def generate_embeddings(df):
    texts = df['enhanced_text'].tolist()
    embeddings = model.encode(texts, convert_to_tensor=True)
    return embeddings

def search_similar_events(query, df, embeddings, top_k=5):
    query_embedding = model.encode(query, convert_to_tensor=True)
    cosine_scores = util.pytorch_cos_sim(query_embedding, embeddings)[0]

    top_results = torch.topk(cosine_scores, k=top_k)

    results = []
    for score, idx in zip(top_results[0], top_results[1]):
        event = df.iloc[int(idx)]
        results.append({
            "event_id": event["event_id"],
            "event_time": event["event_time"],
            "event_location": event["event_location"],
            "event_person": event["event_person"],
            "original_text": event["event_text"],
            "enhanced_text": event["enhanced_text"],
            "similarity_score": float(score)
        })
    return results
