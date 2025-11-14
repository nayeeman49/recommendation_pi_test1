import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Set page configuration
st.set_page_config(
    page_title="Pure Intentions Matchmaking Dashboard",
    page_icon="🕌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define differing columns at the module level
DIFFERING_COLUMNS = {
    'pref_love_lang_acts_of_service', 'pref_love_lang_gifts', 'pref_love_lang_physical_touch',
    'pref_love_lang_quality_time', 'pref_love_lang_sending_memes', 'pref_love_lang_words_of_affirmation',
    'self_ethnicity_african', 'self_ethnicity_caribbean', 'self_ethnicity_other_asian',
    'self_goals_other.1', 'self_goals_other.2', 'self_marital_dated_never_married',
    'self_marital_talking_stages', 'self_sect_shia'
}

class CrossDatasetMuslimMatchmaker:
    
    def __init__(self, data, df1_std=None, df2_std=None):
        self.data = data.copy()
        self.user_ids = data['self_identification'].tolist()
        self.gender_map = data.set_index('self_identification')['self_gender'].to_dict()
        self.age_map = data.set_index('self_identification')['self_age'].to_dict()

        # Track which dataset each user came from
        self.dataset_source = {}
        for user_id in self.user_ids:
            try:
                user_name = data[data['self_identification'] == user_id]['self_full_name'].iloc[0]
                # Simple heuristic: if name exists in original df1, it's from dataset 1
                if df1_std is not None and user_name in df1_std['self_full_name'].values:
                    self.dataset_source[user_id] = 'dataset_1'
                elif df2_std is not None and user_name in df2_std['self_full_name'].values:
                    self.dataset_source[user_id] = 'dataset_2'
                else:
                    self.dataset_source[user_id] = 'unknown'
            except (IndexError, KeyError):
                self.dataset_source[user_id] = 'unknown'

    def preprocess_features(self):
        """Extract and preprocess features for matching"""
        # 1. Geographic Compatibility
        self._extract_geographic_features()
        # 2. Religious Practice Similarity
        self._extract_religious_features()
        # 3. Personal Goals Similarity
        self._extract_goals_features()
        # 4. Values and Personality Compatibility
        self._extract_values_features()
        # 5. Marriage Goals Features
        self._extract_marriage_goals_features()
        # 6. Dealbreaker Analysis
        self._identify_dealbreakers()

    def _extract_geographic_features(self):
        """Extract geographic compatibility features"""
        # Current location encoding
        location_mapping = {
            'London': 1, 'Leicester': 2, 'Birmingham': 2, 'Manchester': 2,
            'Scotland': 3, 'Ireland': 3, 'International': 4, 'Oxford': 2,
            'Cambridge': 2, 'Leeds': 2, 'Glasgow': 3, 'Edinburgh': 3,
            'Cardiff': 2, 'Bristol': 2, 'Liverpool': 2, 'Newcastle': 2
        }

        self.data['location_score'] = self.data['self_current_location'].apply(
            lambda x: next((score for loc, score in location_mapping.items()
                          if str(loc).lower() in str(x).lower()), 4)
        )

        # Relocation willingness score
        relocation_cols = ['self_willing_move_uk_only', 'self_willing_move_internationally']
        for col in relocation_cols:
            if col not in self.data.columns:
                self.data[col] = 0

        self.data['relocation_score'] = (
            self.data['self_willing_move_uk_only'].fillna(0) +
            self.data['self_willing_move_internationally'].fillna(0) * 0.5
        )

    def _extract_religious_features(self):
        """Extract religious practice features"""
        religious_columns = [
            'self_prayer_level', 'self_growth_pray_salah', 'self_growth_read_quran',
            'self_growth_islamic_course', 'self_growth_complete_umrah',
            'self_growth_complete_hajj', 'self_growth_charity', 'self_growth_complete_hifz',
            'self_goals_improve_religion', 'self_goals_umrah_hajj'
        ]

        # Ensure all religious columns exist
        for col in religious_columns:
            if col not in self.data.columns:
                self.data[col] = 0

        # Sum religious practice indicators
        self.data['religious_practice_score'] = self.data[religious_columns].sum(axis=1)

        # Sect alignment - handle both datasets
        sunni_cols = ['pref_sect_Sunni', 'self_sect_sunni']
        shia_cols = ['pref_sect_Shia', 'self_sect_shia']

        for col in sunni_cols + shia_cols:
            if col not in self.data.columns:
                self.data[col] = 0

        self.data['sect_alignment'] = self.data.apply(
            lambda row: 1 if (row['self_sect_sunni'] == 1 and row['pref_sect_Sunni'] == 1) or
                            (row.get('self_sect_shia', 0) == 1 and row.get('pref_sect_Shia', 0) == 1)
                        else 0.5,
            axis=1
        )

    def _extract_goals_features(self):
        """Extract personal goals compatibility"""
        goals_columns = [
            'self_goals_buy_house', 'self_goals_get_married', 'self_goals_have_children',
            'self_goals_travel', 'self_goals_build_business', 'self_goals_time_friends_family',
            'self_goals_promotion', 'self_goals_get_pet'
        ]

        # Ensure all goals columns exist
        for col in goals_columns:
            if col not in self.data.columns:
                self.data[col] = 0

        self.data['goals_alignment'] = self.data[goals_columns].sum(axis=1)

    def _extract_values_features(self):
        """Extract values and personality compatibility"""
        # Self personality traits
        personality_prefix = 'self_personality_traits_'
        personality_cols = [col for col in self.data.columns if col.startswith(personality_prefix)]

        if personality_cols:
            self.data['self_personality_score'] = self.data[personality_cols].sum(axis=1)
        else:
            self.data['self_personality_score'] = 0

        # Preferred values alignment
        values_prefix = 'pref_values_'
        values_cols = [col for col in self.data.columns if col.startswith(values_prefix)]

        if values_cols:
            self.data['preferred_values_score'] = self.data[values_cols].sum(axis=1)
        else:
            self.data['preferred_values_score'] = 0

    def _extract_marriage_goals_features(self):
        """Extract and weight marriage goals based on ranking"""
        # Marriage goals ranking columns
        marriage_goals_columns = [
            'marriage_goals_rank_1', 'marriage_goals_rank_2', 'marriage_goals_rank_3',
            'marriage_goals_rank_4', 'marriage_goals_rank_5', 'marriage_goals_rank_6',
            'marriage_goals_rank_7', 'marriage_goals_rank_8', 'marriage_goals_rank_9',
            'marriage_goals_rank_10', 'marriage_goals_rank_11', 'marriage_goals_rank_12',
            'marriage_goals_rank_13', 'marriage_goals_rank_14', 'marriage_goals_rank_15',
            'marriage_goals_rank_16', 'marriage_goals_rank_17'
        ]

        # Ensure all marriage goals columns exist
        for col in marriage_goals_columns:
            if col not in self.data.columns:
                self.data[col] = ""

        # Create weighted marriage goals score
        self.data['marriage_goals_score'] = self.data.apply(
            lambda row: self._calculate_weighted_goals_score(row, marriage_goals_columns),
            axis=1
        )

    def _calculate_weighted_goals_score(self, user_row, goals_columns):
        """Calculate weighted score based on goal rankings"""
        weighted_score = 0
        max_possible_score = 0

        for i, col in enumerate(goals_columns):
            goal_value = user_row[col]
            if pd.notna(goal_value) and goal_value != "":
                # Higher weight for higher priority (lower rank number)
                weight = len(goals_columns) - i  # Rank 1 gets highest weight
                weighted_score += weight
                max_possible_score += weight

        return weighted_score / max_possible_score if max_possible_score > 0 else 0

    def _calculate_marriage_goals_compatibility(self, user1_data, user2_data):
        """Calculate compatibility based on marriage goals ranking"""
        marriage_goals_columns = [f'marriage_goals_rank_{i}' for i in range(1, 18)]

        compatibility_score = 0
        matched_priorities = 0

        for i, col in enumerate(marriage_goals_columns):
            if col in user1_data and col in user2_data:
                user1_goal = user1_data[col]
                user2_goal = user2_data[col]

                if (pd.notna(user1_goal) and pd.notna(user2_goal) and
                    user1_goal != "" and user2_goal != "" and
                    user1_goal == user2_goal):
                    # Higher weight for higher priority matches
                    weight = len(marriage_goals_columns) - i
                    compatibility_score += weight
                    matched_priorities += 1

        max_possible_score = sum(len(marriage_goals_columns) - i for i in range(len(marriage_goals_columns)))
        return compatibility_score / max_possible_score if max_possible_score > 0 else 0

    def _identify_dealbreakers(self):
        """Identify potential dealbreakers between users"""
        dealbreaker_cols = [col for col in self.data.columns if col.startswith('dealbreaker_')]
        self.dealbreaker_mapping = {}

        for user_id in self.user_ids:
            try:
                user_data = self.data[self.data['self_identification'] == user_id].iloc[0]
                dealbreakers = []

                for col in dealbreaker_cols:
                    if col in user_data and user_data[col] == 1:  # This is a dealbreaker for the user
                        dealbreakers.append(col.replace('dealbreaker_', ''))

                self.dealbreaker_mapping[user_id] = dealbreakers
            except (IndexError, KeyError):
                self.dealbreaker_mapping[user_id] = []

    def _check_age_compatibility(self, user1_id, user2_id):
        """Check if age difference is within 10 years"""
        age1 = self.age_map.get(user1_id)
        age2 = self.age_map.get(user2_id)

        if pd.isna(age1) or pd.isna(age2) or age1 == 0 or age2 == 0:
            return False

        return abs(age1 - age2) <= 10

    def _check_dealbreakers(self, user1_id, user2_id):
        """Check if user2 has any dealbreakers for user1"""
        user1_dealbreakers = self.dealbreaker_mapping.get(user1_id, [])
        
        try:
            user2_data = self.data[self.data['self_identification'] == user2_id].iloc[0]
        except (IndexError, KeyError):
            return 0

        dealbreaker_count = 0
        for dealbreaker in user1_dealbreakers:
            # Map dealbreaker names to actual columns in user2's data
            if dealbreaker == 'doesnt_pray' and user2_data.get('self_prayer_level', 0) < 2:
                dealbreaker_count += 1
            elif dealbreaker == 'alcohol_drugs' and user2_data.get('self_healing_went_rehab', 0) == 1:
                dealbreaker_count += 1
            elif dealbreaker == 'different_sect':
                try:
                    user1_pref_sunni = self.data[self.data['self_identification'] == user1_id]['pref_sect_Sunni'].iloc[0]
                    if (user2_data.get('self_sect_sunni', 0) == 1 and user1_pref_sunni == 0):
                        dealbreaker_count += 1
                    elif (user2_data.get('self_sect_shia', 0) == 1 and 
                          self.data[self.data['self_identification'] == user1_id].get('pref_sect_Shia', 0).iloc[0] == 0):
                        dealbreaker_count += 1
                except (IndexError, KeyError):
                    continue
            elif dealbreaker == 'education_level':
                try:
                    user1_edu = self.data[self.data['self_identification'] == user1_id]['self_education_level'].iloc[0]
                    user2_edu = user2_data.get('self_education_level', 0)
                    if abs(user1_edu - user2_edu) > 1:  # Allow one level difference
                        dealbreaker_count += 1
                except (IndexError, KeyError):
                    continue
            elif dealbreaker == 'has_children' and user2_data.get('self_children_info', 0) == 1:
                dealbreaker_count += 1
            elif dealbreaker == 'previous_relationships' and user2_data.get('self_marital_divorced', 0) == 1:
                dealbreaker_count += 1

        return dealbreaker_count

    def get_demographic_summary(self):
        """Get summary of available users by gender and age"""
        male_users = []
        female_users = []

        for user_id in self.user_ids:
            gender = self.gender_map.get(user_id)
            age = self.age_map.get(user_id)

            # Safely get user details with error handling
            try:
                name = self.data[self.data['self_identification'] == user_id]['self_full_name'].iloc[0]
                location = self.data[self.data['self_identification'] == user_id]['self_current_location'].iloc[0]

                user_info = {
                    'id': user_id,
                    'name': name,
                    'age': age,
                    'location': location,
                    'dataset': self.dataset_source.get(user_id, 'unknown')
                }

                if gender == 'Male':
                    male_users.append(user_info)
                elif gender == 'Female':
                    female_users.append(user_info)
            except (IndexError, KeyError) as e:
                continue

        return {
            'male_users': sorted(male_users, key=lambda x: x['age'] if pd.notna(x['age']) and x['age'] != 0 else 0),
            'female_users': sorted(female_users, key=lambda x: x['age'] if pd.notna(x['age']) and x['age'] != 0 else 0),
            'total_males': len(male_users),
            'total_females': len(female_users),
            'total_users': len(male_users) + len(female_users)
        }

    def calculate_compatibility(self, user1_id, user2_id):
        """Calculate compatibility score between two users"""
        # Skip same gender matches
        if self.gender_map.get(user1_id) == self.gender_map.get(user2_id):
            return 0

        # Check age compatibility (max 10 years difference)
        if not self._check_age_compatibility(user1_id, user2_id):
            return 0

        try:
            user1_data = self.data[self.data['self_identification'] == user1_id].iloc[0]
            user2_data = self.data[self.data['self_identification'] == user2_id].iloc[0]
        except IndexError:
            return 0

        # Check dealbreakers
        dealbreakers_1_to_2 = self._check_dealbreakers(user1_id, user2_id)
        dealbreakers_2_to_1 = self._check_dealbreakers(user2_id, user1_id)
        total_dealbreakers = dealbreakers_1_to_2 + dealbreakers_2_to_1

        if total_dealbreakers > 2:  # Too many dealbreakers
            return 0

        # Calculate similarity scores
        scores = {}

        # 1. Geographic compatibility (25% weight)
        loc_diff = abs(user1_data.get('location_score', 0) - user2_data.get('location_score', 0))
        relocation_compat = min(user1_data.get('relocation_score', 0), user2_data.get('relocation_score', 0))
        scores['geographic'] = (1 - loc_diff/4) * 0.5 + relocation_compat * 0.5

        # 2. Religious compatibility (30% weight)
        relig_diff = abs(user1_data.get('religious_practice_score', 0) - user2_data.get('religious_practice_score', 0))
        max_relig = max(user1_data.get('religious_practice_score', 0), user2_data.get('religious_practice_score', 0))
        relig_similarity = 1 - (relig_diff / max_relig) if max_relig > 0 else 0.5

        sect_similarity = (user1_data.get('sect_alignment', 0) + user2_data.get('sect_alignment', 0)) / 2
        scores['religious'] = (relig_similarity * 0.7 + sect_similarity * 0.3)

        # 3. Goals compatibility (25% weight)
        goals1 = user1_data.get('goals_alignment', 0)
        goals2 = user2_data.get('goals_alignment', 0)
        basic_goals_similarity = 1 - abs(goals1 - goals2) / max(goals1, goals2) if max(goals1, goals2) > 0 else 0.5

        # Marriage goals ranking compatibility
        marriage_goals_similarity = self._calculate_marriage_goals_compatibility(user1_data, user2_data)

        # Combine basic goals and marriage goals ranking
        scores['goals'] = (basic_goals_similarity * 0.4 + marriage_goals_similarity * 0.6)

        # 4. Values compatibility (20% weight)
        personality_diff = abs(user1_data.get('self_personality_score', 0) - user2_data.get('self_personality_score', 0))
        max_personality = max(user1_data.get('self_personality_score', 0), user2_data.get('self_personality_score', 0))
        personality_similarity = 1 - (personality_diff / max_personality) if max_personality > 0 else 0.5

        values_diff = abs(user1_data.get('preferred_values_score', 0) - user2_data.get('preferred_values_score', 0))
        max_values = max(user1_data.get('preferred_values_score', 0), user2_data.get('preferred_values_score', 0))
        values_similarity = 1 - (values_diff / max_values) if max_values > 0 else 0.5

        scores['values'] = (personality_similarity * 0.6 + values_similarity * 0.4)

        # Weighted final score
        weights = {'geographic': 0.15, 'religious': 0.30, 'goals': 0.35, 'values': 0.20}
        final_score = sum(scores[domain] * weight for domain, weight in weights.items())

        # Apply dealbreaker penalty
        dealbreaker_penalty = total_dealbreakers * 0.2
        final_score = max(0, final_score - dealbreaker_penalty)

        return final_score

    def find_matches(self, user_id, top_n=5):
        """Find top matches for a given user across both datasets"""
        matches = []
        user_age = self.age_map.get(user_id, 0)
        user_gender = self.gender_map.get(user_id)

        if not user_gender:
            return matches

        for other_user_id in self.user_ids:
            if other_user_id != user_id:
                # Skip if same gender
                if self.gender_map.get(other_user_id) == user_gender:
                    continue

                # Check age difference
                other_user_age = self.age_map.get(other_user_id, 0)
                if user_age and other_user_age:
                    if abs(user_age - other_user_age) > 10:
                        continue

                score = self.calculate_compatibility(user_id, other_user_id)
                if score > 0:  # Only include viable matches
                    try:
                        matches.append({
                            'user_id': other_user_id,
                            'score': score,
                            'name': self.data[self.data['self_identification'] == other_user_id]['self_full_name'].iloc[0],
                            'gender': self.gender_map.get(other_user_id, 'Unknown'),
                            'location': self.data[self.data['self_identification'] == other_user_id]['self_current_location'].iloc[0],
                            'age': self.age_map.get(other_user_id, 'N/A'),
                            'age_difference': abs(user_age - other_user_age) if user_age and other_user_age else 'N/A',
                            'dataset': self.dataset_source.get(other_user_id, 'Unknown')
                        })
                    except (IndexError, KeyError):
                        continue

        # Sort by score and return top matches
        matches.sort(key=lambda x: x['score'], reverse=True)
        return matches[:top_n]

    def get_user_details(self, user_id):
        """Get comprehensive details for a user"""
        if user_id not in self.user_ids:
            return None

        try:
            user_data = self.data[self.data['self_identification'] == user_id].iloc[0]
        except IndexError:
            return None

        details = {
            'basic_info': {
                'name': user_data.get('self_full_name', 'N/A'),
                'gender': user_data.get('self_gender', 'N/A'),
                'age': user_data.get('self_age', 'N/A'),
                'location': user_data.get('self_current_location', 'N/A'),
                'education_level': user_data.get('self_education_level', 'N/A'),
                'uk_based': user_data.get('self_uk_based', 'N/A'),
                'dataset': self.dataset_source.get(user_id, 'Unknown')
            },
            'religious_info': {
                'prayer_level': user_data.get('self_prayer_level', 'N/A'),
                'sect': 'Sunni' if user_data.get('self_sect_sunni', 0) == 1 else 'Shia' if user_data.get('self_sect_shia', 0) == 1 else 'N/A',
                'completed_umrah': 'Yes' if user_data.get('self_umrah_hajj_umrah', 0) == 1 or user_data.get('self_umrah_hajj_both', 0) == 1 else 'No',
                'completed_hajj': 'Yes' if user_data.get('self_umrah_hajj_both', 0) == 1 else 'No'
            },
            'top_priorities': {
                'buy_house': 'Yes' if user_data.get('self_goals_buy_house', 0) == 1 else 'No',
                'get_married': 'Yes' if user_data.get('self_goals_get_married', 0) == 1 else 'No',
                'have_children': 'Yes' if user_data.get('self_goals_have_children', 0) == 1 else 'No',
                'travel': 'Yes' if user_data.get('self_goals_travel', 0) == 1 else 'No',
                'build_business': 'Yes' if user_data.get('self_goals_build_business', 0) == 1 else 'No',
                'improve_in_deen': 'Yes' if user_data.get('self_goals_improve_religion',0)== 1 else 'No'
            },
            'dealbreakers': self.dealbreaker_mapping.get(user_id, [])
        }

        return details

    def get_match_analysis(self, user1_id, user2_id):
        """Get detailed analysis of why two users match"""
        score = self.calculate_compatibility(user1_id, user2_id)
        if score == 0:
            return "No compatible match due to dealbreakers, age difference, or same gender"

        try:
            user1_data = self.data[self.data['self_identification'] == user1_id].iloc[0]
            user2_data = self.data[self.data['self_identification'] == user2_id].iloc[0]
        except IndexError:
            return "Error: User data not found"

        analysis = {
            'overall_score': score,
            'age_difference': abs(self.age_map.get(user1_id, 0) - self.age_map.get(user2_id, 0)),
            'geographic_compatibility': 1 - abs(user1_data.get('location_score', 0) - user2_data.get('location_score', 0))/4,
            'religious_similarity': 1 - abs(user1_data.get('religious_practice_score', 0) - user2_data.get('religious_practice_score', 0)) / max(user1_data.get('religious_practice_score', 0), user2_data.get('religious_practice_score', 0)) if max(user1_data.get('religious_practice_score', 0), user2_data.get('religious_practice_score', 0)) > 0 else 0.5,
            'goals_alignment': 1 - abs(user1_data.get('goals_alignment', 0) - user2_data.get('goals_alignment', 0)) / max(user1_data.get('goals_alignment', 0), user2_data.get('goals_alignment', 0)) if max(user1_data.get('goals_alignment', 0), user2_data.get('goals_alignment', 0)) > 0 else 0.5,
            'dealbreakers': self._check_dealbreakers(user1_id, user2_id) + self._check_dealbreakers(user2_id, user1_id)
        }

        return analysis

def standardize_datasets(df1, df2):
    """Standardize both datasets to have compatible columns"""
    # Add missing columns to each dataset with default values
    for col in DIFFERING_COLUMNS:
        if col not in df1.columns:
            df1[col] = 0
        if col not in df2.columns:
            df2[col] = 0

    # Handle love language columns mapping
    love_language_mapping = {
        'pref_love_language_1': 'pref_love_lang_words_of_affirmation',
        'pref_love_language_2': 'pref_love_lang_quality_time',
        'pref_love_language_3': 'pref_love_lang_acts_of_service',
        'pref_love_language_4': 'pref_love_lang_physical_touch',
        'pref_love_language_5': 'pref_love_lang_gifts',
        'pref_love_language_6': 'pref_love_lang_sending_memes'
    }

    # Map love language columns in df1 to match df2 format
    for old_col, new_col in love_language_mapping.items():
        if old_col in df1.columns and new_col not in df1.columns:
            df1[new_col] = df1[old_col]

    return df1, df2

def handle_file_upload():
    """Handle manual file uploads for both datasets"""
    st.sidebar.header("📁 Upload Your Datasets")
    
    st.sidebar.markdown("""
    **Instructions:**
    1. Upload your first dataset (CSV format)
    2. Upload your second dataset (CSV format)  
    3. Click 'Process Datasets' to start matching
    """)
    
    # File uploaders
    uploaded_file1 = st.sidebar.file_uploader(
        "Upload First Dataset (CSV)", 
        type=['csv'],
        key="file1",
        help="Upload your first dataset in CSV format"
    )
    
    uploaded_file2 = st.sidebar.file_uploader(
        "Upload Second Dataset (CSV)", 
        type=['csv'],
        key="file2", 
        help="Upload your second dataset in CSV format"
    )
    
    # Process button
    process_clicked = st.sidebar.button("🚀 Process Datasets", type="primary")
    
    return uploaded_file1, uploaded_file2, process_clicked

def load_and_prepare_data(uploaded_file1, uploaded_file2):
    """Load and prepare datasets from uploaded files"""
    
    if uploaded_file1 is None or uploaded_file2 is None:
        return None, None, None
    
    try:
        # Read uploaded files
        df1 = pd.read_csv(uploaded_file1)
        df2 = pd.read_csv(uploaded_file2)
        
        # Show dataset info
        st.success(f"✅ Dataset 1 loaded: {len(df1)} users")
        st.success(f"✅ Dataset 2 loaded: {len(df2)} users")
        
        # Display basic info about datasets
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"**Dataset 1 Columns:** {len(df1.columns)}")
            st.dataframe(df1.head(3), use_container_width=True)
        
        with col2:
            st.info(f"**Dataset 2 Columns:** {len(df2.columns)}")
            st.dataframe(df2.head(3), use_container_width=True)
        
        # Standardize datasets
        with st.spinner("🔄 Standardizing datasets..."):
            df1_std, df2_std = standardize_datasets(df1.copy(), df2.copy())
        
        # Combine datasets
        combined_df = pd.concat([df1_std, df2_std], ignore_index=True)
        combined_df = combined_df.fillna(0)
        
        st.success(f"✅ Combined dataset ready: {len(combined_df)} total users")
        
        return combined_df, df1_std, df2_std
        
    except Exception as e:
        st.error(f"❌ Error processing files: {str(e)}")
        return None, None, None

def initialize_matchmaker(combined_df, df1_std, df2_std):
    """Initialize the matchmaker with loaded data"""
    if combined_df is None or combined_df.empty:
        return None
        
    try:
        with st.spinner("🔄 Initializing matchmaker..."):
            matchmaker = CrossDatasetMuslimMatchmaker(combined_df, df1_std, df2_std)
            matchmaker.preprocess_features()
        
        st.success("✅ Matchmaker initialized successfully!")
        return matchmaker
        
    except Exception as e:
        st.error(f"❌ Error initializing matchmaker: {str(e)}")
        return None

def display_user_details(details): #user_details
    """Display user details in an organized format"""
    st.subheader("Basic Information")
    basic_info = details['basic_info']
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.write("**Name:**", basic_info['name'])
        st.write("**Gender:**", basic_info['gender'])
    
    with col2:
        st.write("**Age:**", basic_info['age'])
        st.write("**Location:**", basic_info['location'])
    
    with col3:
        st.write("**Education Level:**", basic_info['education_level'])
        st.write("**UK Based:**", basic_info['uk_based'])
    
    with col4:
        st.write("**Dataset:**", basic_info['dataset'])
    
    st.divider()
    
    # Religious Information
    st.subheader("Religious Information")
    religious_info = details['religious_info']
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.write("**Prayer Level:**", religious_info['prayer_level'])
    
    with col2:
        st.write("**Sect:**", religious_info['sect'])
    
    with col3:
        st.write("**Completed Umrah:**", religious_info['completed_umrah'])
    
    with col4:
        st.write("**Completed Hajj:**", religious_info['completed_hajj'])
    
    st.divider()
    
    # Goals
    st.subheader("Life Goals")
    goals = details['top_priorities']
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**Buy House:**", goals['buy_house'])
        st.write("**Get Married:**", goals['get_married'])
    
    with col2:
        st.write("**Have Children:**", goals['have_children'])
        st.write("**Travel:**", goals['travel'])
    
    with col3:
        st.write("**Build Business:**", goals['build_business'])
        st.write("**Improve in Deen:**", goals['improve_in_deen'])
    
    st.divider()
    
    # Dealbreakers
    st.subheader("Dealbreakers")
    dealbreakers = details['dealbreakers']
    if dealbreakers:
        for dealbreaker in dealbreakers:
            st.write(f"• {dealbreaker.replace('_', ' ').title()}")
    else:
        st.write("No specific dealbreakers listed")

def show_dashboard(matchmaker):
    """Display the main dashboard"""
    st.header("📊 Matchmaking Dashboard")
    
    # Get demographic summary
    demographics = matchmaker.get_demographic_summary()
    
    # Display key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Users", demographics['total_users'])
    
    with col2:
        st.metric("Male Users", demographics['total_males'])
    
    with col3:
        st.metric("Female Users", demographics['total_females'])
    
    with col4:
        st.metric("Datasets", "2 Combined")
    
    # Age distribution charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("👨 Male Users")
        if demographics['male_users']:
            male_ages = [user['age'] for user in demographics['male_users'] if user['age'] and user['age'] != 0]
            if male_ages:
                fig = px.histogram(x=male_ages, nbins=10, title="Male Age Distribution")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No age data available for male users")
        else:
            st.info("No male users in dataset")
    
    with col2:
        st.subheader("👩 Female Users")
        if demographics['female_users']:
            female_ages = [user['age'] for user in demographics['female_users'] if user['age'] and user['age'] != 0]
            if female_ages:
                fig = px.histogram(x=female_ages, nbins=10, title="Female Age Distribution")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No age data available for female users")
        else:
            st.info("No female users in dataset")
    
    # User lists
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Male Users List")
        if demographics['male_users']:
            male_df = pd.DataFrame(demographics['male_users'])
            st.dataframe(male_df[['name', 'age', 'location', 'dataset']], use_container_width=True)
        else:
            st.info("No male users available")
    
    with col2:
        st.subheader("Female Users List")
        if demographics['female_users']:
            female_df = pd.DataFrame(demographics['female_users'])
            st.dataframe(female_df[['name', 'age', 'location', 'dataset']], use_container_width=True)
        else:
            st.info("No female users available")

def find_matches_section(matchmaker):
    """Section for finding matches for a specific user"""
    st.header("👤 Find Compatible Matches")
    
    # Get all user IDs
    user_ids = matchmaker.user_ids
    
    if not user_ids:
        st.error("No users found in the dataset")
        return
    
    # User selection
    selected_user = st.selectbox(
        "Select a user to find matches for:",
        options=user_ids,
        format_func=lambda x: f"{x} - {matchmaker.gender_map.get(x, 'Unknown')}"
    )
    
    # Number of matches to show
    top_n = st.slider("Number of matches to show:", min_value=1, max_value=20, value=5)
    
    if st.button("Find Matches", type="primary"):
        with st.spinner("Finding compatible matches..."):
            matches = matchmaker.find_matches(selected_user, top_n=top_n)
        
        if matches:
            st.success(f"Found {len(matches)} compatible matches!")
            
            # Display matches in a nice format
            for i, match in enumerate(matches, 1):
                with st.container():
                    col1, col2, col3 = st.columns([3, 2, 1])
                    
                    with col1:
                        st.subheader(f"{i}. {match['name']}")
                        st.write(f"**Age:** {match['age']} | **Location:** {match['location']}")
                        st.write(f"**Dataset:** {match['dataset']} | **Age Difference:** {match['age_difference']} years")
                    
                    with col2:
                        # Progress bar for compatibility score
                        score_percent = match['score'] * 100
                        st.progress(float(match['score']))
                        st.write(f"**Compatibility:** {score_percent:.1f}%")
                    
                    with col3:
                        if st.button("View Details", key=f"details_{i}"):
                            st.session_state[f"show_match_{i}"] = not st.session_state.get(f"show_match_{i}", False)
                    
                    # Show detailed analysis when button is clicked
                    if st.session_state.get(f"show_match_{i}", False):
                        analysis = matchmaker.get_match_analysis(selected_user, match['user_id'])
                        with st.expander("Match Analysis", expanded=True):
                            if isinstance(analysis, dict):
                                st.write(f"**Overall Score:** {analysis['overall_score']:.2f}")
                                st.write(f"**Age Difference:** {analysis['age_difference']} years")
                                st.write(f"**Geographic Compatibility:** {analysis['geographic_compatibility']:.2f}")
                                st.write(f"**Religious Similarity:** {analysis['religious_similarity']:.2f}")
                                st.write(f"**Goals Alignment:** {analysis['goals_alignment']:.2f}")
                                st.write(f"**Dealbreakers:** {analysis['dealbreakers']}")
                            else:
                                st.write(analysis)
                        
                        # Show user details
                        user_details = matchmaker.get_user_details(match['user_id'])
                        if user_details:
                            with st.expander("User Profile", expanded=False):
                                display_user_details(user_details)
                
                st.divider()
        else:
            st.warning("No compatible matches found for this user.")

def user_details_section(matchmaker):
    """Section for viewing detailed user information"""
    st.header("📊 User Details")
    
    user_ids = matchmaker.user_ids
    
    if not user_ids:
        st.error("No users found in the dataset")
        return
    
    selected_user = st.selectbox(
        "Select a user to view details:",
        options=user_ids,
        key="user_details_select"
    )
    
    if st.button("Show User Details", type="primary"):
        user_details = matchmaker.get_user_details(selected_user)
        
        if user_details:
            display_user_details(user_details)
        else:
            st.error("Could not retrieve user details")

def analytics_section(matchmaker):
    """Section for analytics and insights"""
    st.header("📈 Matchmaking Analytics")
    
    demographics = matchmaker.get_demographic_summary()
    
    # Compatibility score distribution (sample calculation)
    st.subheader("Sample Compatibility Analysis")
    
    # Calculate some sample compatibilities for demonstration
    if len(matchmaker.user_ids) >= 4:
        sample_users = matchmaker.user_ids[:4]
        compatibility_matrix = []
        
        with st.spinner("Calculating sample compatibilities..."):
            for i, user1 in enumerate(sample_users):
                row = []
                for j, user2 in enumerate(sample_users):
                    if i != j and matchmaker.gender_map.get(user1) != matchmaker.gender_map.get(user2):
                        score = matchmaker.calculate_compatibility(user1, user2)
                        row.append(score)
                    else:
                        row.append(0)
                compatibility_matrix.append(row)
        
        # Display compatibility matrix
        if compatibility_matrix:
            fig = px.imshow(
                compatibility_matrix,
                x=[f"User {i+1}" for i in range(len(sample_users))],
                y=[f"User {i+1}" for i in range(len(sample_users))],
                title="Sample Compatibility Matrix",
                color_continuous_scale="Viridis"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Dataset distribution
    st.subheader("Dataset Distribution")
    source_counts = {}
    for user_id in matchmaker.user_ids:
        source = matchmaker.dataset_source.get(user_id, 'unknown')
        source_counts[source] = source_counts.get(source, 0) + 1
    
    if source_counts:
        fig = px.pie(
            values=list(source_counts.values()),
            names=list(source_counts.keys()),
            title="User Distribution by Dataset"
        )
        st.plotly_chart(fig, use_container_width=True)

        
def show_data_upload_section():
    """Show the main data upload interface"""
    st.header("🕌 Pure Intentions Matchmaking Recommendation System")
    st.markdown("Find compatible matches based on religious values, goals, and preferences")
    
    st.markdown("---")
    
    # Show upload instructions
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📋 How to Use:")
        st.markdown("""
        1. **Upload Datasets** - Use the sidebar to upload two CSV files
        2. **Process Data** - Click 'Process Datasets' to combine and standardize
        3. **Find Matches** - Use the navigation to explore matches and analytics
        4. **View Details** - See detailed user profiles and compatibility analysis
        """)
    
    with col2:
        st.subheader("🔍 Required Data Columns:")
        st.markdown("""
        Your CSV files should include:
        - User identification
        - Gender and age information  
        - Religious practices
        - Personal goals
        - Location data
        - Dealbreakers and preferences
        """)
    
    st.info("👈 **Start by uploading your datasets in the sidebar**")

def main():
    # Initialize session state for matchmaker
    if 'matchmaker' not in st.session_state:
        st.session_state.matchmaker = None
    if 'data_processed' not in st.session_state:
        st.session_state.data_processed = False
    
    # Handle file uploads
    uploaded_file1, uploaded_file2, process_clicked = handle_file_upload()
    
    # Process datasets when button is clicked
    if process_clicked and uploaded_file1 and uploaded_file2:
        with st.spinner("Processing your datasets..."):
            combined_df, df1_std, df2_std = load_and_prepare_data(uploaded_file1, uploaded_file2)
            
            if combined_df is not None:
                matchmaker = initialize_matchmaker(combined_df, df1_std, df2_std)
                if matchmaker:
                    st.session_state.matchmaker = matchmaker
                    st.session_state.data_processed = True
                    st.rerun()
    
    # Show appropriate content based on data state
    if st.session_state.data_processed and st.session_state.matchmaker:
        # Show the main app with navigation
        matchmaker = st.session_state.matchmaker
        
        # Sidebar for navigation
        st.sidebar.header("🎯 Navigation")
        app_mode = st.sidebar.selectbox(
            "Choose a section",
            ["🏠 Dashboard", "👤 Find Matches", "📊 User Details", "📈 Analytics", "🔄 Upload New Data"]
        )
        
        if app_mode == "🏠 Dashboard":
            show_dashboard(matchmaker)
        elif app_mode == "👤 Find Matches":
            find_matches_section(matchmaker)
        elif app_mode == "📊 User Details":
            user_details_section(matchmaker)
        elif app_mode == "📈 Analytics":
            analytics_section(matchmaker)
        elif app_mode == "🔄 Upload New Data":
            # Reset and allow new uploads
            if st.sidebar.button("Start New Session"):
                st.session_state.matchmaker = None
                st.session_state.data_processed = False
                st.rerun()
            show_data_upload_section()
    
    else:
        # Show upload interface
        show_data_upload_section()
        
        # Show demo data option (optional)
        st.sidebar.markdown("---")
        st.sidebar.info("💡 **Tip:** Make sure your CSV files have compatible column structures for best results")

if __name__ == "__main__":
    main()
