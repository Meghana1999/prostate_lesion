import pandas as pd
import re

class PathologyDataProcessor:
    def __init__(self, df): 
        self.df_bx1 = df 
        self.df_bx2 = df

    @staticmethod
    def extract_sections(text):
        text = str(text) if pd.notna(text) else ''
        result = re.split(r'FINAL DIAGNOSIS', text, flags=re.IGNORECASE)
        if len(result) < 2:
            return [], text 
        
        before_paragraph = result[0].strip()
        final_diagnosis_paragraph = "FINAL DIAGNOSIS" + result[1].strip()

        section_headers = re.finditer(r'\b([A-Z])([:.])', before_paragraph)
        section_indices = [(match.start(), match.group(1)) for match in section_headers]
        section_content = {}

        for i in range(len(section_indices)):
            start_index, section_name = section_indices[i]
            end_index = section_indices[i + 1][0] if i + 1 < len(section_indices) else len(before_paragraph)
            section_content[section_name] = before_paragraph[start_index:end_index]

        roi_pattern = re.compile(r'\bProstate-?ROI-?\d*|\bROI-?\d*')
        sections_with_roi = [section for section, content in section_content.items() if roi_pattern.search(content)]

        target_sections = []
        for section_name in sections_with_roi:
            regex_pattern = re.compile(fr'{section_name}\.([\s\S]*?)(?=[A-Z]\.|$)', re.IGNORECASE)
            matches = regex_pattern.finditer(final_diagnosis_paragraph)
            for match in matches:
                section_content = match.group(1).strip()
                target_sections.append(section_content)
                # Remove the extracted section from the final_diagnosis_paragraph
                final_diagnosis_paragraph = final_diagnosis_paragraph.replace(match.group(0), '')

        return target_sections, final_diagnosis_paragraph.strip()

    @staticmethod
    def extract_required_fields(section_text):
        section_text = str(section_text) if pd.notna(section_text) else ''
        if section_text:
            # Pattern to extract tissue percent and cores information
            pattern = re.compile(r'(\d+)%.*?(\d+) of (\d+) cores', re.DOTALL)
            match = pattern.search(section_text)
            if match:
                tissue_percent, cores_positive, cores_total = match.groups()
                return tissue_percent, cores_positive, cores_total

            # Alternative pattern , if the first one doesn't match
            pattern_alt = re.compile(r'Tumor involves (\d+)% of.*?(\d+) of (\d+) cores', re.DOTALL)
            match_alt = pattern_alt.search(section_text)
            if match_alt:
                tissue_percent, cores_positive, cores_total = match_alt.groups()
                return tissue_percent, cores_positive, cores_total

        # Return None values if no pattern matches or if section_text is None
        return None, None, None
    
 
    def extract_roi_fields(self, df, target_column):
        def extract_fields(row):
            target_text = str(row[target_column]) if pd.notna(row[target_column]) else ''
            # Checking if the target column value is None or NaN 
            if pd.isna(row[target_column]):
                return pd.Series([None, None, None, None, None])
            
            # Extract location
            location_match = re.search(r"Prostate, (?:ROI[- ]?\d+[-: ]?)?(.*?)(?:,| needle)", target_text)
            location = location_match.group(1).strip() if location_match else None 

            # Extract Gleason score
            gleason_match = re.search(r"Gleason (?:score|grade)[\s:]*?(\d\s?\+\s?\d)", target_text, re.IGNORECASE)
            gleason_score = gleason_match.group(1).replace(" ", "") if gleason_match else None

            # Extract number of cores taken
            cores_taken_match = re.search(r"(\d+) cores\)", target_text)
            number_cores_taken = cores_taken_match.group(1) if cores_taken_match else None

            # Extract number of cores positive
            cores_positive_match = re.search(r"(\d+) of (\d+) cores", target_text)
            number_cores_positive = cores_positive_match.group(1) if cores_positive_match else None

            # Extract total percentage positive
            percent_positive_match = re.search(r"(?:Tumor involves|involving) (<\d+|\d+-\d+|\d+)% of (?:overall )?specimen", target_text) 
            total_percent_positive = percent_positive_match.group(1) if percent_positive_match else None

            return pd.Series([location, gleason_score, number_cores_taken, number_cores_positive, total_percent_positive])

        # Applying the extraction function to each row of the DataFrame
        new_columns = [f'{target_column}_location', f'{target_column}_gleason', 
                       f'{target_column}_number_cores_taken', f'{target_column}_number_cores_positive', 
                       f'{target_column}_total_percent_positive']
        df[new_columns] = df.apply(extract_fields, axis=1)
        return df



    @staticmethod
    def find_and_extract_systematic_info(df, column_name):
        def extract_info(text):
            text = str(text) if pd.notna(text) else ''
            location, score, section_content = PathologyDataProcessor.find_highest_gleason_section_and_content(text)
            if location:
                tissue_percent, cores_positive, cores_total = PathologyDataProcessor.extract_required_fields(section_content)
                return pd.Series([location, score, section_content, tissue_percent, cores_positive, cores_total])
            else:
                # Alternative method , if the first method fails
                max_info = PathologyDataProcessor.find_highest_gleason_info(text)
                if max_info:
                    return pd.Series([
                        max_info.get('location'), max_info.get('gleason_score'), None, 
                        max_info.get('tissue_percent'), max_info.get('cores_positive'), max_info.get('cores_total')
                    ])
                else:
                    return pd.Series([None, None, None, None, None, None])

        new_columns = ['Systematic_Location', 'Systematic_Gleason_Score', 'Systematic_Content', 
                       'Systematic_Tissue_Percent', 'Systematic_Cores_Positive', 'Systematic_Cores_Total']

        df[new_columns] = df[column_name].apply(extract_info)
        return df
    

    @staticmethod
    def find_highest_gleason_section_and_content(text): 
        text = str(text) if pd.notna(text) else ''
        pattern = re.compile(r'([A-Z])\.\s.*?Prostate, (.*?), needle core biopsy:.*?Gleason score (\d+)\+(\d+)', re.DOTALL)
        matches = pattern.findall(text)

        max_score = (0, 0)
        max_location = None
        section_content = None
        for section, location, primary, secondary in matches:
            primary, secondary = int(primary), int(secondary)
            if (primary + secondary) > sum(max_score):
                max_score = (primary, secondary)
                max_location = location
                section_pattern = re.compile(rf'{section}\.\s(.*?)(?=\n[A-Z]\.\s|\Z)', re.DOTALL)
                match = section_pattern.search(text)
                section_content = match.group(1).strip() if match else None

        formatted_score = f"{max_score[0]}+{max_score[1]}={sum(max_score)}"
        return max_location, formatted_score, section_content

    @staticmethod
    def find_highest_gleason_info(text):
        text = str(text) if pd.notna(text) else ''
        pattern = re.compile(
            r'([A-Z][).-])\s*Prostate, (.*?),\sneedle core biopsy:.*?'
            r'(?:Gleason score|Grade Group and Gleason score|Grade Group \d+ \(Gleason Score):\s*'
            r'(?:Grade Group\s*\d+\s*\()?(?:Gleason Score\s*)?(\d+)\+(\d+)=\d+(?:\))?'
            r'.*?(?:Percentage of pattern 4:.*?NA|Percentage of pattern 4:.*?<\d+%|Percentage of pattern 4:.*?\d+%|Tumor involves\s*<?\s*(\d+)% of overall specimen|Multifocal tumor involves\s*<?\s*(\d+)% of overall specimen)?'
            r'.*?(\d+) of (\d+) cores?', 
            re.DOTALL | re.IGNORECASE
        )
        matches = pattern.findall(text)
        
        max_score = 0
        max_info = None
        for match in matches:
            section, location, primary, secondary, tissue_percent_a, tissue_percent_b, cores_positive, cores_total = match
            primary, secondary = int(primary), int(secondary)
            gleason_score = primary + secondary
            tissue_percent = tissue_percent_a or tissue_percent_b
            tissue_percent = tissue_percent.replace('<', '').strip()
            cores_positive = cores_positive.strip()
            cores_total = cores_total.strip()

            if gleason_score > max_score:
                max_score = gleason_score
                max_info = {
                    'location': location,
                    'gleason_score': f"{primary}+{secondary}={gleason_score}",
                    'cores_total': cores_total,
                    'cores_positive': cores_positive,
                    'tissue_percent': tissue_percent if tissue_percent else 'NA'
                }

        return max_info

    

    def process_dataframe_bx1(self):
        self.df_bx1[['target_paragraph', 'remaining_final_diagnosis_content']] = self.df_bx1['PATH_RES_1'].apply(
            lambda x: pd.Series(self.extract_sections(x))
        )

        max_sections = max(self.df_bx1['target_paragraph'].apply(len), default=0)
        columns = [f'target_paragraph{i+1}' for i in range(max_sections)]
        df_expanded = self.df_bx1['target_paragraph'].apply(pd.Series)
        df_expanded.columns = columns

        self.df_bx1 = self.df_bx1.join(df_expanded)

        for col in columns:
            if col in self.df_bx1.columns:
                self.df_bx1 = self.extract_roi_fields(self.df_bx1, col)

        self.df_bx1 = self.find_and_extract_systematic_info(self.df_bx1, 'remaining_final_diagnosis_content')
        return self.df_bx1

    def process_dataframe_bx2(self):
        self.df_bx2[['target_paragraph', 'remaining_final_diagnosis_content']] = self.df_bx2['PATH_RES_2'].apply(
            lambda x: pd.Series(self.extract_sections(x))
        )

        max_sections = max(self.df_bx2['target_paragraph'].apply(len), default=0)
        columns = [f'target_paragraph{i+1}' for i in range(max_sections)]
        df_expanded = self.df_bx2['target_paragraph'].apply(pd.Series)
        df_expanded.columns = columns

        self.df_bx2 = self.df_bx2.join(df_expanded)

        for col in columns:
            if col in self.df_bx2.columns:
                self.df_bx2 = self.extract_roi_fields(self.df_bx2, col)

        self.df_bx2 = self.find_and_extract_systematic_info(self.df_bx2, 'remaining_final_diagnosis_content')
        return self.df_bx2




