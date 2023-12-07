import pandas as pd
import re

class ProstateDataProcessor:
    def __init__(self):
        self.df = None

    def load_data(self, input_file):
        try:
            self.df = pd.read_excel(input_file)
            print('Data loaded successfully.')
        except Exception as e:
            print(f"Error loading the input file: {e}")

    def extract_lesion_paragraph(self):
        self.df['extracted_lesion_paragraph'] = ''
        self.df['extracted_lesion_no'] = None
        for index, row in self.df.iterrows():
            comment = row['NARRATIVE']
            if isinstance(comment, str):
                # Extracting the content between "Lesion #" and "LOCAL STAGING"
                extracted = re.search(r'Lesion #.*?(?=LOCAL STAGING|LYMPH NODES)', comment, re.DOTALL)
                if extracted:
                    extracted_comment = extracted.group(0).strip()
                    self.df.at[index, 'extracted_lesion_paragraph'] = extracted_comment
                    # Extracting lesion number
                    lesion_numbers = re.findall(r'Lesion # (\d+)', extracted_comment)
                    if lesion_numbers:
                        highest_lesion_no = max(map(int, lesion_numbers))
                        self.df.at[index, 'extracted_lesion_no'] = highest_lesion_no

    def extract_lesion_contents(self):
        # New column for extracted lesion contents
        self.df['Lesion_contents'] = self.df['NARRATIVE'].apply(self._extract_lesion_contents)
        max_lesion_count = self.df['Lesion_contents'].apply(len).max()
        # Column names for all possible lesions
        column_names = [f'Lesion{i}_content' for i in range(1, max_lesion_count + 1)]
        # Separate Columns
        df_new = pd.concat([self.df['Lesion_contents'].apply(lambda x: x.get(f'Lesion{i}_content', '')) for i in range(1, max_lesion_count + 1)], axis=1)
        df_new.columns = column_names
        self.df = pd.concat([self.df, df_new], axis=1)

    def _extract_lesion_contents(self, text):
        lesion_contents = {}
        # Content between Lesion # and the next Lesion #
        lesion_matches = re.split(r'Lesion # \d+', text)
        lesion_matches = lesion_matches[1:]
        for i, content in enumerate(lesion_matches, start=1):
            lesion_contents[f'Lesion{i}_content'] = content.strip()
        return lesion_contents


    def extract_lesion_measurements(self):
        #Scores and values extrcation for all lesions
        for index, row in self.df.iterrows():
            num = row['extracted_lesion_no']
            if num:
                try:
                    for i in range(1, num + 1):
                        comment = row[f'Lesion{i}_content']
                        if isinstance(comment, str):
                            # Extract lesion size
                            lesion_size_match = re.search(r'((\d+(\.\d+)?) ?x ?(\d+(\.\d+)?) ?(x ?\d+(\.\d+)?)? ?(?:mm|cm))', comment)
                            if lesion_size_match:
                                extracted_lesion_size = lesion_size_match.group(1).strip()
                                self.df.at[index, f'extracted_lesion{i}_size'] = extracted_lesion_size

                            # Extract lesion volume
                            lesion_volume_match = re.search(r'(\d+(\.\d+)?) ?cc', comment)
                            if lesion_volume_match:
                                extracted_lesion_volume = float(lesion_volume_match.group(1))
                                self.df.at[index, f'extracted_lesion{i}_volume'] = extracted_lesion_volume


                            # Update extracted_PZ and extracted_TZ
                            if re.search(r'\bPeripheral zone\b|\bperipheral\b|\bPeripheral\b', comment):
                                self.df.at[index, f'extracted_PZ{i}'] = 1
                            if re.search(r'\bTransitional zone\b|\btransitional\b|\bTransitional\b', comment):
                                self.df.at[index, f'extracted_TZ{i}'] = 1


                            # Update location presence columns
                            if re.search(r'\bapex\b', comment):
                                self.df.at[index, f'extracted_location_apex{i}'] = 1
                            if re.search(r'\bbase\b', comment):
                                self.df.at[index, f'extracted_location_base{i}'] = 1
                            if re.search(r'\bmid gland\b|\bmidgland\b', comment):
                                self.df.at[index, f'extracted_location_midgland{i}'] = 1


                            # Extract laterality
                            if re.search(r'\bleft\b', comment, re.IGNORECASE):
                                self.df.at[index, f'extracted_later{i}'] = 'left'
                            elif re.search(r'\bright\b', comment, re.IGNORECASE):
                                self.df.at[index, f'extracted_later{i}'] = 'right'

                            # Extract T2WI score
                            t2wi_score_match = re.search(r'T2WI score (\d+)', comment, re.IGNORECASE)
                            if t2wi_score_match:
                                self.df.at[index, f'extracted_T2{i}'] = t2wi_score_match.group(1)

                            # Extract DWI score
                            dwi_score_match = re.search(r'DWI score (\d+)', comment, re.IGNORECASE)
                            if dwi_score_match:
                                self.df.at[index, f'extracted_DWI{i}'] = dwi_score_match.group(1)

                            # Extract ADC
                            adc_match = re.search(r'ADC\s*([<>]?=?\s?\d+)', comment, re.IGNORECASE)
                            if adc_match:
                                extracted_adc = adc_match.group(1).strip()
                                self.df.at[index, f'extracted_ADC{i}'] = extracted_adc
                            
                            # Extract DCE
                            dce_match = re.search(r'DCE: (\w+)', comment, re.IGNORECASE)
                            if dce_match:
                                extracted_dce = dce_match.group(1).strip().lower()
                                if extracted_dce == 'positive':
                                    self.df.at[index, f'extracted_DCE{i}'] = 'P'
                                elif extracted_dce == 'negative':
                                    self.df.at[index, f'extracted_DCE{i}'] = 'N'

                            # Extract Overall PIRADS category and update extracted_overall_pirads_category column
                            pirads_match = re.search(r'Overall category: (PIRADS\s?\d)', comment, re.IGNORECASE)
                            if pirads_match:
                                extracted_pirads = pirads_match.group(1).lower()
                                if 'pirads 1' in extracted_pirads or 'pirads1' in extracted_pirads:
                                    self.df.at[index, f'extracted_overall_pirads_category{i}'] = 1
                                elif 'pirads 2' in extracted_pirads or 'pirads2' in extracted_pirads:
                                    self.df.at[index, f'extracted_overall_pirads_category{i}'] = 2
                                elif 'pirads 3' in extracted_pirads or 'pirads3' in extracted_pirads:
                                    self.df.at[index, f'extracted_overall_pirads_category{i}'] = 3
                                elif 'pirads 4' in extracted_pirads or 'pirads4' in extracted_pirads:
                                    self.df.at[index, f'extracted_overall_pirads_category{i}'] = 4
                                elif 'pirads 5' in extracted_pirads or 'pirads5' in extracted_pirads:
                                    self.df.at[index, f'extracted_overall_pirads_category{i}'] = 5
                except Exception as e:
                    print(e)

    def create_max_size_columns(self):
        def convert_to_cm(dimensions):
            cm_values = []
            dimension_matches = re.findall(r'(\d+(\.\d+)?)\s*(?:mm)?', dimensions)
            if dimension_matches:
                for match in dimension_matches:
                    cm_value = float(match[0]) / 10 if 'mm' in dimensions else float(match[0])
                    cm_values.append(cm_value)
            return max(cm_values) if cm_values else None

        for i in range(1, 6):
            self.df[f'max_extracted_lesion{i}_size(cm)'] = self.df[f'extracted_lesion{i}_size'].astype(str).apply(convert_to_cm)


    def extract_local_staging_paragraph(self):
        self.df['extracted_local_staging_paragraph'] = ''
        for index, row in self.df.iterrows():
            comment = row['NARRATIVE']
            if isinstance(comment, str):  
                # Content between "LOCAL STAGING" and "LYMPH NODES"
                local_staging_extracted = re.search(r'LOCAL STAGING.*?(?=LYMPH NODES)', comment, re.DOTALL)
                if local_staging_extracted:
                    local_staging_comment = local_staging_extracted.group(0).strip()
                    self.df.at[index, 'extracted_local_staging_paragraph'] = local_staging_comment


    def extract_capsule_invasion(self):
        self.df['extracted_capsule_invasion'] = ''
        for index, row in self.df.iterrows():
            local_staging_comment = row['extracted_local_staging_paragraph']
            if isinstance(local_staging_comment, str):
                # Content between "Capsule:" and "Neurovascular bundle invasion:"
                capsule_to_neurovascular_match = re.search(r'Capsule:(.*?)Neurovascular bundle invasion:', local_staging_comment, re.DOTALL | re.IGNORECASE)
                if capsule_to_neurovascular_match:
                    capsule_to_neurovascular_text = capsule_to_neurovascular_match.group(1).strip() 
                    if not capsule_to_neurovascular_text.strip():
                        self.df.at[index, 'extracted_capsule_invasion'] = ''
                    else:
                        if re.search(r'\b(?:negative|intact|absent)\b', capsule_to_neurovascular_text, re.IGNORECASE):
                            self.df.at[index, 'extracted_capsule_invasion'] = 'absent'
                        elif re.search(r'\b(?:equivocal|indeterminate|no definite evidence of extracapsular extension|no macroscopic evidence of transcapsular extension|no evidence of macroscopic transcapsular extension|no macroscopic evidence of extracapsular extension|no macroscopic extraprostatic extension)\b', capsule_to_neurovascular_text, re.IGNORECASE):
                            self.df.at[index, 'extracted_capsule_invasion'] = 'equivocal'
                        elif re.search(r'\b(?:positive|present)\b', capsule_to_neurovascular_text, re.IGNORECASE):
                            self.df.at[index, 'extracted_capsule_invasion'] = 'present'
                        else:
                            self.df.at[index, 'extracted_capsule_invasion'] = ''


    def extract_neurovascular_invasion(self):
        self.df['extracted_neurovascular_invasion'] = ''
        for index, row in self.df.iterrows():
            local_staging_comment = row['extracted_local_staging_paragraph']
            if isinstance(local_staging_comment, str):
                # Content between "Neurovascular bundle invasion:" and "Seminal vesicles invasion:"
                neurovascular_to_seminal_match = re.search(r'Neurovascular bundle invasion:(.*?)Seminal vesicles invasion:', local_staging_comment, re.DOTALL | re.IGNORECASE)
                
                if neurovascular_to_seminal_match:
                    neurovascular_to_seminal_text = neurovascular_to_seminal_match.group(1).strip() 
                    
                    if not neurovascular_to_seminal_text.strip():
                        self.df.at[index, 'extracted_neurovascular_invasion'] = ''
                    else:
                        # Update the extracted_neurovascular_invasion column
                        if re.search(r'\b(?:negative|intact|absent)\b', neurovascular_to_seminal_text, re.IGNORECASE):
                            self.df.at[index, 'extracted_neurovascular_invasion'] = 'absent'
                        elif re.search(r'\b(?:equivocal|indeterminate|no definite evidence of involvement)\b', neurovascular_to_seminal_text, re.IGNORECASE):
                            self.df.at[index, 'extracted_neurovascular_invasion'] = 'equivocal'
                        elif re.search(r'\b(?:positive|present)\b', neurovascular_to_seminal_text, re.IGNORECASE):
                            self.df.at[index, 'extracted_capsule_invasion'] = 'present'
                        else:
                            self.df.at[index, 'extracted_capsule_invasion'] = ''


    def extract_seminal_vesicle_invasion(self):
        self.df['extracted_seminal_vesicle_invasion'] = ''
        for index, row in self.df.iterrows():
            local_staging_comment = row['extracted_local_staging_paragraph']
            if isinstance(local_staging_comment, str):
                # Content between "Seminal vesicles invasion:" and "Other organ invasion:"
                seminal_to_other_match = re.search(r'Seminal vesicles invasion:(.*?)Other organ invasion:', local_staging_comment, re.DOTALL | re.IGNORECASE)
                
                if seminal_to_other_match:
                    seminal_to_other_text = seminal_to_other_match.group(1).strip() 
                    if not seminal_to_other_text.strip():
                        self.df.at[index, 'extracted_seminal_vesicle_invasion'] = ''
                    else:
                        # update the extracted_seminal_vesicle_invasion column
                        if re.search(r'\b(?:negative|intact|absent)\b', seminal_to_other_text, re.IGNORECASE):
                            self.df.at[index, 'extracted_seminal_vesicle_invasion'] = 'absent'
                        elif re.search(r'\b(?:equivocal|indeterminate|under distended|without definite evidence of invasion|without definite invasion|no evidence of direct invasion|not definite|underdistention)\b', seminal_to_other_text, re.IGNORECASE):
                            self.df.at[index, 'extracted_seminal_vesicle_invasion'] = 'equivocal'
                        elif re.search(r'\b(?:positive|present)\b', seminal_to_other_text, re.IGNORECASE):
                            self.df.at[index, 'extracted_capsule_invasion'] = 'present'
                        else:
                            self.df.at[index, 'extracted_capsule_invasion'] = ''


    def extract_lymph_nodes(self):
        for index, row in self.df.iterrows():
            comment = row['NARRATIVE']
            if isinstance(comment, str):  
                # Content between "LYMPH NODES" and "BONES"
                lymph_nodes_extracted = re.search(r'LYMPH NODES.*?(?=BONES)', comment, re.DOTALL)
                if lymph_nodes_extracted:
                    lymph_nodes_comment = lymph_nodes_extracted.group(0).strip()
                    self.df.at[index, 'extracted_lymph_nodes_paragraph'] = lymph_nodes_comment

                    # Categorize lymph nodes information
                    lymph_nodes_comment = self.df.at[index, 'extracted_lymph_nodes_paragraph']
                    if not lymph_nodes_comment.strip():
                        self.df.at[index, 'extracted_lymph_nodes'] = ''
                    else:
                        if re.search(r'\b(?:negative|intact|absent|no enlarged pelvic lymph nodes)\b', lymph_nodes_comment, re.IGNORECASE):
                            self.df.at[index, 'extracted_lymph_nodes'] = 'absent'
                        elif re.search(r'\b(?:equivocal|indeterminate)\b', lymph_nodes_comment, re.IGNORECASE):
                            self.df.at[index, 'extracted_lymph_nodes'] = 'equivocal'
                        elif re.search(r'\b(?:positive|present)\b', lymph_nodes_comment, re.IGNORECASE):
                            self.df.at[index, 'extracted_lymph_nodes'] = 'present'
                        else:
                            self.df.at[index, 'extracted_lymph_nodes'] = '' 


    def extract_bones_information(self):
        self.df['extracted_bones_paragraph'] = ''
        for index, row in self.df.iterrows():
            comment = row['NARRATIVE']
            if isinstance(comment, str):    
                # Content between "BONES" and "OTHER FINDINGS"
                bones_to_other_findings = re.search(r'BONES.*?(?=OTHER FINDINGS)', comment, re.DOTALL)
                if bones_to_other_findings:
                    bones_comment = bones_to_other_findings.group(0).strip()
                    self.df.at[index, 'extracted_bones_paragraph'] = bones_comment
                else:
                    # Not found, then check for "BONES" and "Other Findings"
                    bones_to_other_findings_alt = re.search(r'BONES.*?(?=Other Findings)', comment, re.DOTALL | re.IGNORECASE)
                    if bones_to_other_findings_alt:
                        bones_comment = bones_to_other_findings_alt.group(0).strip()
                        self.df.at[index, 'extracted_bones_paragraph'] = bones_comment
                    else:
                        # Still not found, check for "BONES" and "PROSTATE MRI TECHNIQUE"
                        bones_to_prostate_mri = re.search(r'BONES.*?(?=PROSTATE MRI TECHNIQUE)', comment, re.DOTALL | re.IGNORECASE)
                        if bones_to_prostate_mri:
                            bones_comment = bones_to_prostate_mri.group(0).strip()
                            self.df.at[index, 'extracted_bones_paragraph'] = bones_comment

        self.df['extracted_bones'] = ''
        for index, row in self.df.iterrows():
            bones_comment = row['extracted_bones_paragraph'] 
            if not bones_comment.strip():
                self.df.at[index, 'extracted_bones'] = ''
            else: 
                if re.search(r'\b(?:negative|intact|absent|no aggressive bone lesions|no acute or suspicious|no suspicious bone)\b', bones_comment, re.IGNORECASE):
                    self.df.at[index, 'extracted_bones'] = 'absent'
                elif re.search(r'\b(?:equivocal|indeterminate)\b', bones_comment, re.IGNORECASE):
                    self.df.at[index, 'extracted_bones'] = 'equivocal'
                elif re.search(r'\b(?:positive|present)\b', bones_comment, re.IGNORECASE):
                            self.df.at[index, 'extracted_bones'] = 'present'
                else:
                    self.df.at[index, 'extracted_bones'] = ''  



    def save_data(self, output_file):
        try:
            self.df.to_excel(output_file)
            print(f"Data has been processed and saved to {output_file}")
        except Exception as e:
            print(f"Error saving the output file: {e}")

    def process_prostate_data(self, input_file, output_file):
        self.load_data(input_file)
        self.extract_lesion_paragraph()
        self.extract_lesion_contents()
        self.extract_lesion_measurements()
        self.create_max_size_columns()
        self.extract_local_staging_paragraph()
        self.extract_capsule_invasion()
        self.extract_neurovascular_invasion()
        self.extract_seminal_vesicle_invasion()
        self.extract_lymph_nodes()
        self.extract_bones_information()
        self.save_data(output_file)
