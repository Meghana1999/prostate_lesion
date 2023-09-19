import pandas as pd
import re
import pandas as pd
import re
import numpy as np
import nltk
from nltk.translate.bleu_score import sentence_bleu 
from sklearn.metrics import r2_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer


class ProstateDataProcessor:
    def process_prostate_data(self, input_file, output_file):
        print('Started CODE EXCECUTION')
        # Load data from the input Excel file
        try:
            df = pd.read_excel(input_file)
        except Exception as e:
            print(f"Error loading the input file: {e}")
            return

        
        df['extracted_lesion_paragraph'] = ''
        df['extracted_lesion_no'] = None

        
        for index, row in df.iterrows():
            comment = row['NARRATIVE']
            if isinstance(comment, str):
                # Extracting the content between "Lesion #" and "LOCAL STAGING"
                extracted = re.search(r'Lesion #.*?(?=LOCAL STAGING|LYMPH NODES)', comment, re.DOTALL)
                if extracted:
                    extracted_comment = extracted.group(0).strip()
                    df.at[index, 'extracted_lesion_paragraph'] = extracted_comment

                    # Extracting lesion number
                    lesion_numbers = re.findall(r'Lesion # (\d+)', extracted_comment)
                    if lesion_numbers:
                        highest_lesion_no = max(map(int, lesion_numbers))
                        df.at[index, 'extracted_lesion_no'] = highest_lesion_no

        # Extract content for each lesion
        def extract_lesion_contents(text):
            lesion_contents = {}

            # Content between Lesion # and the next Lesion #
            lesion_matches = re.split(r'Lesion # \d+', text)
            lesion_matches = lesion_matches[1:]
            for i, content in enumerate(lesion_matches, start=1):
                lesion_contents[f'Lesion{i}_content'] = content.strip()
            return lesion_contents

        # New column for extracted lesion contents
        df['Lesion_contents'] = df['NARRATIVE'].apply(extract_lesion_contents)
        max_lesion_count = df['Lesion_contents'].apply(len).max()

        # Generate column names for all possible lesions
        column_names = [f'Lesion{i}_content' for i in range(1, max_lesion_count + 1)]
        # Separate columns with generated column names
        df_new = pd.concat([df['Lesion_contents'].apply(lambda x: x.get(f'Lesion{i}_content', '')) for i in range(1, max_lesion_count + 1)], axis=1)
        df_new.columns = column_names

    
        df = pd.concat([df, df_new], axis=1)
        
        #Scores and values extrcation for all lesions
        for index, row in df.iterrows():
            num = row['extracted_lesion_no']
            if num:
                try:
                    for i in range(1,num+1):
                        comment = row[f'Lesion{i}_content']
                        if isinstance(comment, str):
                            extracted_comment = comment  
                            # Extracting Lesion Size 
                            lesion_size_match = re.search(r'((\d+(\.\d+)?) ?x ?(\d+(\.\d+)?) ?(x ?\d+(\.\d+)?)? ?(?:mm|cm))', extracted_comment)
                            if lesion_size_match:
                                extracted_lesion_size = lesion_size_match.group(1).strip()
                                df.at[index, f'extracted_lesion{i}_size'] = extracted_lesion_size

                            # Extract lesion volume
                            lesion_volume_match = re.search(r'(\d+(\.\d+)?) ?cc', extracted_comment)
                            if lesion_volume_match:
                                extracted_lesion_volume = float(lesion_volume_match.group(1))
                                df.at[index, f'extracted_lesion{i}_volume'] = extracted_lesion_volume

                            # Check for keywords in the extracted comment to update extracted_PZ and extracted_TZ
                            if re.search(r'\bPeripheral zone\b|\bperipheral\b|\bPeripheral\b', extracted_comment):
                                df.at[index, f'extracted_PZ{i}'] = 1
                            if re.search(r'\bTransitional zone\b|\btransitional\b|\bTransitional\b', extracted_comment):
                                df.at[index, f'extracted_TZ{i}'] = 1

                            # Check for keywords in the extracted comment to update location presence columns
                            if re.search(r'\bapex\b', extracted_comment):
                                df.at[index, f'extracted_location_apex{i}'] = 1
                            if re.search(r'\bbase\b', extracted_comment):
                                df.at[index, f'extracted_location_base{i}'] = 1
                            if re.search(r'\bmid gland\b|\bmidgland\b', extracted_comment):
                                df.at[index, f'extracted_location_midgland{i}'] = 1

                            # Check for keywords in the extracted comment to update laterality column
                            if re.search(r'\bleft\b|\bLeft\b', extracted_comment):
                                df.at[index, f'extracted_later{i}'] = 'left'
                            if re.search(r'\bright\b|\bRight\b', extracted_comment):
                                df.at[index, f'extracted_later{i}'] = 'right'

                            t2wi_score_match = re.search(r'T2WI score (\d+)', extracted_comment, re.IGNORECASE)
                            if t2wi_score_match:
                                df.at[index, f'extracted_T2{i}'] = t2wi_score_match.group(1)
                            
                            # Extract DWI score
                            dwi_score_match = re.search(r'DWI score (\d+)', extracted_comment, re.IGNORECASE)
                            if dwi_score_match:
                                df.at[index, f'extracted_DWI{i}'] = dwi_score_match.group(1)
                            
                            # Extract ADC 
                            adc_match = re.search(r'ADC\s*([<>]?=?\s?\d+)', extracted_comment, re.IGNORECASE)
                            if adc_match:
                                extracted_adc = adc_match.group(1).strip()
                                df.at[index, f'extracted_ADC{i}'] = extracted_adc
                            
                            # Extract DCE and update extracted_DCE column
                            dce_match = re.search(r'DCE: (\w+)', extracted_comment, re.IGNORECASE)
                            if dce_match:
                                extracted_dce = dce_match.group(1).strip().lower()
                                if extracted_dce == 'positive':
                                    df.at[index, f'extracted_DCE{i}'] = 'P'
                                elif extracted_dce == 'negative':
                                    df.at[index, f'extracted_DCE{i}'] = 'N' 

                            # Extract Overall PIRADS category and update extracted_overall_pirads_category column
                            pirads_match = re.search(r'Overall category: (PIRADS\s?\d)', extracted_comment, re.IGNORECASE)
                            if pirads_match:
                                extracted_pirads = pirads_match.group(1).lower()
                                if 'pirads 1' in extracted_pirads or 'pirads1' in extracted_pirads:
                                    df.at[index, f'extracted_overall_pirads_category{i}'] = 1
                                elif 'pirads 2' in extracted_pirads or 'pirads2' in extracted_pirads:
                                    df.at[index, f'extracted_overall_pirads_category{i}'] = 2
                                elif 'pirads 3' in extracted_pirads or 'pirads3' in extracted_pirads:
                                    df.at[index, f'extracted_overall_pirads_category{i}'] = 3
                                elif 'pirads 4' in extracted_pirads or 'pirads4' in extracted_pirads:
                                    df.at[index, f'extracted_overall_pirads_category{i}'] = 4
                                elif 'pirads 5' in extracted_pirads or 'pirads5' in extracted_pirads:
                                    df.at[index, f'extracted_overall_pirads_category{i}'] = 5
                except Exception as e:
                    print(e)
                            
        #Lesion max size in centimeters cm
        def convert_to_cm(dimensions):
            cm_values = []  
            dimension_matches = re.findall(r'(\d+(\.\d+)?)\s*(?:mm)?', dimensions) 
            if dimension_matches: 
                for match in dimension_matches:
                    cm_value = float(match[0]) / 10 if 'mm' in dimensions else float(match[0])
                    cm_values.append(cm_value) 
            return max(cm_values) if cm_values else None



        #create the 'max size' column
        df['max_extracted_lesion1_size(cm)'] = df['extracted_lesion1_size'].astype(str).apply(convert_to_cm)
        df['max_extracted_lesion2_size(cm)'] = df['extracted_lesion2_size'].astype(str).apply(convert_to_cm)
        df['max_extracted_lesion3_size(cm)'] = df['extracted_lesion3_size'].astype(str).apply(convert_to_cm)
        df['max_extracted_lesion4_size(cm)'] = df['extracted_lesion4_size'].astype(str).apply(convert_to_cm)
        df['max_extracted_lesion5_size(cm)'] = df['extracted_lesion5_size'].astype(str).apply(convert_to_cm)


        # EXTRACTING LOCAL STAGING PARAGRAPH
        df['extracted_local_staging_paragraph'] = ''
        for index, row in df.iterrows():
            comment = row['NARRATIVE']
            if isinstance(comment, str):  
                # Extract content between "LOCAL STAGING" and "LYMPH NODES"
                local_staging_extracted = re.search(r'LOCAL STAGING.*?(?=LYMPH NODES)', comment, re.DOTALL)
                if local_staging_extracted:
                    local_staging_comment = local_staging_extracted.group(0).strip()
                    df.at[index, 'extracted_local_staging_paragraph'] = local_staging_comment
                    

        df['extracted_capsule_invasion'] = ''
        for index, row in df.iterrows():
            local_staging_comment = row['extracted_local_staging_paragraph']
            # Extract the portion between "Capsule:" and "Neurovascular bundle invasion:"
            capsule_to_neurovascular_match = re.search(r'Capsule:(.*?)Neurovascular bundle invasion:', local_staging_comment, re.DOTALL | re.IGNORECASE)
            if capsule_to_neurovascular_match:
                capsule_to_neurovascular_text = capsule_to_neurovascular_match.group(1).strip()
                if not capsule_to_neurovascular_text.strip():
                    df.at[index, 'extracted_capsule_invasion'] = ''
                else:
                    if re.search(r'\b(?:negative|intact|absent)\b', capsule_to_neurovascular_text, re.IGNORECASE):
                        df.at[index, 'extracted_capsule_invasion'] = 'absent'
                    elif re.search(r'\b(?:equivocal|indeterminate|no definite evidence of extracapsular extension|no macroscopic evidence of transcapsular extension|no evidence of macroscopic transcapsular extension|no macroscopic evidence of extracapsular extension|no macroscopic extraprostatic extension)\b', capsule_to_neurovascular_text, re.IGNORECASE):
                        df.at[index, 'extracted_capsule_invasion'] = 'equivocal'
                    else:
                        df.at[index, 'extracted_capsule_invasion'] = 'present'
                        
                        

        df['extracted_neurovascular_invasion'] = ''
        for index, row in df.iterrows():
            local_staging_comment = row['extracted_local_staging_paragraph']

            # Extract the portion between "Neurovascular bundle invasion:" and "Seminal vesicles invasion:"
            neurovascular_to_seminal_match = re.search(r'Neurovascular bundle invasion:(.*?)Seminal vesicles invasion:', local_staging_comment, re.DOTALL | re.IGNORECASE)
            
            if neurovascular_to_seminal_match:
                neurovascular_to_seminal_text = neurovascular_to_seminal_match.group(1).strip() 
                
                if not neurovascular_to_seminal_text.strip():
                    df.at[index, 'extracted_neurovascular_invasion'] = ''
                else:
                    # Check for specific conditions and update the extracted_neurovascular_invasion column
                    if re.search(r'\b(?:negative|intact|absent)\b', neurovascular_to_seminal_text, re.IGNORECASE):
                        df.at[index, 'extracted_neurovascular_invasion'] = 'absent'
                    elif re.search(r'\b(?:equivocal|indeterminate|no definite evidence of involvement)\b', neurovascular_to_seminal_text, re.IGNORECASE):
                        df.at[index, 'extracted_neurovascular_invasion'] = 'equivocal'
                    else:
                        df.at[index, 'extracted_neurovascular_invasion'] = 'present'
                        

        df['extracted_seminal_vesicle_invasion'] = ''
        for index, row in df.iterrows():
            local_staging_comment = row['extracted_local_staging_paragraph']

            # Extract the portion between "Seminal vesicles invasion:" and "Other organ invasion:"
            seminal_to_other_match = re.search(r'Seminal vesicles invasion:(.*?)Other organ invasion:', local_staging_comment, re.DOTALL | re.IGNORECASE)
            
            if seminal_to_other_match:
                seminal_to_other_text = seminal_to_other_match.group(1).strip() 
                if not seminal_to_other_text.strip():
                    df.at[index, 'extracted_seminal_vesicle_invasion'] = ''
                else:
                    # Check for specific conditions and update the extracted_seminal_vesicle_invasion column
                    if re.search(r'\b(?:negative|intact|absent)\b', seminal_to_other_text, re.IGNORECASE):
                        df.at[index, 'extracted_seminal_vesicle_invasion'] = 'absent'
                    elif re.search(r'\b(?:equivocal|indeterminate|under distended|without definite evidence of invasion|without definite invasion|no evidence of direct invasion|not definite|underdistention)\b', seminal_to_other_text, re.IGNORECASE):
                        df.at[index, 'extracted_seminal_vesicle_invasion'] = 'equivocal'
                    else:
                        df.at[index, 'extracted_seminal_vesicle_invasion'] = 'present'
                        

        ## Extracting Lymph Node
        df['extracted_lymph_nodes_paragraph'] = ''
        for index, row in df.iterrows():
            comment = row['NARRATIVE']
            if isinstance(comment, str):  
                # Extracting content between "LYMPH NODES" and "BONES"
                lymph_nodes_extracted = re.search(r'LYMPH NODES.*?(?=BONES)', comment, re.DOTALL)
                if lymph_nodes_extracted:
                    lymph_nodes_comment = lymph_nodes_extracted.group(0).strip()
                    df.at[index, 'extracted_lymph_nodes_paragraph'] = lymph_nodes_comment 
                    

        df['extracted_lymph_nodes'] = ''
        for index, row in df.iterrows():
            lymph_nodes_comment = row['extracted_lymph_nodes_paragraph'] 
            if not lymph_nodes_comment.strip():
                df.at[index, 'extracted_lymph_nodes'] = ''
            else:
                if re.search(r'\b(?:negative|intact|absent|no enlarged pelvic lymph nodes)\b', lymph_nodes_comment, re.IGNORECASE):
                    df.at[index, 'extracted_lymph_nodes'] = 'absent'
                elif re.search(r'\b(?:equivocal|indeterminate)\b', lymph_nodes_comment, re.IGNORECASE):
                    df.at[index, 'extracted_lymph_nodes'] = 'equivocal'
                else:
                    df.at[index, 'extracted_lymph_nodes'] = 'present'
                    

        ## EXTRACTING BONES
        df['extracted_bones_paragraph'] = ''
        for index, row in df.iterrows():
            comment = row['NARRATIVE']
            if isinstance(comment, str):    
                #Extracting content between "BONES" and "OTHER FINDINGS"
                bones_to_other_findings = re.search(r'BONES.*?(?=OTHER FINDINGS)', comment, re.DOTALL)
                if bones_to_other_findings:
                    bones_comment = bones_to_other_findings.group(0).strip()
                    df.at[index, 'extracted_bones_paragraph'] = bones_comment
                else:
                    #If not found, check for "BONES" and "Other Findings"
                    bones_to_other_findings_alt = re.search(r'BONES.*?(?=Other Findings)', comment, re.DOTALL | re.IGNORECASE)
                    if bones_to_other_findings_alt:
                        bones_comment = bones_to_other_findings_alt.group(0).strip()
                        df.at[index, 'extracted_bones_paragraph'] = bones_comment
                    else:
                        #If still not found, check for "BONES" and "PROSTATE MRI TECHNIQUE"
                        bones_to_prostate_mri = re.search(r'BONES.*?(?=PROSTATE MRI TECHNIQUE)', comment, re.DOTALL | re.IGNORECASE)
                        if bones_to_prostate_mri:
                            bones_comment = bones_to_prostate_mri.group(0).strip()
                            df.at[index, 'extracted_bones_paragraph'] = bones_comment
                            

        df['extracted_bones'] = ''
        for index, row in df.iterrows():
            bones_comment = row['extracted_bones_paragraph'] 
            if not bones_comment.strip():
                df.at[index, 'extracted_bones'] = ''
            else: 
                if re.search(r'\b(?:negative|intact|absent|no aggressive bone lesions|no acute or suspicious|no suspicious bone)\b', bones_comment, re.IGNORECASE):
                    df.at[index, 'extracted_bones'] = 'absent'
                elif re.search(r'\b(?:equivocal|indeterminate)\b', bones_comment, re.IGNORECASE):
                    df.at[index, 'extracted_bones'] = 'equivocal'
                else:
                    df.at[index, 'extracted_bones'] = 'present'


        # df.to_excel("/mnt/storage/RAD_PATH/updated_lesion_results.xlsx") 
        # Saving processed data to the output Excel file
        try:
            df.to_excel(output_file)
            print(f"Data has been processed and saved to {output_file}")
        except Exception as e:
            print(f"Error saving the output file: {e}")
            return