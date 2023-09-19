# Python code here
import os
import requests
import zipfile
import io
import glob
import pandas as pd
import numpy as np
import pathlib
from tqdm import tqdm
#for now i choose scipy as it offers a lot  without having to install additional libraries but maybe librosa can also be an option
import scipy
from conf import CUSTOM_SETTINGS,RAVDESS_EMOTION_TO_LABEL,RAVDESS_LABEL_TO_EMOTION,MAIN_FOLDER,DATASETS_FOLDER,OUTPUTS_FOLDER

#TODO: add comments and dockstrings etc
def example_run():
    """
    A basic print function to verify if docker is running.
    :return: None
    """
    print(f'Docker for preprocessing component Audio has run. Conf from configuration.json file: {CUSTOM_SETTINGS}')

def preprocess():
    dataset_name = CUSTOM_SETTINGS['dataset_config']['dataset_name']
    full_dataset_path = os.path.join(DATASETS_FOLDER,dataset_name)

    #add check if ravdess folder exist, if not : download and create
    if not os.path.isdir(full_dataset_path):
        print(f"no existing {dataset_name} folder found, download will start")
        zip_file_url = "https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip?download=1"
        r = requests.get(zip_file_url, stream=True)
        progress_bar = tqdm(total=int(r.headers.get('content-length', 0)), unit='B', unit_scale=True, desc='download progress of RAVDESS dataset')
        dat = b''.join(x for x in r.iter_content(chunk_size=16384) if progress_bar.update(len(x)) or True)
        z = zipfile.ZipFile(io.BytesIO(dat))
        z.extractall(full_dataset_path)
    else:
        print(f"{dataset_name} folder exists at {full_dataset_path}, will use available data")

    all_subject_dirs = os.listdir(full_dataset_path)
    print(f"found a total of {len(all_subject_dirs)} subjects inside the {dataset_name} dataset")

    train_split,val_split,test_split = process_dataset(full_dataset_path,all_subject_dirs)

    print ('writing CSV files containing the splits to storage')
    pd.DataFrame.from_dict(train_split).to_csv(os.path.join(OUTPUTS_FOLDER,'train.csv'))
    pd.DataFrame.from_dict(val_split).to_csv(os.path.join(OUTPUTS_FOLDER,'val.csv'))
    pd.DataFrame.from_dict(test_split).to_csv(os.path.join(OUTPUTS_FOLDER,'test.csv'))

def process_dataset(full_dataset_path,all_subjects_dirs):
    train_split = {'files':[],'labels':[]}
    val_split = {'files':[],'labels':[]}
    test_split = {'files':[],'labels':[]}

    if CUSTOM_SETTINGS['pre_processing_config']['create_splits']:
        train_subjects = all_subjects_dirs[int(0.2*len(all_subjects_dirs)):-int(0.2*len(all_subjects_dirs))]
        val_subjects = all_subjects_dirs[:int(0.2*len(all_subjects_dirs))]
        test_subjects = all_subjects_dirs[-int(0.2*len(all_subjects_dirs)):]
    else:
        train_subjects = all_subjects_dirs
        val_subjects = []
        test_subjects = []

    print('train: ',train_subjects)
    print('val: ', val_subjects)
    print('test: ', test_subjects)

    splits_phase = {'train':train_split,'val':val_split,'test':test_split}
    subjects_phase = {'train':train_subjects,'val':val_subjects,'test':test_subjects}

    # get the right function to use, and create path to save files to
    self_functions = {"normalize":normalize,'standardize':standardize,'only_resample':no_preprocessing}
    preprocessing_to_aply = self_functions[CUSTOM_SETTINGS['pre_processing_config']['process']]
    pathlib.Path(os.path.join(OUTPUTS_FOLDER,'preprocessed',CUSTOM_SETTINGS['pre_processing_config']['process'])).mkdir(parents=True, exist_ok=True)

    for phase in ['train','val','test']:
        split = splits_phase[phase]
        subjects = subjects_phase[phase]
        for subject_path in tqdm(subjects,desc=f"preprocessing {phase} set"):
            all_subject_audio_files = glob.glob(os.path.join(f"{full_dataset_path}\{subject_path}", '*.wav'))
            all_subject_audio = []
            loaded_files = []
            for audio_path in all_subject_audio_files:
                sr,audio = scipy.io.wavfile.read(audio_path)

                #check for multi-channel audio
                if len(audio.shape)>1:
                    audio = np.mean(audio,axis=1)

                #TODO: check if resampling should happen before or after standardization or not
                resampled_audio = resample_audio_signal(audio,sr,CUSTOM_SETTINGS['pre_processing_config']['target_sr'])
                #TODO check if resampling rate is as desired by looking at ration between origin/target +/- tolerance
                all_subject_audio.append(resampled_audio)
                #TODO: padding ?

                loaded_files.append(audio_path)

            all_subject_audio_standardized = preprocessing_to_aply(all_subject_audio)

            processed_file_paths = []
            processed_file_labels = []
            for file_name,processed_audio in zip(loaded_files,all_subject_audio_standardized):
                filename = '_'.join(file_name.split('\\')[-3:])
                processed_file_labels.append(RAVDESS_LABEL_TO_EMOTION[file_name.split('-')[2]])
                filepath = os.path.join(OUTPUTS_FOLDER,'preprocessed',CUSTOM_SETTINGS['pre_processing_config']['process'],filename)
                processed_file_paths.append(filepath)
                scipy.io.wavfile.write(filepath, CUSTOM_SETTINGS['pre_processing_config']['target_sr'], processed_audio.astype(np.float32))

            split['files'].extend(processed_file_paths)
            split['labels'].extend(processed_file_labels)

    
    return train_split,val_split,test_split
    
def normalize(subject_all_audio):
    # normalize : transformed into a range between -1 and 1 by normalization for each speaker
    #TODO: check if actually works
    min = np.min(np.hstack(subject_all_audio))
    max = np.max(np.hstack(subject_all_audio))

    subject_all_normalized_audio = [2*(au-min)/(max-min)-1 for au in subject_all_audio]
    return subject_all_normalized_audio

def standardize(subject_all_audio):
    # standardize : divided by the standard deviation after the mean has been subtracted
    # 0 mean, unit variance for each speaker

    #as the encoding is float32, could maybe cause under/overflow?
    mean = np.mean(np.hstack(subject_all_audio))
    std = np.std(np.hstack(subject_all_audio))
    subject_all_standardized_audio = [(au-mean)/std for au in subject_all_audio]

    # TODO: add check if mean is 0+-tol and std 1+-tol

    return subject_all_standardized_audio

def no_preprocessing(subject_all_audio):
    # don't apply any pre-processing and return audio as is
    return subject_all_audio

def resample_audio_signal(audio,sample_rate,target_rate):
    number_of_samples = round(len(audio) * float(target_rate) / sample_rate)
    resampled_audio = scipy.signal.resample(audio, number_of_samples)
    return resampled_audio



if __name__ == '__main__':
    preprocess()
    #example_run()
