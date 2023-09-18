# Python code here
import os
import glob
import pandas as pd
import numpy as np

#for now i choose scipy as it offers a lot  without having to install additional libraries but maybe librosa can also be an option
import scipy
from conf import CUSTOM_SETTINGS

#TODO: add comments and dockstrings etc
def example_run():
    """
    A basic print function to verify if docker is running.
    :return: None
    """
    print(f'Docker for preprocessing component Audio has run. Conf from configuration.json file: {CUSTOM_SETTINGS}')

def preprocess():
    dataset_name = CUSTOM_SETTINGS['dataset_config']['dataset_name']
    full_dataset_path = f"Pre_processing\Pre_processing_Audio_Modality\datasets\{dataset_name}"
    all_subject_dirs = os.listdir(full_dataset_path)

    print(f"found a total of {len(all_subject_dirs)} subjects inside the {dataset_name} dataset")

    train_split,val_split,test_split = process_dataset(full_dataset_path,all_subject_dirs)

    train_df = pd.DataFrame.from_dict(train_split)
    train_df.to_csv("Pre_processing\Pre_processing_Audio_Modality\outputs\\train.csv")
    pd.DataFrame.from_dict(val_split).to_csv('Pre_processing\Pre_processing_Audio_Modality\outputs\\val.csv')
    pd.DataFrame.from_dict(test_split).to_csv('Pre_processing\Pre_processing_Audio_Modality\outputs\\test.csv')


    #TODO: write dictionaries to csvs

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

    splits = [train_split,val_split,test_split]
    subjects = [train_subjects,val_subjects,test_subjects]
    for split,subjects in zip(splits,subjects):
        for subject_path in subjects:
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

                loaded_files.append(audio_path)

            all_subject_audio_standardized = standardize(all_subject_audio)

            print(len(all_subject_audio_standardized))
            print(len(loaded_files))

            processed_file_names = []
            processed_file_labels = []
            for file_name,processed_audio in zip(loaded_files,all_subject_audio_standardized):
                filename = '_'.join(file_name.split('\\')[-3:])
                processed_file_labels.append(CUSTOM_SETTINGS['dataset_config']['label_to_emotion'][file_name.split('-')[2]])
                filepath = f"Pre_processing\Pre_processing_Audio_Modality\datasets\preprocessed\{filename}"
                processed_file_names.append(filename)
                scipy.io.wavfile.write(filepath, CUSTOM_SETTINGS['pre_processing_config']['target_sr'], processed_audio.astype(np.float32))

            split['files'].extend(processed_file_names)
            split['labels'].extend(processed_file_labels)

    
    return train_split,val_split,test_split
    
def normalize(subject_all_audio):
    # normalize : transformed into a range between -1 and 1 by normalization for each speaker

    subject_all_normalized_audio = []
    return subject_all_normalized_audio

def standardize(subject_all_audio):
    # standardize : divided by the standard deviation after the mean has been subtracted
    # 0 mean, unit variance for each speaker

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
