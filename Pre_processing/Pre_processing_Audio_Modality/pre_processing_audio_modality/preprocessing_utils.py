import numpy as np
import scipy


def normalize(subject_all_audio):
    """
    normalize: transformed into a range between -1 and 1 by normalization for each speaker (min-max scaling)

    Args:
        subject_all_audio: a list containing the numpy arrays with all the audio data of the subject

    Returns:
        subject_all_normalized_audio: a list containing the normalized numpy arrays with audio from a subject
    """

    # could be heavily influenced by outliers
    min = np.min(np.hstack(subject_all_audio))
    max = np.max(np.hstack(subject_all_audio))

    subject_all_normalized_audio = [2 * (au - min) / (max - min) - 1 for au in subject_all_audio]

    return subject_all_normalized_audio


def standardize(subject_all_audio):
    """
    z-normalization to zero mean and unit variance for each speaker

    Args:
        subject_all_audio: a list containing the numpy arrays with all the audio data of the subject

    Returns:
        subject_all_standardized_audio: a list containing the standardized numpy arrays with audio from a subject
    """

    # as the encoding is float32, could maybe cause under/overflow?
    mean = np.mean(np.hstack(subject_all_audio))
    std = np.std(np.hstack(subject_all_audio))
    subject_all_standardized_audio = [(au - mean) / std for au in subject_all_audio]

    # check if mean is 0+-tol and std 1+-tol
    assert -0.05 < np.mean(np.hstack(subject_all_standardized_audio)) < 0.05, "mean is not equal to 0"
    assert 0.95 < np.std(np.hstack(subject_all_standardized_audio)) < 1.05, "std is not equal to 1"

    return subject_all_standardized_audio


def no_preprocessing(subject_all_audio):
    """
    No pre-processing applied

    Args:
        subject_all_audio: a list containing the numpy arrays with all the audio data of the subject

    Returns:
        subject_all_audio: a list containing the numpy arrays with all the audio data of the subject
    """
    return subject_all_audio


def resample_audio_signal(
        audio: np.ndarray,
        sample_rate: int,
        target_rate: int
):
    """
    Resample the given (audio) signal to a target frequency

    Args:
        audio: a numpy array with the audio data to resample
        sample_rate: the sample rate of the original signal
        target_rate: the target sample rate
    Returns:
        subject_all_audio: a list containing the numpy arrays with all the audio data of the subject
    """
    number_of_samples = round(len(audio) * float(target_rate) / sample_rate)
    resampled_audio = scipy.signal.resample(audio, number_of_samples)
    return resampled_audio
