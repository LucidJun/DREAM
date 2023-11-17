import dream

FEATURES_JSON = dream.__path__[0] + '/feature_extraction/features.json'

settings0 = dream.load_json(FEATURES_JSON)

settings1 = dream.get_features_by_domain('statistical')

settings2 = dream.get_features_by_domain('temporal')

settings3 = dream.get_features_by_domain('spectral')

settings4 = dream.get_features_by_domain(None)

# settings5 = dream.extract_sheet('Features')

settings6 = dream.get_features_by_tag('audio')

settings7 = dream.get_features_by_tag('inertial')

settings8 = dream.get_features_by_tag('ecg')

settings9 = dream.get_features_by_tag('eeg')

settings10 = dream.get_features_by_tag('emg')

settings11 = dream.get_features_by_tag(None)
