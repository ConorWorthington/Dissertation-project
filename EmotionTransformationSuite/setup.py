from setuptools import setup

APP = ['main.py']
APP_NAME = "Emotion Transformation Suite"
DATA_FILES = ['genderModel.xml','emotion_model.h5','keypoints_model.h5','default.jpg','maleHappy.jpg'
    ,'maleSad.jpg','maleAngry.jpg','maleDisgust.jpg','maleNeutral.jpg','maleScared.jpg','maleSurprise.jpg','sad.jpg'
    ,'angry.jpg','neutral.jpg','scared.jpg','surprised.jpg','disgust.jpg','happy.jpg']
OPTIONS = {
    'iconfile' : 'newLogo.icns',
    'argv_emulation': True,
    'plist': {
        'CFBundleName': APP_NAME,
        'CFBundleDisplayName': APP_NAME,
        'CFBundleGetInfoString': "Emotional transformations",
        'CFBundleVersion': "0.1.0",
        'CFBundleShortVersionString': "0.1.0",
        'NSHumanReadableCopyright': "Copyright © 2019, Conor Worthington, All Rights Reserved"
    }
}

setup(
    app=APP,
    name=APP_NAME,
    data_files=DATA_FILES,
    options={'py2app': OPTIONS},
    setup_requires=['py2app'],
)