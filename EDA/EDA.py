

import os
import sys
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
subprocess.check_call([sys.executable,'-m','pip','install','parsinorm']) # !pip install parsinorm
import parsinorm

def Correction(df,column):
    mail_url_cleaner = parsinorm.Mail_url_cleaner()
    date_time_to_text = parsinorm.Date_time_to_text()
    general_normalization = parsinorm.General_normalization()
    telephone_number = parsinorm.Telephone_number()
    abbreviation = parsinorm.Abbreviation()
    TTS_normalization = parsinorm.TTS_normalization()
    special_numbers = parsinorm.Special_numbers()
    # tokenizer = parsinorm.Tokenizer()
    for idx in df.index:
        sen = df[column][idx]
        sen = mail_url_cleaner.find_mails_clean(sen)
        sen = mail_url_cleaner.find_urls_clean(sen)
        sen = date_time_to_text.date_to_text(sen)
        sen = date_time_to_text.time_to_text(sen)
        sen = general_normalization.alphabet_correction(sen)
        sen = general_normalization.semi_space_correction(sen)
        sen = general_normalization.english_correction(sen)
        sen = general_normalization.html_correction(sen)
        sen = general_normalization.arabic_correction(sen)
        sen = general_normalization.punctuation_correction(sen)
        sen = general_normalization.specials_chars(sen)
        sen = general_normalization.remove_emojis(sen)
        sen = general_normalization.unique_floating_point(sen)
        sen = general_normalization.remove_comma_between_numbers(sen)
        sen = general_normalization.number_correction(sen)
        sen = general_normalization.remove_not_desired_chars(sen)
        sen = general_normalization.remove_repeated_punctuation(sen)
        sen = telephone_number.find_phones_replace(sen)
        sen = abbreviation.replace_date_abbreviation(sen)
        sen = abbreviation.replace_persian_label_abbreviation(sen)
        sen = abbreviation.replace_law_abbreviation(sen)
        sen = abbreviation.replace_book_abbreviation(sen)
        sen = abbreviation.replace_other_abbreviation(sen)
        sen = abbreviation.replace_English_abbrevations(sen)
        sen = TTS_normalization.math_correction(sen)
        sen = TTS_normalization.replace_currency(sen)
        sen = TTS_normalization.replace_symbols(sen)
        sen = special_numbers.convert_numbers_to_text(sen)
        sen = special_numbers.replace_national_code(sen)
        sen = special_numbers.replace_shaba(sen)
        df[column][idx] = sen
    return df

def ReadDataFrame():
    if os.path.exists("./datasets/FarsTail/Train-word-Corrected.csv"):
        df_train = pd.read_csv('./datasets/FarsTail/Train-word-Corrected.csv')
        df_dev = pd.read_csv('./datasets/FarsTail/Val-word-Corrected.csv')
        df_test = pd.read_csv('./datasets/FarsTail/Test-word-Corrected.csv')
    else:
        df_train = pd.read_csv('./datasets/FarsTail/Train-word.csv', sep='	')
        df_dev = pd.read_csv('./datasets/FarsTail/Val-word.csv', sep='	')
        df_test = pd.read_csv('./datasets/FarsTail/Test-word.csv', sep='	')
        df_train = Correction(df_train, 'premise') ; print('train premise corrected.')
        df_train = Correction(df_train, 'hypothesis') ; print('train hypothesis corrected.')
        df_dev = Correction(df_dev, 'premise') ; print('val premise corrected.')
        df_dev = Correction(df_dev, 'hypothesis') ; print('val hypothesis corrected.')
        df_test = Correction(df_test, 'premise') ; print('test premise corrected.')
        df_test = Correction(df_test, 'hypothesis') ; print('test hypothesis corrected.')
        label_encoder = LabelEncoder()
        df_train['label_id'] = label_encoder.fit_transform(df_train['label'])
        df_dev['label_id'] = label_encoder.transform(df_dev['label'])
        df_test['label_id'] = label_encoder.transform(df_test['label'])
        df_train.to_csv('./datasets/FarsTail/Train-word-Corrected.csv', index=False)
        df_dev.to_csv('./datasets/FarsTail/Val-word-Corrected.csv', index=False)
        df_test.to_csv('./datasets/FarsTail/Test-word-Corrected.csv', index=False)
    return df_train, df_dev, df_test

def PlotReport(report):
    fig, axs = plt.subplots(1,2,figsize=(8,2.5))
    axs[0].set_title("Train and Validation Loss", y=0.95, fontsize=8)
    axs[0].plot(report['train_loss'], linewidth=2, label="Train")
    axs[0].plot(report['valid_loss'], linewidth=2, label="Validation")
    axs[0].set_xlabel("Epoch", fontsize=7) ; axs[0].set_ylabel("Loss", fontsize=7)
    axs[0].grid(axis="y", alpha=0.5) ; axs[0].legend(loc=0, prop={"size": 7})
    axs[0].tick_params(axis="x", labelsize=7) ; axs[0].tick_params(axis="y", labelsize=7)
    axs[1].set_title("Train and Validation Accuracy", y=0.95, fontsize=8)
    axs[1].plot(report['train_acc'], linewidth=2, label="Train")
    axs[1].plot( report['valid_acc'], linewidth=2, label="Validation")
    axs[1].set_xlabel("Epoch", fontsize=7) ; axs[1].set_ylabel("Accuracy", fontsize=7)
    axs[1].grid(axis="y", alpha=0.5) ; axs[1].legend(loc=0, prop={"size": 7})
    axs[1].tick_params(axis="x", labelsize=7) ; axs[1].tick_params(axis="y", labelsize=7)
    plt.show()

