import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import periodogram, spectrogram


def get_data():
    # wczytanie bazy, czesniej zostala przekonwertowana z .m do .csv
    df = pd.read_csv('baza.csv', header=None)

    return df


def moving_average(df, number, size, choice, plot):
    # pozbycie sie trendu wykorzystujac podana w instrukcji funkcje
    column_number = number
    window_size = size
    rolling_mean = df.iloc[:, column_number].rolling(window=window_size).mean()
    df.iloc[:, column_number] = df.iloc[:, column_number] - rolling_mean

    if plot:
        # Stworzenie wykresu pokazujacego roznice przed i po operacji
        fig, ax = plt.subplots()
        ax.plot(df.iloc[:, column_number], label='Po usunięciu trendu')
        ax.plot(df.iloc[:, column_number].rolling(window=window_size).mean(), label='Średnia ruchoma')
        ax.plot(rolling_mean, label='Średnia ruchoma (oryginalna)')
        ax.legend()
        plt.show()
    if choice:
        # zapis oczyszczonych danych do pliku .csv
        df.to_csv('dane_clear.csv', index=False, header=False)

    return df


def split_data(df1, choice):

    #podzial danych na etapy zgodnie z oznaczeniami czasowymi w instrukcji:
    # baseline 909 - 42909
    # 0.1HZ 42910 - 84979
    # 01HZ+R 84980 - 127049
    # recovery 127050 - 169119

    stg = ['BaseLine', '01HZ', '01HZ_R', 'recovery']
    index = 909
    for i in range(len(stg)):
        stg[i] = df1.iloc[index:index + 42000].reset_index(drop=True)
        index += 42070
        if choice:
            print(stg[i])
    return stg


def make_plots(data, fs, N, overlap, x, y, z, a, num):
    # Stworzenie 4 periodogramow dla kazdego "etapu"
    stg = [x, y, z, a]
    for i in range(num):
        for col in stg[i].columns:
            f, Pxx = periodogram(df_BaseLine[col], fs=fs, window='hann', nfft=N)
            plt.semilogy(f, Pxx, label=col)

        plt.legend()
        plt.title('Periodogram')
        plt.xlabel('Częstotliwość [Hz]')
        plt.ylabel('Gęstość mocy [V**2/Hz]')
        plt.show()

    # Stworzenie spektogramu dla oczyszczonych wczesniej danych
    f, t, Sxx = spectrogram(data.values.ravel(), fs=fs, window='hann', nperseg=N, noverlap=overlap)
    plt.pcolormesh(t, f, 10 * np.log10(Sxx), cmap='inferno')
    plt.title('Spektrogram')
    plt.xlabel('Czas [s]')
    plt.ylabel('Częstotliwość [Hz]')
    plt.ylim([0, 200])
    plt.colorbar(label='Gęstość mocy [dB/Hz]')
    plt.show()


dataFrame = get_data()
dataFrame_clear = moving_average(dataFrame, number=0, size=10, choice=False, plot=False)


df_BaseLine, df_01HZ, df_01HZ_R, Recovery = split_data(dataFrame_clear, choice=True)


dataframe_all = pd.concat([df_BaseLine, df_01HZ, df_01HZ_R, Recovery], axis=1, keys=['etap1', 'etap2', 'etap3', 'etap4'])

full = dataFrame_clear.iloc[909:169119]

#deklaracja zmiennych do utworznia wykresow
fs = 1000
N = 4096
overlap = 4000

make_plots(dataFrame_clear, fs, N, overlap, df_BaseLine, df_01HZ, df_01HZ_R, Recovery, num=4)

# Grzegorz Grazewicz 183208

