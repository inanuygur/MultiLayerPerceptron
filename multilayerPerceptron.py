import os
from pathlib import Path
import math
import numpy as np
import matplotlib.pyplot as plt

real_filepath = os.path.realpath(__file__)

''' ------------------------------------------------ '''
def init_weigth_for_1_layer(dims=(1,1), lower_margin=0.1, upper_margin=0.1):
    """[summary]
        - Yaratilan bir noron icin degisken dimension (boyut) ile beraber uyumlu 
        agirlik degerleri uretme fonksiyonu - agirliklar -1 ve 1 arasinda, 
        kullanicinin belirleyebilecegi bir marjin ile olusturulmaktadir.

    Args:
        dims (tuple, optional): [katmana dair boyutlar]. Defaults to (1,1).
        lower_margin (float, optional): [alt sinir]. Defaults to 0.1.
        upper_margin (float, optional): [ust sinir]. Defaults to 0.1.

    Returns:
        [float]: [olusturulan agirliklar]
    """
    weights = (2 - (upper_margin + lower_margin)) * np.random.rand(dims[0], dims[1]) + lower_margin  - 1
    return weights

''' ------------------------------------------------ '''
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

''' ------------------------------------------------ '''
def der_sigmoid(x):
    return (sigmoid(x) * (1 - sigmoid(x)))

''' ------------------------------------------------ '''
# tanh aktivasyon fonksiyonu
def tanh(x):
    return (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x))

''' ------------------------------------------------ '''
# tanh aktivasyon fonksiyonu turevi
def der_tanh(x):
    return (1 - (tanh(x) ** 2))

''' ------------------------------------------------ '''
def init_parameters(layers_dimensions):
    """[summary]
        - Verilen katman bilgilerine gore 
            1. agirlik
            2. bias
            3. yerel_grad 
        degidkenlerinin baslangicini uygun bicimlerde gerceklestiren fonksiyon.
        - Parametrelerin rahat kontrolü adina bir dictionary icerisinde 
        saklanip yeri geldiginde o sekilde kullanilmasini uygun gordugum icin
        tum islemleri "parameters" dict'i uzerinden gerceklestirdim.

    Args:
        layers_dimensions ([type]): [description]

    Returns:
        [dict]: [parameters dictionary]
    """
    parameters = {}
    # layer_number = len(layers_dimensions)

    # parameters['W0'] => giris vektoru
    parameters['b0'] = np.ones((1, 1))
    parameters['b1'] = np.ones((1, 1))
    parameters['b2'] = np.ones((1, 1))
    parameters['b3'] = np.ones((1, 1))
    parameters['b4'] = np.ones((1, 1))
    parameters['b5'] = np.ones((1, 1))
    
    parameters['W1'] = init_weigth_for_1_layer(dims=(layers_dimensions[1], layers_dimensions[0] + 1))
    parameters['W2'] = init_weigth_for_1_layer(dims=(layers_dimensions[2], layers_dimensions[1] + 1))
    parameters['W3'] = init_weigth_for_1_layer(dims=(layers_dimensions[3], layers_dimensions[2] + 1))
    parameters['W4'] = init_weigth_for_1_layer(dims=(layers_dimensions[4], layers_dimensions[3] + 1))
    parameters['W5'] = init_weigth_for_1_layer(dims=(layers_dimensions[5], layers_dimensions[4] + 1))

    parameters['grad1'] = np.zeros((layers_dimensions[1], 1))
    parameters['grad2'] = np.zeros((layers_dimensions[2], 1))
    parameters['grad3'] = np.zeros((layers_dimensions[3], 1))
    parameters['grad4'] = np.zeros((layers_dimensions[4], 1))
    parameters['grad5'] = np.zeros((layers_dimensions[5], 1))
    
    return parameters

''' ------------------------------------------------ '''
def ileriye_adim(inputs, parameters, step):
    """[summary]
        - Yaratilan ag icerisinde giristen cikisa dogru bir adim atan fonksiyon.
        - Giris olarak her adimda farkli verilerle calisir ve sonuc dondurur.
        - Her adim icin farkli aktivasyon fonksiyonu secimi yapilabilir.

    Args:
        inputs ([type]): [description]
        parameters ([type]): [description]
        step ([type]): [description]

    Returns:
        [type]: [description]
    """
    
    weights = parameters['W' + str(step)]

    bias = parameters['b' + str(step - 1)]

    inputs = np.append(inputs, bias)
    
    weighted_sum = np.dot(weights, inputs)
    # y = []
    if step == 5:
        y = sigmoid(weighted_sum)
    if step == 4:
        y = sigmoid(weighted_sum)
    if step == 3:
        y = sigmoid(weighted_sum)
    if step == 2:
        y = sigmoid(weighted_sum)
    if step == 1:
        y = sigmoid(weighted_sum)

    return np.array(y), np.array(weighted_sum)

''' ------------------------------------------------ '''
########################################################
########################################################
########################################################
''' ------------------------------------------------ '''
# Veri setlerine cekebilmek icin kod dosyasinin dizinine ulasma
parent_dir = Path(real_filepath).parent

# veri setlerini cekme
training_set = np.load(str(parent_dir) + "\\veriseti\\numpies\\train_images.npy")
test_set = np.load(str(parent_dir) + "\\veriseti\\numpies\\test_images.npy")
training_set_labels = np.load(str(parent_dir) + "\\veriseti\\numpies\\train_labels.npy")
test_set_labels = np.load(str(parent_dir) + "\\veriseti\\numpies\\test_labels.npy")

# agin katman bilgilerini belirleme
giris_boyutu = training_set[0].shape[0] * training_set[0].shape[1]
layers_dimensions = [giris_boyutu, 155, 125, 100, 25, 10] # giris katmani = 784 boyutlu

# Veri setlerini kontrol etme
print()
print('-' * 15)
print('Egitim Seti Boyutu: ', training_set.shape)
print('Test Seti Boyutu: ', test_set.shape)
print('Egitim Seti Etiketleri Boyutu: ', training_set_labels.shape)
print('Test Seti Etiketleri Boyutu: ', test_set_labels.shape)

''' ------------------------------------------------ '''
# parametre baslatma ve boyut/bicim kontrolu islemi yapma
parameters = init_parameters(layers_dimensions)


# agirlik vb. bilgilerin boyut kontrolu
print()
print('-' * 15)
print('W1 shape', parameters['W1'].shape)
print('b1 shape', parameters['b1'].shape)
print('grad1 shape', parameters['grad1'].shape)
print('W2 shape', parameters['W2'].shape)
print('b2 shape', parameters['b2'].shape)
print('grad2 shape', parameters['grad2'].shape)
print('W3 shape', parameters['W3'].shape)
print('b3 shape', parameters['b3'].shape)
print('grad3 shape', parameters['grad3'].shape)
print('W4 shape', parameters['W4'].shape)
print('b4 shape', parameters['b4'].shape)
print('grad4 shape', parameters['grad4'].shape)
print('W5 shape', parameters['W5'].shape)
print('b5 shape', parameters['b5'].shape)
print('grad5 shape', parameters['grad5'].shape)

print()
print('-' * 15)
print('-' * 15)
print('-' * 15)

# giris_index = 1
''' ------------------------------------------------ '''
import time

def egitim (parameters, epsilon = 0.01):
    """[summary]
        - egitim fonksiyonu yuklenen veri seti uzerinden agirliklari
        modeli egitip agirliklari guncelleyerek bir tum ileri islemdeki 
        hatayi epsilon hata durdurma kriterinin altina indirmeye calisir.
        - "parameters" dictionary girisi ile gerekli tum agirlik, bias ve 
        yerel_grad degiskenlerine erisim ve duzenleme saglar.  

    Args:
        parameters ([dict]): [agirlik, bias ve yerel_grad]
        epsilon (float, optional): [hata durdurma kriteri]. Defaults to 0.01.

    Returns:
        [int]: [egitimdeki hatanin hata durdurma kriterinin 
                altina inme iterasyon degeri]
    """

    # t0 = time.time()

    max_iter = 2000 # egitim sirasindaki maksimum iterasyon degeri
    hatalar = [] # egitim sirasinda hatalari tutacak list

    # sirasiyla katmanlar icin momentum ve ogrenme hizi degerleri
    learning_rate_5 = 0.02
    learning_rate_4 = 0.03
    learning_rate_3 = 0.04
    learning_rate_2 = 0.05
    learning_rate_1 = 0.06

    momentum_5 = 0.4
    momentum_4 = 0.5
    momentum_3 = 0.6
    momentum_2 = 0.7
    momentum_1 = 0.8

    for iter_num in range(max_iter):

        tot_error = []

        # t1 = time.time()

        # momentum icin eski agirlik degiskenlerinin kaydi
        parameters['older_W5'] = parameters['W5']
        parameters['older_W4'] = parameters['W4']
        parameters['older_W3'] = parameters['W3']
        parameters['older_W2'] = parameters['W2']
        parameters['older_W1'] = parameters['W1']

        parameters['old_W5'] = parameters['W5']
        parameters['old_W4'] = parameters['W4']
        parameters['old_W3'] = parameters['W3']
        parameters['old_W2'] = parameters['W2']
        parameters['old_W1'] = parameters['W1']

        for giris_index in range(training_set.shape[0]): # training_set.shape[0]
            
            # Bir iterasyon icin egitim setindeki her veriye bakilacak dongu.

            # t2 = time.time()

            ''' ------------------------------------------------ '''
            # egitim setindeki veriyi dilim halinde alinmasi ve bicim kontrolu 
            # burada 28x28 giris matrisini duzlestirip 784x1 hale getiriyorum
            training_data = training_set[giris_index].flatten()
            training_data = np.reshape(training_data, (training_data.shape[0], 1))

            ''' ------------------------------------------------ '''

            y1, v1 = ileriye_adim(training_data, parameters, 1)

            ''' ------------------------------------------------ '''
            y2, v2 = ileriye_adim(y1, parameters, 2)

            ''' ------------------------------------------------ '''
            y3, v3 = ileriye_adim(y2, parameters, 3)

            ''' ------------------------------------------------ '''
            y4, v4 = ileriye_adim(y3, parameters, 4)

            ''' ------------------------------------------------ '''
            y5, v5 = ileriye_adim(y4, parameters, 5)

            ''' ------------------------------------------------ '''

            # y cikislari ve v degerlerinin bicim kontrolu ve duzenlemesi
            v5 = np.reshape(v5, (v5.shape[0], 1))
            v4 = np.reshape(v4, (v4.shape[0], 1))
            v3 = np.reshape(v3, (v3.shape[0], 1))
            v2 = np.reshape(v2, (v2.shape[0], 1))
            v1 = np.reshape(v1, (v1.shape[0], 1))

            # y cikislari ve v degerlerinin bicim kontrolu ve duzenlemesi
            y5 = np.reshape(y5, (y5.shape[0], 1))
            y4 = np.reshape(y4, (y4.shape[0], 1))
            y3 = np.reshape(y3, (y3.shape[0], 1))
            y2 = np.reshape(y2, (y2.shape[0], 1))
            y1 = np.reshape(y1, (y1.shape[0], 1))

            ''' ------------------------------------------------ '''

            # cikis katmani sonucu y3 ile giris verisinin gercek ciktisi arasindaki hata hesabi
            
            label_deger = np.reshape(training_set_labels[giris_index], y5.shape)
            small_error = label_deger.astype(np.int64) - y5

            small_error = np.reshape(np.array(small_error), (len(training_set_labels[giris_index]), 1))
            # toplam ani hata hesabi
            big_error = np.dot(small_error.T, small_error)
            # toplam ani hatanin biriktirilmesi
            tot_error.append(big_error)

            # t3 = time.time()

            ''' ------------------------------------------------ '''

            # geriye dogru atilan adimlarin hazirliklari

            # katmanlardaki agirlikli toplamlarin, v degerlerinin, ileri adim atilirken
            # uygulanan aktivasyon fonksiyonlarinin turevleri ile tekrar hesaplanmasi
            v5 = der_sigmoid(v5)

            v4 = der_sigmoid(v4)

            v3 = der_sigmoid(v3)

            v2 = der_sigmoid(v2)

            v1 = der_sigmoid(v1)

            # t4 = time.time()

            ''' ------------------------------------------------ '''

            # son katmana dair yerel gradyan hesabi yapilmasi
            yerel_grad_5 = small_error * v5

            ''' ------------------------------------------------ ''' 

            # dorduncu gizli katman ile cikis katmani arasindaki agirliklardan
            # bias degerlerinin cikartilmasi
            biassiz_agirlik_5 = parameters['W5'][:,:-1]

            # dorduncu gizli katmana dair yerel gradyan hesabi yapilmasi
            yerel_grad_4 = (np.dot(biassiz_agirlik_5.T, yerel_grad_5)) * v4

            ''' ------------------------------------------------ '''

            biassiz_agirlik_4 = parameters['W4'][:,:-1]

            yerel_grad_3 = (np.dot(biassiz_agirlik_4.T, yerel_grad_4)) * v3

            ''' ------------------------------------------------ '''

            biassiz_agirlik_3 = parameters['W3'][:,:-1]

            yerel_grad_2 = (np.dot(biassiz_agirlik_3.T, yerel_grad_3)) * v2

            ''' ------------------------------------------------ '''

            biassiz_agirlik_2 = parameters['W2'][:,:-1]

            yerel_grad_1 = (np.dot(biassiz_agirlik_2.T, yerel_grad_2)) * v1

            ''' ------------------------------------------------ '''

            # t5 = time.time()

            # momentumda kullanilacak eski ve yeni agirlik bilgilerinin ayarlamasi
            parameters['older_W5'] = parameters['old_W5']
            parameters['older_W4'] = parameters['old_W4']
            parameters['older_W3'] = parameters['old_W3']
            parameters['older_W2'] = parameters['old_W2']
            parameters['older_W1'] = parameters['old_W1']

            parameters['old_W5'] = parameters['W5']
            parameters['old_W4'] = parameters['W4']
            parameters['old_W3'] = parameters['W3']
            parameters['old_W2'] = parameters['W2']
            parameters['old_W1'] = parameters['W1']
            
            # momentum ve ogrenme hizi kullanarak agirlik guncellenmesi
            parameters['W5'] = parameters['W5'] + learning_rate_5 * np.dot(yerel_grad_5, np.c_[y4.T, parameters['b4']])
            parameters['W4'] = parameters['W4'] + learning_rate_4 * np.dot(yerel_grad_4, np.c_[y3.T, parameters['b3']])
            parameters['W3'] = parameters['W3'] + learning_rate_3 * np.dot(yerel_grad_3, np.c_[y2.T, parameters['b2']])
            parameters['W2'] = parameters['W2'] + learning_rate_2 * np.dot(yerel_grad_2, np.c_[y1.T, parameters['b1']])
            parameters['W1'] = parameters['W1'] + learning_rate_1 * np.dot(yerel_grad_1, np.c_[training_data.T, parameters['b0']])
            
            parameters['W5'] = parameters['W5'] + momentum_5 * (parameters['old_W5'] - parameters['older_W5'])
            parameters['W4'] = parameters['W4'] + momentum_4 * (parameters['old_W4'] - parameters['older_W4'])
            parameters['W3'] = parameters['W3'] + momentum_3 * (parameters['old_W3'] - parameters['older_W3'])
            parameters['W2'] = parameters['W2'] + momentum_2 * (parameters['old_W2'] - parameters['older_W2'])
            parameters['W1'] = parameters['W1'] + momentum_1 * (parameters['old_W1'] - parameters['older_W1'])

            # t6 = time.time()

        # egitim kumesindeki ortalama hatanin hesaplanmasi 
        ort_error = sum(tot_error) / len(tot_error)
        print('ort_error: ', ort_error , ' - iter_num: ', iter_num + 1)
        hatalar.append(ort_error)

        # t_end = time.time()

        # print('t1 - t0: ', (t1 - t0) * 100)
        # print('t2 - t1: ', (t2 - t1) * 100)
        # print('t3 - t2: ', (t3 - t2) * 100)
        # print('t4 - t3: ', (t4 - t3) * 100)
        # print('t5 - t4: ', (t5 - t4) * 100)
        # print('t6 - t5: ', (t6 - t5) * 100)
        # print('t_end - t6: ', (t_end - t6) * 100)
        # print('t_end - t0: ', (t_end - t0) * 100)

        if iter_num % 1 == 0: 
            # her iterasyonda bir parametre ve hata degeri kaydi 
            np.save(str(parent_dir) + "\\parameters_W5", parameters['W5'])
            np.save(str(parent_dir) + "\\parameters_W4", parameters['W4'])
            np.save(str(parent_dir) + "\\parameters_W3", parameters['W3'])
            np.save(str(parent_dir) + "\\parameters_W2", parameters['W2'])
            np.save(str(parent_dir) + "\\parameters_W1", parameters['W1'])
            np.save(str(parent_dir) + "\\hatalar", np.array(hatalar))
            np.save(str(parent_dir) + "\\parameters", parameters)

        # hata durdurma epsilon degeri
        if ort_error  <= epsilon:
            print('ort_error: ', ort_error , ' - iter_num: ', iter_num + 1)
            print('hata degeri epsilonun altina indi.')
            break

''' ------------------------------------------------ '''
# 
def test(parameters, epsilon):
    """[summary]
        - bu fonksiyon egitim fonksiyonunun ayni islemlerini gerceklestirir.
        - geryiye yayilim kismini icermez, agda sadece ileriye gider ve y3
        cikis degeri ile gercek cikis arasinda karsilastirma yaparak hata
        uzerinden dogruluk orani hesaplar. 

    Args:
        parameters ([dict]): [parametreler]
        epsilon ([float]): [test hata toleransi]

    Returns:
        [float]: [ortalama dogru test sayisi]
    """
    dogru_test_sayisi = 0
    test_karmasiklik_hatalari = []

    print("test islemi baslatiyor...")

    for giris_index in range(test_set.shape[0]):
        ''' ------------------------------------------------ '''
        test_data = test_set[giris_index].flatten()
        test_data = np.reshape(test_data, (test_data.shape[0], 1))

        y1, v1 = ileriye_adim(test_data, parameters, 1)

        ''' ------------------------------------------------ '''
        y2, v2 = ileriye_adim(y1, parameters, 2)

        ''' ------------------------------------------------ '''
        y3, v3 = ileriye_adim(y2, parameters, 3)

        ''' ------------------------------------------------ '''
        y4, v4 = ileriye_adim(y3, parameters, 4)

        ''' ------------------------------------------------ '''
        y5, v5 = ileriye_adim(y4, parameters, 5)

        ''' ------------------------------------------------ '''

        v5 = np.reshape(v5, (v5.shape[0], 1))
        v4 = np.reshape(v4, (v4.shape[0], 1))
        v3 = np.reshape(v3, (v3.shape[0], 1))
        v2 = np.reshape(v2, (v2.shape[0], 1))
        v1 = np.reshape(v1, (v1.shape[0], 1))

        y5 = np.reshape(y5, (y5.shape[0], 1))
        y4 = np.reshape(y4, (y4.shape[0], 1))
        y3 = np.reshape(y3, (y3.shape[0], 1))
        y2 = np.reshape(y2, (y2.shape[0], 1))
        y1 = np.reshape(y1, (y1.shape[0], 1))

        ''' ------------------------------------------------ '''
        # tahmin verisindeki kucuk e hatasini hesaplama
        small_error = []
        for deger_index in range(len(test_set_labels[giris_index])):
            label_deger = test_set_labels[giris_index][deger_index]
            small_error.append(int(label_deger) - y5[deger_index])
        
        # test sirasindaki tahminleri ayiklamak ve tahmin rakamini elde etmek amacli 
        en_iyi_tahmin_ixs = np.where(y5 == np.amax(y5))
        dogru_deger_ixs = np.where(test_set_labels[giris_index] == np.amax(test_set_labels[giris_index]))
        test_karmasiklik_hatalari.append(np.array([en_iyi_tahmin_ixs, dogru_deger_ixs]))

        small_error = np.reshape(np.array(small_error), (len(test_set_labels[giris_index]), 1))

        big_error = np.dot(small_error.T, small_error)
        if big_error <= epsilon:
            dogru_test_sayisi += 1

    ortalama_dogru_test_sayisi = dogru_test_sayisi / test_set.shape[0]
    print("ortalama dogru test sayisi %: ", 100 * ortalama_dogru_test_sayisi)

    test_karmasiklik_hatalari = np.array(test_karmasiklik_hatalari)
    # print('test_karmasiklik_hatalari.shape: ', test_karmasiklik_hatalari.shape)
    # np.save(str(parent_dir) + "\\test_karmasiklik_hatalari", test_karmasiklik_hatalari)

''' ------------------------------------------------ '''
# 
def tahmin(parameters, test_ornek_index=0):
    """[summary]
        - test setindeki spesifik bir ornek icin agin tahminini
        ve test verisinin gercek degerini gosteren fonksiyon

    Args:
        parameters ([dict]): [parametreler]
        test_ornek_index (int, optional): [test setindeki verinin indeksi]. Defaults to 0.
    """
    # epsilon = 0.1
    # dogru_test_sayisi = 0
    print(test_ornek_index, ". data icin tahmin islemi baslatiyor...")

    ''' ------------------------------------------------ '''
    test_data = test_set[test_ornek_index].flatten()
    test_data = np.reshape(test_data, (test_data.shape[0], 1))
    # print('test_data.shape:',test_data.shape)

    y1, v1 = ileriye_adim(test_data, parameters, 1)

    ''' ------------------------------------------------ '''
    y2, v2 = ileriye_adim(y1, parameters, 2)

    ''' ------------------------------------------------ '''
    y3, v3 = ileriye_adim(y2, parameters, 3)

    ''' ------------------------------------------------ '''
    y4, v4 = ileriye_adim(y3, parameters, 4)

    ''' ------------------------------------------------ '''
    y5, v5 = ileriye_adim(y4, parameters, 5)

    ''' ------------------------------------------------ '''
    v5 = np.reshape(v5, (v5.shape[0], 1))
    v4 = np.reshape(v4, (v4.shape[0], 1))
    v3 = np.reshape(v3, (v3.shape[0], 1))
    v2 = np.reshape(v2, (v2.shape[0], 1))
    v1 = np.reshape(v1, (v1.shape[0], 1))

    y5 = np.reshape(y5, (y5.shape[0], 1))
    y4 = np.reshape(y4, (y4.shape[0], 1))
    y3 = np.reshape(y3, (y3.shape[0], 1))
    y2 = np.reshape(y2, (y2.shape[0], 1))
    y1 = np.reshape(y1, (y1.shape[0], 1))

    print("\tmodelin tahmini: ", y5.T, "\n\tgercek deger: ", test_set_labels[test_ornek_index])

''' ------------------------------------------------ '''
# egitim gerceklestirme ve sonunda agirlik kaydi
egitim(parameters) # <<<<<<<<<<<<<<< egitim islemi <<<<<<<<<<<<<<< 

''' ------------------------------------------------ '''
# test veri setinde test islemi 
print()
print('-' * 15)

test(parameters, epsilon=0.01)

''' ------------------------------------------------ '''
# # spesifik veri tahmini icin
print()
print('-' * 15)

tahmin(parameters, test_ornek_index=420)
plt.imshow(test_set[420])
plt.show()

''' ------------------------------------------------ '''
# KOD SONU # 