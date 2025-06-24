import os
import pathlib
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from tensorflow.keras.preprocessing.image import ImageDataGenerator

#Cnns
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.applications.resnet_v2 import ResNet50V2
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.convnext import ConvNeXtSmall
from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2B0, EfficientNetV2B3
from tensorflow.keras.applications import MobileNetV3Small

from keras import utils
import time

!pip freeze > requirements.txt
!pip list --format=freeze > requirements.txt

from google.colab import drive

drive.mount('/content/drive')

base_dir =  "/content/drive/MyDrive/dataset-reduzido-copia/"

!ls "{base_dir}"

data_dir = pathlib.Path(base_dir)

classe_path = sorted([ x for x in data_dir.iterdir() if x.is_dir()])
#adicionei sorted() para garantir a ordenação do elementos pelo nome [Prof Diego]

#classes_names = ['Grau1', 'Grau2', 'Grau3', 'Grau4', 'Grau5', 'Grau6', 'Grau7', 'Grau8']
#classes_names = ['Verde', 'Em amadurecimento', 'Madura', 'Passada']
classes_names = ['Verde', 'Madura', 'Passada']

classe_path

"""## Data Augmentation"""

#DATA AUGMENTATION 01

from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img, array_to_img

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

def criarSinteticas(paths, className, nImage=5):
    output_class_dir = os.path.join(base_dir, className, 'aug')
    os.makedirs(output_class_dir, exist_ok=True)

    for img_path in paths:
        img = load_img(img_path, target_size=(244, 244))
        x = img_to_array(img)
        x = x.reshape((1,) + x.shape)  # reshape para (1, 244, 244, 3)

        count = 0
        for batch in datagen.flow(x, batch_size=1, save_to_dir=output_class_dir, save_prefix='aug', save_format='jpg'):
            count += 1
            if count >= nImage:
                break

#DATA AUGMENTATION 02
'''
========ATENÇÃO===================================================
 Não rodar esse bloco se você já tem imagens sintéticas nas pastas!
==================================================================
'''
def getImagePaths(class_path):
    return [str(p) for p in class_path.glob('*.jpg')]

for class_name, class_path in zip(classes_names, classe_path):
    print(f'Gerando imagens aumentadas para {class_name}...')
    image_paths = getImagePaths(class_path)
    #criarSinteticas(image_paths, class_name) #Remova o comentário no inicío desta linha se, e somente se, você realmente precisar gerar imagens

#DATA AUGMENTATION 03

def globComSinteticas(classDir, withAugmentation=True):
    images = []

    augDir = classDir / 'aug'

    if withAugmentation:
        if augDir.exists():
            images.extend(list(augDir.glob('*.jpg')))
        images.extend(list(classDir.glob('*.jpg')))
    else:
        images.extend(list(classDir.glob('*.jpg')))

    return images

chooseValueAugmentation = True

Grau1 = globComSinteticas(classe_path[0], withAugmentation= chooseValueAugmentation)
Grau2 = globComSinteticas(classe_path[1], withAugmentation= chooseValueAugmentation)
Grau3 = globComSinteticas(classe_path[2], withAugmentation= chooseValueAugmentation)
Grau4 = globComSinteticas(classe_path[3], withAugmentation= chooseValueAugmentation)
Grau5 = globComSinteticas(classe_path[4], withAugmentation= chooseValueAugmentation)
Grau6 = globComSinteticas(classe_path[5], withAugmentation= chooseValueAugmentation)
Grau7 = globComSinteticas(classe_path[6], withAugmentation= chooseValueAugmentation)
Grau8 = globComSinteticas(classe_path[7], withAugmentation= chooseValueAugmentation)

# 8 Classes
'''qtd_amostras_total =  len(Grau1) + len(Grau2) + len(Grau3) + len(Grau4) + len(Grau5) + len(Grau6) + len(Grau7) + len(Grau8)
qtd_amostras_per_class = [len(Grau1), len(Grau2), len(Grau3), len(Grau4), len(Grau5), len(Grau6), len(Grau7), len(Grau8)]

print('Quantidade total de amostras: ', qtd_amostras_total)
print('Quantidade amostra Grau1: ', len(Grau1))
print('Quantidade amostra Grau2: ', len(Grau2))
print('Quantidade amostra Grau3: ', len(Grau3))
print('Quantidade amostra Grau4: ', len(Grau4))
print('Quantidade amostra Grau5: ', len(Grau5))
print('Quantidade amostra Grau6: ', len(Grau6))
print('Quantidade amostra Grau7: ', len(Grau7))
print('Quantidade amostra Grau8: ', len(Grau8))'''

# 4 Classes

'''Verde = Grau1
Em_amadurecimento = Grau2 + Grau3
Madura = Grau4 + Grau5 + Grau6 + Grau7
Passada = Grau8

qtd_amostras_total =  len(Verde) + len(Em_amadurecimento) + len(Madura) + len(Passada)
qtd_amostras_per_class = [len(Verde), len(Em_amadurecimento), len(Madura), len(Passada)]

print('Quantidade total de amostras: ', qtd_amostras_total)
print('Quantidade amostra Grau1: ', len(Verde))
print('Quantidade amostra Grau2: ', len(Em_amadurecimento))
print('Quantidade amostra Grau3: ', len(Madura))
print('Quantidade amostra Grau4: ', len(Passada))'''

# 3 Classes
Verde = Grau1 + Grau2 + Grau3
Madura = Grau4 + Grau5 + Grau6 + Grau7
Passada = Grau8

qtd_amostras_total =  len(Verde) + len(Madura) + len(Passada)
qtd_amostras_per_class = [len(Verde), len(Madura), len(Passada)]

print('Quantidade total de amostras: ', qtd_amostras_total)
print('Quantidade amostra Verde: ', len(Verde))
print('Quantidade amostra Madura: ', len(Madura))
print('Quantidade amostra Passada: ', len(Passada))

Y = np.zeros(qtd_amostras_total)

indice_img = 0
label = 0
for i in qtd_amostras_per_class:
  print ("Label:%2d\tFamily: %15s\tNumber of images: %d" % (label, classes_names[label], i))
  for j in range(i):
   Y[indice_img] = label
   indice_img += 1
  label += 1

num_of_classes = label

"""### Preprocess 8 Classes"""

from keras.applications.imagenet_utils import preprocess_input
from keras.utils import img_to_array, load_img

width, height, channels = (244, 244, 3)
X = np.zeros((qtd_amostras_total, width, height, channels))
count = 0
list_caminhos = []

#Computando as fetures
# 8 Classes

print("Processando imagens ...")
for i in range(len(classes_names)):
  if i == 0:
    for caminho in Grau1:
      list_caminhos.append(str(caminho))
      img = load_img(caminho, target_size=(244,244))
      x = img_to_array(img)
      x = np.expand_dims(x, axis=0)
      x = preprocess_input(x)
      X[count] = x
      count += 1
  elif i == 1:
    for caminho in Grau2:
      list_caminhos.append(str(caminho))
      img = load_img(caminho, target_size=(244,244))
      x = img_to_array(img)
      x = np.expand_dims(x, axis=0)
      x = preprocess_input(x)
      X[count] = x
      count += 1
  elif i == 2:
    for caminho in Grau3:
      list_caminhos.append(str(caminho))
      img = load_img(caminho, target_size=(244,244))
      x = img_to_array(img)
      x = np.expand_dims(x, axis=0)
      x = preprocess_input(x)
      X[count] = x
      count += 1
  elif i == 3:
    for caminho in Grau4:
      list_caminhos.append(str(caminho))
      img = load_img(caminho, target_size=(244,244))
      x = img_to_array(img)
      x = np.expand_dims(x, axis=0)
      x = preprocess_input(x)
      X[count] = x
      count += 1
  elif i == 4:
    for caminho in Grau5:
      list_caminhos.append(str(caminho))
      img = load_img(caminho, target_size=(244,244))
      x = img_to_array(img)
      x = np.expand_dims(x, axis=0)
      x = preprocess_input(x)
      X[count] = x
      count += 1
  elif i == 5:
    for caminho in Grau6:
      list_caminhos.append(str(caminho))
      img = load_img(caminho, target_size=(244,244))
      x = img_to_array(img)
      x = np.expand_dims(x, axis=0)
      x = preprocess_input(x)
      X[count] = x
      count += 1
  elif i == 6:
    for caminho in Grau7:
      list_caminhos.append(str(caminho))
      img = load_img(caminho, target_size=(244,244))
      x = img_to_array(img)
      x = np.expand_dims(x, axis=0)
      x = preprocess_input(x)
      X[count] = x
      count += 1
  elif i == 7:
    for caminho in Grau8:
      list_caminhos.append(str(caminho))
      img = load_img(caminho, target_size=(244,244))
      x = img_to_array(img)
      x = np.expand_dims(x, axis=0)
      x = preprocess_input(x)
      X[count] = x
      count += 1

print(len(list_caminhos), len(X))

"""### Preprocess 4 Classes"""

from keras.applications.imagenet_utils import preprocess_input
from keras.utils import img_to_array, load_img

width, height, channels = (244, 244, 3)
X = np.zeros((qtd_amostras_total, width, height, channels))
count = 0
count0 = 0
count1 = 0
count2 = 0
count3 = 0
list_caminhos = []

print("Processando imagens ...")
for i in range(len(classes_names)):
  if i == 0:
    for caminho in Verde:
      list_caminhos.append(str(caminho))
      img = load_img(caminho, target_size=(244,244))
      x = img_to_array(img)
      x = np.expand_dims(x, axis=0)
      x = preprocess_input(x)
      X[count] = x
      count += 1
      count0 += 1
  elif i == 1:
    for caminho in Em_amadurecimento:
      list_caminhos.append(str(caminho))
      img = load_img(caminho, target_size=(244,244))
      x = img_to_array(img)
      x = np.expand_dims(x, axis=0)
      x = preprocess_input(x)
      X[count] = x
      count += 1
      count1 += 1
  elif i == 2:
    for caminho in Madura:
      list_caminhos.append(str(caminho))
      img = load_img(caminho, target_size=(244,244))
      x = img_to_array(img)
      x = np.expand_dims(x, axis=0)
      x = preprocess_input(x)
      X[count] = x
      count += 1
      count2 += 1
  elif i == 3:
    for caminho in Passada:
      list_caminhos.append(str(caminho))
      img = load_img(caminho, target_size=(244,244))
      x = img_to_array(img)
      x = np.expand_dims(x, axis=0)
      x = preprocess_input(x)
      X[count] = x
      count += 1
      count3 += 1

print(count0)
print(count1)
print(count2)
print(count3)
print(len(list_caminhos), len(X))

"""### Preprocess 3 Classes"""

from keras.applications.imagenet_utils import preprocess_input
from keras.utils import img_to_array, load_img

width, height, channels = (244, 244, 3)
X = np.zeros((qtd_amostras_total, width, height, channels))
count = 0
count0 = 0
count1 = 0
count2 = 0
count3 = 0
list_caminhos = []

print("Processando imagens ...")
for i in range(len(classes_names)):
  if i == 0:
    for caminho in Verde:
      list_caminhos.append(str(caminho))
      img = load_img(caminho, target_size=(244,244))
      x = img_to_array(img)
      x = np.expand_dims(x, axis=0)
      x = preprocess_input(x)
      X[count] = x
      count += 1
      count0 += 1
  elif i == 1:
    for caminho in Madura:
      list_caminhos.append(str(caminho))
      img = load_img(caminho, target_size=(244,244))
      x = img_to_array(img)
      x = np.expand_dims(x, axis=0)
      x = preprocess_input(x)
      X[count] = x
      count += 1
      count1 += 1
  elif i == 2:
    for caminho in Passada:
      list_caminhos.append(str(caminho))
      img = load_img(caminho, target_size=(244,244))
      x = img_to_array(img)
      x = np.expand_dims(x, axis=0)
      x = preprocess_input(x)
      X[count] = x
      count += 1
      count2 += 1

print(count0)
print(count1)
print(count2)
print(len(list_caminhos), len(X))

encoder = LabelEncoder()
encoder.fit(Y)
y_encoded = encoder.transform(Y)
y = utils.to_categorical(y_encoded)

#Extraindo e salvando
image_shape = (244, 244, 3)

nomeExtrator = ''
modelo = 10

if modelo == 1:
  base_model = VGG16(include_top=False, weights="imagenet", input_shape=image_shape, pooling='max_')
  nomeArquivo = "dados_bananas_vgg16.npy"
  nomeExtrator = "VGG16"

elif modelo == 2:
  base_model = ResNet50(include_top=False, weights="imagenet", input_shape=image_shape, pooling='max_')
  nomeArquivo = "dados_bananas_resnet50.npy"
  nomeExtrator = "ResNet50"

elif modelo == 3:
  base_model = InceptionV3(include_top=False, weights="imagenet", input_shape=image_shape, pooling='max_')
  nomeArquivo = "dados_bananas_inceptionV3.npy"
  nomeExtrator = "InceptionV3"

elif modelo == 4:
  base_model = ResNet50V2(include_top=False, weights="imagenet", input_shape=image_shape, pooling='max_')
  nomeArquivo = "dados_bananas_resnet50V2.npy"
  nomeExtrator = "ResNet50V2"

elif modelo == 5:
  base_model = InceptionResNetV2(include_top=False, weights="imagenet", input_shape=image_shape, pooling='max_')
  nomeArquivo = "dados_bananas_inceptionresnet50V2.npy"
  nomeExtrator = "InceptionResNetV2"

elif modelo == 6:
  base_model = VGG19(include_top=False, weights="imagenet", input_shape=image_shape, pooling='max_')
  nomeArquivo = "dados_bananas_vgg19.npy"
  nomeExtrator = "VGG19"

elif modelo == 7:
  base_model = ConvNeXtSmall(include_top=False, weights="imagenet", input_shape=image_shape, pooling='max_')
  nomeArquivo = "dados_bananas_convneXtSmall.npy"
  nomeExtrator = "ConvNeXtSmall"

elif modelo == 8:
  base_model = EfficientNetV2B0(include_top=False, weights="imagenet", input_shape=image_shape, pooling='max_')
  nomeArquivo = "dados_bananas_efficienteNetv2B0.npy"
  nomeExtrator = "EfficientNetV2B0"

elif modelo == 9:
  base_model = MobileNetV3Small(include_top=False, weights="imagenet", input_shape=image_shape, pooling='max_')
  nomeArquivo = "dados_bananas_mobileNetV3Small.npy"
  nomeExtrator = "MobileNetV3Small"

elif modelo == 10:
  base_model = EfficientNetV2B3(include_top=False, weights="imagenet", input_shape=image_shape, pooling='max_')
  nomeArquivo = "dados_bananas_efficienteNetv2B3.npy"
  nomeExtrator = "EfficientNetV2B3"


if os.path.exists("/content/drive/MyDrive/dataset-reduzido-copia/"+nomeArquivo):
  print("Carregando caracteristicas ...")
  ExtractorFeatures = np.load("/content/drive/MyDrive/dataset-reduzido-copia/"+nomeArquivo)

else:
  inicio_extracao = time.time()
  print("Extraindo com " +nomeExtrator  + "...")
  ExtractorFeatures = base_model.predict(X)
  print("Salvando informações no arquivo  ...")
  np.save("/content/drive/MyDrive/dataset-reduzido-copia/"+nomeArquivo, ExtractorFeatures)
  np.save("/content/drive/MyDrive/dataset-reduzido-copia/labels_bananas.npy", y)
  fim_extracao = time.time()

output2 = open("tempo_extração.txt","w")
output2.write("\n\nTempo de extração: "+str((fim_extracao-inicio_extracao)))
output2.close()

from sklearn.model_selection  import StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sea

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix,accuracy_score, precision_score, f1_score, recall_score

kfold = 15 # no. of folds
skf = StratifiedKFold(n_splits=kfold, shuffle=True, random_state=1)
skfind = [None] * kfold  # skfind[i][0] -> train indices, skfind[i][1] -> test indices
cnt = 0
for index in skf.split(X, Y):
    skfind[cnt] = index
    cnt += 1

nome_do_arquivo = "metricas_mlp_" + nomeExtrator + ".txt"

conf_mat = np.zeros((len(classes_names), len(classes_names)))
tempo_treinamento = []
tempo_teste = []
acc = []
prec = []
rec = []
f1s = []
mat_conf = []
prec0 = []
prec1 = []
prec2 = []
prec3 = []
prec4 = []
prec5 = []
prec6 = []
prec7 = []

acc_best = -99
prec_best  = -99
rec_best  = -99
f1_best  = -99
matriz_best  = -99
prec0_best = -99
prec1_best = -99
prec2_best = -99
prec3_best = -99
prec4_best = -99
prec5_best = -99
prec6_best = -99
prec7_best = -99

for i in range(kfold):
  train_indices = skfind[i][0]
  test_indices = skfind[i][1]

  X_train_aux = ExtractorFeatures[train_indices]
  nsamples, nx, ny, nz = X_train_aux.shape
  X_train = X_train_aux.reshape((nsamples, nx*ny*nz))
  y_train = Y[train_indices]

  X_test_aux= ExtractorFeatures[test_indices]
  nsamples, x, y, z = X_test_aux.shape
  X_test = X_test_aux.reshape((nsamples, x*y*z))
  y_test = Y[test_indices]

  #Treinando o modelo
  print("--------------------------------------------------")
  print("Treinando...", end='\n')
  mlp_model = MLPClassifier(random_state=1, max_iter=200)
  inicio_treinamento = time.time()
  mlp_model.fit(X_train, y_train)
  fim_treinamento = time.time()

  #tempo de treinamento
  tempo_treinamento.append((fim_treinamento - inicio_treinamento))

  #Testando o modelo
  print("Testando...", end='\n')
  inicio_teste = time.time()
  y_pred = mlp_model.predict(X_test)
  fim_teste = time.time()

  print("[%d] Test acurracy: %.4f" %(i,accuracy_score(y_test,y_pred)))
  #print(y_pred)
  #print(y_test)
  cm = confusion_matrix(y_test, y_pred)
  print('===== CONFUSION MATRIX ======')
  print(cm, end='\n')
  conf_mat += cm

  #tempo de teste
  tempo_teste.append((fim_teste - inicio_teste))

  acc.append(accuracy_score(y_test,y_pred))
  prec.append(precision_score(y_test, y_pred, average='macro'))
  rec.append(recall_score(y_test, y_pred, average='macro'))
  f1s.append(f1_score(y_test, y_pred, average='macro'))

  # 8 Classes
  '''prec0.append(cm[0,0]/(cm[0,0]+cm[0,1]+cm[0,2]+cm[0,3]+cm[0,4]+cm[0,5]+cm[0,6]+cm[0,7]))
  prec1.append(cm[1,1]/(cm[1,0]+cm[1,1]+cm[1,2]+cm[1,3]+cm[1,4]+cm[1,5]+cm[1,6]+cm[1,7]))
  prec2.append(cm[2,2]/(cm[2,0]+cm[2,1]+cm[2,2]+cm[2,3]+cm[2,4]+cm[2,5]+cm[2,6]+cm[2,7]))
  prec3.append(cm[3,3]/(cm[3,0]+cm[3,1]+cm[3,2]+cm[3,3]+cm[3,4]+cm[3,5]+cm[3,6]+cm[3,7]))
  prec4.append(cm[4,4]/(cm[4,0]+cm[4,1]+cm[4,2]+cm[4,3]+cm[4,4]+cm[4,5]+cm[4,6]+cm[4,7]))
  prec5.append(cm[5,5]/(cm[5,0]+cm[5,1]+cm[5,2]+cm[5,3]+cm[5,4]+cm[5,5]+cm[5,6]+cm[5,7]))
  prec6.append(cm[6,6]/(cm[6,0]+cm[6,1]+cm[6,2]+cm[6,3]+cm[6,4]+cm[6,5]+cm[6,6]+cm[6,7]))
  prec7.append(cm[7,7]/(cm[7,0]+cm[7,1]+cm[7,2]+cm[7,3]+cm[7,4]+cm[7,5]+cm[7,6]+cm[7,7]))'''

  # 4 Classes
  '''prec0.append(cm[0,0]/(cm[0,0]+cm[0,1]+cm[0,2]+cm[0,3]))
  print("cm[0,0]: ", cm[0,0])
  print("cm[0,1]: ", cm[0,1])
  print("cm[0,2]: ", cm[0,2])
  print("cm[0,3]: ", cm[0,3])
  print("prec0: ", cm[0,0]/(cm[0,0]+cm[0,1]+cm[0,2]+cm[0,3]))
  prec1.append(cm[1,1]/(cm[1,0]+cm[1,1]+cm[1,2]+cm[1,3]))
  prec2.append(cm[2,2]/(cm[2,0]+cm[2,1]+cm[2,2]+cm[2,3]))
  prec3.append(cm[3,3]/(cm[3,0]+cm[3,1]+cm[3,2]+cm[3,3]))'''

  # 3 Classes
  prec0.append(cm[0,0]/(cm[0,0]+cm[0,1]+cm[0,2]))
  print("cm[0,0]: ", cm[0,0])
  print("cm[0,1]: ", cm[0,1])
  print("cm[0,2]: ", cm[0,2])
  print("prec0: ", cm[0,0]/(cm[0,0]+cm[0,1]+cm[0,2]))
  prec1.append(cm[1,1]/(cm[1,0]+cm[1,1]+cm[1,2]))
  prec2.append(cm[2,2]/(cm[2,0]+cm[2,1]+cm[2,2]))

  mat_conf.append(cm)


  teste_acc = accuracy_score(y_test, y_pred)

  if teste_acc > acc_best:
      acc_best = teste_acc
      prec_best = precision_score(y_test, y_pred, average='macro')
      rec_best = recall_score(y_test, y_pred, average='macro')
      f1_best = f1_score(y_test,y_pred, average='macro')

      #8 Classes
      '''prec0_best = (cm[0,0]/(cm[0,0]+cm[0,1]+cm[0,2]+cm[0,3]+cm[0,4]+cm[0,5]+cm[0,6]+cm[0,7]))
      prec1_best = (cm[1,1]/(cm[1,0]+cm[1,1]+cm[1,2]+cm[1,3]+cm[1,4]+cm[1,5]+cm[1,6]+cm[1,7]))
      prec2_best = (cm[2,2]/(cm[2,0]+cm[2,1]+cm[2,2]+cm[2,3]+cm[2,4]+cm[2,5]+cm[2,6]+cm[2,7]))
      prec3_best = (cm[3,3]/(cm[3,0]+cm[3,1]+cm[3,2]+cm[3,3]+cm[3,4]+cm[3,5]+cm[3,6]+cm[3,7]))
      prec4_best = (cm[4,4]/(cm[4,0]+cm[4,1]+cm[4,2]+cm[4,3]+cm[4,4]+cm[4,5]+cm[4,6]+cm[4,7]))
      prec5_best = (cm[5,5]/(cm[5,0]+cm[5,1]+cm[5,2]+cm[5,3]+cm[5,4]+cm[5,5]+cm[5,6]+cm[5,7]))
      prec6_best = (cm[6,6]/(cm[6,0]+cm[6,1]+cm[6,2]+cm[6,3]+cm[6,4]+cm[6,5]+cm[6,6]+cm[6,7]))
      prec7_best = (cm[7,7]/(cm[7,0]+cm[7,1]+cm[7,2]+cm[7,3]+cm[7,4]+cm[7,5]+cm[7,6]+cm[7,7]))'''

      #4 Classes
      '''prec0.append(cm[0,0]/(cm[0,0]+cm[0,1]+cm[0,2]+cm[0,3]))
      prec1.append(cm[1,1]/(cm[1,0]+cm[1,1]+cm[1,2]+cm[1,3]))
      prec2.append(cm[2,2]/(cm[2,0]+cm[2,1]+cm[2,2]+cm[2,3]))
      prec3.append(cm[3,3]/(cm[3,0]+cm[3,1]+cm[3,2]+cm[3,3]))'''

      #3 Classes
      prec0.append(cm[0,0]/(cm[0,0]+cm[0,1]+cm[0,2]))
      prec1.append(cm[1,1]/(cm[1,0]+cm[1,1]+cm[1,2]))
      prec2.append(cm[2,2]/(cm[2,0]+cm[2,1]+cm[2,2]))

      matriz_best = cm

desvio_acc = np.std(acc)
desvio_precision = np.std(prec)
desvio_recall = np.std(rec)
desvio_f1score = np.std(f1s)

# 8 Classes
'''desvio_prec0 = np.std(prec0)
desvio_prec1 = np.std(prec1)
desvio_prec2 = np.std(prec2)
desvio_prec3 = np.std(prec3)
desvio_prec4 = np.std(prec4)
desvio_prec5 = np.std(prec5)
desvio_prec6 = np.std(prec6)
desvio_prec7 = np.std(prec7)'''

# 4 Classes
'''desvio_prec0 = np.std(prec0)
desvio_prec1 = np.std(prec1)
desvio_prec2 = np.std(prec2)
desvio_prec3 = np.std(prec3)'''

# 3 Classes
desvio_prec0 = np.std(prec0)
desvio_prec1 = np.std(prec1)
desvio_prec2 = np.std(prec2)

acc_final = sum(acc)/kfold
prec_final = sum(prec)/kfold
recall_final = sum(rec)/kfold
f1_final = sum(f1s)/kfold

# 8 Classes
'''prec0_final = sum(prec0)/kfold
prec1_final = sum(prec1)/kfold
prec2_final = sum(prec2)/kfold
prec3_final = sum(prec3)/kfold
prec4_final = sum(prec4)/kfold
prec5_final = sum(prec5)/kfold
prec6_final = sum(prec6)/kfold
prec7_final = sum(prec7)/kfold'''

# 4 Classes
'''prec0_final = sum(prec0)/kfold
print("Prec0_final: ", prec0)
prec1_final = sum(prec1)/kfold
prec2_final = sum(prec2)/kfold
prec3_final = sum(prec3)/kfold'''

# 3 Classes
prec0_final = sum(prec0)/kfold
prec1_final = sum(prec1)/kfold
prec2_final = sum(prec2)/kfold


with open(nome_do_arquivo, "w") as arquivo:

  arquivo.write(str(conf_mat))
  arquivo.write("\n\nAcc: "+str(acc_final*100))
  arquivo.write("\n\nDesvio Padrao ACC: "+str(desvio_acc*100))
  arquivo.write("\n\nF1_score: "+str(f1_final*100))
  arquivo.write("\n\nDesvio Padrao f1_score: "+str(desvio_f1score*100))
  arquivo.write("\n\nRecall: "+str(recall_final*100))
  arquivo.write("\n\nDesvio Padrao Recall: "+str(desvio_recall*100))
  arquivo.write("\n\nPrecision: "+str(prec_final*100))
  arquivo.write("\n\nDesvio Padrao Precision: "+str(desvio_precision*100))

  # 8 Classes
  '''arquivo.write("\n\nPrecision Classe 0: "+str(prec0_final*100))
  arquivo.write("\n\nDesvio Padrao Precision Classe 0: "+str(desvio_prec0*100))
  arquivo.write("\n\nPrecision Classe 1: "+str(prec1_final*100))
  arquivo.write("\n\nDesvio Padrao Precision Classe 1: "+str(desvio_prec1*100))
  arquivo.write("\n\nPrecision Classe 2: "+str(prec2_final*100))
  arquivo.write("\n\nDesvio Padrao Precision Classe 2: "+str(desvio_prec2*100))
  arquivo.write("\n\nPrecision Classe 3: "+str(prec3_final*100))
  arquivo.write("\n\nDesvio Padrao Precision Classe 3: "+str(desvio_prec3*100))
  arquivo.write("\n\nPrecision Classe 4: "+str(prec4_final*100))
  arquivo.write("\n\nDesvio Padrao Precision Classe 4: "+str(desvio_prec4*100))
  arquivo.write("\n\nPrecision Classe 5: "+str(prec5_final*100))
  arquivo.write("\n\nDesvio Padrao Precision Classe 5: "+str(desvio_prec5*100))
  arquivo.write("\n\nPrecision Classe 6: "+str(prec6_final*100))
  arquivo.write("\n\nDesvio Padrao Precision Classe 6: "+str(desvio_prec6*100))
  arquivo.write("\n\nPrecision Classe 7: "+str(prec7_final*100))
  arquivo.write("\n\nDesvio Padrao Precision Classe 7: "+str(desvio_prec7*100))'''

  # 4 Classes
  '''arquivo.write("\n\nPrecision Classe 0: "+str(prec0_final*100))
  arquivo.write("\n\nDesvio Padrao Precision Classe 0: "+str(desvio_prec0*100))
  arquivo.write("\n\nPrecision Classe 1: "+str(prec1_final*100))
  arquivo.write("\n\nDesvio Padrao Precision Classe 1: "+str(desvio_prec1*100))
  arquivo.write("\n\nPrecision Classe 2: "+str(prec2_final*100))
  arquivo.write("\n\nDesvio Padrao Precision Classe 2: "+str(desvio_prec2*100))
  arquivo.write("\n\nPrecision Classe 3: "+str(prec3_final*100))
  arquivo.write("\n\nDesvio Padrao Precision Classe 3: "+str(desvio_prec3*100))'''

  # 3 Classes
  arquivo.write("\n\nPrecision Classe 0: "+str(prec0_final*100))
  arquivo.write("\n\nDesvio Padrao Precision Classe 0: "+str(desvio_prec0*100))
  arquivo.write("\n\nPrecision Classe 1: "+str(prec1_final*100))
  arquivo.write("\n\nDesvio Padrao Precision Classe 1: "+str(desvio_prec1*100))
  arquivo.write("\n\nPrecision Classe 2: "+str(prec2_final*100))
  arquivo.write("\n\nDesvio Padrao Precision Classe 2: "+str(desvio_prec2*100))


  arquivo.write("\n\n ---------------- Tempos --------------------")
  arquivo.write("\n\nTempo de treinamento: "+str((sum(tempo_treinamento)/kfold)))
  arquivo.write("\nTempo de teste: "+str((sum(tempo_teste)/kfold)))

  arquivo.write("\n\n ------------- melhor resultado --------------------")
  arquivo.write("\n\nACC: "+str(acc_best*100))
  arquivo.write("\n\nPrecision: "+str(prec_best*100))

  # 8 Classes
  '''arquivo.write("\n\nPrecision Classe 0: "+str(prec0_best*100))
  arquivo.write("\n\nPrecision Classe 1: "+str(prec1_best*100))
  arquivo.write("\n\nPrecision Classe 2: "+str(prec2_best*100))
  arquivo.write("\n\nPrecision Classe 3: "+str(prec3_best*100))
  arquivo.write("\n\nPrecision Classe 4: "+str(prec4_best*100))
  arquivo.write("\n\nPrecision Classe 5: "+str(prec5_best*100))
  arquivo.write("\n\nPrecision Classe 6: "+str(prec6_best*100))
  arquivo.write("\n\nPrecision Classe 7: "+str(prec7_best*100))'''

  # 4 Classes
  '''arquivo.write("\n\nPrecision Classe 0: "+str(prec0_best*100))
  arquivo.write("\n\nPrecision Classe 1: "+str(prec1_best*100))
  arquivo.write("\n\nPrecision Classe 2: "+str(prec2_best*100))
  arquivo.write("\n\nPrecision Classe 3: "+str(prec3_best*100))'''

  # 3 Classes
  arquivo.write("\n\nPrecision Classe 0: "+str(prec0_best*100))
  arquivo.write("\n\nPrecision Classe 1: "+str(prec1_best*100))
  arquivo.write("\n\nPrecision Classe 2: "+str(prec2_best*100))

  arquivo.write("\n\nRecall: "+str(rec_best*100))
  arquivo.write("\n\nF1-score: "+str(f1_best*100))
  arquivo.write("\n\nVetor Matriz: "+str(matriz_best))

  arquivo.write("\n\n ------------- Valores 15 interações --------------------")
  arquivo.write("Acc: "+str(acc))
  arquivo.write("\nPrecision: " +str(prec))

  # 8 Classes
  '''arquivo.write("\n\nPrecision Classe 0: "+str(prec0))
  arquivo.write("\n\nPrecision Classe 1: "+str(prec1))
  arquivo.write("\n\nPrecision Classe 2: "+str(prec2))
  arquivo.write("\n\nPrecision Classe 3: "+str(prec3))
  arquivo.write("\n\nPrecision Classe 4: "+str(prec4))
  arquivo.write("\n\nPrecision Classe 5: "+str(prec5))
  arquivo.write("\n\nPrecision Classe 6: "+str(prec6))
  arquivo.write("\n\nPrecision Classe 7: "+str(prec7))'''

  # 4 Classes
  '''arquivo.write("\n\nPrecision Classe 0: "+str(prec0))
  arquivo.write("\n\nPrecision Classe 1: "+str(prec1))
  arquivo.write("\n\nPrecision Classe 2: "+str(prec2))
  arquivo.write("\n\nPrecision Classe 3: "+str(prec3))'''

  # 3 Classes
  arquivo.write("\n\nPrecision Classe 0: "+str(prec0))
  arquivo.write("\n\nPrecision Classe 1: "+str(prec1))
  arquivo.write("\n\nPrecision Classe 2: "+str(prec2))

  arquivo.write("\nF1_score:" +str(rec))
  arquivo.write("\nRecall: "+str(rec))
  arquivo.write("\n\nVetor Matriz: "+str(mat_conf))
  arquivo.close()

kfold = 15 # no. of folds
skf = StratifiedKFold(kfold, shuffle=True,random_state=1)
skfind = [None] * kfold  # skfind[i][0] -> train indices, skfind[i][1] -> test indices
cnt = 0
for index in skf.split(X, Y):
    skfind[cnt] = index
    cnt += 1

nome_do_arquivo = "metricas_knn_" + nomeExtrator + ".txt"

conf_mat = np.zeros((len(classes_names), len(classes_names)))
tempo_treinamento = []
tempo_teste = []
acc = []
prec = []
rec = []
f1s = []
mat_conf = []
prec0 = []
prec1 = []
prec2 = []
prec3 = []
prec4 = []
prec5 = []
prec6 = []
prec7 = []

acc_best = -99
prec_best  = -99
rec_best  = -99
f1_best  = -99
matriz_best  = -99
prec0_best = -99
prec1_best = -99
prec2_best = -99
prec3_best = -99
prec4_best = -99
prec5_best = -99
prec6_best = -99
prec7_best = -99

for i in range(kfold):
  train_indices = skfind[i][0]
  test_indices = skfind[i][1]

  X_train_aux = ExtractorFeatures[train_indices]
  nsamples, nx, ny, nz = X_train_aux.shape
  X_train = X_train_aux.reshape((nsamples, nx*ny*nz))
  y_train = Y[train_indices]

  X_test_aux= ExtractorFeatures[test_indices]
  nsamples, x, y, z = X_test_aux.shape
  X_test = X_test_aux.reshape((nsamples, x*y*z))
  y_test = Y[test_indices]

  #Treinando o modelo
  print("--------------------------------------------------")
  print("Treinando...", end='\n')
  knn_model = KNeighborsClassifier(n_neighbors=3)
  inicio_treinamento = time.time()
  knn_model.fit(X_train, y_train)
  fim_treinamento = time.time()

  #tempo de treinamento
  tempo_treinamento.append((fim_treinamento - inicio_treinamento))

  #Testando o modelo
  print("Testando...", end='\n')
  inicio_teste = time.time()
  y_pred = knn_model.predict(X_test)
  fim_teste = time.time()

  print("[%d] Test acurracy: %.4f" %(i,accuracy_score(y_test,y_pred)))
  cm = confusion_matrix(y_test, y_pred)
  print('===== CONFUSION MATRIX ======')
  print(cm, end='\n')
  conf_mat += cm

  #tempo de teste
  tempo_teste.append((fim_teste - inicio_teste))

  acc.append(accuracy_score(y_test,y_pred))
  prec.append(precision_score(y_test, y_pred, average='macro'))
  rec.append(recall_score(y_test, y_pred, average='macro'))
  f1s.append(f1_score(y_test, y_pred, average='macro'))

  # 8 Classes
  '''prec0.append(cm[0,0]/(cm[0,0]+cm[0,1]+cm[0,2]+cm[0,3]+cm[0,4]+cm[0,5]+cm[0,6]+cm[0,7]))
  prec1.append(cm[1,1]/(cm[1,0]+cm[1,1]+cm[1,2]+cm[1,3]+cm[1,4]+cm[1,5]+cm[1,6]+cm[1,7]))
  prec2.append(cm[2,2]/(cm[2,0]+cm[2,1]+cm[2,2]+cm[2,3]+cm[2,4]+cm[2,5]+cm[2,6]+cm[2,7]))
  prec3.append(cm[3,3]/(cm[3,0]+cm[3,1]+cm[3,2]+cm[3,3]+cm[3,4]+cm[3,5]+cm[3,6]+cm[3,7]))
  prec4.append(cm[4,4]/(cm[4,0]+cm[4,1]+cm[4,2]+cm[4,3]+cm[4,4]+cm[4,5]+cm[4,6]+cm[4,7]))
  prec5.append(cm[5,5]/(cm[5,0]+cm[5,1]+cm[5,2]+cm[5,3]+cm[5,4]+cm[5,5]+cm[5,6]+cm[5,7]))
  prec6.append(cm[6,6]/(cm[6,0]+cm[6,1]+cm[6,2]+cm[6,3]+cm[6,4]+cm[6,5]+cm[6,6]+cm[6,7]))
  prec7.append(cm[7,7]/(cm[7,0]+cm[7,1]+cm[7,2]+cm[7,3]+cm[7,4]+cm[7,5]+cm[7,6]+cm[7,7]))'''

  # 4 Classes
  '''prec0.append(cm[0,0]/(cm[0,0]+cm[0,1]+cm[0,2]+cm[0,3]))
  prec1.append(cm[1,1]/(cm[1,0]+cm[1,1]+cm[1,2]+cm[1,3]))
  prec2.append(cm[2,2]/(cm[2,0]+cm[2,1]+cm[2,2]+cm[2,3]))
  prec3.append(cm[3,3]/(cm[3,0]+cm[3,1]+cm[3,2]+cm[3,3]))'''

  # 3 Classes
  prec0.append(cm[0,0]/(cm[0,0]+cm[0,1]+cm[0,2]))
  prec1.append(cm[1,1]/(cm[1,0]+cm[1,1]+cm[1,2]))
  prec2.append(cm[2,2]/(cm[2,0]+cm[2,1]+cm[2,2]))

  mat_conf.append(cm)

  teste_acc = accuracy_score(y_test, y_pred)

  if teste_acc > acc_best:
      acc_best = teste_acc
      prec_best = precision_score(y_test, y_pred, average='macro')
      rec_best = recall_score(y_test, y_pred, average='macro')
      f1_best = f1_score(y_test,y_pred, average='macro')

      # 8 Classes
      '''prec0_best = (cm[0,0]/(cm[0,0]+cm[0,1]+cm[0,2]+cm[0,3]+cm[0,4]+cm[0,5]+cm[0,6]+cm[0,7]))
      prec1_best = (cm[1,1]/(cm[1,0]+cm[1,1]+cm[1,2]+cm[1,3]+cm[1,4]+cm[1,5]+cm[1,6]+cm[1,7]))
      prec2_best = (cm[2,2]/(cm[2,0]+cm[2,1]+cm[2,2]+cm[2,3]+cm[2,4]+cm[2,5]+cm[2,6]+cm[2,7]))
      prec3_best = (cm[3,3]/(cm[3,0]+cm[3,1]+cm[3,2]+cm[3,3]+cm[3,4]+cm[3,5]+cm[3,6]+cm[3,7]))
      prec4_best = (cm[4,4]/(cm[4,0]+cm[4,1]+cm[4,2]+cm[4,3]+cm[4,4]+cm[4,5]+cm[4,6]+cm[4,7]))
      prec5_best = (cm[5,5]/(cm[5,0]+cm[5,1]+cm[5,2]+cm[5,3]+cm[5,4]+cm[5,5]+cm[5,6]+cm[5,7]))
      prec6_best = (cm[6,6]/(cm[6,0]+cm[6,1]+cm[6,2]+cm[6,3]+cm[6,4]+cm[6,5]+cm[6,6]+cm[6,7]))
      prec7_best = (cm[7,7]/(cm[7,0]+cm[7,1]+cm[7,2]+cm[7,3]+cm[7,4]+cm[7,5]+cm[7,6]+cm[7,7]))'''

      # 4 Classes
      '''prec0.append(cm[0,0]/(cm[0,0]+cm[0,1]+cm[0,2]+cm[0,3]))
      prec1.append(cm[1,1]/(cm[1,0]+cm[1,1]+cm[1,2]+cm[1,3]))
      prec2.append(cm[2,2]/(cm[2,0]+cm[2,1]+cm[2,2]+cm[2,3]))
      prec3.append(cm[3,3]/(cm[3,0]+cm[3,1]+cm[3,2]+cm[3,3]))'''

      # 3 Classes
      prec0.append(cm[0,0]/(cm[0,0]+cm[0,1]+cm[0,2]))
      prec1.append(cm[1,1]/(cm[1,0]+cm[1,1]+cm[1,2]))
      prec2.append(cm[2,2]/(cm[2,0]+cm[2,1]+cm[2,2]))

      matriz_best = cm

desvio_acc = np.std(acc)
desvio_precision = np.std(prec)
desvio_recall = np.std(rec)
desvio_f1score = np.std(f1s)

# 8 Classes
'''desvio_prec0 = np.std(prec0)
desvio_prec1 = np.std(prec1)
desvio_prec2 = np.std(prec2)
desvio_prec3 = np.std(prec3)
desvio_prec4 = np.std(prec4)
desvio_prec5 = np.std(prec5)
desvio_prec6 = np.std(prec6)
desvio_prec7 = np.std(prec7)'''


# 4 Classes
'''desvio_prec0 = np.std(prec0)
desvio_prec1 = np.std(prec1)
desvio_prec2 = np.std(prec2)
desvio_prec3 = np.std(prec3)'''

# 3 Classes
desvio_prec0 = np.std(prec0)
desvio_prec1 = np.std(prec1)
desvio_prec2 = np.std(prec2)

acc_final = sum(acc)/kfold
prec_final = sum(prec)/kfold
recall_final = sum(rec)/kfold
f1_final = sum(f1s)/kfold

# 8 Classes
'''prec0_final = sum(prec0)/kfold
prec1_final = sum(prec1)/kfold
prec2_final = sum(prec2)/kfold
prec3_final = sum(prec3)/kfold
prec4_final = sum(prec4)/kfold
prec5_final = sum(prec5)/kfold
prec6_final = sum(prec6)/kfold
prec7_final = sum(prec7)/kfold'''

# 4 Classes
'''prec0_final = sum(prec0)/kfold
prec1_final = sum(prec1)/kfold
prec2_final = sum(prec2)/kfold
prec3_final = sum(prec3)/kfold'''

# 3 Classes
prec0_final = sum(prec0)/kfold
prec1_final = sum(prec1)/kfold
prec2_final = sum(prec2)/kfold

with open(nome_do_arquivo, "w") as arquivo:

  arquivo.write(str(conf_mat))
  arquivo.write(str(conf_mat))
  arquivo.write("\n\nAcc: "+str(acc_final*100))
  arquivo.write("\n\nDesvio Padrao ACC: "+str(desvio_acc*100))
  arquivo.write("\n\nF1_score: "+str(f1_final*100))
  arquivo.write("\n\nDesvio Padrao f1_score: "+str(desvio_f1score*100))
  arquivo.write("\n\nRecall: "+str(recall_final*100))
  arquivo.write("\n\nDesvio Padrao Recall: "+str(desvio_recall*100))
  arquivo.write("\n\nPrecision: "+str(prec_final*100))
  arquivo.write("\n\nDesvio Padrao Precision: "+str(desvio_precision*100))

  # 8 Classes
  '''arquivo.write("\n\nPrecision Classe 0: "+str(prec0_final*100))
  arquivo.write("\n\nDesvio Padrao Precision Classe 0: "+str(desvio_prec0*100))
  arquivo.write("\n\nPrecision Classe 1: "+str(prec1_final*100))
  arquivo.write("\n\nDesvio Padrao Precision Classe 1: "+str(desvio_prec1*100))
  arquivo.write("\n\nPrecision Classe 2: "+str(prec2_final*100))
  arquivo.write("\n\nDesvio Padrao Precision Classe 2: "+str(desvio_prec2*100))
  arquivo.write("\n\nPrecision Classe 3: "+str(prec3_final*100))
  arquivo.write("\n\nDesvio Padrao Precision Classe 3: "+str(desvio_prec3*100))
  arquivo.write("\n\nPrecision Classe 4: "+str(prec4_final*100))
  arquivo.write("\n\nDesvio Padrao Precision Classe 4: "+str(desvio_prec4*100))
  arquivo.write("\n\nPrecision Classe 5: "+str(prec5_final*100))
  arquivo.write("\n\nDesvio Padrao Precision Classe 5: "+str(desvio_prec5*100))
  arquivo.write("\n\nPrecision Classe 6: "+str(prec6_final*100))
  arquivo.write("\n\nDesvio Padrao Precision Classe 6: "+str(desvio_prec6*100))
  arquivo.write("\n\nPrecision Classe 7: "+str(prec7_final*100))
  arquivo.write("\n\nDesvio Padrao Precision Classe 7: "+str(desvio_prec7*100))'''

  # 4 Classes
  '''arquivo.write("\n\nPrecision Classe 0: "+str(prec0_final*100))
  arquivo.write("\n\nDesvio Padrao Precision Classe 0: "+str(desvio_prec0*100))
  arquivo.write("\n\nPrecision Classe 1: "+str(prec1_final*100))
  arquivo.write("\n\nDesvio Padrao Precision Classe 1: "+str(desvio_prec1*100))
  arquivo.write("\n\nPrecision Classe 2: "+str(prec2_final*100))
  arquivo.write("\n\nDesvio Padrao Precision Classe 2: "+str(desvio_prec2*100))
  arquivo.write("\n\nPrecision Classe 3: "+str(prec3_final*100))
  arquivo.write("\n\nDesvio Padrao Precision Classe 3: "+str(desvio_prec3*100))'''

  # 3 Classes
  arquivo.write("\n\nPrecision Classe 0: "+str(prec0_final*100))
  arquivo.write("\n\nDesvio Padrao Precision Classe 0: "+str(desvio_prec0*100))
  arquivo.write("\n\nPrecision Classe 1: "+str(prec1_final*100))
  arquivo.write("\n\nDesvio Padrao Precision Classe 1: "+str(desvio_prec1*100))
  arquivo.write("\n\nPrecision Classe 2: "+str(prec2_final*100))
  arquivo.write("\n\nDesvio Padrao Precision Classe 2: "+str(desvio_prec2*100))

  arquivo.write("\n\n ---------------- Tempos --------------------")
  arquivo.write("\n\nTempo de treinamento: "+str((sum(tempo_treinamento)/kfold)))
  arquivo.write("\nTempo de teste: "+str((sum(tempo_teste)/kfold)))

  arquivo.write("\n\n ------------- melhor resultado --------------------")
  arquivo.write("\n\nACC: "+str(acc_best*100))
  arquivo.write("\n\nPrecision: "+str(prec_best*100))

  # 8 Classes
  '''arquivo.write("\n\nPrecision Classe 0: "+str(prec0_best*100))
  arquivo.write("\n\nPrecision Classe 1: "+str(prec1_best*100))
  arquivo.write("\n\nPrecision Classe 2: "+str(prec2_best*100))
  arquivo.write("\n\nPrecision Classe 3: "+str(prec3_best*100))
  arquivo.write("\n\nPrecision Classe 4: "+str(prec4_best*100))
  arquivo.write("\n\nPrecision Classe 5: "+str(prec5_best*100))
  arquivo.write("\n\nPrecision Classe 6: "+str(prec6_best*100))
  arquivo.write("\n\nPrecision Classe 7: "+str(prec7_best*100))'''

  # 4 Classes
  '''arquivo.write("\n\nPrecision Classe 0: "+str(prec0_best*100))
  arquivo.write("\n\nPrecision Classe 1: "+str(prec1_best*100))
  arquivo.write("\n\nPrecision Classe 2: "+str(prec2_best*100))
  arquivo.write("\n\nPrecision Classe 3: "+str(prec3_best*100))'''

  # 3 Classes
  arquivo.write("\n\nPrecision Classe 0: "+str(prec0_best*100))
  arquivo.write("\n\nPrecision Classe 1: "+str(prec1_best*100))
  arquivo.write("\n\nPrecision Classe 2: "+str(prec2_best*100))

  arquivo.write("\n\nRecall: "+str(rec_best*100))
  arquivo.write("\n\nF1-score: "+str(f1_best*100))
  arquivo.write("\n\nVetor Matriz: "+str(matriz_best))

  arquivo.write("\n\n ------------- Valores 15 interações --------------------")
  arquivo.write("Acc: "+str(acc))
  arquivo.write("\nPrecision: " +str(prec))

  # 8 Classes
  ''' arquivo.write("\n\nPrecision Classe 0: "+str(prec0))
  arquivo.write("\n\nPrecision Classe 1: "+str(prec1))
  arquivo.write("\n\nPrecision Classe 2: "+str(prec2))
  arquivo.write("\n\nPrecision Classe 3: "+str(prec3))
  arquivo.write("\n\nPrecision Classe 4: "+str(prec4))
  arquivo.write("\n\nPrecision Classe 5: "+str(prec5))
  arquivo.write("\n\nPrecision Classe 6: "+str(prec6))
  arquivo.write("\n\nPrecision Classe 7: "+str(prec7))'''

  # 4 Classes
  '''arquivo.write("\n\nPrecision Classe 0: "+str(prec0))
  arquivo.write("\n\nPrecision Classe 1: "+str(prec1))
  arquivo.write("\n\nPrecision Classe 2: "+str(prec2))
  arquivo.write("\n\nPrecision Classe 3: "+str(prec3))'''

  # 3 Classes
  arquivo.write("\n\nPrecision Classe 0: "+str(prec0))
  arquivo.write("\n\nPrecision Classe 1: "+str(prec1))
  arquivo.write("\n\nPrecision Classe 2: "+str(prec2))

  arquivo.write("\nF1_score:" +str(rec))
  arquivo.write("\nRecall: "+str(rec))
  arquivo.write("\n\nVetor Matriz: "+str(mat_conf))
  arquivo.close()

kfold = 15 # no. of folds
skf = StratifiedKFold(kfold, shuffle=True,random_state=1)
skfind = [None] * kfold  # skfind[i][0] -> train indices, skfind[i][1] -> test indices
cnt = 0
for index in skf.split(X, Y):
    skfind[cnt] = index
    cnt += 1

nome_do_arquivo = "metricas_svm_" + nomeExtrator + ".txt"

conf_mat = np.zeros((len(classes_names), len(classes_names)))
tempo_treinamento = []
tempo_teste = []
acc = []
prec = []
rec = []
f1s = []
mat_conf = []
prec0 = []
prec1 = []
prec2 = []
prec3 = []
prec4 = []
prec5 = []
prec6 = []
prec7 = []

acc_best = -99
prec_best  = -99
rec_best  = -99
f1_best  = -99
matriz_best  = -99
prec0_best = -99
prec1_best = -99
prec2_best = -99
prec3_best = -99
prec4_best = -99
prec5_best = -99
prec6_best = -99
prec7_best = -99

for i in range(kfold):
  train_indices = skfind[i][0]
  test_indices = skfind[i][1]

  X_train_aux = ExtractorFeatures[train_indices]
  nsamples, nx, ny, nz = X_train_aux.shape
  X_train = X_train_aux.reshape((nsamples, nx*ny*nz))
  y_train = Y[train_indices]

  X_test_aux= ExtractorFeatures[test_indices]
  nsamples, x, y, z = X_test_aux.shape
  X_test = X_test_aux.reshape((nsamples, x*y*z))
  y_test = Y[test_indices]

  #Treinando o modelo
  print("--------------------------------------------------")
  print("Treinando...", end='\n')
  svm_model = SVC(kernel="linear")
  inicio_treinamento = time.time()
  svm_model.fit(X_train, y_train)
  fim_treinamento = time.time()

  #tempo de treinamento
  tempo_treinamento.append((fim_treinamento - inicio_treinamento))

  #Testando o modelo
  print("Testando...", end='\n')
  inicio_teste = time.time()
  y_pred = svm_model.predict(X_test)
  fim_teste = time.time()

  print("[%d] Test acurracy: %.4f" %(i,accuracy_score(y_test,y_pred)))
  cm = confusion_matrix(y_test, y_pred)
  print('===== CONFUSION MATRIX ======')
  print(cm, end='\n')
  conf_mat += cm

  #tempo de teste
  tempo_teste.append((fim_teste - inicio_teste))

  acc.append(accuracy_score(y_test,y_pred))
  prec.append(precision_score(y_test, y_pred, average='macro'))
  rec.append(recall_score(y_test, y_pred, average='macro'))
  f1s.append(f1_score(y_test, y_pred, average='macro'))

  # 8 Classes
  '''prec0.append(cm[0,0]/(cm[0,0]+cm[0,1]+cm[0,2]+cm[0,3]+cm[0,4]+cm[0,5]+cm[0,6]+cm[0,7]))
  prec1.append(cm[1,1]/(cm[1,0]+cm[1,1]+cm[1,2]+cm[1,3]+cm[1,4]+cm[1,5]+cm[1,6]+cm[1,7]))
  prec2.append(cm[2,2]/(cm[2,0]+cm[2,1]+cm[2,2]+cm[2,3]+cm[2,4]+cm[2,5]+cm[2,6]+cm[2,7]))
  prec3.append(cm[3,3]/(cm[3,0]+cm[3,1]+cm[3,2]+cm[3,3]+cm[3,4]+cm[3,5]+cm[3,6]+cm[3,7]))
  prec4.append(cm[4,4]/(cm[4,0]+cm[4,1]+cm[4,2]+cm[4,3]+cm[4,4]+cm[4,5]+cm[4,6]+cm[4,7]))
  prec5.append(cm[5,5]/(cm[5,0]+cm[5,1]+cm[5,2]+cm[5,3]+cm[5,4]+cm[5,5]+cm[5,6]+cm[5,7]))
  prec6.append(cm[6,6]/(cm[6,0]+cm[6,1]+cm[6,2]+cm[6,3]+cm[6,4]+cm[6,5]+cm[6,6]+cm[6,7]))
  prec7.append(cm[7,7]/(cm[7,0]+cm[7,1]+cm[7,2]+cm[7,3]+cm[7,4]+cm[7,5]+cm[7,6]+cm[7,7]))'''

  # 4 Classes
  '''prec0.append(cm[0,0]/(cm[0,0]+cm[0,1]+cm[0,2]+cm[0,3]))
  prec1.append(cm[1,1]/(cm[1,0]+cm[1,1]+cm[1,2]+cm[1,3]))
  prec2.append(cm[2,2]/(cm[2,0]+cm[2,1]+cm[2,2]+cm[2,3]))
  prec3.append(cm[3,3]/(cm[3,0]+cm[3,1]+cm[3,2]+cm[3,3]))'''

  # 3 Classes
  prec0.append(cm[0,0]/(cm[0,0]+cm[0,1]+cm[0,2]))
  prec1.append(cm[1,1]/(cm[1,0]+cm[1,1]+cm[1,2]))
  prec2.append(cm[2,2]/(cm[2,0]+cm[2,1]+cm[2,2]))

  mat_conf.append(cm)

  teste_acc = accuracy_score(y_test, y_pred)

  if teste_acc > acc_best:
      acc_best = teste_acc
      prec_best = precision_score(y_test, y_pred, average='macro')
      rec_best = recall_score(y_test, y_pred, average='macro')
      f1_best = f1_score(y_test,y_pred, average='macro')

      # 8 Classes
      '''prec0_best = (cm[0,0]/(cm[0,0]+cm[0,1]+cm[0,2]+cm[0,3]+cm[0,4]+cm[0,5]+cm[0,6]+cm[0,7]))
      prec1_best = (cm[1,1]/(cm[1,0]+cm[1,1]+cm[1,2]+cm[1,3]+cm[1,4]+cm[1,5]+cm[1,6]+cm[1,7]))
      prec2_best = (cm[2,2]/(cm[2,0]+cm[2,1]+cm[2,2]+cm[2,3]+cm[2,4]+cm[2,5]+cm[2,6]+cm[2,7]))
      prec3_best = (cm[3,3]/(cm[3,0]+cm[3,1]+cm[3,2]+cm[3,3]+cm[3,4]+cm[3,5]+cm[3,6]+cm[3,7]))
      prec4_best = (cm[4,4]/(cm[4,0]+cm[4,1]+cm[4,2]+cm[4,3]+cm[4,4]+cm[4,5]+cm[4,6]+cm[4,7]))
      prec5_best = (cm[5,5]/(cm[5,0]+cm[5,1]+cm[5,2]+cm[5,3]+cm[5,4]+cm[5,5]+cm[5,6]+cm[5,7]))
      prec6_best = (cm[6,6]/(cm[6,0]+cm[6,1]+cm[6,2]+cm[6,3]+cm[6,4]+cm[6,5]+cm[6,6]+cm[6,7]))
      prec7_best = (cm[7,7]/(cm[7,0]+cm[7,1]+cm[7,2]+cm[7,3]+cm[7,4]+cm[7,5]+cm[7,6]+cm[7,7]))'''

      # 4 Classes
      '''prec0.append(cm[0,0]/(cm[0,0]+cm[0,1]+cm[0,2]+cm[0,3]))
      prec1.append(cm[1,1]/(cm[1,0]+cm[1,1]+cm[1,2]+cm[1,3]))
      prec2.append(cm[2,2]/(cm[2,0]+cm[2,1]+cm[2,2]+cm[2,3]))
      prec3.append(cm[3,3]/(cm[3,0]+cm[3,1]+cm[3,2]+cm[3,3]))'''

      # 3 Classes
      prec0.append(cm[0,0]/(cm[0,0]+cm[0,1]+cm[0,2]))
      prec1.append(cm[1,1]/(cm[1,0]+cm[1,1]+cm[1,2]))
      prec2.append(cm[2,2]/(cm[2,0]+cm[2,1]+cm[2,2]))

      matriz_best = cm

desvio_acc = np.std(acc)
desvio_precision = np.std(prec)
desvio_recall = np.std(rec)
desvio_f1score = np.std(f1s)

# 8 Classes
'''desvio_prec0 = np.std(prec0)
desvio_prec1 = np.std(prec1)
desvio_prec2 = np.std(prec2)
desvio_prec3 = np.std(prec3)
desvio_prec4 = np.std(prec4)
desvio_prec5 = np.std(prec5)
desvio_prec6 = np.std(prec6)
desvio_prec7 = np.std(prec7)'''

# 4 Classes
'''desvio_prec0 = np.std(prec0)
desvio_prec1 = np.std(prec1)
desvio_prec2 = np.std(prec2)
desvio_prec3 = np.std(prec3)'''

# 3 Classes
desvio_prec0 = np.std(prec0)
desvio_prec1 = np.std(prec1)
desvio_prec2 = np.std(prec2)

acc_final = sum(acc)/kfold
prec_final = sum(prec)/kfold
recall_final = sum(rec)/kfold
f1_final = sum(f1s)/kfold

# 8 Classes
'''prec0_final = sum(prec0)/kfold
prec1_final = sum(prec1)/kfold
prec2_final = sum(prec2)/kfold
prec3_final = sum(prec3)/kfold
prec4_final = sum(prec4)/kfold
prec5_final = sum(prec5)/kfold
prec6_final = sum(prec6)/kfold
prec7_final = sum(prec7)/kfold'''

# 4 Classes
'''prec0_final = sum(prec0)/kfold
prec1_final = sum(prec1)/kfold
prec2_final = sum(prec2)/kfold
prec3_final = sum(prec3)/kfold'''

# 3 Classes
prec0_final = sum(prec0)/kfold
prec1_final = sum(prec1)/kfold
prec2_final = sum(prec2)/kfold

with open(nome_do_arquivo, "w") as arquivo:

  arquivo.write(str(conf_mat))
  arquivo.write("\n\nAcc: "+str(acc_final*100))
  arquivo.write("\n\nDesvio Padrao ACC: "+str(desvio_acc*100))
  arquivo.write("\n\nF1_score: "+str(f1_final*100))
  arquivo.write("\n\nDesvio Padrao f1_score: "+str(desvio_f1score*100))
  arquivo.write("\n\nRecall: "+str(recall_final*100))
  arquivo.write("\n\nDesvio Padrao Recall: "+str(desvio_recall*100))
  arquivo.write("\n\nPrecision: "+str(prec_final*100))
  arquivo.write("\n\nDesvio Padrao Precision: "+str(desvio_precision*100))

  # 8 Classes
  '''arquivo.write("\n\nPrecision Classe 0: "+str(prec0_final*100))
  arquivo.write("\n\nDesvio Padrao Precision Classe 0: "+str(desvio_prec0*100))
  arquivo.write("\n\nPrecision Classe 1: "+str(prec1_final*100))
  arquivo.write("\n\nDesvio Padrao Precision Classe 1: "+str(desvio_prec1*100))
  arquivo.write("\n\nPrecision Classe 2: "+str(prec2_final*100))
  arquivo.write("\n\nDesvio Padrao Precision Classe 2: "+str(desvio_prec2*100))
  arquivo.write("\n\nPrecision Classe 3: "+str(prec3_final*100))
  arquivo.write("\n\nDesvio Padrao Precision Classe 3: "+str(desvio_prec3*100))
  arquivo.write("\n\nPrecision Classe 4: "+str(prec4_final*100))
  arquivo.write("\n\nDesvio Padrao Precision Classe 4: "+str(desvio_prec4*100))
  arquivo.write("\n\nPrecision Classe 5: "+str(prec5_final*100))
  arquivo.write("\n\nDesvio Padrao Precision Classe 5: "+str(desvio_prec5*100))
  arquivo.write("\n\nPrecision Classe 6: "+str(prec6_final*100))
  arquivo.write("\n\nDesvio Padrao Precision Classe 6: "+str(desvio_prec6*100))
  arquivo.write("\n\nPrecision Classe 7: "+str(prec7_final*100))
  arquivo.write("\n\nDesvio Padrao Precision Classe 7: "+str(desvio_prec7*100))'''

  # 4 Classes
  '''arquivo.write("\n\nPrecision Classe 0: "+str(prec0_final*100))
  arquivo.write("\n\nDesvio Padrao Precision Classe 0: "+str(desvio_prec0*100))
  arquivo.write("\n\nPrecision Classe 1: "+str(prec1_final*100))
  arquivo.write("\n\nDesvio Padrao Precision Classe 1: "+str(desvio_prec1*100))
  arquivo.write("\n\nPrecision Classe 2: "+str(prec2_final*100))
  arquivo.write("\n\nDesvio Padrao Precision Classe 2: "+str(desvio_prec2*100))
  arquivo.write("\n\nPrecision Classe 3: "+str(prec3_final*100))
  arquivo.write("\n\nDesvio Padrao Precision Classe 3: "+str(desvio_prec3*100))'''

  # 3 Classes
  arquivo.write("\n\nPrecision Classe 0: "+str(prec0_final*100))
  arquivo.write("\n\nDesvio Padrao Precision Classe 0: "+str(desvio_prec0*100))
  arquivo.write("\n\nPrecision Classe 1: "+str(prec1_final*100))
  arquivo.write("\n\nDesvio Padrao Precision Classe 1: "+str(desvio_prec1*100))
  arquivo.write("\n\nPrecision Classe 2: "+str(prec2_final*100))
  arquivo.write("\n\nDesvio Padrao Precision Classe 2: "+str(desvio_prec2*100))

  arquivo.write("\n\n ---------------- Tempos --------------------")
  arquivo.write("\n\nTempo de treinamento: "+str((sum(tempo_treinamento)/kfold)))
  arquivo.write("\nTempo de teste: "+str((sum(tempo_teste)/kfold)))

  arquivo.write("\n\n ------------- melhor resultado --------------------")
  arquivo.write("\n\nACC: "+str(acc_best*100))
  arquivo.write("\n\nPrecision: "+str(prec_best*100))

  # 8 Classes
  '''arquivo.write("\n\nPrecision Classe 0: "+str(prec0_best*100))
  arquivo.write("\n\nPrecision Classe 1: "+str(prec1_best*100))
  arquivo.write("\n\nPrecision Classe 2: "+str(prec2_best*100))
  arquivo.write("\n\nPrecision Classe 3: "+str(prec3_best*100))
  arquivo.write("\n\nPrecision Classe 4: "+str(prec4_best*100))
  arquivo.write("\n\nPrecision Classe 5: "+str(prec5_best*100))
  arquivo.write("\n\nPrecision Classe 6: "+str(prec6_best*100))
  arquivo.write("\n\nPrecision Classe 7: "+str(prec7_best*100))'''

  # 4 Classes
  '''arquivo.write("\n\nPrecision Classe 0: "+str(prec0_best*100))
  arquivo.write("\n\nPrecision Classe 1: "+str(prec1_best*100))
  arquivo.write("\n\nPrecision Classe 2: "+str(prec2_best*100))
  arquivo.write("\n\nPrecision Classe 3: "+str(prec3_best*100))'''

  # 3 Classes
  arquivo.write("\n\nPrecision Classe 0: "+str(prec0_best*100))
  arquivo.write("\n\nPrecision Classe 1: "+str(prec1_best*100))
  arquivo.write("\n\nPrecision Classe 2: "+str(prec2_best*100))

  arquivo.write("\n\nRecall: "+str(rec_best*100))
  arquivo.write("\n\nF1-score: "+str(f1_best*100))
  arquivo.write("\n\nVetor Matriz: "+str(matriz_best))

  arquivo.write("\n\n ------------- Valores 15 interações --------------------")
  arquivo.write("Acc: "+str(acc))
  arquivo.write("\nPrecision: " +str(prec))

  # 8 Classes
  '''arquivo.write("\n\nPrecision Classe 0: "+str(prec0))
  arquivo.write("\n\nPrecision Classe 1: "+str(prec1))
  arquivo.write("\n\nPrecision Classe 2: "+str(prec2))
  arquivo.write("\n\nPrecision Classe 3: "+str(prec3))
  arquivo.write("\n\nPrecision Classe 4: "+str(prec4))
  arquivo.write("\n\nPrecision Classe 5: "+str(prec5))
  arquivo.write("\n\nPrecision Classe 6: "+str(prec6))
  arquivo.write("\n\nPrecision Classe 7: "+str(prec7))'''

  # 4 Classes
  '''arquivo.write("\n\nPrecision Classe 0: "+str(prec0))
  arquivo.write("\n\nPrecision Classe 1: "+str(prec1))
  arquivo.write("\n\nPrecision Classe 2: "+str(prec2))
  arquivo.write("\n\nPrecision Classe 3: "+str(prec3))'''

  # 3 Classes
  arquivo.write("\n\nPrecision Classe 0: "+str(prec0))
  arquivo.write("\n\nPrecision Classe 1: "+str(prec1))
  arquivo.write("\n\nPrecision Classe 2: "+str(prec2))

  arquivo.write("\nF1_score:" +str(rec))
  arquivo.write("\nRecall: "+str(rec))
  arquivo.write("\n\nVetor Matriz: "+str(mat_conf))
  arquivo.close()