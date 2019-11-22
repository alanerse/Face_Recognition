import cv2
import numpy as np
import os

#Funções---------------------------------------------------
def equaliza(img):

    #equalização de cada canal da imagem
    R, G, B = cv2.split(img)

    output1_R = cv2.equalizeHist(R)
    output1_G = cv2.equalizeHist(G)
    output1_B = cv2.equalizeHist(B)

    equ = cv2.merge((output1_R, output1_G, output1_B))

    return equ

def geraBase():

    video_capture = cv2.VideoCapture(0)
    cascade_path = 'haar.xml'
    clf = cv2.CascadeClassifier(cascade_path)

    name = input("Nome da nova pessoa: ")

    #cria a pasta da nova pessoa
    path = 'images'
    directory = os.path.join(path, name)
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok='True')

    qnt = 0
    max = 50
    count = 0

    #captura 50 imagens da pessoa e salva na pasta relacionada
    while qnt < max:

        ret, frame = video_capture.read()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = clf.detectMultiScale(frame_gray, 1.3, 5)

        for (x, y, w, h) in faces:  # marcação das faces
            frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
            nframe = frame[y:y + h, x:x + w]
            nframe = cv2.resize(nframe, (255, 255))
            cv2.imwrite(os.path.join(directory, str(str(count) + '.jpg')), nframe)
            qnt += 1
            count += 1
            cv2.waitKey(0)

        cv2.imshow('Video', frame)

def criaArquivoDeRotulo(pasta):
    label = 0
    f = open("TRAIN", "w+")
    for dirPrincipal, nomeDirs, nomeArqs in os.walk(pasta):
        for subDir in nomeDirs:
            caminhoPasta = os.path.join(dirPrincipal, subDir)
            for filename in os.listdir(caminhoPasta):
                caminhoAbs = caminhoPasta + "\\" + filename
                f.write(caminhoAbs + ";" + str(label) + "\n")
            label = label + 1
    f.close()

def criaDicionarioDeImagens(fPoint):
    lines = fPoint.readlines()

    dicionarioDeFotos = {}
    for line in lines:
        filename, label = line.rstrip().split(';')
        if int(label) in dicionarioDeFotos.keys():
            dicionarioDeFotos.get(int(label)).append(cv2.imread(filename, 0))
        else:
            dicionarioDeFotos[int(label)] = [cv2.imread(filename, 0)]

    return dicionarioDeFotos

def treinaModelo(dicionarioDePessoas):
    model = cv2.face.EigenFaceRecognizer_create()
    listkey = []
    listvalue = []
    for key in dicionarioDePessoas.keys():
       for value in dicionarioDePessoas[key]:
           listkey.append(key)
           listvalue.append(value)

    model.train(np.array(listvalue), np.array(listkey))
    return model

def reconheceImagem(modelo,path):
    face_cascade = cv2.CascadeClassifier('haar.xml')
    img = cv2.imread(path)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        rosto = img[y:y + h, x:x + w]

        if w > 100 and h > 100:
            rosto = cv2.resize(rosto, (255, 255))
            rosto = equaliza(rosto)
            rosto = cv2.cvtColor(rosto, cv2.COLOR_BGR2GRAY)

            label = modelo.predict(rosto)

            font = cv2.FONT_HERSHEY_SIMPLEX
            if (label[0] == 0):
                cv2.putText(img, 'Alan', (x - 20, y + h + 60), font, 3, (255, 0, 0), 5, cv2.LINE_AA)
                img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

            if (label[0] == 1):
                cv2.putText(img, 'Jonata', (x - 20, y + h + 60), font, 3, (0, 0, 255), 5, cv2.LINE_AA)
                img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

            if (label[0] == 2):
                cv2.putText(img, 'Lorena', (x - 20, y + h + 60), font, 3, (255, 0, 255), 5, cv2.LINE_AA)
                img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)

    img = cv2.resize(img, (int(0.75 * img.shape[1]), int(0.75 * img.shape[0])))

    cv2.imshow("reconhecimento", img)
    cv2.waitKey(0)

def main():

    opc = '-'
    while(opc!='3'):

        opc = input('1- Inserir alguem na base \n2- Realizar reconhecimento \n3- Sair \nOpção :')

        if(opc=='1'):
            geraBase()
        if(opc=='2'):

            image_path = input('Digite o nome do arquivo da imagemcom sua extensão(lembre-se que a imagem deve estar na mesma pasta deste código): ')
            #image_path = 'alanlorena.jpeg'  # imagem

            criaArquivoDeRotulo("images")
            fPoint = open("TRAIN", "r")

            dicionarioDeFotos = criaDicionarioDeImagens(fPoint)
            modelo = treinaModelo(dicionarioDeFotos)

            reconheceImagem(modelo, str('figs/' + image_path))
#-------------------------------------------------------------

if __name__ == "__main__":
    main()